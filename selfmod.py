"""Self-modification: Sandy can propose and -- only after Ruk explicitly
approves in chat -- apply a code change to her own repo, then push it so
HF rebuilds. Never triggered by Sandy on her own initiative; always
starts from Ruk explicitly asking for a specific change.

Also gives Ruk /chat visibility into edit history and rollback (git log
/ git revert), so a bad push can be undone from chat, not just by hand.
"""
import ast
import difflib
import json
import os
import subprocess

from llm import call_llm_with_fallback, strip_fence, log

REPO_DIR = "/app"

# session_id -> pending proposal, so what Ruk approves is EXACTLY what
# gets applied (an LLM regenerating the diff on the approval turn could
# produce something slightly different -- this avoids that mismatch).
_pending: dict[str, dict] = {}

_PERSIST_KEY = "selfmod_pending"      # sandy_config mirror for restart durability
_MAX_DIFF_LINES = 400                 # sanity cap: a "small edit" proposal bigger
                                      # than this is suspicious -- reject at the gate


def _persist_pending() -> None:
    """Mirror _pending into sandy_config so a Space restart (which wipes
    process memory) doesn't orphan an approved-but-unapplied edit. Best
    effort: if Supabase blips the in-memory copy still works this boot."""
    try:
        import config
        config.set_config(_PERSIST_KEY, json.dumps(_pending))
    except Exception as e:
        log(f"[selfmod] pending-mirror write failed (non-fatal): {e!r}")


def _load_persisted_pending() -> None:
    """Boot-time restore of proposals saved before a restart. Called once
    from main's lifespan; failures leave _pending empty and non-fatal."""
    if _pending:
        return
    try:
        import config
        raw = config.get_config(_PERSIST_KEY)
        if raw:
            _pending.update(json.loads(raw))
            log(f"[selfmod] restored {len(_pending)} pending proposal(s) from sandy_config")
    except Exception as e:
        log(f"[selfmod] pending-mirror read failed (non-fatal): {e!r}")


def _clear_persisted_pending() -> None:
    try:
        import config
        config.delete_config(_PERSIST_KEY)
    except Exception as e:
        log(f"[selfmod] pending-mirror clear failed (non-fatal): {e!r}")


def _risk_label(diff: str) -> str:
    """Purely informational tag so Ruk can eyeball blast radius. 'high'
    means the edit touches control flow of risky modules or is large;
    it does NOT change the approval requirement -- every apply still
    needs his explicit yes."""
    n_lines = len(diff.splitlines())
    risky_markers = ("subprocess", "os.system", "eval(", "exec(",
                     "_run_git", "HF_WRITE_TOKEN")
    touches_risky = any(m in diff for m in risky_markers)
    return "high" if (touches_risky or n_lines > 100) else "low"


class GitOpError(Exception):
    pass


def ensure_git_ready() -> None:
    """One-time-per-call safety: mark /app as a safe git directory (newer
    git refuses to operate on repos it doesn't own otherwise), and fail
    clearly + early if there's no .git here at all, instead of a
    confusing error three steps later."""
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", REPO_DIR],
        capture_output=True, text=True, timeout=10,
    )
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        raise GitOpError(
            "No .git found in the running container -- self-modification "
            "can't push from here. This needs checking: does the Docker "
            "build actually copy .git in, or does it need to be cloned "
            "fresh at container startup instead?"
        )


def _run_git(*args: str) -> str:
    """Runs git with the HF write token supplied via GIT_ASKPASS instead
    of embedding it in a remote URL: a URL lands in /proc/<pid>/cmdline
    (readable by every process in this container) for as long as git
    runs; an askpass helper only ever exists inside this process's own
    environment. Error scrubbing stays as defense-in-depth."""
    result = subprocess.run(
        ["git", "-C", REPO_DIR, *args],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **_git_auth_env()},
    )
    if result.returncode != 0:
        err = result.stderr.strip() or f"git {args[0]} failed"
        token = os.environ.get("HF_WRITE_TOKEN")
        if token:
            err = err.replace(token, "***")
        raise GitOpError(err)
    return result.stdout.strip()


_ASKPASS_PATH = os.path.join("/tmp", "sandy_askpass.sh")


def _git_auth_env() -> dict[str, str]:
    """Writes a tiny askpass script (once per process) that echoes
    HF_WRITE_TOKEN, and returns the env vars pointing git at it."""
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise GitOpError("HF_WRITE_TOKEN not set -- can't talk to origin")
    if not os.path.exists(_ASKPASS_PATH):
        with open(_ASKPASS_PATH, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho \"$HF_WRITE_TOKEN\"\n")
        os.chmod(_ASKPASS_PATH, 0o700)
    return {
        "GIT_ASKPASS": _ASKPASS_PATH,
        "GIT_USERNAME": "oauth",
        # Remote used for fetch/push must be plain (no token in it).
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "",
    }


def _origin_url() -> str:
    return "https://huggingface.co/spaces/Rukmany/RuksHome"


def _assert_up_to_date() -> None:
    """Refuse to edit/push on a stale checkout -- if origin has moved
    since this container was built (e.g. you pushed something by hand),
    don't risk a confusing push. Tell Ruk to redeploy first instead.

    Fetches via the authenticated URL (not the plain 'origin' remote,
    which has no credentials in the container) -- fetching by explicit
    URL doesn't update the origin/main tracking ref, so FETCH_HEAD is
    what actually holds the result here."""
    _run_git("fetch", "origin", "main")
    if _run_git("rev-parse", "HEAD") != _run_git("rev-parse", "FETCH_HEAD"):
        raise GitOpError(
            "Ruk, is container ka code origin/main se peeche hai (shayad "
            "kahin aur se push hua hai). Pehle Space ko restart/redeploy "
            "karo, phir dobara try karo."
        )


def _contained_path(file_path: str) -> str | None:
    """Repo-rooted absolute path, or None if file_path tries to escape
    the repo (path traversal). Same guard as codebase.read_file --
    selfmod WRITES files, so it needs this even more."""
    full = os.path.normpath(os.path.join(REPO_DIR, file_path))
    if not full.startswith(os.path.normpath(REPO_DIR) + os.sep):
        return None
    return full


def propose_edit(session_id: str, file_path: str, instruction: str) -> str:
    """Step 1: generates the proposed new file content and shows a diff
    -- writes NOTHING to disk yet, commits nothing, pushes nothing."""
    full_path = _contained_path(file_path)
    if full_path is None:
        return f"Ruk, {file_path} repo ke bahar ja raha hai -- sirf apne repo ki files edit kar sakti hoon."
    if not os.path.isfile(full_path):
        return f"Ruk, {file_path} naam ki file nahi mili repo mein."

    with open(full_path, encoding="utf-8") as f:
        old_content = f.read()

    gen_prompt = (
        f"Here is the current content of {file_path}:\n\n{old_content}\n\n"
        f"Apply this change: {instruction}\n\n"
        "BEFORE writing the new content, think through for real (briefly, in your own "
        "reasoning, not in the output): is this change actually necessary; does it break "
        "anything else in this file or files that import from it; is the thing being "
        "'fixed' actually broken (verify against the real code above, don't assume); does "
        "this match what Ruk actually asked for, not a guessed-at version of it; would "
        "enhancing something already here be better than adding something new; will this "
        "really accomplish the instruction, not just look like it does; could it introduce "
        "a new logic error; and how will Ruk actually use this once it's live, what could go "
        "wrong for him. Only after that reasoning, apply the change.\n\n"
        "Respond with ONLY the complete new file content -- no explanation, "
        "no markdown fences, just the raw file content."
    )
    new_content = strip_fence(call_llm_with_fallback("gemini", [{"role": "user", "content": gen_prompt}]))

    if file_path.endswith(".py"):
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            return (
                f"Ruk, generated edit mein Python syntax error hai (line {e.lineno}: {e.msg}) "
                "-- proposal reject kar di, kuch push nahi hua. Instruction thoda aur specific "
                "de ke phir try karo."
            )

    diff = "\n".join(difflib.unified_diff(
        old_content.splitlines(), new_content.splitlines(),
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="",
    ))
    if not diff:
        return f"Ruk, koi actual change nahi bana {file_path} mein us instruction se."
    if len(diff.splitlines()) > _MAX_DIFF_LINES:
        return (
            f"Ruk, ye proposal {_MAX_DIFF_LINES} lines se bada diff ban raha hai "
            f"({len(diff.splitlines())}) -- itna bada 'chhota edit' galat direction "
            "mein jaa raha hai. Instruction ko chhote steps mein tod ke try karo. "
            "Kuch pending nahi hua."
        )

    _pending[session_id] = {
        "file_path": file_path,
        "new_content": new_content,
        "commit_message": f"Sandy self-edit: {instruction[:72]}",
    }
    _persist_pending()
    risk = _risk_label(diff)
    risk_note = ("(risk: high -- risky APIs ya bada diff, dhyan se padhna)"
                 if risk == "high" else "(risk: low)")
    return (f"Proposed change to {file_path} {risk_note}:\n\n{diff}\n\n"
            "Confirm karoge to apply + push kar dungi.")


def apply_pending(session_id: str) -> str:
    """Step 2: ONLY called after Ruk has explicitly approved in chat.
    Applies EXACTLY what was proposed (not a freshly regenerated diff),
    commits, and pushes -- HF rebuilds automatically from there."""
    pending = _pending.pop(session_id, None)
    if not pending:
        return "Ruk, koi pending edit nahi hai confirm karne ke liye."

    ensure_git_ready()
    _assert_up_to_date()

    full_path = _contained_path(pending["file_path"])
    if full_path is None:
        return "Ruk, ye pending edit repo ke bahar likhta -- reject kar diya."
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(pending["new_content"])

    _run_git("add", pending["file_path"])
    _run_git("-c", "user.email=sandy@ruks-home.local", "-c", "user.name=Sandy",
              "commit", "-m", pending["commit_message"])
    _run_git("push", "origin", "main")
    _clear_persisted_pending()
    return f"Done, Ruk — {pending['file_path']} push ho gaya, HF rebuild ho raha hai ab."


def recent_history(limit: int = 10) -> str:
    ensure_git_ready()
    log = _run_git("log", f"-{limit}", "--pretty=%h  %ad  %s", "--date=short")
    return log or "Koi edit history nahi hai abhi tak."


def rollback_to(commit_hash: str) -> str:
    """Reverts (not hard-resets) so history stays honest -- what was
    undone is still visible and re-doable later, same as a normal
    git revert."""
    ensure_git_ready()
    _assert_up_to_date()
    try:
        _run_git("revert", "--no-edit", commit_hash)
    except GitOpError:
        # A conflicting revert leaves the repo mid-operation -- every
        # future git call would break until someone cleans this up by
        # hand. Abort it ourselves so the repo stays usable, and give
        # Ruk an honest reason instead of a raw git error.
        _run_git("revert", "--abort")
        return (
            f"Ruk, {commit_hash} ko cleanly revert nahi kar payi — baad ke "
            "commits usी jagah ko phir se change kar chuke hain, conflict "
            "aa raha hai. Manually resolve karna padegा, auto-revert safe nahi hai yahan."
        )
    _run_git("push", "origin", "main")
    return f"Done, Ruk — {commit_hash} revert ho gaya, redeploy ho raha hai."
