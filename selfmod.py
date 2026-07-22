"""Self-modification: Sandy can propose and -- only after Ruk explicitly
approves in chat -- apply a code change to her own repo, then push it so
HF rebuilds. Never triggered by Sandy on her own initiative; always
starts from Ruk explicitly asking for a specific change.

Also gives Ruk /chat visibility into edit history and rollback (git log
/ git revert), so a bad push can be undone from chat, not just by hand.
"""
import base64
import difflib
import json
import os
import subprocess

from llm import call_llm_with_fallback

REPO_DIR = "/app"

# session_id -> pending proposal, so what Ruk approves is EXACTLY what
# gets applied (an LLM regenerating the diff on the approval turn could
# produce something slightly different -- this avoids that mismatch).
_pending: dict[str, dict] = {}


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
    result = subprocess.run(
        ["git", "-C", REPO_DIR, *args],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise GitOpError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def _auth_header() -> str:
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise GitOpError("HF_WRITE_TOKEN not set -- can't push")
    basic = base64.b64encode(f"user:{token}".encode()).decode()
    return f"AUTHORIZATION: basic {basic}"


def _assert_up_to_date() -> None:
    """Refuse to edit/push on a stale checkout -- if origin has moved
    since this container was built (e.g. you pushed something by hand),
    don't risk a confusing push. Tell Ruk to redeploy first instead."""
    _run_git("fetch", "origin", "main")
    if _run_git("rev-parse", "HEAD") != _run_git("rev-parse", "origin/main"):
        raise GitOpError(
            "Ruk, is container ka code origin/main se peeche hai (shayad "
            "kahin aur se push hua hai). Pehle Space ko restart/redeploy "
            "karo, phir dobara try karo."
        )


def extract_selfmod_request(message: str) -> dict | None:
    """Classifies a chat message into one of: edit a file, view recent
    history, or roll back a specific commit. Returns None if the message
    isn't any of these."""
    prompt = (
        "Does this message ask Sandy to (a) edit/change her own code, "
        "(b) show recent code-change history, or (c) undo/rollback a "
        "specific past change? \n"
        f'Message: "{message}"\n'
        "Respond with ONLY one of these JSON shapes:\n"
        '{"action": "edit", "file_path": "...", "instruction": "..."}\n'
        '{"action": "history"}\n'
        '{"action": "rollback", "commit_hash": "..."}\n'
        '{"action": "none"}'
    )
    try:
        raw = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}])
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None if data.get("action") == "none" else data


def propose_edit(session_id: str, file_path: str, instruction: str) -> str:
    """Step 1: generates the proposed new file content and shows a diff
    -- writes NOTHING to disk yet, commits nothing, pushes nothing."""
    full_path = os.path.join(REPO_DIR, file_path)
    if not os.path.isfile(full_path):
        return f"Ruk, {file_path} naam ki file nahi mili repo mein."

    with open(full_path, encoding="utf-8") as f:
        old_content = f.read()

    gen_prompt = (
        f"Here is the current content of {file_path}:\n\n{old_content}\n\n"
        f"Apply this change: {instruction}\n\n"
        "Respond with ONLY the complete new file content -- no explanation, "
        "no markdown fences, just the raw file content."
    )
    new_content = call_llm_with_fallback("gemini", [{"role": "user", "content": gen_prompt}])

    diff = "\n".join(difflib.unified_diff(
        old_content.splitlines(), new_content.splitlines(),
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="",
    ))
    if not diff:
        return f"Ruk, koi actual change nahi bana {file_path} mein us instruction se."

    _pending[session_id] = {
        "file_path": file_path,
        "new_content": new_content,
        "commit_message": f"Sandy self-edit: {instruction[:72]}",
    }
    return f"Proposed change to {file_path}:\n\n{diff}\n\nConfirm karoge to apply + push kar dungi."


def apply_pending(session_id: str) -> str:
    """Step 2: ONLY called after Ruk has explicitly approved in chat.
    Applies EXACTLY what was proposed (not a freshly regenerated diff),
    commits, and pushes -- HF rebuilds automatically from there."""
    pending = _pending.pop(session_id, None)
    if not pending:
        return "Ruk, koi pending edit nahi hai confirm karne ke liye."

    ensure_git_ready()
    _assert_up_to_date()

    full_path = os.path.join(REPO_DIR, pending["file_path"])
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(pending["new_content"])

    _run_git("add", pending["file_path"])
    _run_git("-c", "user.email=sandy@ruks-home.local", "-c", "user.name=Sandy",
              "commit", "-m", pending["commit_message"])
    _run_git("-c", f"http.extraheader={_auth_header()}", "push", "origin", "main")
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
    _run_git("-c", f"http.extraheader={_auth_header()}", "push", "origin", "main")
    return f"Done, Ruk — {commit_hash} revert ho gaya, redeploy ho raha hai."
