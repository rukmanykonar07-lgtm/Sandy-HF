"""Read-only whole-codebase access -- lets Sandy actually answer "scan
your code and tell me what's wrong" instead of admitting she can't.

No approval needed here (read-only, changes nothing) -- unlike
selfmod.py's propose->approve->push flow, which exists specifically
because THAT one writes files and pushes to git.
"""
import os
import re

from llm import call_llm_with_fallback

REPO_DIR = "/app"

# Extensions worth reading for a code review. Skips binaries, images,
# lockfiles, etc -- nothing a code review needs to see as raw bytes.
_READABLE_EXT = {".py", ".yaml", ".yml", ".json", ".html", ".js", ".md", ".sh", ".txt", ".conf"}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".hermes"}


def list_files() -> list[str]:
    """Repo-relative paths of every readable source file."""
    out = []
    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in _READABLE_EXT:
                out.append(os.path.relpath(os.path.join(root, f), REPO_DIR))
    return sorted(out)


def read_file(rel_path: str) -> str | None:
    """One file's content by its repo-relative path. None if it doesn't
    exist or tries to escape the repo dir (no path traversal)."""
    full = os.path.normpath(os.path.join(REPO_DIR, rel_path))
    if not full.startswith(os.path.normpath(REPO_DIR) + os.sep):
        return None
    if not os.path.isfile(full):
        return None
    with open(full, encoding="utf-8", errors="replace") as f:
        return f.read()


def analyze(instruction: str) -> str:
    """Reads the repo and asks an LLM to answer Ruk's specific request
    against the actual current code -- not a guess, not a summary of
    what a codebase like this might contain.

    scode: real credit-burn bug this fixes -- this used to send EVERY
    file in the repo (main.py alone is 1,300+ lines; the whole repo is
    several hundred KB) into ONE gemini call for every single codebase
    question, even a narrow one like "is there a bug in search.py."
    Gemini is Sandy's smallest free-tier quota, shared with chat/Mem0/
    healing -- one broad-cost call for a narrow question was a real,
    avoidable waste. If the instruction names real file(s) from the repo
    (checked against the actual file list, not guessed), only those get
    sent. A genuinely broad ask ("review everything", "scan your whole
    code") that names no specific file still gets the full repo, exactly
    as before -- that's a real use case, not the bug."""
    files = list_files()

    def _mentions(path: str) -> bool:
        # Word-boundary match: "search.py" matches in "is there a bug in
        # search.py" but NOT inside "research.py" (no boundary before
        # 's' there). Full rel paths ("plugins/foo.py") checked the same
        # way, so a directory prefix can't create false hits either.
        name = re.escape(os.path.basename(path))
        full = re.escape(path)
        return bool(
            re.search(rf"(?<![\w.-]){name}(?![\w-])", instruction)
            or re.search(rf"(?<![\w.-]){full}(?![\w-])", instruction)
        )

    named = [f for f in files if _mentions(f)]
    scoped = named or files
    contents = ((p, read_file(p)) for p in scoped)
    parts = [f"=== {p} ===\n{c}" for p, c in contents if c is not None]
    combined = "\n\n".join(parts)

    scope_label = f"the specific file(s) Ruk named" if named else "Sandy's entire current codebase"
    prompt = (
        f"Here is {scope_label} ({len(scoped)} of {len(files)} files):\n\n"
        f"{combined}\n\n"
        f"Ruk's request: {instruction}\n\n"
        "Answer specifically and concretely, referencing actual file names "
        "and real detail from the code above. Don't pad with generic advice "
        "-- only comment on what's actually there."
    )
    return call_llm_with_fallback("gemini", [{"role": "user", "content": prompt}])


def read_recent_logs(lines: int = 60) -> str:
    """Last N lines of Sandy's own runtime log (errors from failed
    provider calls, /status read failures, etc) -- for when Ruk asks
    what went wrong recently. Written by llm.py's log() helper."""
    try:
        with open("/tmp/sandy.log", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:]) or "(log file exists but is empty -- nothing's gone wrong recently)"
    except FileNotFoundError:
        return "(no log file yet this session -- nothing's been logged since the last restart)"


if __name__ == "__main__":
    files = list_files()
    assert files, "list_files() found nothing -- REPO_DIR wrong or repo empty"
    assert "main.py" in files, f"expected main.py in repo, got: {files}"
    content = read_file("main.py")
    assert content and "FastAPI" in content, "read_file() didn't return real main.py content"
    print(f"codebase.py: found {len(files)} files, read_file() OK ->", files[:5])

    # scode self-check: naming a real file scopes the request down to it;
    # naming nothing keeps the old full-repo behavior. Mirrors analyze()'s
    # real word-boundary matching -- the old substring self-check kept
    # passing while validating logic analyze() no longer uses.
    def _mentions_like_analyze(path: str, instruction: str) -> bool:
        name = re.escape(os.path.basename(path))
        full = re.escape(path)
        return bool(
            re.search(rf"(?<![\w.-]){name}(?![\w-])", instruction)
            or re.search(rf"(?<![\w.-]){full}(?![\w-])", instruction)
        )

    named = [f for f in files if _mentions_like_analyze(f, "is there a bug in search.py")]
    assert named == ["search.py"], f"expected only search.py to match, got: {named}"
    assert not _mentions_like_analyze("search.py", "research.py explains it"), (
        "word boundary broken -- 'research.py' scoped the scan to search.py"
    )
    named_none = [f for f in files if _mentions_like_analyze(f, "review everything please")]
    assert named_none == [], f"expected no file to match a broad request, got: {named_none}"
    print("codebase.py: analyze() scoping logic OK")
