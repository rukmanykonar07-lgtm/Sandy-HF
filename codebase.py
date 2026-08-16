"""Read-only whole-codebase access -- lets Sandy actually answer "scan
your code and tell me what's wrong" instead of admitting she can't.

No approval needed here (read-only, changes nothing) -- unlike
selfmod.py's propose->approve->push flow, which exists specifically
because THAT one writes files and pushes to git.
"""
import os

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
    """Reads every file in the repo for real and asks an LLM to answer
    Ruk's specific request against the actual current code -- not a
    guess, not a summary of what a codebase like this might contain."""
    files = list_files()
    parts = [f"=== {p} ===\n{read_file(p)}" for p in files if read_file(p) is not None]
    combined = "\n\n".join(parts)

    prompt = (
        f"Here is Sandy's entire current codebase ({len(files)} files):\n\n"
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
