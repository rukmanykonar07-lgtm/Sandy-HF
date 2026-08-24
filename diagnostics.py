"""Sandy's own internal awareness -- real env-key presence checks and
real git/build state, plus the Hermes gateway's own real log files
(new -- previously only visible via /dev/stdout, invisible to Sandy's
own process). Sandy's OWN process logs are already covered by
codebase.read_recent_logs() (reads /tmp/sandy.log) -- reused here, not
duplicated.

Nothing here is guessed or narrated from memory; every function either
reads a real file/env var this turn or says plainly that it couldn't.

What this does NOT cover, honestly: anything before the current
container boot (logs don't survive a restart), and any process outside
this container entirely.
"""
import os
import subprocess

import codebase
from llm import log, PROVIDER_API_KEY_ENV

_GATEWAY_LOG = "/tmp/hermes_gateway.log"
_GATEWAY_ERR_LOG = "/tmp/hermes_gateway.err.log"

# scode: LLM provider keys now come straight from llm.PROVIDER_API_KEY_ENV --
# one source of truth instead of a second hand-typed list here. That
# second copy is exactly how this ended up checking TAVILY_API_KEY/
# EXA_API_KEY/LINKUP_API_KEY (never-real names) instead of the actual
# TAVILY/EXA/LINKUP secrets search.py reads -- a hand-maintained list
# drifts, an imported one can't. Also widens real provider coverage from
# 3 to all 14 llm.py actually supports, for free.
_SEARCH_AND_INFRA_KEYS = (
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_KEY", "SUPABASE_DB_CONNECTION_STRING",
    "TAVILY", "EXA", "LINKUP",
    "WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "RUK_WHATSAPP_NUMBER",
)
_KNOWN_KEYS = tuple(sorted(set(PROVIDER_API_KEY_ENV.values()) | set(_SEARCH_AND_INFRA_KEYS)))


def _tail(path: str, lines: int = 100) -> str:
    """Real tail -- last N lines of a real file. Empty string (not an
    exception) if the file doesn't exist yet, which is a legitimate
    state (e.g. gateway hasn't logged anything since this boot)."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except Exception as e:
        log(f"[diagnostics] couldn't read {path}: {e!r}")
        return ""


def get_gateway_logs(lines: int = 100) -> str:
    """The Hermes cron subprocess's real recent stdout/stderr -- new:
    previously only ever reached HF's live log viewer via /dev/stdout,
    completely unreadable from Sandy's own process. This is why a
    Hermes-side failure (like the 401 auth traceback) used to be
    invisible to her no matter how she was asked about it."""
    out = _tail(_GATEWAY_LOG, lines)
    err = _tail(_GATEWAY_ERR_LOG, lines)
    return (
        f"--- Hermes gateway stdout ({_GATEWAY_LOG}) ---\n{out or '(empty)'}\n\n"
        f"--- Hermes gateway stderr ({_GATEWAY_ERR_LOG}) ---\n{err or '(empty)'}"
    )


def env_key_matrix() -> dict[str, bool]:
    """Which known keys are SET -- presence only, never the value.
    Real os.environ check, not a guess."""
    return {k: bool(os.environ.get(k)) for k in _KNOWN_KEYS}


def git_push_summary() -> str:
    """Last real commit hash/message/changed files, if git metadata is
    actually present in this deployment. Honest fallback if not --
    never invents a commit that isn't really there."""
    try:
        info = subprocess.run(
            ["git", "log", "-1", "--format=%H%n%ci%n%s"],
            cwd="/app", capture_output=True, text=True, timeout=5,
        )
        if info.returncode != 0 or not info.stdout.strip():
            return "Git metadata nahi mila is deployment mein -- commit history se compare nahi kar sakti."
        commit_hash, commit_date, subject = info.stdout.strip().split("\n", 2)
        changed = subprocess.run(
            ["git", "show", "--stat", "--format=", "HEAD"],
            cwd="/app", capture_output=True, text=True, timeout=5,
        )
        files = changed.stdout.strip() if changed.returncode == 0 else "(file list unavailable)"
        return f"Last commit: {commit_hash[:12]} ({commit_date})\n{subject}\n\nChanged files:\n{files}"
    except Exception as e:
        return f"Git metadata read nahi ho paya: {e!r}"


def inspect_system_health() -> str:
    """The one real call Sandy should make FIRST for any internal
    error/status question -- before ever reaching for web search.
    Combines: her own recent logs (via codebase.read_recent_logs, reused
    not duplicated), the gateway's recent logs (new), and which keys are
    actually present (not their values)."""
    keys = env_key_matrix()
    missing = [k for k, present in keys.items() if not present]
    lines = [
        "=== Env key presence (not values) ===",
        ", ".join(f"{k}={'set' if v else 'MISSING'}" for k, v in keys.items()),
    ]
    if missing:
        lines.append(f"⚠️ Missing: {', '.join(missing)}")
    lines.append("")
    lines.append("=== Sandy's own recent logs ===")
    lines.append(codebase.read_recent_logs(lines=60))
    lines.append("")
    lines.append("=== Hermes gateway's recent logs ===")
    lines.append(get_gateway_logs(lines=60))
    return "\n".join(lines)
