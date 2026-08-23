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
from llm import log

_GATEWAY_LOG = "/tmp/hermes_gateway.log"
_GATEWAY_ERR_LOG = "/tmp/hermes_gateway.err.log"

# Real env vars this deployment actually uses (from requirements.txt /
# config.py / llm.py / projects.py) -- presence-checked only, values
# NEVER read or exposed here.
#
# scode: real bug fixed here -- TAVILY/EXA/LINKUP's real HF secret names
# have NO "_API_KEY" suffix (confirmed against Ruk's actual live secrets
# list), unlike every other provider. This function is the FIRST thing
# Sandy checks for any internal diagnostic question -- with the wrong
# names, it reported all three as "MISSING" every single time even
# though they're set and search.py reads them fine, actively misleading
# both Sandy's own diagnosis and Ruk.
_KNOWN_KEYS = (
    "GROQ_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY",
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_KEY", "SUPABASE_DB_CONNECTION_STRING",
    "TAVILY", "EXA", "LINKUP",
    "WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "RUK_WHATSAPP_NUMBER",
)


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
