"""Raw, verbatim chat log — separate from memory.py's Mem0 fact-extraction.

Mem0 extracts facts from messages (great for Sandy recalling things about
Ruk), but it doesn't preserve exact message text in order. This does --
purely so Ruk's Home can show the real word-for-word conversation on
page refresh. Called automatically alongside memory.remember(), same as
memory itself: no one ever has to ask Sandy to log anything.
"""
import os
from datetime import datetime, timedelta, timezone

from supabase import create_client

_client = None


def _c():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _client


def log(message: str, role: str) -> None:
    """Append one message, and opportunistically prune raw text older
    than 30 days. Mem0 already extracted permanent facts from every
    message the moment it was first logged, so nothing is actually lost
    -- only the verbatim copy ages out. Best-effort -- logging/pruning
    hiccups should never break the actual chat response, which is why
    this only ever runs as a background task (see main.py's _remember())."""
    try:
        _c().table("chat_log").insert({"role": role, "message": message}).execute()
    except Exception:
        pass
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        _c().table("chat_log").delete().lt("created_at", cutoff).execute()
    except Exception:
        pass


def get_history(limit: int = 200) -> list[dict]:
    """Most recent `limit` messages, returned oldest-first (ready to
    render top-to-bottom in the chat UI)."""
    try:
        res = (
            _c()
            .table("chat_log")
            .select("role, message, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(reversed(res.data))
    except Exception:
        return []


if __name__ == "__main__":
    # ponytail self-check: needs live SUPABASE_URL/SUPABASE_SERVICE_KEY
    # and the chat_log table to already exist.
    log("test message from chatlog.py self-check", role="user")
    hist = get_history(limit=5)
    assert any(h["message"] == "test message from chatlog.py self-check" for h in hist), (
        f"logged message didn't come back in history: {hist}"
    )
    print("chatlog.py: log + get_history OK ->", hist[-1])
