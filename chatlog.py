"""Raw, verbatim chat log — separate from memory.py's Mem0 fact-extraction.

Mem0 extracts facts from messages (great for Sandy recalling things about
Ruk), but it doesn't preserve exact message text in order. This does --
purely so Ruk's Home can show the real word-for-word conversation on
page refresh. Called automatically alongside memory.remember(), same as
memory itself: no one ever has to ask Sandy to log anything.
"""
import os
import re
from datetime import date, datetime, timedelta, timezone

from supabase import create_client

_N_DAYS_AGO_RE = re.compile(r"(\d+)\s*din\s*pehle|(\d+)\s*days?\s*ago", re.IGNORECASE)
_LAST_WEEK_RE = re.compile(r"\blast week\b|\bpichhle hafte\b|\bpichle hafte\b", re.IGNORECASE)
_THIS_WEEK_RE = re.compile(r"\bthis week\b|\bis hafte\b", re.IGNORECASE)
_RELATIVE_DAY_PATTERNS = [
    (r"\bday before yesterday\b|\bparso\b", 2),
    (r"\byesterday\b", 1),
]


def extract_date_range(message: str) -> tuple[str, str] | None:
    """Detects an explicit relative-date reference (yesterday, 3 days
    ago, last week, etc) and returns (start_iso, end_iso) for that
    period, or None if there's no clear date reference. Deterministic
    keyword matching, not an LLM call -- there are already 6 classifier
    calls per message, this stays cheap and fast rather than adding a 7th.
    Note: doesn't handle ambiguous Hindi words like "kal" (can mean
    yesterday OR tomorrow) -- only unambiguous phrases."""
    lower = message.lower()
    today = date.today()

    m = _N_DAYS_AGO_RE.search(lower)
    if m:
        n = int(m.group(1) or m.group(2))
        d = today - timedelta(days=n)
        return (d.isoformat(), (d + timedelta(days=1)).isoformat())

    if _LAST_WEEK_RE.search(lower):
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=7)
        return (start.isoformat(), end.isoformat())

    if _THIS_WEEK_RE.search(lower):
        start = today - timedelta(days=today.weekday())
        return (start.isoformat(), (today + timedelta(days=1)).isoformat())

    for pattern, days_back in _RELATIVE_DAY_PATTERNS:
        if re.search(pattern, lower):
            d = today - timedelta(days=days_back)
            return (d.isoformat(), (d + timedelta(days=1)).isoformat())

    return None

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


def get_history_in_range(start_date: str, end_date: str) -> list[dict]:
    """Messages between start_date (inclusive) and end_date (exclusive),
    both ISO date strings ('2026-08-01'). For explicit date-referenced
    questions ('what did we do yesterday') -- separate from
    get_history()'s plain recent-N-messages."""
    try:
        res = (
            _c()
            .table("chat_log")
            .select("role, message, created_at")
            .gte("created_at", start_date)
            .lt("created_at", end_date)
            .order("created_at", desc=False)
            .execute()
        )
        return res.data
    except Exception:
        return []


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
