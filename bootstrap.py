"""Composition root -- the ONE place a Supabase client gets constructed.

Before this file existed, config.py, chatlog.py, events.py, and
projects.py each had their own private `_client: Client | None = None`
+ `_db()`/`_c()` lazy-singleton, all doing the exact same
create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY-or-SUPABASE_KEY) dance.
Five copies of the same construction logic means five places that can
drift out of sync (e.g. one forgets the service-key-first fallback), and
five places a test has to separately monkeypatch to fully isolate a test
from the network.

Everything now goes through config.get_client(), which delegates here --
so the existing `fake_db` pytest fixture (which patches
config.get_client) transparently covers chatlog/events/projects too, not
just config.py itself. That's not just deduplication for its own sake:
it's the concrete testability win the repository-pattern skill calls out.

ponytail: still a plain function, not a class-based DI container --
Sandy has exactly one real dependency to wire (Supabase), so a
container would be solving a problem this codebase doesn't have yet.
Revisit if a second swappable backend genuinely shows up.
"""
import os
import threading

from supabase import create_client, Client

_client: Client | None = None
_lock = threading.Lock()


def get_supabase_client() -> Client:
    """Real Supabase client, constructed once, reused everywhere.

    Backend/server code should bypass RLS (RLS exists to restrict
    untrusted client access, not trusted server code) -- uses the
    service_role key if it's set, falls back to the old anon key
    otherwise so this doesn't break before the new secret exists.

    Thread-safe double-checked lock: main.py's /chat handlers, the
    background healing loop, and BackgroundTasks workers can all call
    this concurrently (FastAPI's threadpool for sync routes means this
    is genuinely multi-threaded, not just async-concurrent) -- the bare
    `if _client is None` your other modules used had a benign but real
    race (worst case: two Client objects briefly constructed, one
    discarded) that a lock removes for free.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:  # re-check inside the lock -- another thread may have won the race
                url = os.environ["SUPABASE_URL"]
                key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
                _client = create_client(url, key)
    return _client


def reset_client_for_tests() -> None:
    """Test-only escape hatch -- forces the next get_supabase_client()
    call to construct fresh. Production code never calls this; a real
    client is meant to live for the whole process lifetime."""
    global _client
    with _lock:
        _client = None
