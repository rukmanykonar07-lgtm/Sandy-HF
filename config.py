"""
Sandy's live-editable config: LLM credit caps + preferences.
Stored as simple key/value rows in Supabase (table: sandy_config).
Sandy reads/writes this via chat — no files to open, no redeploy.

One-time setup (run once in Supabase SQL editor):

    create table sandy_config (
        key text primary key,
        value jsonb not null,
        updated_at timestamptz default now()
    );

ponytail: a flat key/value table, not a rules engine. Add structure
only when a real need for it shows up (e.g. per-user caps).

--- in-process cache (added) -------------------------------------------
Every call_llm() call was doing 3 Supabase round-trips just for cap
bookkeeping: get_config("caps"), get_config("usage"), set_config("usage",
...). On a multi-call tier (judge rounds, orchestrator rounds) that's
3x N network hops before any actual LLM work happens.

This is safe to cache in-process with NO ttl (not "cache for 30s and
hope") because sandy_config has exactly one writer: this module, in
this one process. Dockerfile runs a single uvicorn process (no
--workers), so there is no second process that could mutate the table
out from under this cache. Every write path below goes through
set_config/delete_config, which update the cache in the same breath --
so the cache literally cannot go stale while this process is alive.
If Sandy is ever scaled to multiple replicas/workers, this cache
becomes wrong and needs to switch to a real ttl or a pub/sub
invalidation -- flagging that assumption explicitly here so it isn't
forgotten later.

atomic_update() also fixes a real (pre-existing, not introduced by
this cache) lost-update race: two concurrent requests (FastAPI's
threadpool makes overlapping /chat calls genuinely concurrent, not just
async-interleaved) could both read usage, both increment their own
local copy, then both write -- the second write silently clobbers the
first's increment, undercounting real usage against the cap. A single
process-wide lock around the whole read-check-write closes this for
free, because everything is one process.
"""
import json
import os
import threading
from pathlib import Path

from supabase import Client

import bootstrap

_cache: dict = {}
_cache_lock = threading.Lock()
_cap_lock = threading.Lock()  # separate lock: guards the cap-bump critical section, not just cache access


def _db() -> Client:
    """Kept as a thin alias to bootstrap's shared client -- every other
    module in this file already calls _db(), so this is the smallest
    possible change that removes config.py's own separate client copy
    without touching every call site below."""
    return bootstrap.get_supabase_client()


DEFAULTS = {
    "caps": {"gemini": None, "groq": None, "cerebras": None},  # None = no cap
}


def get_config(key: str):
    """Read one config key. Falls back to DEFAULTS if never set."""
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    res = _db().table("sandy_config").select("value").eq("key", key).execute()
    value = res.data[0]["value"] if res.data else DEFAULTS.get(key)
    with _cache_lock:
        _cache[key] = value
    return value


def set_config(key: str, value) -> None:
    """Write/overwrite one config key. This is what Sandy calls when
    Ruk says 'change the Gemini cap to 100' etc.

    NEVER call this with value=None to "clear" a key -- sandy_config's
    real schema has `value jsonb NOT NULL` (see the table definition
    above); this confirmed crashes live with Postgres error 23502 (null
    value in column "value" violates not-null constraint). Use
    delete_config() below to actually remove a key."""
    if value is None:
        raise ValueError("set_config(key, None) is not allowed -- sandy_config.value is NOT NULL. Use delete_config(key) to clear a key.")
    _db().table("sandy_config").upsert({"key": key, "value": value}).execute()
    with _cache_lock:
        _cache[key] = value


def delete_config(key: str) -> None:
    """Actually removes the row -- the correct way to clear a key.
    set_config(key, None) looks like it should mean the same thing but
    silently violates the real NOT NULL constraint on `value` instead."""
    _db().table("sandy_config").delete().eq("key", key).execute()
    with _cache_lock:
        _cache.pop(key, None)


def get_all_config() -> dict:
    res = _db().table("sandy_config").select("key,value").execute()
    stored = {row["key"]: row["value"] for row in res.data}
    with _cache_lock:
        _cache.update(stored)
    return {**DEFAULTS, **stored}


def atomic_update(key: str, mutate):
    """Generic read-modify-write under one process-wide lock: reads
    `key` (cache-hit in the common case, so effectively free), calls
    mutate(current_value) -> new_value, persists new_value, returns it.

    This is the fix for the 3-Supabase-round-trips-per-LLM-call cost
    AND the lost-update race described in the module docstring above --
    but it's deliberately generic (doesn't know about "caps" or "daily
    reset") so that domain logic (llm.py's date-rollover rule, what
    counts as "exceeded") stays in llm.py where it belongs. config.py
    owns storage + atomicity, not cap policy.
    """
    with _cap_lock:
        current = get_config(key)
        new_value = mutate(current)
        set_config(key, new_value)
        return new_value


def get_client() -> Client:
    """Real Supabase client, for modules that own their own proper
    relational tables (native_mastery.py, healing.py) instead of the
    generic sandy_config key-value store. Public wrapper around _db() --
    other modules shouldn't reach into a "_private" function.

    Also now the ONE function every other module's DB access routes
    through (chatlog.py, events.py, projects.py, healing.py all call
    this instead of keeping their own client) -- which means the
    existing `fake_db` pytest fixture, which patches exactly this
    function, now actually isolates those modules from the network too
    instead of silently missing them."""
    return _db()


def reset_cache_for_tests() -> None:
    """Test-only escape hatch. The module docstring's "provably
    correct" claim for the cache rests on one real Supabase client
    living for the whole process -- true in production (bootstrap's
    client is a genuine singleton), but NOT true across pytest tests
    that each construct a fresh FakeSupabaseClient and monkeypatch
    config._db() directly (test_config.py's real_config_over_fake_db
    fixture does exactly this). Confirmed by reproducing it directly:
    without this reset, a value cached against test A's fake client
    was still being served to test B's completely fresh, empty fake
    client. Any fixture that swaps config._db() must call this too, or
    it inherits the previous test's cached values."""
    global _cache
    with _cache_lock:
        _cache = {}


if __name__ == "__main__":
    # ponytail self-check: not a real network test (needs live Supabase),
    # just confirms defaults shape is sane before anything imports this.
    assert set(DEFAULTS["caps"].keys()) == {"gemini", "groq", "cerebras"}
    print("config.py: defaults OK")


# --- Hermes cron jobs.json persistence -------------------------------
# HF Spaces' filesystem is wiped on every rebuild (including Sandy's own
# self-mod pushes), and Hermes has no persistence of its own -- mastery
# jobs (registered via cron.jobs.create_job, stored at
# ~/.hermes/cron/jobs.json) were silently lost on every rebuild.
#
# scode: this reuses the sandy_config table that already exists for
# caps -- one extra row, not a new Supabase Storage bucket or a new
# credential. jobs.json is a small flat JSON file (schedules + skill
# names, not the actual run history), so storing its whole content as
# one jsonb value is the lazy-and-correct move here. Job *output*
# (progress notes in ~/.hermes/cron/output/) is NOT covered by this --
# that's runtime history, not the registration Sandy needs to keep
# actually running a job, and it's a bigger sync problem for another day.
JOBS_PATH = Path(os.environ.get("HERMES_HOME", "/root/.hermes")) / "cron" / "jobs.json"
_JOBS_BACKUP_KEY = "hermes_jobs_backup"


def backup_hermes_jobs() -> None:
    """Call this right after any mastery job is created/changed so the
    current jobs.json is saved before the next rebuild can wipe it."""
    if not JOBS_PATH.exists():
        return
    set_config(_JOBS_BACKUP_KEY, json.loads(JOBS_PATH.read_text()))


def restore_hermes_jobs() -> None:
    """Call this once at container startup, BEFORE the Hermes gateway
    starts (it reads jobs.json on its own startup) -- writes the last
    backup back to disk if jobs.json doesn't already exist locally."""
    if JOBS_PATH.exists():
        return  # real jobs.json already on disk, don't clobber it
    backup = get_config(_JOBS_BACKUP_KEY)
    if not backup:
        return  # nothing to restore, first-ever boot
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(backup, indent=2))
