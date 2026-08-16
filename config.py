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
"""
import json
import os
from pathlib import Path

from supabase import create_client, Client

_client: Client | None = None


def _db() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        # Backend writes should bypass RLS (it's meant to restrict
        # untrusted client access, not trusted server code) -- use the
        # service_role key if it's set, fall back to the old anon key
        # otherwise so this doesn't break before the new secret exists.
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


DEFAULTS = {
    "caps": {"gemini": None, "groq": None, "cerebras": None},  # None = no cap
}


def get_config(key: str):
    """Read one config key. Falls back to DEFAULTS if never set."""
    res = _db().table("sandy_config").select("value").eq("key", key).execute()
    if res.data:
        return res.data[0]["value"]
    return DEFAULTS.get(key)


def set_config(key: str, value) -> None:
    """Write/overwrite one config key. This is what Sandy calls when
    Ruk says 'change the Gemini cap to 100' etc."""
    _db().table("sandy_config").upsert({"key": key, "value": value}).execute()


def get_all_config() -> dict:
    res = _db().table("sandy_config").select("key,value").execute()
    stored = {row["key"]: row["value"] for row in res.data}
    return {**DEFAULTS, **stored}


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
