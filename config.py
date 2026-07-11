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
import os
from supabase import create_client, Client

_client: Client | None = None


def _db() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


DEFAULTS = {
    "caps": {"gemini": None, "groq": None, "cerebras": None},  # None = no cap
    "approval_required_for": [],   # task descriptors user has opted IN to approval for
    "always_ask_approval": False,  # global override, default off
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
    assert isinstance(DEFAULTS["always_ask_approval"], bool)
    print("config.py: defaults OK")
