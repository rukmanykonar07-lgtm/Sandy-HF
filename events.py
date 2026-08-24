"""Event log for mastery runs -- the real data the orb graph renders.
Every node in the graph is one row here. Nothing gets shown that wasn't
actually logged; this module has no synthesis/narration in it at all,
on purpose (see main.py/native_mastery.py for where LLM explanation of
these events happens, separately, from real stored data).

One-time setup (run once in Supabase SQL editor):

    create table mastery_events (
        id bigint generated always as identity primary key,
        run_id text not null,
        agent text not null,              -- 'sandy' | 'hermes'
        round int not null default 0,
        event_type text not null,         -- planning|worker_call|verify|conflict|retry_similar|synthesis|obstacle|skill_saved|output
        provider text,
        summary text not null,
        detail text,
        parent_event_id bigint,
        related_event_ids bigint[],
        created_at timestamptz default now()
    );
    create index mastery_events_run_idx on mastery_events (run_id);

ponytail: a flat event table, not a graph database -- edges are just
parent_event_id/related_event_ids columns. Good enough for what the
frontend needs (build a force-graph client-side from these rows);
add a real graph store only if this genuinely stops being enough.
"""
import difflib
import uuid
from typing import Any

from supabase import Client

import config


def _db() -> Client:
    """Routes through config.get_client() -- the one real Supabase
    client for the process -- instead of keeping its own copy."""
    return config.get_client()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(
    run_id: str, agent: str, event_type: str, summary: str,
    round: int = 0, provider: str | None = None, detail: str | None = None,
    parent_event_id: int | None = None,
) -> int | None:
    """Real event, written as it happens. Returns the new row's id (used
    as parent_event_id by later related events), or None on failure --
    best-effort: a logging failure should never break the actual mastery
    work happening around it, same principle as memory.remember()."""
    row = {
        "run_id": run_id, "agent": agent, "round": round, "event_type": event_type,
        "provider": provider, "summary": summary, "detail": detail,
        "parent_event_id": parent_event_id,
    }
    try:
        result = _db().table("mastery_events").insert(row).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        from llm import log
        log(f"[events.log_event] failed, continuing without it: {e!r}")
        return None


def get_events(run_id: str) -> list[dict]:
    """Every real event for a run, oldest first -- what the graph
    actually renders. Real data straight from Supabase, not a mock."""
    try:
        result = _db().table("mastery_events").select("*").eq("run_id", run_id).order("id").execute()
        return result.data or []
    except Exception as e:
        from llm import log
        log(f"[events.get_events] failed: {e!r}")
        return []


def link_similar_events(events: list[dict], threshold: float = 0.72) -> None:
    """Deterministic, explainable similarity check across a run's events
    -- flags 'similar work across rounds' WITHOUT an LLM guessing at it.
    difflib.SequenceMatcher on event summaries; no embeddings/vector
    infra needed for this. Updates related_event_ids in place in
    Supabase for any pair that crosses the threshold. Real signal from
    real text, not an invented pattern."""
    from llm import log
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a["round"] == b["round"] or a["event_type"] != b["event_type"]:
                continue  # only cross-round, same-event-type comparisons count as "similar work"
            ratio = difflib.SequenceMatcher(None, a["summary"], b["summary"]).ratio()
            if ratio >= threshold:
                try:
                    existing = a.get("related_event_ids") or []
                    _db().table("mastery_events").update(
                        {"related_event_ids": existing + [b["id"]]}
                    ).eq("id", a["id"]).execute()
                except Exception as e:
                    log(f"[events.link_similar_events] update failed: {e!r}")


def list_runs() -> list[dict]:
    """One row per real mastery run (both agents), for Ruk's Home's run
    picker -- derived from the real event log, not a separate table to
    keep in sync. Real data: the first event of each run_id/agent pair
    (usually a 'planning' event) carries the skill in its summary."""
    from llm import log
    try:
        result = _db().table("mastery_events").select("run_id,agent,event_type,summary,created_at").order("id").execute()
    except Exception as e:
        log(f"[events.list_runs] failed: {e!r}")
        return []
    seen = {}
    for row in result.data or []:
        key = (row["run_id"], row["agent"])
        if key not in seen:
            seen[key] = {"run_id": row["run_id"], "agent": row["agent"], "started_at": row["created_at"], "summary": row["summary"]}
    return list(seen.values())
