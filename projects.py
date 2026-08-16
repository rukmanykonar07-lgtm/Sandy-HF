"""
Sandy's Projects: real work tracking -- Ruk's own client/personal
projects AND reviewed coding-bounty work (Algora, Bountysource, Opire
-- see the handoff notes on why gig-labor-marketplace autonomy is
explicitly out of scope; this is for real, human-reviewed work).
Replaces the honest "Knowledge Base" placeholder.

One-time setup (run once in Supabase SQL editor):

    create table projects (
        id uuid primary key default gen_random_uuid(),
        name text not null,
        type text not null default 'personal',  -- 'personal' | 'bounty'
        description text,
        model_limit jsonb,
        status text default 'active',           -- 'active' | 'paused' | 'done'
        trusted_submissions int default 0,
        requires_approval boolean default true,
        created_at timestamptz default now()
    );

    create table project_events (
        id uuid primary key default gen_random_uuid(),
        project_id uuid references projects(id),
        event_type text not null,   -- 'action' | 'response' | 'payment' | 'alert'
        content text,
        payment_status text,        -- 'paid' | 'unpaid' | null
        created_at timestamptz default now()
    );

Also needs (new secret, not yet in HF): RUK_WHATSAPP_NUMBER -- Ruk's own
number to send alerts to. WHATSAPP_TOKEN/WHATSAPP_PHONE_NUMBER_ID were
already sitting in HF secrets unused (Ruk's Home replaced WhatsApp as
the primary interface a while back) -- reused here for alerts only,
not as a chat interface.
"""
import os

import requests
from supabase import create_client, Client

APPROVAL_GRADUATION_THRESHOLD = 3  # after this many Ruk-approved submissions on a
                                    # project, stop asking -- exactly what Ruk asked for

_client: Client | None = None


def _db() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def _default_model_limit(project_type: str) -> dict:
    # ponytail: bounty work tends to need more reasoning (review,
    # correctness matters more since a stranger judges the output) --
    # personal projects get a lighter default budget.
    return (
        {"groq": 60, "gemini": 30, "cerebras": 20}
        if project_type == "bounty"
        else {"groq": 30, "gemini": 15, "cerebras": 10}
    )


def create_project(name: str, project_type: str, description: str = "", model_limit: dict | None = None) -> dict:
    """project_type: 'personal' or 'bounty'. model_limit auto-set if not
    given -- 'if no setted sandy will set that automatically' per Ruk's spec."""
    if model_limit is None:
        model_limit = _default_model_limit(project_type)
    row = {
        "name": name,
        "type": project_type,
        "description": description,
        "model_limit": model_limit,
        "status": "active",
        "trusted_submissions": 0,
        "requires_approval": project_type == "bounty",
    }
    return _db().table("projects").insert(row).execute().data[0]


def list_projects(status: str | None = None) -> list[dict]:
    q = _db().table("projects").select("*")
    if status:
        q = q.eq("status", status)
    return q.order("created_at", desc=True).execute().data


def get_project(project_id: str) -> dict | None:
    result = _db().table("projects").select("*").eq("id", project_id).execute()
    return result.data[0] if result.data else None


def log_event(project_id: str, event_type: str, content: str, payment_status: str | None = None) -> None:
    """event_type: 'action' (what Sandy did), 'response' (what the
    client/task-poster said back), 'payment' (paid/unpaid update),
    'alert' (limit/notification events). Best-effort -- a logging
    failure should never break the actual task."""
    try:
        _db().table("project_events").insert({
            "project_id": project_id,
            "event_type": event_type,
            "content": content,
            "payment_status": payment_status,
        }).execute()
    except Exception:
        pass


def get_events(project_id: str, limit: int = 100) -> list[dict]:
    return (
        _db().table("project_events").select("*")
        .eq("project_id", project_id).order("created_at", desc=True).limit(limit)
        .execute().data
    )


def _cheapest_fallback(model_limit: dict, exhausted: str) -> str | None:
    order = ["groq", "cerebras", "gemini"]  # roughly fastest/cheapest first
    for p in order:
        if p != exhausted and p in model_limit:
            return p
    return None


def check_limit(project_id: str, provider: str, used_this_project: int) -> tuple[bool, str | None]:
    """Returns (still_within_limit, fallback_provider_or_None). Call
    before spending on a provider for this project. If exhausted:
    alerts Ruk AND returns a cheaper fallback to keep going on until he
    responds -- per Ruk's explicit choice, not a hard stop."""
    project = get_project(project_id)
    if not project or not project.get("model_limit"):
        return True, None
    limit = project["model_limit"].get(provider)
    if limit is None or used_this_project < limit:
        return True, None
    fallback = _cheapest_fallback(project["model_limit"], provider)
    _notify_limit_exhausted(project, provider, fallback)
    return False, fallback


def _notify_limit_exhausted(project: dict, provider: str, fallback: str | None) -> None:
    msg = (
        f"Ruk, project '{project['name']}' ne {provider} ka limit khatam kar diya. "
        + (
            f"Fallback pe chal rahi hu ({fallback}) jab tak tu reply na kare."
            if fallback else
            "Koi fallback available nahi hai is project ke liye, task pause ho gaya."
        )
    )
    log_event(project["id"], "alert", msg)
    send_whatsapp(msg)


def send_whatsapp(message: str) -> bool:
    """Best-effort -- a failed notification should never crash whatever
    triggered it. Uses the WhatsApp Business Cloud API (Meta) -- the
    secrets were already sitting in HF unused."""
    try:
        phone_id = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
        token = os.environ["WHATSAPP_TOKEN"]
        to = os.environ["RUK_WHATSAPP_NUMBER"]
        requests.post(
            f"https://graph.facebook.com/v20.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": message}},
            timeout=10,
        )
        return True
    except Exception:
        return False


def needs_approval(project_id: str) -> bool:
    """First APPROVAL_GRADUATION_THRESHOLD external submissions on a
    bounty project need Ruk's explicit OK; auto-trust after that."""
    project = get_project(project_id)
    if not project or project.get("type") != "bounty":
        return False
    return project.get("trusted_submissions", 0) < APPROVAL_GRADUATION_THRESHOLD


def record_approved_submission(project_id: str) -> None:
    """Call after Ruk approves a submission -- counts toward graduating
    out of the approval requirement for this project."""
    project = get_project(project_id)
    if not project:
        return
    _db().table("projects").update(
        {"trusted_submissions": project.get("trusted_submissions", 0) + 1}
    ).eq("id", project_id).execute()
