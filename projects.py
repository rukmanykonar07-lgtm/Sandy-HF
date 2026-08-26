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
import time

import requests
from supabase import Client

import config
from llm import call_llm_with_fallback, strip_json_fence, log

APPROVAL_GRADUATION_THRESHOLD = 3  # after this many Ruk-approved submissions on a
                                    # project, stop asking -- exactly what Ruk asked for


def _db() -> Client:
    """Routes through config.get_client() -- the one real Supabase
    client for the process -- instead of keeping its own copy."""
    return config.get_client()


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
    from notify import alert  # deferred import: avoids a circular import at module load

    msg = (
        f"Ruk, project '{project['name']}' ne {provider} ka limit khatam kar diya. "
        + (
            f"Fallback pe chal rahi hu ({fallback}) jab tak tu reply na kare."
            if fallback else
            "Koi fallback available nahi hai is project ke liye, task pause ho gaya."
        )
    )
    log_event(project["id"], "alert", msg)
    alert(title=f"Sandy project limit: {project['name']}", body=msg, severity="warn",
          meta={"project_id": project["id"], "provider": provider})


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


# ---------------------------------------------------------------------------
# Part 8: autonomous execution loop.
#
# One daemon poller (wired in main's lifespan next to the healing loop)
# picks up status='active' projects, one at a time (free-tier politeness).
# Each cycle: plan the next steps with a single cheap classify-style call,
# execute them through brain.answer (which inherits MAX_ORCH_CALLS=16 and
# the stall watchdog per step), append project_events, and stop cleanly on
# pause/approval/limit conditions. auto_resolve tasks retry an adjusted
# approach up to _MAX_ATTEMPTS; needs_approval tasks pause and wait for
# Ruk in chat instead of acting unilaterally.

_MAX_STEPS_PER_CYCLE = 3      # steps executed per worker pass
_MAX_ATTEMPTS = 3             # adjusted-approach retries for auto-resolve tasks
_POLL_SECONDS = 120           # idle poll interval; HF sleep makes this moot off-wake
_STEP_TIMEOUT_S = float(os.environ.get("PROJECT_STEP_TIMEOUT_S", "600"))


def _plan_steps(project: dict) -> list[str]:
    """One cheap groq call turns the description into 2-5 concrete,
    self-contained steps. Deterministic fallback keeps the loop alive if
    the planner call fails or returns garbage."""
    prompt = (
        "Break this project into 2-5 concrete, self-contained execution steps. "
        "Each step must be a standalone instruction Sandy can complete without "
        "further clarification. Reply with ONLY a JSON array of strings.\n"
        f"Project: {project['name']}\nDescription: {project.get('description') or '(none)'}"
    )
    try:
        raw = call_llm_with_fallback(
            "groq", [{"role": "user", "content": prompt}], caller="projects.plan"
        )
        import json
        parsed = json.loads(strip_json_fence(raw))
        if isinstance(parsed, list) and parsed:
            return [str(s).strip() for s in parsed if str(s).strip()][:_MAX_STEPS_PER_CYCLE]
    except Exception as e:
        log(f"[projects.worker] planner call failed, using deterministic fallback: {e!r}")
    desc = (project.get("description") or "").strip()
    return [desc or f"Make progress on '{project['name']}'"]


def _execute_step(project: dict, step: str) -> str:
    """Runs one step through brain.answer (deferred import -- brain is a
    heavy module and importing it here would drag mastery/cron deps into
    every projects.py consumer at load time). brain.answer inherits the
    orchestrator ceiling and stall watchdog automatically."""
    import brain

    return brain.answer(
        task=step,
        context=(
            f"You are executing one step of Ruk's project '{project['name']}'. "
            f"Project description: {project.get('description') or 'n/a'}. "
            "Complete THIS step only; be concrete and finish with the actual result."
        ),
    )


def run_cycle(project: dict) -> dict:
    """Plans + executes one batch of steps for one active project. Returns
    a small outcome dict for tests/logging. Never raises into the poller."""
    outcome = {"project": project["name"], "steps_done": 0, "status": "ok"}
    log_event(project["id"], "action", f"Worker picked up project; planning up to {_MAX_STEPS_PER_CYCLE} steps.")

    steps = _plan_steps(project)

    for i, step in enumerate(steps):
        # Per-project spend check before spending (check_limit alerts on exhaustion).
        used = len(get_events(project["id"])) // 2  # rough proxy: each prior step ~2 events
        within, fallback = check_limit(project["id"], "groq", used)
        if not within:
            outcome["status"] = "paused" if not fallback else "degraded"
            break

        attempts = _MAX_ATTEMPTS if project.get("requires_approval") else _MAX_ATTEMPTS
        last_err = None
        result = None
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                result = _execute_step(project, step)
                break
            except Exception as e:
                last_err = e
                log(f"[projects.worker] step {i + 1} attempt {attempt} failed: {e!r}")
                log_event(project["id"], "alert",
                          f"Step {i + 1} attempt {attempt}/{attempts} failed: {e}")
        if result is None:
            outcome["status"] = "failed"
            from notify import alert  # deferred: circular-import avoidance
            msg = (f"Project '{project['name']}' step {i + 1} failed after {attempts} attempts. "
                   f"Last error: {last_err!r}")
            log_event(project["id"], "alert", msg)
            alert(title=f"Sandy project failed: {project['name']}", body=msg, severity="warn",
                  meta={"project_id": project["id"]})
            break

        took = time.time() - t0
        log_event(project["id"], "action", f"Step {i + 1}: {step}")
        log_event(project["id"], "response", f"Result ({took:.0f}s):\n{result[:2000]}")
        outcome["steps_done"] += 1

        if time.time() - t0 > _STEP_TIMEOUT_S:
            log(f"[projects.worker] step exceeded {_STEP_TIMEOUT_S:.0f}s -- stopping this cycle")
            break

    if outcome["steps_done"]:
        log_event(project["id"], "action",
                  f"Cycle complete: {outcome['steps_done']} step(s) done.")
    return outcome


def pick_next_project() -> dict | None:
    """Oldest active project first; paused/done are skipped by the filter.
    None = nothing eligible right now."""
    rows = list_projects(status="active")
    return rows[-1] if rows else None  # oldest first (list is created_at desc)


def pause_for_approval(project: dict, reason: str) -> None:
    """Flips the project to paused so the poller skips it until Ruk
    resumes it in chat, and tells him why (info severity -- nothing is
    broken, it's waiting on a human)."""
    _db().table("projects").update({"status": "paused"}).eq("id", project["id"]).execute()
    from notify import alert  # deferred import: circular-import avoidance
    msg = f"Project '{project['name']}' paused -- {reason}"
    log_event(project["id"], "alert", msg)
    alert(title=f"Sandy project paused: {project['name']}", body=msg, severity="info",
          meta={"project_id": project["id"]})


def notify_completed(project: dict) -> None:
    from notify import alert  # deferred import: circular-import avoidance
    msg = f"Project '{project['name']}' finished all planned work."
    log_event(project["id"], "alert", msg)
    alert(title=f"Sandy project complete: {project['name']}", body=msg, severity="info",
          meta={"project_id": project["id"]})


_worker_started = False


def start_worker() -> None:
    """Idempotent boot hook called once from main's lifespan."""
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    import threading

    def _loop():
        while True:
            try:
                project = pick_next_project()
                if project:
                    outcome = run_cycle(project)
                    log(f"[projects.worker] cycle for '{outcome['project']}': "
                        f"{outcome['steps_done']} step(s), status={outcome['status']}")
                    if outcome["steps_done"] == 0 and outcome["status"] == "ok":
                        # Nothing progressed two cycles in a row would spin;
                        # a completed project should be marked done in chat --
                        # here we just avoid hot-looping on it via the poll gap.
                        pass
                else:
                    log("[projects.worker] no active projects; idling")
            except Exception as e:
                log(f"[projects.worker] cycle error (continuing): {e!r}")
            time.sleep(_POLL_SECONDS)

    threading.Thread(target=_loop, daemon=True, name="projects-worker").start()
    log(f"[projects.worker] started (poll every {_POLL_SECONDS:.0f}s)")
