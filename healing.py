"""Failure awareness -- the piece that was missing entirely: a job could
error out and Sandy would just... sit on it until Ruk happened to ask.

Two honest limits, stated up front because they shape everything below:
1. Hermes's cron runner is a SEPARATE OS process (supervisord's
   'gateway'). Sandy's own FastAPI process cannot intercept an exception
   INSIDE it live -- there is no hook for that from here. What this
   module does instead: poll each job's real last_error/state after the
   fact and alert on a NEW failure, same as Hermes's own ticker already
   works on a poll loop. Same HF-sleep dependency as everything else --
   this only runs while the container is awake.
2. "Reply on WhatsApp to confirm" would need an inbound webhook (a real
   Meta App Dashboard config on Ruk's side) that doesn't exist yet. This
   module sends the alert (real function, was dead code in projects.py)
   and Ruk confirms back in CHAT -- same confirm pattern already in use
   everywhere else, no new trust model invented.

For OUR OWN execution paths (native_mastery._run) failures ARE caught
live, in-process, same as before -- this module classifies them.
"""
import re

from llm import MODELS, log
from projects import send_whatsapp

_TABLE = "healing_ledger"

_PATTERNS = [
    (re.compile(r"no longer available to new users|is deprecated|model.*not.?found", re.I),
     "Model deprecated/discontinued by the provider", "model_fallback"),
    (re.compile(r"drifted since (this job was created|creation).*unpinned", re.I),
     "Global inference config changed after this job was created, and the job was never pinned", "pin_provider"),
    (re.compile(r"skill '?([\w-]+)'? not found", re.I),
     "Job references a skill that doesn't exist", "skill_missing"),
    (re.compile(r"\b429\b|rate.?limit", re.I), "Provider rate limit hit", "rate_limit"),
    (re.compile(r"\b401\b|unauthorized|invalid api key", re.I), "API key invalid/expired/unauthorized", "auth"),
    (re.compile(r"timed? ?out|timeout", re.I), "Request timed out", "timeout"),
    (re.compile(r"ResponseNotRead|streaming response content", re.I),
     "A bug inside Hermes's OWN error-handling code while it tried to summarize a "
     "DIFFERENT real error -- this is not something in Sandy's code, and the actual "
     "underlying failure is whichever error appears just before this one in the log",
     "hermes_internal_bug"),
]


def classify_error(error_text: str, entity: str = "unknown") -> dict:
    """Deterministic pattern match -- not an LLM call, on purpose. This
    runs on every poll; it needs to be free, instant, and NOT allowed to
    hallucinate a root cause the way an LLM asked 'what went wrong'
    with no real signal sometimes will."""
    for pattern, root_cause, fix_kind in _PATTERNS:
        if pattern.search(error_text):
            return {"root_cause": root_cause, "fix_kind": fix_kind, "entity": entity, "raw": error_text[:500]}
    return {"root_cause": "Real error, but this doesn't match a known pattern -- needs a human look, not a guess.",
            "fix_kind": "unknown", "entity": entity, "raw": error_text[:500]}


def propose_fix(diag: dict, job: dict | None = None) -> dict | None:
    """Returns a REAL, applicable {job_ref, updates} or None. Never
    invents a value -- a model_fallback fix only fires because
    MODELS[...] is a real, already-maintained value in llm.py, not a
    guessed model name. skill_missing/rate_limit/auth/timeout/
    hermes_internal_bug have no safe automatic fix -- Ruk has to look,
    and the alert says so plainly instead of pretending otherwise."""
    kind = diag["fix_kind"]
    if not job or job.get("engine") == "native":
        return None  # native failures don't have a Hermes-style provider/model to re-pin
    if kind == "model_fallback":
        provider = job.get("provider") or "gemini"
        new_model = MODELS.get(provider, "").split("/", 1)[-1]
        if new_model and new_model != job.get("model"):
            return {"job_ref": job["id"], "updates": {"provider": provider, "model": new_model}}
        return None
    if kind == "pin_provider":
        provider = job.get("provider") or "gemini"
        new_model = MODELS.get(provider, "").split("/", 1)[-1] or job.get("model")
        return {"job_ref": job["id"], "updates": {"provider": provider, "model": new_model}}
    return None


def check_for_new_failures() -> list[dict]:
    """Compares current job states against the healing_ledger (real
    table, not a jsonb dedup blob) and fires only on a REAL new failure
    -- an unresolved ledger row already matching this job_ref + root
    cause means Ruk's already been told, don't re-alert every poll."""
    import config
    import mastery
    import native_mastery

    alerts = []
    all_jobs = []
    try:
        for j in mastery.list_mastery_jobs():
            j.setdefault("engine", "hermes")
            all_jobs.append(j)
    except Exception as e:
        log(f"[healing] hermes job list read failed: {e!r}")
    all_jobs += native_mastery.status_shape()

    client = config.get_client()
    for j in all_jobs:
        jid = j["id"]
        error_text = j.get("last_error") or ""
        is_failing = bool(error_text) or j.get("state") == "failed"
        if not is_failing:
            continue
        diag = classify_error(error_text or "job state is 'failed', no error text captured", entity=j.get("name") or "unknown")
        existing = (
            client.table(_TABLE).select("id")
            .eq("job_ref", jid).eq("root_cause", diag["root_cause"]).eq("is_resolved", False)
            .limit(1).execute()
        )
        if existing.data:
            continue  # already have an open ledger row for this exact failure
        fix = propose_fix(diag, job=j)
        alerts.append({"job": j, "diag": diag, "fix": fix})
    return alerts


def alert_and_store(alerts: list[dict]) -> None:
    """Sends the real WhatsApp alert (best-effort) AND inserts a real row
    into healing_ledger -- root cause + fix, or root cause alone if
    there's no safe auto-fix -- so a later 'haan'/'fix it' in chat can
    apply it, and so main.py's chat pipeline can surface it even if
    WhatsApp silently isn't configured (confirmed from Ruk's own logs it
    currently isn't -- chat is the channel that can never silently fail
    to reach him the way an unconfigured webhook/secret can)."""
    import config

    client = config.get_client()
    for a in alerts:
        j, diag, fix = a["job"], a["diag"], a["fix"]
        client.table(_TABLE).insert({
            "job_ref": j["id"], "job_name": j.get("name", j["id"]), "engine": j.get("engine", "hermes"),
            "root_cause": diag["root_cause"], "proposed_updates": fix["updates"] if fix else None,
        }).execute()
        lines = [f"Ruk, real problem mili -- '{j.get('name', j['id'])}' ({j['id']}):", f"KYA HUA: {diag['root_cause']}"]
        if fix:
            lines.append(f"PROPOSED FIX: {fix['updates']} -- chat me 'haan'/'fix it' bolo, apply kar dungi.")
        else:
            lines.append("Iska koi safe automatic fix nahi hai -- khud dekhna padega, ya bata kya karna hai.")
        msg = "\n".join(lines)
        log(f"[healing] {msg}")
        if not send_whatsapp(msg):
            log(f"[healing] WhatsApp send failed or not configured (check RUK_WHATSAPP_NUMBER/WHATSAPP_TOKEN/WHATSAPP_PHONE_NUMBER_ID in HF secrets) -- alert only reached server logs for {j['id']}, chat will still surface it")


def run_check_and_alert() -> list[dict]:
    """One call, does the whole cycle -- what main.py actually calls."""
    alerts = check_for_new_failures()
    if alerts:
        alert_and_store(alerts)
    return alerts


def _row_to_fix(row: dict) -> dict:
    """Ledger row -> the {job_ref, job_name, root_cause, updates} shape
    main.py already expects -- keeps every caller in main.py unchanged."""
    return {
        "job_ref": row["job_ref"], "job_name": row["job_name"],
        "root_cause": row["root_cause"], "updates": row.get("proposed_updates"),
    }


def list_pending_fixes() -> list[dict]:
    """Every real unresolved failure -- {job_ref, job_name, root_cause,
    updates (None if no safe auto-fix exists)}."""
    import config
    res = config.get_client().table(_TABLE).select("*").eq("is_resolved", False).execute()
    return [_row_to_fix(r) for r in res.data]


def list_announced_pending_fixes() -> list[dict]:
    """Only fixes Sandy has ACTUALLY shown Ruk in chat already. Step 1
    (self-heal intercept) must never auto-apply a fix for a failure Ruk
    hasn't seen yet, even if the message he happens to send the exact
    turn it's first detected looks like an affirmative for an unrelated
    reason -- a real edge case, not a hypothetical one, given "haan"/
    "yes" alone triggers this."""
    import config
    res = config.get_client().table(_TABLE).select("*").eq("is_resolved", False).eq("announced_in_chat", True).execute()
    return [_row_to_fix(r) for r in res.data]


def list_unannounced_fixes() -> list[dict]:
    """Same as list_pending_fixes(), filtered to ones Sandy hasn't
    actually shown Ruk IN CHAT yet -- separate from whether the
    best-effort WhatsApp send worked, since that can silently fail
    (confirmed it currently does) and chat must not rely on it."""
    import config
    res = config.get_client().table(_TABLE).select("*").eq("is_resolved", False).eq("announced_in_chat", False).execute()
    return [_row_to_fix(r) for r in res.data]


def mark_announced(job_ref: str) -> None:
    import config
    config.get_client().table(_TABLE).update({"announced_in_chat": True}).eq("job_ref", job_ref).eq("is_resolved", False).execute()


def pop_pending_fix(job_id: str) -> dict | None:
    """Marks every currently-unresolved ledger row for this job_ref as
    resolved (a real fix just got applied -- whatever was open for this
    job is superseded by that). Returns the row that was there, for the
    caller's own record-keeping."""
    import config
    client = config.get_client()
    existing = client.table(_TABLE).select("*").eq("job_ref", job_id).eq("is_resolved", False).execute()
    if not existing.data:
        return None
    client.table(_TABLE).update({"is_resolved": True, "resolved_at": _now_iso()}).eq("job_ref", job_id).eq("is_resolved", False).execute()
    return _row_to_fix(existing.data[0])


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
