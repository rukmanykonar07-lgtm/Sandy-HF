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
    """Compares current job states against the last-seen snapshot
    (persisted in sandy_config -- survives restarts) and fires only on a
    REAL transition into failure, not on every poll of an already-known
    one (would otherwise re-alert Ruk every few minutes forever)."""
    import config
    import mastery
    import native_mastery

    seen = config.get_config("healing_seen_failures") or {}
    alerts = []
    all_jobs = []
    try:
        for j in mastery.list_mastery_jobs():
            j.setdefault("engine", "hermes")
            all_jobs.append(j)
    except Exception as e:
        log(f"[healing] hermes job list read failed: {e!r}")
    all_jobs += native_mastery.status_shape()

    for j in all_jobs:
        jid = j["id"]
        error_text = j.get("last_error") or ""
        is_failing = bool(error_text) or j.get("state") == "failed"
        if not is_failing:
            continue
        fingerprint = str(hash(error_text or j.get("state")))
        if seen.get(jid) == fingerprint:
            continue  # already alerted on this exact failure
        diag = classify_error(error_text or "job state is 'failed', no error text captured", entity=j.get("name") or "unknown")
        fix = propose_fix(diag, job=j)
        alerts.append({"job": j, "diag": diag, "fix": fix})
        seen[jid] = fingerprint

    if alerts:
        config.set_config("healing_seen_failures", seen)
    return alerts


def _pending_index_key() -> str:
    return "healing_pending_ids"


def alert_and_store(alerts: list[dict]) -> None:
    """Sends the real WhatsApp alert (best-effort) AND stores a full
    record -- root cause + fix, or root cause alone if there's no safe
    auto-fix -- so a later 'haan'/'fix it' in chat can apply it, and so
    main.py's chat pipeline can surface it even if WhatsApp silently
    isn't configured (confirmed from Ruk's own logs it currently isn't --
    chat is the channel that can never silently fail to reach him the
    way an unconfigured webhook/secret can).

    Maintains its OWN index (healing_pending_ids) rather than reusing
    healing_seen_failures -- real bug found via actual testing: this
    function is called from TWO places (the poller's
    check_for_new_failures, AND native_mastery.py's direct in-process
    alert on a live exception, which never touches
    healing_seen_failures at all). Piggy-backing the pending-fix index on
    the poller's own dedup state meant a native job's failure record was
    written but permanently undiscoverable by list_pending_fixes(). This
    index is now updated here, the one real place a record is created,
    regardless of which caller triggered it."""
    import config

    ids = set(config.get_config(_pending_index_key()) or [])
    for a in alerts:
        j, diag, fix = a["job"], a["diag"], a["fix"]
        record = {
            "job_ref": j["id"], "job_name": j.get("name", j["id"]),
            "root_cause": diag["root_cause"], "updates": fix["updates"] if fix else None,
        }
        config.set_config(f"pending_fix:{j['id']}", record)
        ids.add(j["id"])
        lines = [f"Ruk, real problem mili -- '{record['job_name']}' ({record['job_ref']}):", f"KYA HUA: {record['root_cause']}"]
        if fix:
            lines.append(f"PROPOSED FIX: {fix['updates']} -- chat me 'haan'/'fix it' bolo, apply kar dungi.")
        else:
            lines.append("Iska koi safe automatic fix nahi hai -- khud dekhna padega, ya bata kya karna hai.")
        msg = "\n".join(lines)
        log(f"[healing] {msg}")
        if not send_whatsapp(msg):
            log(f"[healing] WhatsApp send failed or not configured (check RUK_WHATSAPP_NUMBER/WHATSAPP_TOKEN/WHATSAPP_PHONE_NUMBER_ID in HF secrets) -- alert only reached server logs for {j['id']}, chat will still surface it")
    config.set_config(_pending_index_key(), list(ids))


def run_check_and_alert() -> list[dict]:
    """One call, does the whole cycle -- what main.py actually calls."""
    alerts = check_for_new_failures()
    if alerts:
        alert_and_store(alerts)
    return alerts


def list_pending_fixes() -> list[dict]:
    """Every real unresolved failure record -- {job_ref, job_name,
    root_cause, updates (None if no safe auto-fix exists)}."""
    import config
    ids = config.get_config(_pending_index_key()) or []
    out = []
    for jid in ids:
        record = config.get_config(f"pending_fix:{jid}")
        if record:
            out.append(record)
    return out


def list_announced_pending_fixes() -> list[dict]:
    """Only fixes Sandy has ACTUALLY shown Ruk in chat already. Step 1
    (self-heal intercept) must never auto-apply a fix for a failure Ruk
    hasn't seen yet, even if the message he happens to send the exact
    turn it's first detected looks like an affirmative for an unrelated
    reason -- a real edge case, not a hypothetical one, given "haan"/
    "yes" alone triggers this."""
    import config
    announced = set(config.get_config("healing_chat_announced") or [])
    return [f for f in list_pending_fixes() if f["job_ref"] in announced]


def list_unannounced_fixes() -> list[dict]:
    """Same as list_pending_fixes(), filtered to ones Sandy hasn't
    actually shown Ruk IN CHAT yet -- separate from whether the
    best-effort WhatsApp send worked, since that can silently fail
    (confirmed it currently does) and chat must not rely on it."""
    import config
    announced = set(config.get_config("healing_chat_announced") or [])
    return [f for f in list_pending_fixes() if f["job_ref"] not in announced]


def mark_announced(job_ref: str) -> None:
    import config
    announced = set(config.get_config("healing_chat_announced") or [])
    announced.add(job_ref)
    config.set_config("healing_chat_announced", list(announced))


def pop_pending_fix(job_id: str) -> dict | None:
    import config
    record = config.get_config(f"pending_fix:{job_id}")
    if record:
        config.set_config(f"pending_fix:{job_id}", None)
        ids = set(config.get_config(_pending_index_key()) or [])
        ids.discard(job_id)
        config.set_config(_pending_index_key(), list(ids))
        # Clear the announced flag too -- otherwise a FUTURE new failure
        # on this same job_id would be silently suppressed by a stale
        # "already shown" marker from a previous, now-resolved failure.
        announced = set(config.get_config("healing_chat_announced") or [])
        announced.discard(job_id)
        config.set_config("healing_chat_announced", list(announced))
    return record
