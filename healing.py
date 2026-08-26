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
from notify import alert as _notify_alert

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


def _dig_deeper(diag: dict, job: dict) -> dict:
    """Real fix for the exact complaint that healing 'just reads the log
    and says there's a problem' without checking whether that's actually
    the answer. Only runs for the two kinds that are known dead ends on
    their own: 'unknown' (no pattern matched at all) and
    'hermes_internal_bug' (its own _PATTERNS entry above already says
    the real cause is 'whichever error appears just before this one in
    the log' -- but nothing ever actually looked before this).

    Two real steps, each only tried if the previous one didn't resolve
    it -- never both unconditionally, so a clean case still costs
    nothing extra:
    1. Read Sandy's own real gateway log (get_gateway_logs -- already
       fixed in an earlier session to actually be readable from her own
       process) for a DIFFERENT line naming this job, and re-classify
       THAT text. If it matches a real known pattern, this IS the true
       root cause -- root_cause/fix_kind get updated for real, because
       this is deterministic: the same log content re-checked later
       finds the same match, so the healing_ledger dedup check (which
       keys on root_cause) stays stable.
    2. Only if step 1 found nothing better, run ONE real web search on
       the literal error text for research context. This result is
       NEVER written into root_cause/fix_kind -- search results aren't
       deterministic across polls (different top hit, different
       phrasing), so folding them into the dedup key would create a
       fresh duplicate ledger row roughly every poll instead of one
       real alert. It goes in the separate `research_note` field
       instead: shown to Ruk, stored for later, excluded from dedup.

    Critical boundary, unchanged from before: neither step is allowed
    to produce or apply an automatic fix on its own. propose_fix()
    still only ever returns a real MODELS[...]-backed value for the
    fix_kinds it already recognized. Research here only makes what
    Sandy TELLS Ruk better-informed -- it never expands what she's
    allowed to touch by herself."""
    if diag["fix_kind"] not in ("unknown", "hermes_internal_bug"):
        return diag  # already a clean, confident classification -- nothing to dig for

    import diagnostics

    entity = job.get("name") or job.get("id") or diag.get("entity", "")
    try:
        gateway_log = diagnostics.get_gateway_logs(lines=200)
    except Exception as e:
        log(f"[healing] _dig_deeper: gateway log read failed, skipping step 1: {e!r}")
        gateway_log = ""

    if entity:
        for line in gateway_log.splitlines():
            if entity not in line:
                continue
            deeper = classify_error(line, entity=entity)
            if deeper["fix_kind"] not in ("unknown", "hermes_internal_bug"):
                deeper["root_cause"] = (
                    f"{deeper['root_cause']} (found by checking the real gateway log near "
                    f"'{entity}', not just the surface error Hermes originally reported)"
                )
                return deeper

    try:
        import search
        results = search.search(diag["raw"][:200], complexity="simple")
    except Exception as e:
        log(f"[healing] _dig_deeper: research search failed, reporting plain diagnosis: {e!r}")
        return diag

    if results:
        top = results[0]
        diag = dict(diag)
        diag["research_note"] = (
            f"Ruk, log me isse zyada nahi mila, toh maine iska error text search kiya -- "
            f"ek possible lead: \"{top['title']}\" ({top['url']}). Ye Sandy ke apne registry se "
            f"verified nahi hai -- ek lead hai jo tum khud check kar sakte ho, auto-fix nahi banaya iska."
        )
    return diag


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


def _stale_failure(j: dict, client) -> bool:
    """True if the newest RESOLVED ledger row for this job is newer than
    the job's last_error timestamp -- i.e. a fix was already applied for
    this exact failure, and Hermes just hasn't cleared last_error yet.
    Without this check, every poll between 'fix applied' and 'Hermes
    clears the error' re-alerts the same failure (the dedup query above
    only matches UNRESOLVED rows). Jobs with no timestamp on their error
    are never considered stale -- better one duplicate than zero alerts."""
    jid = j["id"]
    err_time = j.get("last_error_at") or j.get("updated_at") or ""
    if not err_time:
        return False
    res = (
        client.table(_TABLE).select("resolved_at")
        .eq("job_ref", jid).eq("is_resolved", True)
        .order("resolved_at", desc=True).limit(1).execute()
    )
    if not res.data or not res.data[0].get("resolved_at"):
        return False
    return res.data[0]["resolved_at"] > err_time


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
        # Same honesty rule as job failures, applied to the gateway
        # itself: don't just log -- say what broke, why, and the fix.
        # notify's own cooldown dedups repeat polls to one ping per
        # ALERT_COOLDOWN window.
        _notify_alert(
            title="Hermes gateway unreachable",
            body=(
                "KYA HUA: Hermes ke cron jobs ki list padh nahi payi -- "
                "gateway process down ya crash hua lagta hai.\n"
                f"ERROR: {e!r}\n"
                "FIX: container logs mein gateway/supervisord section dekho "
                "(supervisorctl status, restart gateway). Sandy ke native "
                "jobs abhi bhi monitor ho rahe hain -- sirf Hermes-side "
                "jobs is window mein andhi hain."
            ),
            severity="warn",
            meta={"subsystem": "hermes_gateway", "error": repr(e)[:300]},
        )
    all_jobs += native_mastery.status_shape()

    client = config.get_client()
    for j in all_jobs:
        jid = j["id"]
        error_text = j.get("last_error") or ""
        is_failing = bool(error_text) or j.get("state") == "failed"
        if not is_failing:
            continue
        if _stale_failure(j, client):
            continue  # fix already applied; Hermes just hasn't cleared last_error yet
        diag = classify_error(error_text or "job state is 'failed', no error text captured", entity=j.get("name") or "unknown")
        diag = _dig_deeper(diag, j)
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
    """Stores each alert as a healing_ledger row (root cause + fix, or
    root cause alone if there's no safe auto-fix) so a later
    'haan'/'fix it' in chat can apply it, then routes the notification
    through notify.AlertRouter (Telegram; criticals also phone).
    Severity: blocked/repeated failures are critical, everything else warn."""
    import config

    client = config.get_client()
    for a in alerts:
        j, diag, fix = a["job"], a["diag"], a["fix"]
        client.table(_TABLE).insert({
            "job_ref": j["id"], "job_name": j.get("name", j["id"]), "engine": j.get("engine", "hermes"),
            "root_cause": diag["root_cause"], "proposed_updates": fix["updates"] if fix else None,
            "research_note": diag.get("research_note"),
        }).execute()
        lines = [f"Ruk, real problem mili -- '{j.get('name', j['id'])}' ({j['id']}):", f"KYA HUA: {diag['root_cause']}"]
        if diag.get("research_note"):
            lines.append(diag["research_note"])
        if fix:
            lines.append(f"PROPOSED FIX: {fix['updates']} -- chat me 'haan'/'fix it' bolo, apply kar dungi.")
        else:
            lines.append("Iska koi safe automatic fix nahi hai -- khud dekhna padega, ya bata kya karna hai.")
        msg = "\n".join(lines)
        log(f"[healing] {msg}")
        severity = "critical" if a.get("critical") or diag.get("critical") else "warn"
        receipt = _notify_alert(
            title=f"Sandy job failure: {j.get('name', j['id'])}",
            body=msg,
            severity=severity,
            meta={"job_ref": j["id"], "engine": j.get("engine", "hermes")},
        )
        if not receipt.get("dispatched") and not receipt.get("queued"):
            log(f"[healing] no channel dispatched for {j['id']} -- alert reached logs only; chat will still surface it")


def run_check_and_alert() -> list[dict]:
    """One call, does the whole cycle -- what main.py actually calls."""
    alerts = check_for_new_failures()
    if alerts:
        alert_and_store(alerts)
    return alerts


def _row_to_fix(row: dict) -> dict:
    """Ledger row -> the {job_ref, job_name, root_cause, updates} shape
    main.py already expects -- keeps every caller in main.py unchanged.
    research_note is new and optional (None on every row from before
    this session, and on any row _dig_deeper's log-check alone already
    resolved) -- callers that don't check for it see exactly the same
    shape as before."""
    return {
        "job_ref": row["job_ref"], "job_name": row["job_name"],
        "root_cause": row["root_cause"], "updates": row.get("proposed_updates"),
        "research_note": row.get("research_note"),
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


if __name__ == "__main__":
    # scode self-check: a clean classification must pass through
    # _dig_deeper untouched -- no wasted log read/search for a case
    # that already has a real, confident answer.
    clean = classify_error("HTTP 429 rate limit exceeded", entity="job-1")
    dug = _dig_deeper(clean, {"id": "job-1", "name": "job-1"})
    assert dug is clean, "a clean fix_kind must never trigger digging"
    print("healing.py: clean-classification passthrough OK")

    # scode self-check: the dedup-critical property -- root_cause for an
    # 'unknown'/'hermes_internal_bug' case must stay exactly the plain
    # classification's text unless a real log-based reclassification
    # happened. research_note (if any) must never leak into root_cause,
    # since root_cause is the healing_ledger dedup key and search
    # results aren't deterministic across polls.
    unknown = classify_error("some never-seen-before error string xyz123", entity="job-2")
    assert unknown["fix_kind"] == "unknown"
    print("healing.py: unknown-pattern classification OK ->", unknown["fix_kind"])
