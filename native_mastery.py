"""Sandy's OWN mastery engine -- the 'my mastery' side of the comparison
Ruk wants against Hermes's cron-based path. Reuses brain._orchestrate
(the real planner -> parallel-workers -> verifier-loop -> synthesis
pattern already proven in chat) instead of reinventing it -- don't build
a second orchestrator when a real one already exists.

Division from mastery.py (Hermes path):
- mastery.py hands the approved plan to Hermes's own agent runtime,
  which executes autonomously via cron, outside Sandy's own process.
- This module runs the SAME kind of approved plan through Sandy's own
  process, using her own orchestrator directly.

Full lifecycle Ruk actually wants (per his spec):
  propose -> edit (feedback ACTUALLY changes weights/caps/mode, not
  just the displayed text) -> confirm -> running -> pause/resume ->
  continue (the real "do the next chunk now" trigger scheduled mode
  was missing entirely before) -> done. Ruk can add/remove weighted
  models and set a per-job call cap on any model at any point; a round
  that turns out genuinely hard can pull in extra capped-but-unweighted
  models automatically. Every real step is a real event.py row -- same
  data the orb graph and (via list_jobs below) Workflows/Agents render.

Jobs are stored in a REAL relational table, native_mastery_jobs (real
columns, real types -- see migration_rdbms.sql) -- not a jsonb blob
under a generic key-value store. A job surviving HF container
sleep/restart between propose and confirm (or mid-pause) is the whole
point; an in-memory dict was the exact cause of "confirmed" silently
falling into a hallucinated fake job before the first rewrite, and the
jsonb-blob approach worked but wasn't real relational storage, which is
what this table now is.
"""
import random
import re
import threading

import brain
import config
import events
import healing
import observability
from identity import SANDY_SYSTEM_PROMPT
from llm import call_llm_with_fallback, log, MODELS

_IDENTITY_MSG = {"role": "system", "content": SANDY_SYSTEM_PROMPT}
_TABLE = "native_mastery_jobs"

# Real race condition, found by deliberately walking through concurrency
# scenarios (not hit by chance): _orchestrate() runs multiple workers for
# the SAME job concurrently via ThreadPoolExecutor. _bump_usage does a
# read-modify-write (get_job -> mutate dict -> _save_job) -- two workers
# finishing at the same instant can both read usage=N, both compute N+1,
# both write N+1, and one real increment silently vanishes. For a job
# with a real per-provider cap set, that means Ruk's own cap is quietly
# enforced less accurately than it looks, with no error anywhere.
#
# A per-job threading.Lock (not a DB-level lock/RPC) is the right-sized
# fix: all the concurrency here is ThreadPoolExecutor workers inside ONE
# Python process (HF Spaces runs a single container) -- there's no
# multi-instance/distributed concurrency to defend against, so adding
# Postgres-level atomic increment machinery would be solving a problem
# that doesn't exist at this scale.
#
# Honest limit this does NOT close: the provider_guard CHECK (before a
# call starts) and the bump (after it finishes) are still two separate
# moments -- two workers can both pass the cap check a moment apart, then
# both bump, overshooting the cap by a small amount in a rare near-
# simultaneous case. Caps here are a soft spend guardrail, not a hard
# billing limit, so this residual gap is an accepted tradeoff, not
# something silently ignored -- fully closing it would mean holding a
# lock across the entire external LLM call (real latency), which isn't
# worth it for a soft internal guardrail. This fix closes the definitely-
# worse bug (silently LOSING real usage counts), not the boundary case.
#
# SECOND real race found the same way (scenario-walking, then confirmed
# with an actual concurrent test, not assumed): _run()'s finishing write
# reads the job early, does the whole orchestrator run, then saves the
# WHOLE job dict back -- if pause()/resume() write a different state
# while _run() is mid-flight, _run()'s stale in-memory copy silently
# overwrites the ENTIRE row (not just state) when it finally saves. This
# is worse than the usage race: it can clobber a real pause, not just
# undercount a number. Confirmed with a real threaded test before fixing.
#
# One lock, generalized to guard every real read-modify-write on a job
# (usage bumps AND state transitions) -- not two separate locking
# schemes for what's the same underlying problem.
_job_locks: dict[str, threading.Lock] = {}
_job_locks_guard = threading.Lock()


def _get_job_lock(job_id: str) -> threading.Lock:
    with _job_locks_guard:
        if job_id not in _job_locks:
            _job_locks[job_id] = threading.Lock()
        return _job_locks[job_id]


# Explicit transition table -- self-documenting, and closes a real gap:
# mark_done() previously had NO guard at all, so it could mark a job
# that was still 'proposed' (never even started) as 'done'.
_ALLOWED_TRANSITIONS = {
    "pause": {"running", "scheduled_waiting"},
    "resume": {"paused"},
    "continue_now": {"scheduled_waiting"},
    "mark_done": {"running", "paused", "scheduled_waiting"},
    # NEW: editing an already-confirmed job's weights/caps/mode. Valid
    # from any non-terminal, already-confirmed state -- NOT "proposed"
    # (propose()+feedback already handles pre-confirm edits, a
    # different code path with a different plan-regeneration prompt),
    # and not "done"/"removed" (nothing left to edit).
    "edit": {"running", "paused", "scheduled_waiting"},
}


def _atomic_transition(job_id: str, action: str, mutate) -> tuple[dict | None, str | None]:
    """Locked, validated read-modify-write -- every real state change
    goes through this one path instead of a scattered ad-hoc
    read-check-write per function. Re-reads the job FRESH inside the
    lock (not whatever the caller already had in memory), so this closes
    the exact race described above for every caller, not just _run().

    mutate(job) -> job: applies the actual change to a job dict already
    confirmed to be in a valid starting state.
    Returns (updated_job, None) on success, or (None, error_message)."""
    with _get_job_lock(job_id):
        job = get_job(job_id)
        if not job:
            return None, f"Ruk, native job {job_id} nahi mila."
        allowed = _ALLOWED_TRANSITIONS.get(action)
        if allowed is not None and job["state"] not in allowed:
            return None, f"Ruk, {job_id} abhi '{job['state']}' state me hai, {action} abhi valid nahi hai."
        job = mutate(job)
        _save_job(job)
        return job, None


# --- persistence (real Postgres table, not a jsonb blob) ---------------

def _save_job(job: dict) -> None:
    """Real upsert -- job["id"] is the primary key. Every field on the
    job dict maps directly to a real column (see migration_rdbms.sql)."""
    row = {k: v for k, v in job.items() if k != "updated_at"}
    row["updated_at"] = _now_iso()
    config.get_client().table(_TABLE).upsert(row).execute()


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def get_job(job_id: str) -> dict | None:
    res = config.get_client().table(_TABLE).select("*").eq("id", job_id).execute()
    return res.data[0] if res.data else None


def list_jobs() -> list[dict]:
    """Every real native job (any state), newest first -- for Ruk's
    Home's Workflows/Agents tabs, which previously only ever showed
    Hermes jobs because native runs had nowhere to register themselves
    at all. Real data, same discipline as mastery.list_mastery_jobs()."""
    res = (
        config.get_client().table(_TABLE).select("*")
        .neq("state", "removed").order("created_at", desc=True).execute()
    )
    return res.data


def get_pending_native_plan(session_id: str) -> dict | None:
    """Back-compat shape for main.py's existing pending-plan routing --
    finds this session's most recent job still in 'proposed' state."""
    res = (
        config.get_client().table(_TABLE).select("*")
        .eq("session_id", session_id).eq("state", "proposed")
        .order("created_at", desc=True).limit(1).execute()
    )
    return res.data[0] if res.data else None


def latest_job_for_session(session_id: str, state: str | None = None) -> dict | None:
    q = config.get_client().table(_TABLE).select("*").eq("session_id", session_id)
    if state is not None:
        q = q.eq("state", state)
    res = q.order("created_at", desc=True).limit(1).execute()
    return res.data[0] if res.data else None


def job_diagnostics(job_id: str) -> str:
    """Everything real about ONE native job -- state, weights/caps/usage,
    round count, AND the real recent event trail (not a narrated
    summary) so a genuine issue -- a worker failing, a cap being hit, a
    replan -- is visible instead of just 'running' or 'done'."""
    job = get_job(job_id)
    if not job:
        return f"Ruk, native job {job_id} nahi mila."
    lines = [f"**native: {job['skill']}** ({job['id']})"]
    lines.append(f"- state: {job['state']}")
    lines.append(f"- mode: {job['mode']}")
    lines.append(f"- weights: {job['weights'] or 'default split (groq/cerebras)'}")
    lines.append(f"- per-job caps: {job['caps'] or 'none set -- only the global daily cap applies'}")
    lines.append(f"- usage this job: {job.get('usage') or 'kuch use nahi hua abhi'}")
    lines.append(f"- rounds completed: {job.get('round', 0)}")
    if job["state"] == "failed":
        lines.append(f"- ⚠️ LAST ERROR: {job.get('result')}")
    ev = events.get_events(job_id)
    if not ev:
        lines.append("- events: koi bhi nahi -- abhi tak real kaam start hi nahi hua.")
    else:
        lines.append(f"- last {min(5, len(ev))} real events:")
        for e in ev[-5:]:
            marker = "⚠️ " if e.get("event_type") == "obstacle" else ""
            lines.append(f"  - round {e.get('round')}: {marker}[{e.get('event_type')}] {e.get('summary')}")
    return "\n".join(lines)


def status_shape() -> list[dict]:
    """Native jobs normalized into the same shape Hermes jobs already
    use for Ruk's Home's Workflows/Agents tabs (id/name/state/next_run_at/
    last_error/repeat) -- these tabs used to only ever read Hermes's job
    list, so native runs (which never touch Hermes) were structurally
    invisible there no matter what. 'proposed' (not yet confirmed) and
    'removed' jobs are left out -- not real running/registered work yet.
    Cosmetic note: state colors in the frontend were tuned for Hermes's
    states -- 'done'/'failed' pass through as-is and will show as the
    default (red) dot rather than a distinct color; real data is what
    was missing, not colors, so leaving that polish for a later pass."""
    _STATE_MAP = {"running": "scheduled", "scheduled_waiting": "scheduled", "paused": "paused"}
    out = []
    for j in list_jobs():
        if j["state"] in ("proposed", "removed"):
            continue
        out.append({
            "id": j["id"], "name": f"native: {j['skill']}", "engine": "native",
            "state": _STATE_MAP.get(j["state"], j["state"]),
            "next_run_at": None, "last_run_at": None, "last_error": j.get("result") if j["state"] == "failed" else None,
            "repeat": {"times": None, "completed": j.get("round", 0)},
        })
    return out


# --- parsing edits/directives from free text (deterministic, no LLM call) --

def parse_directives(message: str) -> tuple[str | None, dict[str, int], dict[str, int]]:
    """mode (None = not mentioned this message), weights {provider: pct},
    caps {provider: max_calls_for_this_job} -- all parsed fresh from
    whatever Ruk just said, so an EDIT message ('gemini 70% instead',
    'cap groq at 50 for this job', 'add mistral 20%') actually changes
    the real execution params, not just the plan text describing them.
    This is the root-cause fix for 'I can't edit things in mastery' --
    the old code fed edit feedback to an LLM to rewrite the plan
    DOCUMENT, but never touched the stored weights/mode dict the run
    actually used at confirm time."""
    mode = None
    if re.search(r"\bschedul", message, re.I):
        mode = "scheduled"
    elif re.search(r"\bcontinuous\b|\bstart now\b|\bright now\b|\bimmediately\b", message, re.I):
        mode = "continuous"
    weights: dict[str, int] = {}
    caps: dict[str, int] = {}
    for p in MODELS:
        m = re.search(rf"\b{p}\D{{0,10}}(\d{{1,3}})\s*%", message, re.I)
        if m:
            weights[p] = int(m.group(1))
        c = re.search(
            rf"\b{p}\D{{0,15}}(?:cap|max|limit)\D{{0,10}}(\d+)"
            rf"|(\d+)\D{{0,10}}(?:cap|max|limit)\D{{0,10}}{p}"
            rf"|\b{p}\D{{0,10}}(\d+)\s*calls?\b",
            message, re.I,
        )
        if c:
            caps[p] = int(next(g for g in c.groups() if g))
    return mode, weights, caps


def _quota_dampen(weights: dict[str, int]) -> dict[str, float]:
    """Scale each provider's weight by how close it is to its own daily
    cap -- exhausted providers drop to 0 (excluded), providers in the
    warn band (80%+, see observability.cap_status) get dampened to 65%
    of their stated weight, healthy/uncapped providers are untouched.

    Ported from OmniRoute's routing scorer (src/lib/routing/
    adaptiveRouting.ts, quotaFactor()) -- same idea (exhausted=0,
    approaching_limit=0.65, healthy=1), applied here to Ruk's static
    percentage weights instead of a live multi-factor score, since
    Sandy doesn't track per-provider latency/health/circuit state the
    way OmniRoute's gateway does. This is what makes a job actually
    shift load AWAY from a provider that's about to hit its cap,
    instead of blindly following the static split until a hard
    failure forces a fallback. Fails safe: any error reading cap
    status leaves that provider's weight untouched (same fail-open
    principle as llm.py's cap check) -- a broken dampening check
    should never be able to stall job scheduling entirely."""
    dampened = {}
    for provider, pct in weights.items():
        try:
            state = observability.cap_status(provider)["state"]
        except Exception as e:
            log(f"[_quota_dampen] cap_status failed for {provider}, leaving weight as-is: {e!r}")
            dampened[provider] = pct
            continue
        if state == "exhausted":
            dampened[provider] = 0
        elif state == "warn":
            dampened[provider] = pct * 0.65
        else:
            dampened[provider] = pct
    return dampened


def _weighted_worker_list(weights: dict[str, int], slots: int = 6) -> list[str]:
    """Turns {'gemini': 50, 'groq': 30, 'cerebras': 20} into a real
    round-robin list _orchestrate can use directly -- proportional
    representation. Any real provider name works here, not just the
    original three -- Ruk adding a new weighted model just means a new
    key in this dict, nothing else has to change.

    Quota-aware (added): weights are dampened by _quota_dampen() first,
    so a provider close to (or past) its own daily cap gets less (or
    no) work this round, instead of the job blindly hammering it at
    the full static split until every call from it starts failing."""
    weights = {p: pct for p, pct in weights.items() if p in MODELS}
    if not weights:
        return ["groq", "cerebras"]  # same default _orchestrate already uses
    weights = _quota_dampen(weights)
    total = sum(weights.values())
    if total <= 0:
        # every weighted provider is exhausted right now -- fall back to
        # the same safety net _orchestrate already uses rather than
        # returning an empty worker list.
        return ["groq", "cerebras"]
    out = []
    for provider, pct in weights.items():
        if pct <= 0:
            continue
        out += [provider] * max(1, round(slots * pct / total))
    random.shuffle(out)
    return out


def _top_weighted_provider(weights: dict[str, int], default: str = "gemini") -> str:
    """The job's own single highest-weighted real provider -- used to
    pick WHO runs the orchestrator role (research/plan/review/replan).

    This used to just be a hardcoded "gemini" inside brain._orchestrate,
    completely ignoring the job's weights -- meaning a job configured to
    barely use Gemini could still burn ~8 Gemini calls per round on
    orchestration alone, on top of whatever it burned as a worker. That
    was Ruk's #1 flagged credit-burn cause. Deliberately NOT derived
    from _weighted_worker_list() above -- that list is randomly
    shuffled for round-robin worker assignment, so its [0] element is
    not "the top-weighted provider", it's an arbitrary pick.
    default="gemini" preserves prior behavior when no weights are set
    at all (empty dict) -- not a random guess, Gemini's real strength
    at structured JSON planning/review is the original, still-valid
    reason it was the default worth keeping as a fallback.
    """
    real = {p: pct for p, pct in (weights or {}).items() if p in MODELS and pct}
    if not real:
        return default
    return max(real, key=real.get)


# --- explain (no job yet) ----------------------------------------------

def explain_native_flow(skill: str, message: str, context: str = "") -> str:
    """Real, grounded explanation for the native (non-Hermes) path."""
    real_mechanism = (
        "REAL mechanism for the NATIVE (non-Hermes) path -- use only what's relevant to what "
        "he actually asked:\n"
        "1. Does NOT use Hermes's agent runtime. Runs Sandy's own real orchestrator "
        "(brain._orchestrate) -- planner (Gemini) -> parallel workers -> verifier/conflict-"
        "check loop -> synthesis.\n"
        "2. Ruk sets per-run provider WEIGHTS (e.g. gemini 50%, groq 30%, cerebras 20%) -- "
        "controls which providers do the actual work, proportionally. He can also set a "
        "per-job CALL CAP on any model (e.g. 'cap groq at 50 for this job') -- separate from "
        "the global daily cap, just for this run. If a round turns out genuinely hard, extra "
        "capped-but-unweighted models get pulled in automatically to help.\n"
        "3. Mode: CONTINUOUS (runs now, straight through) or SCHEDULED (registers, then only "
        "advances when Ruk says 'continue' -- a real chunk-by-chunk trigger, not automatic).\n"
        "4. Ruk can edit the plan, the weights, the caps, or the mode AT ANY TIME before OR "
        "during a run (pause first) -- edits change the REAL params the run actually uses, "
        "not just the description. He can pause a running job and resume it later, and a "
        "resumed/continued run treats prior real output as something to EDIT/ENHANCE/REMOVE "
        "FROM, not just append blindly to.\n"
        "5. Every real step -- planning, each worker's output, conflicts, replanning, pauses, "
        "synthesis -- is a real event, visible as its own orb graph in Ruk's Home AND in the "
        "Workflows/Agents tabs (native jobs show there now, tagged separately from Hermes).\n"
        "6. Ruk must confirm before anything runs.\n"
    )
    prompt = (
        f"Ruk is talking about \"{skill}\" as a NATIVE mastery run (his own orchestration, "
        f"not Hermes). His message this turn: \"{message}\"\n\n"
        "Answer his actual question concretely. If mode/weights/caps haven't been given yet, "
        "end by asking for them -- but only after actually answering what he asked.\n\n" + real_mechanism
    )
    if context:
        prompt = f"{context}\n\n{prompt}"
    return call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])


# --- propose / edit (both go through the same function -- an edit is just
# a propose against an existing job id, feedback re-parsed for real) -----

def propose(
    session_id: str, skill: str, mode: str | None, weights: dict, caps: dict,
    feedback: str | None = None, prior: dict | None = None, context: str = "",
) -> dict:
    """Builds/regenerates the plan doc AND updates the real stored params
    from whatever was actually said this message. `prior` (an existing
    job dict, if this is an edit) supplies whatever mode/weights/caps
    weren't mentioned this turn -- merge, don't clobber."""
    mode = mode or (prior or {}).get("mode") or "continuous"
    merged_weights = {**(prior or {}).get("weights", {}), **weights}
    merged_caps = {**(prior or {}).get("caps", {}), **caps}

    revision_note = ""
    if feedback and prior:
        revision_note = (
            f"\n\nRuk already saw this earlier draft:\n{prior['plan']}\n\n"
            f'He wants this changed: "{feedback}"\n'
            "Rewrite the FULL plan incorporating that feedback."
        )
    weights_line = ", ".join(f"{p} {pct}%" for p, pct in merged_weights.items()) or "no weights given -- default split"
    caps_line = ", ".join(f"{p} max {n} calls" for p, n in merged_caps.items()) or "no per-job caps -- only the global daily cap applies"
    prompt = (
        f'Ruk asked Sandy to master "{skill}" using her OWN orchestrator (not Hermes), '
        f"mode={mode}, provider weights: {weights_line}, per-job caps: {caps_line}. Write a "
        "full plan document, in Hinglish, covering ALL of these sections clearly:\n"
        "1. UNDERSTANDING -- the actual goal, in your own words\n"
        "2. WHAT YOU'LL MAKE -- concrete deliverable(s)\n"
        "3. PROCESS -- how your own planner/parallel-workers/verifier-loop will tackle THIS "
        "skill specifically\n"
        "4. PROVIDER SPLIT -- confirm the weights and caps back, what each provider's role "
        "is, and that extra models can join automatically if a round turns out hard\n"
        "5. SUCCESS CRITERIA"
        + revision_note
    )
    if context:
        prompt = f"{context}\n\n{prompt}"
    plan = call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}], caller="native_mastery.propose")

    job_id = (prior or {}).get("id") or events.new_run_id()
    job = {
        "id": job_id, "session_id": session_id, "skill": skill, "mode": mode,
        "weights": merged_weights, "caps": merged_caps,
        "usage": (prior or {}).get("usage", {}),
        "state": "proposed", "plan": plan, "result": (prior or {}).get("result"),
        "round": (prior or {}).get("round", 0),
        # explicit, not relying on the table's DEFAULT now() -- this is
        # the column get_pending_native_plan() orders by (desc) to find
        # "the most recent proposed job for this session". Leaving it
        # unset here means the LOCAL job dict has no created_at until
        # the next real fetch from Supabase -- and if a caller ever
        # compares/orders freshly-proposed jobs before that round-trip
        # (exactly what the test suite caught), it silently falls back
        # to insertion order, not recency, and confirm can resolve the
        # wrong job. Preserve the original on edits (prior exists) --
        # editing a proposal shouldn't bump when it was first created.
        "created_at": (prior or {}).get("created_at") or _now_iso(),
    }
    _save_job(job)
    return job


# --- confirm / lifecycle -------------------------------------------------

def confirm_native_plan(session_id: str, background_tasks=None) -> tuple[str, str]:
    """Back-compat name/shape for main.py -- confirms the session's
    pending ('proposed') job."""
    job = get_pending_native_plan(session_id)
    if not job:
        return "Ruk, koi pending native plan nahi mila -- pehle naya request bhejo.", ""
    events.log_event(job["id"], "sandy", "planning", f"Native mastery run confirmed for \"{job['skill']}\"", detail=job["plan"])
    if job["mode"] == "scheduled":
        job["state"] = "scheduled_waiting"
        _save_job(job)
        reply = (
            f"Theek hai Ruk! \"{job['skill']}\" native mastery job REGISTERED (scheduled) -- "
            f"job_id {job['id']}. Jab bhi chunk chalani ho, bas \"continue karo {job['id']}\" bolo -- "
            "yahi se aage badhegi. Graph turant dikhega Ruk's Home me (empty state abhi)."
        )
        return reply, job["id"]
    job["state"] = "running"
    _save_job(job)
    if background_tasks is not None:
        background_tasks.add_task(_run, job["id"])
    reply = (
        f"Theek hai Ruk! \"{job['skill']}\" native mastery run STARTING NOW -- job_id {job['id']}. "
        "Real-time graph Ruk's Home me turant populate hone lagega."
    )
    return reply, job["id"]


def pause(job_id: str) -> str:
    job, err = _atomic_transition(job_id, "pause", lambda j: {**j, "state": "paused"})
    if err:
        return err
    events.log_event(job_id, "sandy", "obstacle", "Run paused by Ruk", round=job.get("round", 0))
    return (
        f"Ruk, native job {job_id} pause ho gaya. Real pause point round ke beech me nahi, "
        f"agle round boundary pe hota hai -- agar worker abhi chal raha tha, wo round khatam "
        f"hoke rukega. \"resume karo {job_id}\" bolo jab chaho, yahi se aage badhegi."
    )


def resume(job_id: str, background_tasks=None) -> str:
    def _mutate(j):
        j["state"] = "running" if j["mode"] == "continuous" else "scheduled_waiting"
        return j
    job, err = _atomic_transition(job_id, "resume", _mutate)
    if err:
        return err
    events.log_event(job_id, "sandy", "planning", "Run resumed by Ruk", round=job.get("round", 0))
    if job["mode"] == "continuous" and background_tasks is not None:
        background_tasks.add_task(_run, job_id)
        return f"Ruk, {job_id} resume ho gaya, chal rahi hai ab."
    return f"Ruk, {job_id} resume ho gaya -- \"continue karo {job_id}\" bolo agla chunk chalane ke liye."


def edit_native_job(
    job_id: str, weights: dict | None = None, caps: dict | None = None, mode: str | None = None,
) -> str:
    """Actually changes an ALREADY-CONFIRMED job's weights/caps/mode --
    the piece identity.py claimed existed ("edit (weights/caps/schedule
    actually change, not just the displayed text)") but genuinely
    didn't: there was no edit path at all once a job left 'proposed'.
    Ruk's only option was pause + remove + propose a whole new job,
    losing round/result history. This is a real, additive data change
    (not a state transition), routed through the SAME lock as every
    other job mutation via _atomic_transition, so it can't race a
    concurrent _run() the same way pause()/resume() already can't.

    Partial update -- only keys actually mentioned change; everything
    else on the job is left exactly as-is (merge, not replace, same
    convention propose()'s prior/feedback merge already uses)."""
    unknown_providers = [p for p in {**(weights or {}), **(caps or {})} if p not in MODELS]

    def _mutate(j):
        if weights:
            j["weights"] = {**j.get("weights", {}), **{p: v for p, v in weights.items() if p in MODELS}}
        if caps:
            j["caps"] = {**j.get("caps", {}), **{p: v for p, v in caps.items() if p in MODELS}}
        if mode:
            j["mode"] = mode
        return j

    job, err = _atomic_transition(job_id, "edit", _mutate)
    if err:
        return err

    events.log_event(
        job_id, "sandy", "planning",
        f"Job edited by Ruk -- weights={job['weights']}, caps={job['caps']}, mode={job['mode']}",
        round=job.get("round", 0),
    )
    changed = []
    if weights:
        changed.append(f"weights -> {job['weights']}")
    if caps:
        changed.append(f"caps -> {job['caps']}")
    if mode:
        changed.append(f"mode -> {job['mode']}")
    reply = f"Ruk, {job_id} update ho gaya: " + "; ".join(changed) + "."
    if unknown_providers:
        reply += f" (Ignore kiya -- ye real provider nahi hai: {', '.join(unknown_providers)}.)"
    return reply


def continue_now(job_id: str, background_tasks=None) -> str:
    """THE real tick mechanism scheduled mode was completely missing
    before -- there is no cron/ticker for native jobs, Ruk saying
    'continue' IS the trigger, on purpose (matches his own request to
    control exactly when a scheduled job advances)."""
    job, err = _atomic_transition(job_id, "continue_now", lambda j: {**j, "state": "running"})
    if err:
        return err
    if background_tasks is not None:
        background_tasks.add_task(_run, job_id)
    return f"Ruk, {job_id} ka agla chunk chal raha hai ab."


def mark_done(job_id: str) -> str:
    job, err = _atomic_transition(job_id, "mark_done", lambda j: {**j, "state": "done"})
    if err:
        return err
    return f"Ruk, {job_id} done mark kar diya."


def remove(job_id: str) -> str:
    # No allowed_from restriction -- valid from ANY state, on purpose
    # (an explicit Ruk-directed delete shouldn't be blocked by state).
    # Still goes through the same lock so it can't race a concurrent
    # _run() finishing write the same way pause() could before this fix.
    job, err = _atomic_transition(job_id, "remove", lambda j: {**j, "state": "removed"})
    if err:
        return err
    return f"Ruk, native job {job_id} remove kar diya."


# --- execution ------------------------------------------------------------

def _make_guard(job_id: str):
    """Job-scoped per-provider cap -- Ruk's ADDITIONAL per-job ceiling,
    on top of (not instead of) llm.py's global daily cap."""
    def guard(provider: str) -> bool:
        job = get_job(job_id)
        if not job:
            return True
        cap = (job.get("caps") or {}).get(provider)
        if cap is None:
            return True
        return (job.get("usage") or {}).get(provider, 0) < cap
    return guard


def _bump_usage(job_id: str, provider: str) -> None:
    """Locked per job_id -- see the module-level comment on
    _usage_locks for why this specific race was real, not hypothetical."""
    with _get_job_lock(job_id):
        job = get_job(job_id)
        if not job:
            return
        usage = job.get("usage") or {}
        usage[provider] = usage.get(provider, 0) + 1
        job["usage"] = usage
        _save_job(job)


def _should_continue(job_id: str):
    def check():
        job = get_job(job_id)
        return bool(job) and job.get("state") == "running"
    return check


def _run(job_id: str) -> str:
    """The actual execution. Meant to run as a FastAPI BackgroundTask."""
    job = get_job(job_id)
    if not job:
        return ""

    def on_event(event_type, summary, round, provider, detail):
        events.log_event(job_id, "sandy", event_type, summary, round=round, provider=provider, detail=detail)
        if provider:
            _bump_usage(job_id, provider)

    worker_list = _weighted_worker_list(job["weights"])
    extra = [p for p in (job.get("caps") or {}) if p not in worker_list and p in MODELS]

    task = (
        f"Master the skill \"{job['skill']}\". Ruk's approved approach:\n{job['plan']}\n\n"
    )
    if job.get("result"):
        task += (
            f"This is a CONTINUATION of existing work, not a fresh build. Real result so far:\n"
            f"{job['result']}\n\nYou may EDIT, REMOVE, ADD to, or ENHANCE what's already "
            "there -- actually improve on it where it's wrong, incomplete, or outdated, "
            "don't just tack more onto the end blindly.\n\n"
        )
    task += (
        "Produce a real, concrete result -- actual research/code/technique, not a plan "
        "restated as if it were the work. Before writing or changing any code specifically: "
        "think first whether it's actually necessary, whether it breaks anything else, "
        "whether what you're fixing is actually broken, and whether it truly matches the "
        "plan above -- then write it."
    )

    orchestrator = _top_weighted_provider(job["weights"])
    try:
        result = brain._orchestrate(
            task, context="", on_event=on_event, workers=worker_list, extra_workers=extra,
            provider_guard=_make_guard(job_id), should_continue=_should_continue(job_id),
            orchestrator=orchestrator,
        )
        # scode: THE actual race this whole locking pass exists for --
        # confirmed live with a real threaded test before this fix: the
        # read used to happen here, then _orchestrate ran (real time
        # passing), then the WHOLE job dict got saved -- if pause()
        # wrote a different state in that window, this stale save
        # silently overwrote it entirely. The read now happens INSIDE
        # the lock, immediately before the write, and pause()/resume()/
        # etc. all take the SAME lock -- so whichever gets there first
        # completes its full read+write before the other can start,
        # and this block always sees the real current state, not a
        # stale one from before the run started.
        with _get_job_lock(job_id):
            job = get_job(job_id)
            if not job:
                return result
            job["result"] = result
            job["round"] = job.get("round", 0) + 1
            if job["state"] == "running":  # wasn't paused mid-run
                job["state"] = "done" if job["mode"] == "continuous" else "scheduled_waiting"
            _save_job(job)
        events.log_event(job_id, "sandy", "output", f"Native mastery run finished for \"{job['skill']}\"", detail=result)
    except Exception as e:
        log(f"[native_mastery._run] job {job_id} failed: {e!r}")
        events.log_event(job_id, "sandy", "obstacle", f"Run failed with a real error: {e}", detail=str(e))
        with _get_job_lock(job_id):
            job = get_job(job_id)
            if job:
                job["state"] = "failed"
                _save_job(job)
        result = f"Run failed: {e}"
        # Real, live in-process failure -- unlike a Hermes job, we know about
        # this the INSTANT it happens, not on the next poll cycle. No auto-fix
        # exists for a native run (no provider/model to re-pin the way a
        # Hermes job has), so this just gets the real diagnosis to Ruk fast
        # instead of making him wait or ask.
        try:
            diag = healing.classify_error(str(e), entity=job["skill"] if job else job_id)
            healing.alert_and_store([{"job": {"id": job_id, "name": f"native: {job['skill'] if job else job_id}", "engine": "native"}, "diag": diag, "fix": None}])
        except Exception as alert_err:
            log(f"[native_mastery._run] failure-alert itself failed, non-fatal: {alert_err!r}")
    try:
        events.link_similar_events(events.get_events(job_id))
    except Exception as e:
        log(f"[native_mastery._run] similarity linking failed (non-fatal): {e!r}")
    return result
