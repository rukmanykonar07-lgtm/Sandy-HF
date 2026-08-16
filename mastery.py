"""Skill-mastery engine: turns a chat request like "master web research,
3 days, 4 hours a day" into a real Hermes cron job. The job runs Sandy's
own Hermes-native agent (not our custom brain.py routing) with real
tools -- web search/extraction, code execution, and skill save/load --
so obstacles hit during a session can become permanent, reusable skills
instead of one-off notes.

Core-code self-modification is deliberately NOT auto-applied here -- see
the prompt template below. That's a separate, higher-risk piece.
"""
import os
import re
from pathlib import Path

from cron.jobs import (
    create_job, list_jobs, pause_job, resume_job, trigger_job, remove_job,
    update_job, resolve_job_ref, AmbiguousJobReference,
)
from identity import SANDY_SYSTEM_PROMPT
from llm import call_llm_with_fallback, log

_IDENTITY_MSG = {"role": "system", "content": SANDY_SYSTEM_PROMPT}

MASTERY_PROMPT_TEMPLATE = """You are Sandy, working on becoming a master at: {skill}

This is one session of a multi-day mastery block for this skill.

Ruk's approved approach for THIS skill (this is what he actually reviewed and confirmed -- follow it, don't substitute your own generic process instead):
{approach}

Your job this session:
1. Check what you already know/have saved from earlier sessions on this skill (your existing skills/notes) -- build on it, don't restart from zero.
2. Do real work toward mastery: research, test tools/techniques, write and run code -- whatever the skill actually needs. For skills commonly taught via video (editing, trading, SMMA, and similar), you have real YouTube transcript access (youtube.py) -- search for relevant tutorials via your web tool, then pull real transcripts to learn from instead of guessing at technique. This is unofficial/best-effort and can fail (no captions, blocked) -- if it fails, say so and move to another source, don't invent transcript content.
2b. BEFORE writing or changing ANY code this session (a script, a saved skill, anything): think it through for real first -- is this actually necessary, could an existing skill/piece be enhanced instead of adding something new, is what you're "fixing" actually broken (verify, don't assume), does this match Ruk's approved approach above rather than a guessed-at version of it, would this really work when Ruk/a future session actually uses it, and could it introduce a new bug. Only write the code after that reasoning, not before.
3. If the approach above calls for genuine PARALLEL multi-model work (e.g. separate workers on different providers, a planner + parallel workers + verifier pattern) -- use your real delegate_task tool with model= / override_provider= per child, in batch/parallel mode. This is the ONLY real way to run distinct providers as actual parallel workers. Do NOT attempt this via execute_code/code_execution -- provider API keys are deliberately stripped from that sandboxed environment for security (Hermes hardening against credential leaks) and any attempt to read them there will silently fail or come back empty. If delegate_task isn't available this session for some reason, say so plainly instead of faking parallel work with a single sequential pass.
4. If you hit an obstacle you can't get past (a technique fails, a site blocks you, etc.), research a real fix and save it as a reusable skill so you never hit it again -- don't just note it and move on.
5. If truly fixing something would require editing Sandy's own core source code (main.py/brain.py/llm.py/memory.py/config.py) rather than just adding a skill -- STOP. Do not edit that code yourself. Write up exactly what you'd change and why, clearly, so Ruk can review and approve it later.
6. End the session with an honest progress note for Ruk: what got done, what's still missing, whether any delegate_task workers actually ran (and on which providers), and whether you're at real master/expert level yet or need more sessions.
7. ALSO include a machine-readable EVENTS section (for the orb graph in Ruk's Home -- separate from the progress note above, one line per REAL thing you actually did this session, in this exact format, nothing invented):
EVENTS:
- type=<planning|worker_call|verify|conflict|retry_similar|synthesis|obstacle|skill_saved> | provider=<real provider name or none> | summary=<one line, what actually happened>
(one line per real event, as many as actually happened -- if nothing notable happened beyond the obvious, it's fine to have very few lines)

Be honest and concrete -- no filler, no claiming mastery -- or claiming parallel work happened -- that you haven't actually verified."""


_MAX_SKILL_NOTES_CHARS = 6000  # generous but bounded -- avoid unbounded growth in Supabase


def _pending_key(session_id: str) -> str:
    return f"mastery_pending:{session_id}"


def _explore_key(session_id: str) -> str:
    return f"mastery_pending_explore:{session_id}"


def save_skill_notes(skill: str, message: str) -> None:
    """Verbatim append -- NOT Mem0 fact-extraction. Real reason this
    exists: memory.recall() goes through Mem0's search, which extracts
    compact atomic facts (e.g. "Ruk prefers X") -- it is not built to
    preserve a multi-paragraph technical design (an orchestration plan,
    specific phases, specific tools) intact. A detailed design Ruk gives
    for a skill risks coming back lossy/compressed, or not at all, via
    recall() alone. This stores his own words for that skill, unmodified,
    so a later plan can quote his actual design instead of a fuzzy
    approximation of it. Best-effort: a config write failure here
    shouldn't block the conversation, so it's caught by the caller."""
    import config  # local import: avoids a cycle at module load, same pattern as backup_hermes_jobs
    key = f"skill_notes:{skill.strip().lower()}"
    existing = config.get_config(key) or ""
    combined = (existing + "\n\n---\n\n" + message) if existing else message
    if len(combined) > _MAX_SKILL_NOTES_CHARS:
        combined = combined[-_MAX_SKILL_NOTES_CHARS:]  # keep the most recent detail, not the oldest
    config.set_config(key, combined)


def get_skill_notes(skill: str) -> str | None:
    import config
    return config.get_config(f"skill_notes:{skill.strip().lower()}")


def set_pending_explore(session_id: str, skill: str) -> None:
    import config
    config.set_config(_explore_key(session_id), skill)


def pop_pending_explore(session_id: str) -> str | None:
    import config
    skill = config.get_config(_explore_key(session_id))
    if skill:
        config.set_config(_explore_key(session_id), None)
    return skill


def get_pending_explore(session_id: str) -> str | None:
    import config
    return config.get_config(_explore_key(session_id))


def explain_flow(skill: str, message: str, context: str = "") -> str:
    """Real, grounded answer for mastery-job talk that isn't yet a full
    skill+days+hours request -- covers both the FIRST exploratory message
    and any follow-up in the same thread (e.g. 'how will Hermes actually
    build it', 'explain so I can review/edit', 'where can I see it').

    Bug this fixes: this used to only take `skill` and always produce the
    same canned mechanism-explanation-then-ask-for-days/hours answer no
    matter what was actually asked -- so three different follow-up
    questions in a row got three near-identical answers, none of which
    engaged with the actual question. Now the real message text drives
    the answer; the mechanism facts are grounding to answer FROM, not a
    script to recite every time.
    """
    real_mechanism = (
        "REAL mechanism, exactly as it works in the actual code (use only what's relevant "
        "to what he actually asked -- don't recite all of this every time):\n"
        "1. A mastery job is a real Hermes cron job (cron.jobs.create_job) -- "
        "it runs through Hermes's OWN native agent runtime, using MASTERY_PROMPT_TEMPLATE, "
        "NOT Sandy's normal brain.py chat routing, and NOT selfmod.py's single-file edit flow.\n"
        "2. Before anything is registered, Sandy proposes a plan (skill, days, hours/day, "
        "tools, and optionally Ruk's own described build approach) and Ruk must confirm it -- "
        "nothing runs unapproved. Ruk CAN give feedback/edits before confirming -- the plan "
        "isn't final until he says confirm.\n"
        "3. Once confirmed, the job is written to ~/.hermes/cron/jobs.json and immediately "
        "backed up to Supabase (config.backup_hermes_jobs) so an HF rebuild can't lose it.\n"
        "4. The Hermes gateway's own background ticker (a separate process from chat) checks "
        "that file every 60 seconds and runs a session when one's due -- fully autonomous, "
        "no chat message needed to trigger a run. The schedule is a fixed daily time (9 AM), "
        "not 'runs immediately when confirmed'.\n"
        "5. Each session runs on HERMES'S OWN generic agent loop (its own think-act-observe "
        "cycle, tool calling, skill save/load) -- it is NOT Sandy's own brain.py multi-model "
        "orchestrator (planner/parallel-workers/verifier-loop). If Ruk wants that specific "
        "orchestration pattern to actually drive a session, it has to be written into the "
        "job's own prompt as explicit instructions (see 'Ruk's approach' below) -- Hermes "
        "doesn't inherit it automatically just because it exists elsewhere in Sandy's code.\n"
        "6. Sessions can research (including pulling YouTube video transcripts for skills "
        "that are commonly taught there, like editing/trading/SMMA), test, write/run code, "
        "and save reusable skills to ~/.hermes/skills/ -- separate from Sandy's own core .py "
        "files. If something would genuinely require editing Sandy's own core code, the job "
        "stops and writes that up for Ruk to review instead of doing it unapproved.\n"
        "7. Every session's real progress note is written to ~/.hermes/cron/output/{job_id}/ "
        "-- visible in Ruk's Home: a summary in Command Center, live status in Workflows, and "
        "the actual output in Agents once a session completes.\n"
    )
    prompt = (
        f"Ruk is talking about \"{skill}\" as a mastery job. His message this turn: "
        f'"{message}"\n\n'
        "Answer THIS specific message directly and concretely -- don't just recite a generic "
        "mechanism speech if he's asking something more specific (e.g. how Hermes's own agent "
        "loop relates to Sandy's own orchestration patterns, whether he can edit the plan, "
        "where output shows up). If the background facts above include Ruk's OWN verbatim "
        "words about this skill (marked as such), that IS his real design -- reflect it back "
        "concretely and specifically, don't flatten it into a generic description. Use the "
        "mechanism facts below only where they're actually relevant to what he asked. If a "
        "time commitment (days/hours-per-day) hasn't been given yet, end by asking for it -- "
        "but only after actually answering his real "
        "question, not instead of it.\n\n" + real_mechanism
    )
    if context:
        prompt = f"{context}\n\n{prompt}"
    return call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])


def propose_plan(
    session_id: str, skill: str, days: int, hours_per_day: float,
    feedback: str | None = None, context: str = "",
) -> str:
    """Full mastery plan document -- not just a one-liner. Before a
    multi-day unattended cron mission starts, Ruk should see exactly
    what Sandy understood, what she'll actually produce, how she'll go
    about it, and a rough day-by-day shape -- concrete enough to catch
    a misunderstanding before days of cron sessions run on it, not
    after. Stored as the pending plan for this session -- a follow-up
    message that isn't an approval gets treated as feedback and this
    regenerates the whole document, not a patch to one line.

    Root-cause fix: an edit message ("actually make it 5 days") used to
    only change the DISPLAYED plan text -- confirm_plan() still used the
    ORIGINAL days/hours_per_day underneath, so what Ruk approved and what
    actually got registered as the cron job could silently differ. Now
    feedback is re-parsed with the same days/hours regex main.py's
    classifier fallback uses, and any number actually mentioned this
    turn overrides the stored value for real."""
    import config
    prior = config.get_config(_pending_key(session_id))
    if feedback and prior:
        days_m = re.search(r"(\d+(?:\.\d+)?)\s*day", feedback, re.I)
        hours_m = re.search(r"(\d+(?:\.\d+)?)\s*hour", feedback, re.I)
        if days_m:
            days = float(days_m.group(1))
        if hours_m:
            hours_per_day = float(hours_m.group(1))
    revision_note = ""
    if feedback and prior:
        revision_note = (
            f"\n\nRuk already saw this earlier draft:\n{prior['plan']}\n\n"
            f'He wants this changed: "{feedback}"\n'
            "Rewrite the FULL plan incorporating that feedback -- don't just patch one line."
        )
    prompt = (
        f'Ruk asked Sandy to master "{skill}" over {days} days, ~{hours_per_day}h/day. '
        "Write a full plan document, in Hinglish, covering ALL of these sections clearly "
        "(use these as headers):\n"
        "1. UNDERSTANDING -- what you understand the actual goal to be, in your own words\n"
        "2. WHAT YOU'LL MAKE -- the concrete deliverable(s), specifically, not vague\n"
        "3. PROCESS -- how you'll actually go about it: research first, then think through "
        "approaches, then build/practice, then review and iterate -- concrete for THIS "
        "specific skill, not generic corporate filler. If Ruk has described a specific "
        "orchestration/loop/multi-step approach for this skill in the background facts "
        "above, or if a multi-phase approach (plan -> parallel work -> verify/cross-check -> "
        "synthesize) genuinely fits the skill, describe it concretely here -- this exact "
        "section is what actually runs each session, word for word, not just a proposal.\n"
        "4. DAY-BY-DAY -- a rough breakdown of what happens each of the " + str(days) + " days\n"
        "5. TOOLS -- which of your real tools you'll actually use, and why: web search, "
        "code execution (research/testing/scripts -- NOT for calling provider APIs "
        "directly, those keys are sandboxed away from it), YouTube transcript access for "
        "video-taught skills like editing/trading/SMMA, and delegate_task (with "
        "model=/override_provider=) if genuine PARALLEL multi-model work is part of the "
        "approach -- that's the only real mechanism for actual parallel providers, not "
        "code execution\n"
        "6. SUCCESS CRITERIA -- concrete, checkable signs of real progress, not vague growth"
        + revision_note
    )
    if context:
        prompt = f"{context}\n\n{prompt}"
    plan = call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])
    config.set_config(_pending_key(session_id), {"skill": skill, "days": days, "hours_per_day": hours_per_day, "plan": plan})
    return plan


def get_pending_plan(session_id: str) -> dict | None:
    import config
    return config.get_config(_pending_key(session_id))


def confirm_plan(session_id: str) -> str:
    """Ruk approved -- start the mission using the STORED params from
    the plan he actually saw and confirmed, not re-parsed from whatever
    his approval message happened to say. The exact plan text he approved
    (including any orchestration/approach detail it described) gets baked
    into the real job prompt below -- what Ruk approved is what actually
    runs, not a generic template that ignores it."""
    import config
    pending = config.get_config(_pending_key(session_id))
    if not pending:
        return "Ruk, koi pending plan nahi mila is session ke liye -- pehle naya mastery request bhejo."
    config.set_config(_pending_key(session_id), None)
    return start_mastery(pending["skill"], pending["days"], pending["hours_per_day"], approach=pending.get("plan"))


def _next_9am_ist() -> str:
    """Real next-run time for the '0 9 * * *' schedule, computed honestly
    instead of implying the job starts immediately. Assumes the
    HERMES_TIMEZONE=Asia/Kolkata Dockerfile fix is deployed -- if it isn't
    yet, this label is wrong until the next rebuild (flagging in code,
    not pretending this is unconditionally correct)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target.strftime("%d %b, %I:%M %p IST")


def start_mastery(skill: str, days: int, hours_per_day: int, approach: str | None = None) -> str:
    """Creates a real Hermes cron job: one mastery session per day, for
    `days` days. hours_per_day currently just shapes the prompt/expectation
    -- Hermes cron triggers a session, it doesn't block for N wall-clock
    hours, so this isn't a literal timer yet (flagging honestly, not
    pretending otherwise).

    `approach` -- the actual plan text Ruk reviewed and confirmed (from
    propose_plan), baked verbatim into the real job prompt below. This is
    what makes Ruk's own described build approach/orchestration for THIS
    skill genuinely drive the session, instead of every job just getting
    the same generic template regardless of what was actually agreed."""
    log(f"[start_mastery] calling real create_job() -- skill={skill!r} days={days} schedule=0 9 * * *")
    job = create_job(
        prompt=MASTERY_PROMPT_TEMPLATE.format(
            skill=skill,
            approach=approach or "No specific approach was given beyond this template -- design a reasonable one yourself and say what it is in your progress note.",
        ),
        schedule="0 9 * * *",
        name=f"mastery-{skill.replace(' ', '-').lower()}",
        repeat=days,
        skills=[],  # scode: "default" isn't a real skill name -- Hermes looked it up literally and
        # warned "skill(s) not found and skipped: default" every single run (confirmed from a real
        # failed-run log). No skill exists for a brand-new mastery topic yet; empty list, not a guess.
        enabled_toolsets=["web", "code_execution", "skills", "delegation"],
        provider="gemini",  # scode: CORRECTED (see config.yaml comment for the full story) --
        # without this, Hermes's resolve_requested_provider() falls through to "auto" -> OpenRouter,
        # which then rejected the model string below as an invalid OpenRouter slug. This matches
        # the named "gemini" entry under config.yaml's providers: section.
        model="gemini-3.5-flash",  # bare name, no "gemini/" prefix -- that litellm-style prefix
        # (used elsewhere in this repo, e.g. llm.py) is NOT Hermes's own convention; with provider
        # given explicitly above, Hermes wants just the bare model name for that provider.
    )
    log(f"[start_mastery] create_job() returned: {job!r}")
    import config  # local import: config.py doesn't need Hermes, avoids a cycle at module load
    config.backup_hermes_jobs()
    log(f"[start_mastery] backup_hermes_jobs() done, jobs.json should now exist on disk")
    try:
        next_run = _next_9am_ist()
    except Exception:
        next_run = "next 9 AM IST slot (couldn't compute the exact date)"
    return (
        f"Theek hai Ruk! \"{skill}\" mastery job REGISTERED — {days} din, roz ek session "
        f"(~{hours_per_day}h target). Job ID: {job.get('id', job)}.\n\n"
        f"Real baat: ye \"started\" nahi hai in the sense of running right now -- pehla real "
        f"session {next_run} ko fire hoga (roz 9 AM slot pe, immediately nahi). Do cheezein "
        "jo first run ko block kar sakti hain: (1) agar HF Space us waqt sleep mein hai "
        "(free tier, 48h inactivity), cron tick hi nahi chalega us run ke liye; (2) hours_per_day "
        "abhi ek literal timer nahi hai, sirf session ke liye target guidance hai. "
        "Confirm karne ka real tarika: Agents tab mein pehla real output tab dikhega jab "
        "session genuinely chal chuka hoga -- 9 AM se pehle kuch na dikhna normal hai."
    )


_EVENT_LINE = re.compile(
    r"type=(\w+)\s*\|\s*provider=([\w\-]+|none)\s*\|\s*summary=(.+)", re.I
)


def backfill_events_from_output(job_id: str) -> int:
    """Parses the EVENTS section out of each real output file for this
    job and logs any not-already-logged ones to events.py, so the
    Hermes-side graph has real data too -- Hermes's own agent can't
    write to Supabase directly (its code_execution sandbox strips
    provider/Supabase keys, same hardening covered in the prompt
    template above), so this reads back what it already wrote to disk
    instead. Idempotent: uses run_id=f"hermes:{job_id}:{timestamp}" per
    file, checked against existing events before inserting, so re-polling
    the same output never double-logs. Returns how many new events were
    added this call."""
    import events
    added = 0
    for out in job_output(job_id):
        run_id = f"hermes:{job_id}:{out['timestamp']}"
        if events.get_events(run_id):
            continue  # already backfilled this file
        lines = out["content"].splitlines()
        try:
            start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("EVENTS:"))
        except StopIteration:
            continue
        for line in lines[start + 1:]:
            m = _EVENT_LINE.search(line)
            if not m:
                continue
            event_type, provider, summary = m.groups()
            events.log_event(
                run_id, "hermes", event_type.lower(), summary.strip(),
                provider=None if provider.lower() == "none" else provider,
            )
            added += 1
        if added:
            events.log_event(run_id, "hermes", "output", f"Session output ({out['timestamp']})", detail=out["content"])
    return added


def list_mastery_jobs() -> list[dict]:
    """Every registered Hermes cron job (mastery sessions and anything
    else registered the same way), for Ruk's Home's Workflows view.
    Real data straight from Hermes -- not a mock."""
    return list_jobs(include_disabled=True)


def pause_mastery_job(job_ref: str) -> str:
    """job_ref: real Hermes job id OR name (e.g. 'mastery-video-editing')
    -- resolve_job_ref (inside pause_job) accepts either. Real Hermes
    function, previously imported by nothing in this repo."""
    try:
        job = pause_job(job_ref)
    except AmbiguousJobReference as e:
        return f"Ruk, '{job_ref}' se multiple jobs match ho rahe hain -- exact job ID do: {e}"
    if not job:
        return f"Ruk, '{job_ref}' naam/id ka Hermes job nahi mila."
    return f"Ruk, Hermes job '{job['name']}' ({job['id']}) pause ho gaya."


def resume_mastery_job(job_ref: str) -> str:
    try:
        job = resume_job(job_ref)
    except ValueError as e:
        return f"Ruk, resume nahi kar sakti: {e}"
    except AmbiguousJobReference as e:
        return f"Ruk, '{job_ref}' se multiple jobs match ho rahe hain -- exact job ID do: {e}"
    if not job:
        return f"Ruk, '{job_ref}' naam/id ka Hermes job nahi mila."
    return f"Ruk, Hermes job '{job['name']}' ({job['id']}) resume ho gaya, agla run {job.get('next_run_at', 'unknown')}."


def trigger_mastery_job_now(job_ref: str) -> str:
    """Real 'run the next chunk right now' -- schedules the job for the
    NEXT scheduler tick (within ~60s) instead of waiting for 9 AM."""
    try:
        job = trigger_job(job_ref)
    except AmbiguousJobReference as e:
        return f"Ruk, '{job_ref}' se multiple jobs match ho rahe hain -- exact job ID do: {e}"
    if not job:
        return f"Ruk, '{job_ref}' naam/id ka Hermes job nahi mila."
    return f"Ruk, Hermes job '{job['name']}' ({job['id']}) agle tick (~60s) pe chalega."


def remove_mastery_job(job_ref: str) -> str:
    try:
        ok = remove_job(job_ref)
    except AmbiguousJobReference as e:
        return f"Ruk, '{job_ref}' se multiple jobs match ho rahe hain -- exact job ID do: {e}"
    return (f"Ruk, '{job_ref}' Hermes job remove ho gaya." if ok
            else f"Ruk, '{job_ref}' naam/id ka Hermes job nahi mila.")


def edit_mastery_job(job_ref: str, updates: dict) -> str:
    """Real param edit on an EXISTING Hermes job -- e.g. change which
    provider/model it uses. Anything Hermes's own update_job() accepts
    (schedule, repeat, provider, model, prompt, etc)."""
    try:
        job = resolve_job_ref(job_ref)
    except AmbiguousJobReference as e:
        return f"Ruk, '{job_ref}' se multiple jobs match ho rahe hain -- exact job ID do: {e}"
    if not job:
        return f"Ruk, '{job_ref}' naam/id ka Hermes job nahi mila."
    updated = update_job(job["id"], updates)
    return f"Ruk, Hermes job '{updated['name']}' update ho gaya: {updates}."


_OUTPUT_DIR = Path(os.environ.get("HERMES_HOME", "/root/.hermes")) / "cron" / "output"


def job_output(job_id: str, limit: int = 5) -> list[dict]:
    """Real progress notes Hermes wrote for this job after each session
    (~/.hermes/cron/output/{job_id}/{timestamp}.md), most recent first,
    for Ruk's Home's Agents view -- what a mastery job actually built,
    not a narrated summary. Empty list if the job hasn't run yet (not
    an error -- a freshly created job legitimately has no output until
    its first scheduled run)."""
    d = _OUTPUT_DIR / job_id
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.md"), reverse=True)[:limit]
    return [
        {"timestamp": f.stem, "content": f.read_text(encoding="utf-8", errors="replace")}
        for f in files
    ]
