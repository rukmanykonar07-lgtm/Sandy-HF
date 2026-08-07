"""Skill-mastery engine: turns a chat request like "master web research,
3 days, 4 hours a day" into a real Hermes cron job. The job runs Sandy's
own Hermes-native agent (not our custom brain.py routing) with real
tools -- web search/extraction, code execution, and skill save/load --
so obstacles hit during a session can become permanent, reusable skills
instead of one-off notes.

Core-code self-modification is deliberately NOT auto-applied here -- see
the prompt template below. That's a separate, higher-risk piece.
"""
import json

from cron.jobs import create_job, list_jobs
from identity import SANDY_SYSTEM_PROMPT
from llm import call_llm_with_fallback, strip_json_fence

_IDENTITY_MSG = {"role": "system", "content": SANDY_SYSTEM_PROMPT}

MASTERY_PROMPT_TEMPLATE = """You are Sandy, working on becoming a master at: {skill}

This is one session of a multi-day mastery block for this skill.

Your job this session:
1. Check what you already know/have saved from earlier sessions on this skill (your existing skills/notes) -- build on it, don't restart from zero.
2. Do real work toward mastery: research, test tools/techniques, write and run code -- whatever the skill actually needs.
3. If you hit an obstacle you can't get past (a technique fails, a site blocks you, etc.), research a real fix and save it as a reusable skill so you never hit it again -- don't just note it and move on.
4. If truly fixing something would require editing Sandy's own core source code (main.py/brain.py/llm.py/memory.py/config.py) rather than just adding a skill -- STOP. Do not edit that code yourself. Write up exactly what you'd change and why, clearly, so Ruk can review and approve it later.
5. End the session with an honest progress note for Ruk: what got done, what's still missing, and whether you're at real master/expert level yet or need more sessions.

Be honest and concrete -- no filler, no claiming mastery you haven't actually reached."""


def extract_mastery_request(message: str) -> dict | None:
    """Ask an LLM whether this message is a mastery request and, if so,
    pull out skill/days/hours_per_day. Returns None if it isn't one."""
    prompt = (
        "Does this message ask Sandy to master/become expert at a skill, "
        "with some kind of time commitment (days, hours/day, etc.)?\n"
        f'Message: "{message}"\n'
        "If yes, respond with ONLY this JSON: "
        '{"is_mastery_request": true, "skill": "...", "days": <int>, "hours_per_day": <int>}\n'
        'If no, respond with ONLY: {"is_mastery_request": false}'
    )
    try:
        raw = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}])
        data = json.loads(strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    # Same fail-safe pattern as selfmod.extract_selfmod_request(): main.py
    # reads data["skill"]/["days"]/["hours_per_day"] unguarded, so a
    # malformed dict missing any of these would crash the same way the
    # selfmod KeyError did.
    if not isinstance(data, dict) or not data.get("is_mastery_request"):
        return None
    if not all(k in data for k in ("skill", "days", "hours_per_day")):
        return None
    if not isinstance(data["days"], (int, float)) or not isinstance(data["hours_per_day"], (int, float)):
        return None  # classifier found a skill but no real duration -- treat as not a complete mastery request
    return data


_pending_plans: dict[str, dict] = {}  # session_id -> {skill, days, hours_per_day, plan}


def propose_plan(session_id: str, skill: str, days: int, hours_per_day: float, feedback: str | None = None) -> str:
    """Full mastery plan document -- not just a one-liner. Before a
    multi-day unattended cron mission starts, Ruk should see exactly
    what Sandy understood, what she'll actually produce, how she'll go
    about it, and a rough day-by-day shape -- concrete enough to catch
    a misunderstanding before days of cron sessions run on it, not
    after. Stored as the pending plan for this session -- a follow-up
    message that isn't an approval gets treated as feedback and this
    regenerates the whole document, not a patch to one line."""
    prior = _pending_plans.get(session_id)
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
        "specific skill, not generic corporate filler\n"
        "4. DAY-BY-DAY -- a rough breakdown of what happens each of the " + str(days) + " days\n"
        "5. TOOLS -- which of your real tools (web search, code execution) you'll actually use, and why"
        + revision_note
    )
    plan = call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])
    _pending_plans[session_id] = {"skill": skill, "days": days, "hours_per_day": hours_per_day, "plan": plan}
    return plan


def get_pending_plan(session_id: str) -> dict | None:
    return _pending_plans.get(session_id)


def confirm_plan(session_id: str) -> str:
    """Ruk approved -- start the mission using the STORED params from
    the plan he actually saw and confirmed, not re-parsed from whatever
    his approval message happened to say."""
    pending = _pending_plans.pop(session_id, None)
    if not pending:
        return "Ruk, koi pending plan nahi mila is session ke liye -- pehle naya mastery request bhejo."
    return start_mastery(pending["skill"], pending["days"], pending["hours_per_day"])


def start_mastery(skill: str, days: int, hours_per_day: int) -> str:
    """Creates a real Hermes cron job: one mastery session per day, for
    `days` days. hours_per_day currently just shapes the prompt/expectation
    -- Hermes cron triggers a session, it doesn't block for N wall-clock
    hours, so this isn't a literal timer yet (flagging honestly, not
    pretending otherwise)."""
    job = create_job(
        prompt=MASTERY_PROMPT_TEMPLATE.format(skill=skill),
        schedule="0 9 * * *",
        name=f"mastery-{skill.replace(' ', '-').lower()}",
        repeat=days,
        skills=["default"],
        enabled_toolsets=["web", "code_execution", "skills"],
    )
    import config  # local import: config.py doesn't need Hermes, avoids a cycle at module load
    config.backup_hermes_jobs()
    return (
        f"Theek hai Ruk! \"{skill}\" mein master banne ka mission shuru — "
        f"{days} din, roz ek session (~{hours_per_day}h target). "
        f"Job ID: {job.get('id', job)}. Progress notes roz milenge."
    )


def list_mastery_jobs() -> list[dict]:
    """Every registered Hermes cron job (mastery sessions and anything
    else registered the same way), for Ruk's Home's Workflows view.
    Real data straight from Hermes -- not a mock."""
    return list_jobs(include_disabled=True)
