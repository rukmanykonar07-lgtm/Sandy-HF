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

from cron.jobs import create_job
from llm import call_llm_with_fallback

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
        data = json.loads(raw)
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
    return data


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
    return (
        f"Theek hai Ruk! \"{skill}\" mein master banne ka mission shuru — "
        f"{days} din, roz ek session (~{hours_per_day}h target). "
        f"Job ID: {job.get('id', job)}. Progress notes roz milenge."
    )
