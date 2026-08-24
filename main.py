"""
Sandy's single entry point. One endpoint: POST /chat.

Flow per message:
  1. recall relevant memory for context
  2. if it's a config change ("set gemini cap to 100") -> apply it, reply, done
  3. if the task is risky (or Ruk opted it into approval) -> ask first
  4. otherwise -> classify/route/orchestrate, remember the exchange, reply
"""
import json
import os
import re
import asyncio
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chatlog
import codebase
import config
import diagnostics
import healing
import search
from identity import SANDY_SYSTEM_PROMPT, CAPABILITIES
import memory
import brain
import mastery
import native_mastery
import events
import selfmod
import youtube
from llm import call_llm, call_llm_with_fallback, CapExceeded, MODELS, strip_json_fence, log

@asynccontextmanager
async def _healing_loop_lifespan(app: FastAPI):
    """Real periodic failure check -- every 5 min while the container is
    awake. Honest limitation: HF Spaces free tier sleeps on idle, so this
    loop sleeps with it -- it's not a substitute for the opportunistic
    check in _handle_chat (which fires on every real message, the more
    reliable of the two), just an addition for whenever the container
    happens to be up with nobody actively chatting."""
    if not os.environ.get("SANDY_AUTH_KEY"):
        log("[auth] SANDY_AUTH_KEY not set -- auth gate is OPEN; set the HF secret to lock /chat down")
    loop_task = asyncio.create_task(_heal_poll_loop())
    yield
    loop_task.cancel()


async def _heal_poll_loop():
    while True:
        try:
            alerts = await asyncio.to_thread(healing.run_check_and_alert)
            if alerts:
                log(f"[healing] periodic loop: {len(alerts)} new failure(s) detected and alerted")
        except Exception as e:
            log(f"[healing] periodic loop error, continuing: {e!r}")
        await asyncio.sleep(300)


app = FastAPI(lifespan=_healing_loop_lifespan)


_TRIVIAL_WORDS = {
    "hey", "hi", "hello", "yo", "sup", "wassup", "sandy", "bro", "ruk",
    "whats", "what's", "up", "kya", "haal", "hai", "kaisa", "kaise", "ho",
    "thanks", "thank", "you", "ok", "okay", "k", "cool", "nice", "great",
    "good", "bye", "gm", "gn", "morning", "night", "acha", "theek", "thik",
    "hows", "how's", "going", "there", "it", "man", "today",
}


def _is_trivial(text: str) -> bool:
    """Greetings and one-word acknowledgments (in any order, however
    Ruk actually addresses her -- 'hey sandy whats up', 'sandy hows it
    going', etc) aren't worth an LLM extraction call. Mem0 was storing
    'Ruk asked what's up on July 20' as if it were a real fact about
    him, then recalling that noise back at him on the next 'hey'.
    Skipping extraction here fixes that AND cuts a real Groq call per
    trivial message. Capped at 6 words: long enough to catch real
    greetings, short enough that a genuine question built from common
    words doesn't accidentally get treated as trivial."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words or len(words) > 6:
        return False
    return all(w in _TRIVIAL_WORDS for w in words)


# scode: real, measured cost this fixes -- classify_message()'s system
# prompt alone is ~2,300 tokens, paid as its OWN Groq round trip before
# Ruk's actual answer even starts generating, on EVERY message that
# isn't a literal _is_trivial greeting. A genuinely ordinary question
# ("explain recursion", "what's 2+2", "how's your day") has zero real
# chance of matching any of the 15 tracked axes (mastery/selfmod/
# config/logs/search/jobs/etc) but paid that full cost anyway.
#
# Deliberately broad/permissive in the direction that matters: every
# one of these 15 fields fails SAFE when wrongly left off -- Sandy just
# answers normally instead of taking a special action, Ruk notices and
# rephrases. There's no unsafe outcome from an over-inclusive keyword
# list here, only a missed savings opportunity -- which is the
# acceptable side to err on. When in doubt, this returns None and the
# real classifier still runs, unchanged.
_SPECIAL_SIGNAL_WORDS = {
    # config/caps
    "cap", "caps", "limit", "quota", "credit", "credits",
    # selfmod / code changes / codebase review
    "edit", "rollback", "revert", "code", ".py", "file", "repo",
    "review", "scan", "codebase", "refactor", "fix",
    # mastery (both engines) + hermes job edits
    "master", "mastery", "skill", "hermes", "native", "job", "jobs",
    "cron", "orchestrat", "pin", "weight", "weights",
    # logs / internal diagnosis
    "log", "logs", "error", "fail", "failed", "failing", "broke",
    "broken", "crash", "bug", "issue", "problem", "diagnose", "debug",
    "why is", "whats wrong", "what's wrong",
    # search
    "search", "look up", "lookup", "google", "find out", "latest",
    "current", "news", "research",
    # job status / push history / capabilities
    "status", "running", "push", "commit", "deploy", "deployed",
    "capable", "capability", "capabilities", "what can you", "can you do",
}


def _fast_classification(message: str) -> dict:
    """Same shape classify_message() returns, every special axis off --
    used only when _maybe_skip_classifier already confirmed none of
    them could plausibly apply."""
    word_count = len(message.split())
    return {
        "config_change": None, "selfmod": None, "mastery": None,
        "mastery_explore": None, "mastery_control": None,
        "hermes_job_edit": None, "native_job_edit": None, "codebase_analysis": False,
        "logs_request": False, "search_needed": False,
        "llm_override": None,
        "complexity": "simple" if word_count <= 12 else "medium",
        "job_status_request": False, "push_history_request": False,
        "capabilities_request": False,
    }


def _maybe_skip_classifier(message: str) -> dict | None:
    """Returns a cheap default classification when this message has no
    real chance of needing the full 15-axis classifier, or None to
    fall through to the real thing (unchanged behavior)."""
    low = message.lower()
    provider_words = {p.lower() for p in MODELS} | {"orchestrator"}
    if any(w in low for w in _SPECIAL_SIGNAL_WORDS) or any(w in low for w in provider_words):
        return None
    return _fast_classification(message)


_AFFIRMATIVE_TOKENS = {
    "yes", "ya", "yeah", "yep", "sure", "ok", "okay",
    "haan", "han", "ha", "hn", "haanji",
    "fix", "it", "kar", "do", "karo", "kardo",
    "solve", "confirm", "apply",
    "theek", "thik", "hai", "sahi",
    "go", "ahead",
}


def _is_short_affirmative(text: str) -> bool:
    """Every real word in the message is in a small affirmative
    whitelist, and there aren't many of them. Token-based on purpose --
    scode: real bug found live -- a rigid phrase-enumeration regex
    required an exact match against one whole alternative, so "haan, fix
    it" (comma-joined) matched none of them and fell through to the
    classifier. Checking word-by-word instead of phrase-by-phrase closes
    the whole class of near-miss, not just that one exact wording."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words or len(words) > 5:
        return False
    return all(w in _AFFIRMATIVE_TOKENS for w in words)


def _remember(background_tasks: BackgroundTasks, text: str, role: str) -> None:
    """chat_log gets written SYNCHRONOUSLY, before the response returns --
    it's a fast, cheap DB insert, and Ruk closing the tab right after
    getting a reply was racing the old background-task write, causing
    the exact 'latest messages disappear on reopen' bug he reported.
    Mem0's fact extraction stays backgrounded since that's the genuinely
    slow, LLM-based part actually worth deferring -- no reason to make
    Ruk wait on it just to see a message he already got."""
    chatlog.log(text, role=role)
    if not _is_trivial(text):
        background_tasks.add_task(memory.remember, text, role=role)

def _recall_context(query: str, skill: str | None = None) -> str:
    """Same recalled-memory framing used on the generic chat path, reused
    here for mastery plan proposals. Two sources, checked in this order:
    1. Verbatim skill notes (mastery.get_skill_notes) -- Ruk's own exact
       words for THIS skill, if he's described one. Not lossy.
    2. Mem0's fact recall -- broader personal context, but fact-extracted
       (see mastery.save_skill_notes's docstring for why that's not
       enough on its own for a detailed technical design).
    Best-effort throughout: memory being briefly unavailable shouldn't
    block a plan."""
    parts = []
    if skill:
        try:
            notes = mastery.get_skill_notes(skill)
        except Exception as e:
            log(f"[_recall_context] get_skill_notes failed: {e!r}")
            notes = None
        if notes:
            parts.append(f"Ruk's OWN exact words about \"{skill}\" from earlier messages (verbatim -- treat this as authoritative over any generic assumption):\n{notes}")
    try:
        recalled = memory.recall(query)
    except Exception as e:
        log(f"[_recall_context] memory.recall failed: {e!r}")
        recalled = []
    if recalled:
        parts.append("Background facts Sandy remembers about Ruk (use only what's relevant):\n" + "\n".join(recalled))
    return "\n\n".join(parts)


app.add_middleware(
    CORSMiddleware,
    # Real frontend origins only -- "*" let any site fire quota-burning
    # and approval-driving requests from a visitor's browser.
    allow_origins=[o for o in [
        os.environ.get("RUKS_HOME_ORIGIN"),   # set this HF secret to the real Ruk's Home URL
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ] if o],
    allow_methods=["GET", "POST"],
    allow_headers=["X-Sandy-Key", "Content-Type"],
)


class AuthGateMiddleware(BaseHTTPMiddleware):
    """Shared-secret gate: every /chat, /status and /api/* request must
    carry X-Sandy-Key matching the SANDY_AUTH_KEY secret. Health/static
    routes stay open so the Space's own liveness checks and the bundled
    frontend assets keep working. With no key configured the gate is
    open (fresh deploys don't lock themselves out), but a missing key is
    logged loudly on startup so it can't stay unnoticed."""

    OPEN_PATH_PREFIXES = ("/health", "/history", "/manifest.json", "/service-worker.js", "/static", "/assets")

    async def dispatch(self, request: Request, call_next):
        expected = os.environ.get("SANDY_AUTH_KEY")
        path = request.url.path
        if not expected or path == "/" or any(path.startswith(p) for p in self.OPEN_PATH_PREFIXES):
            return await call_next(request)
        provided = request.headers.get("x-sandy-key")
        if provided and secrets.compare_digest(provided, expected):
            return await call_next(request)
        log(f"[auth] rejected unauthorized {request.method} {path}")
        return JSONResponse(status_code=401, content={"detail": "invalid or missing X-Sandy-Key"})


app.add_middleware(AuthGateMiddleware)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    approved: bool = False       # frontend resends with this =true after Ruk confirms
    override_llms: list[str] | None = None  # e.g. ["gemini"] or ["orchestrator"]
    search_provider: str | None = None  # "tavily" / "exa" / "linkup", or None = auto


class ChatResponse(BaseModel):
    reply: str
    needs_approval: bool = False


def _extract_search_provider(message: str) -> str | None:
    """Cheap keyword check (no LLM call needed) for an explicit provider
    name in the message -- 'use exa for this' etc."""
    lower = message.lower()
    for p in ("tavily", "exa", "linkup"):
        if p in lower:
            return p
    return None


def _wants_all_providers(message: str) -> bool:
    """Cheap keyword check (no LLM call) for 'use all three search
    engines and show me each separately' style requests. Root-cause fix
    for a real hallucination: the old code only ever called ONE search
    provider, so when Ruk asked for a genuine 3-way breakdown, the
    model had nothing real to split and fabricated one instead."""
    lower = message.lower()
    return ("all three" in lower or "each of" in lower or "separately" in lower) and (
        "search" in lower or "tavily" in lower or "exa" in lower or "linkup" in lower
    )


def classify_message(message: str, pending_skill: str | None = None) -> dict:
    """ONE Groq call that replaces what used to be 7 separate classifier
    calls (config-change, selfmod-extract, mastery-extract, codebase-
    analysis, logs-request, search-need, llm-override) -- that stack was
    the actual root cause of the Groq daily-quota exhaustion, since every
    single message paid for 7-8 classification round trips before any
    real answer even started, on top of Mem0's own internal Groq call.

    temperature=0: this is structured classification, not creative
    writing -- consistency matters far more than variety here, and it
    also cuts down on malformed JSON / classifier misfires (which is
    what was likely making search "randomly not trigger").

    scode: one big prompt beats seven small ones -- same provider, same
    trust boundary, one round trip instead of seven. All the original
    fail-safe guards (never invent a filename, validate mastery field
    types, reject hallucinated provider names) are preserved below,
    checked against this single response instead of seven."""
    providers = list(MODELS)
    prompt = (
        "Classify this one message from Ruk to Sandy across ALL of these axes at once. "
        "Answer ONLY with this exact JSON shape, no markdown fences, no commentary:\n"
        "{\n"
        '  "config_change": {"is": true, "key": "caps", "value": {"provider_name": 100}} or null,\n'
        '  "selfmod": {"action": "edit", "file_path": "...", "instruction": "..."} '
        'or {"action": "history"} or {"action": "rollback", "commit_hash": "..."} or null,\n'
        '  "mastery": {"skill": "...", "days": 3, "hours_per_day": 4, "engine": "hermes"} or null,\n'
        '  "mastery_explore": {"skill": "...", "engine": "hermes"} or null,\n'
        '  "mastery_control": {"action": "pause", "engine": "native", "job_ref": null} or null,\n'
        '  "hermes_job_edit": {"job_ref": "...", "updates": {"provider": "gemini"}} or null,\n'
        '  "native_job_edit": {"job_ref": "...", "updates": {"weights": {"gemini": 70}, "caps": {}, "mode": null}} or null,\n'
        '  "codebase_analysis": false,\n'
        '  "logs_request": false,\n'
        '  "search_needed": false,\n'
        '  "llm_override": ["provider_name"] or null,\n'
        '  "complexity": "simple",\n'
        '  "job_status_request": false,\n'
        '  "push_history_request": false,\n'
        '  "capabilities_request": false\n'
        "}\n\n"
        "Rules for each field:\n"
        "- config_change: true ONLY for changing an LLM credit cap/limit number. NEVER true "
        "for requests to edit Sandy's own code/files, her identity/personality, or how she "
        "talks/behaves -- those go through selfmod or normal conversation, even if worded "
        "like a preference.\n"
        "- selfmod: 'edit' ONLY if a specific filename is literally written in the message -- "
        "a general 'build me X' or 'fix the bug where...' with no filename named is NOT edit, "
        "use null instead. Never invent a filename. 'history' = wants to see past code-change "
        "log. 'rollback' = wants to undo a specific past commit (needs a commit hash).\n"
        "- mastery: only set if Ruk asks Sandy to master/become expert at a skill AND gives "
        "some time commitment (days and/or hours per day). If either is missing, use null.\n"
        "- mastery_explore: set ONLY if Ruk is discussing/planning STARTING or BUILDING "
        "something as/via a NEW mastery job, WITHOUT a full time commitment yet (e.g. 'I want "
        "to build X through mastering jobs', 'what's next for the Y mastery job'). Extract the "
        "skill/project name. If mastery (above) is set, this should be null -- mutually "
        "exclusive. Do NOT set this for a question ABOUT the mastery feature/code itself -- "
        "'what's new in the mastery feature', 'explain your mastery code/files', 'read "
        "mastery.py and say what changed' are codebase_analysis, not mastery_explore, even "
        "though they mention the word 'mastery'.\n"
        "- engine (inside mastery/mastery_explore): 'hermes' by default (Hermes's own cron "
        "agent builds it autonomously). Set 'native' if Ruk explicitly wants HIS OWN "
        "orchestration/his own logic running it directly (not handed to Hermes) -- phrases "
        "like 'my own mastery', 'native', 'my orchestration', 'without hermes'. Set 'both' "
        "if he wants to compare -- 'compare', 'both ways', 'mine vs hermes'.\n"
        "- mastery_control: set ONLY when Ruk wants to PAUSE, RESUME, CONTINUE (run the next "
        "chunk of an already-scheduled job right now), or REMOVE/DELETE an EXISTING mastery "
        "job -- not proposing/starting a new one, not asking about status (that's "
        "job_status_request). engine: 'native' for his own orchestration jobs, 'hermes' for "
        "cron jobs -- guess from context if not explicit. job_ref: the specific job id or "
        "name if he names/quotes one, else null.\n"
        "- hermes_job_edit: set when Ruk wants to PIN, UPDATE, or CHANGE a real PARAMETER "
        "(provider, model, schedule, skills, prompt) on an EXISTING Hermes cron job -- 'pin "
        "job X to gemini', 'update job Y's schedule', 'change the model on Z'. NOT proposing a "
        "new job (mastery/mastery_explore), NOT pause/resume/continue/remove (mastery_control). "
        "job_ref: the job id or name Ruk gave. updates: ONLY the exact key(s)/value(s) Ruk "
        "actually said -- e.g. he says 'pin to gemini' -> {\"provider\": \"gemini\"} only. "
        "NEVER add a key he didn't mention (no inventing a 'skills' or 'model' value he never "
        "gave you, even if it seems like it would help) -- an edit with a value Ruk didn't "
        "actually ask for is not a real edit, it's a fabrication with real side effects.\n"
        "- native_job_edit: set when Ruk wants to CHANGE the provider weights, per-job caps, "
        "or mode on an EXISTING, already-CONFIRMED native (his-own-orchestration) mastery job "
        "-- 'change job X's weights to gemini 70 groq 30', 'set cerebras cap to 20 on Y', "
        "'switch Z to scheduled mode'. NOT proposing a new job, NOT pause/resume/remove "
        "(mastery_control). job_ref: the job id or name Ruk gave. updates.weights/caps: ONLY "
        "the exact provider(s)/number(s) Ruk actually said, as a partial dict -- never invent "
        "or complete a full split he didn't state (e.g. he says 'bump gemini to 70' -> "
        "{\"weights\": {\"gemini\": 70}}, NOT a full 100%-summing dict you made up to fill the "
        "rest). updates.mode: only if he explicitly names continuous/scheduled.\n"
        "- codebase_analysis: true for READ-ONLY review/scan/analyze of Sandy's own "
        "source code -- not asking to change/fix/edit anything (that's selfmod's job). This "
        "INCLUDES questions about a specific feature's code/files -- 'what's new in the "
        "mastery code', 'read mastery.py/brain.py and explain what changed', 'explain your "
        "OSINT code' -- even when the feature name matches another axis's name.\n"
        "- logs_request: true if Ruk is asking why something in Sandy's own systems failed/"
        "broke/errored, asking to check/research/investigate/diagnose an internal problem "
        "(a job, a cron run, an API call, a crash), or asking about her recent runtime logs -- "
        "not her source code (that's codebase_analysis). This covers 'research why is this "
        "happening', 'sandy can you fix this error', 'whats causing this' when the subject is "
        "clearly an internal failure already visible in this conversation, not source code.\n"
        "- search_needed: true if answering well requires current/external info from the web "
        "(current events, specific facts, research, competitor info) rather than reasoning/"
        "writing from existing knowledge. NEVER true at the same time as logs_request -- an "
        "internal error in Sandy's own systems is diagnosed from her own real logs, never from "
        "a web search, no matter how the question is phrased ('research why this is failing' "
        "about an internal job/API failure is logs_request, not search_needed).\n"
        "- llm_override: set ONLY if the message explicitly names WHICH model to use for the "
        "task (not what the task is). Valid: " + ", ".join(providers) + ", or 'orchestrator' "
        "for full multi-round mode. If no model is named, use null.\n"
        "- complexity: how hard THIS message is to answer well -- one of simple, medium, "
        "complex, very_complex. simple = greetings/small talk/one clear fact. medium = a real "
        "question needing some reasoning. complex = multi-part or needs cross-checking. "
        "very_complex = genuinely open-ended/multi-step work.\n"
        "- job_status_request: true ONLY if Ruk is asking whether a mastery/background job is "
        "running, what state it's in, or to check/list his jobs -- NOT asking to start a new "
        "one (that's mastery) and not asking about code/logs.\n"
        "- push_history_request: true if Ruk is asking what a recent push/commit/update "
        "actually changed or did -- 'what does the latest push do', 'what did you just "
        "change', etc.\n"
        "- capabilities_request: true if Ruk is asking what Sandy can do / is capable of, "
        "generally or right now -- 'what can you do now', 'what all can you do', etc. Can "
        "be true at the same time as push_history_request (e.g. 'what changed and what can "
        "you do now').\n\n"
        + (
            (
                'IMPORTANT CONTEXT: you just asked Ruk for days/hours-per-day to build '
                f"'{pending_skill}' as a mastery job. If this message supplies a time "
                "commitment (e.g. '3 days, 4 hours a day') without repeating the skill name, "
                f"fill mastery with skill='{pending_skill}' and the days/hours from this "
                "message -- don't leave mastery null just because he didn't repeat the skill.\n\n"
            )
            if pending_skill else ""
        )
        + f'Message: "{message}"'
    )
    try:
        raw = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}], temperature=0)
        data = json.loads(strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    cfg = data.get("config_change")
    if not isinstance(cfg, dict) or not cfg.get("is") or "key" not in cfg or "value" not in cfg:
        cfg = None

    selfmod_req = data.get("selfmod")
    if isinstance(selfmod_req, dict) and selfmod_req.get("action") == "edit":
        file_path = selfmod_req.get("file_path", "")
        if not file_path or file_path not in message:
            selfmod_req = None  # invented filename -> fail safe, not a real request
    elif not isinstance(selfmod_req, dict) or selfmod_req.get("action") not in ("edit", "history", "rollback"):
        selfmod_req = None

    mastery_req = data.get("mastery")
    if isinstance(mastery_req, dict):
        if not all(k in mastery_req for k in ("skill", "days", "hours_per_day")):
            mastery_req = None
        elif not isinstance(mastery_req.get("days"), (int, float)) or not isinstance(
            mastery_req.get("hours_per_day"), (int, float)
        ):
            mastery_req = None
        elif mastery_req.get("engine") not in ("hermes", "native", "both"):
            mastery_req["engine"] = "hermes"  # fail-safe default -- never silently run an unrecognized engine value
    else:
        mastery_req = None

    mastery_explore = data.get("mastery_explore")
    if isinstance(mastery_explore, dict) and mastery_explore.get("skill"):
        engine = mastery_explore.get("engine")
        mastery_explore = {"skill": mastery_explore["skill"], "engine": engine if engine in ("hermes", "native", "both") else "hermes"}
    else:
        mastery_explore = None
    if mastery_req:
        mastery_explore = None  # mutually exclusive -- fully-specified wins

    # scode: belt-and-suspenders fallback -- don't rely on the LLM alone to
    # stitch pending_skill + this message together. If it missed and this
    # message is clearly just numbers (a direct reply to "how many days/
    # hours?"), extract days/hours with plain regex and fill the skill in
    # from what was pending. Real regression case: Ruk replies "3 days, 4
    # hours a day" without repeating the skill name Sandy just asked about.
    if not mastery_req and pending_skill:
        days_m = re.search(r"(\d+(?:\.\d+)?)\s*day", message, re.I)
        hours_m = re.search(r"(\d+(?:\.\d+)?)\s*hour", message, re.I)
        if days_m and hours_m:
            mastery_req = {
                "skill": pending_skill,
                "days": float(days_m.group(1)),
                "hours_per_day": float(hours_m.group(1)),
            }
            mastery_explore = None

    override = data.get("llm_override")
    if not isinstance(override, list) or not override:
        override = None
    else:
        valid = set(providers) | {"orchestrator"}
        if not all(p in valid for p in override):
            override = None  # hallucinated provider name -> fail safe

    complexity = data.get("complexity")
    if complexity not in {"simple", "medium", "complex", "very_complex"}:
        complexity = "medium"  # fail-safe default, same as brain.classify_complexity's own fallback

    mastery_control = data.get("mastery_control")
    if isinstance(mastery_control, dict) and mastery_control.get("action") in ("pause", "resume", "continue", "remove"):
        mastery_control = {
            "action": mastery_control["action"],
            "engine": mastery_control.get("engine") if mastery_control.get("engine") in ("native", "hermes") else "native",
            "job_ref": mastery_control.get("job_ref") or None,
        }
    else:
        mastery_control = None

    hermes_job_edit = data.get("hermes_job_edit")
    if isinstance(hermes_job_edit, dict) and hermes_job_edit.get("job_ref") and isinstance(hermes_job_edit.get("updates"), dict) and hermes_job_edit["updates"]:
        hermes_job_edit = {"job_ref": hermes_job_edit["job_ref"], "updates": hermes_job_edit["updates"]}
    else:
        hermes_job_edit = None

    native_job_edit = data.get("native_job_edit")
    if isinstance(native_job_edit, dict) and native_job_edit.get("job_ref") and isinstance(native_job_edit.get("updates"), dict):
        nje_updates = native_job_edit["updates"]
        # same discipline as hermes_job_edit above: only real, non-empty
        # fields survive -- an edit with nothing Ruk actually asked for
        # is not a real edit.
        clean_updates = {}
        if isinstance(nje_updates.get("weights"), dict) and nje_updates["weights"]:
            clean_updates["weights"] = nje_updates["weights"]
        if isinstance(nje_updates.get("caps"), dict) and nje_updates["caps"]:
            clean_updates["caps"] = nje_updates["caps"]
        if isinstance(nje_updates.get("mode"), str) and nje_updates["mode"] in ("continuous", "scheduled"):
            clean_updates["mode"] = nje_updates["mode"]
        native_job_edit = {"job_ref": native_job_edit["job_ref"], "updates": clean_updates} if clean_updates else None
    else:
        native_job_edit = None

    logs_req = bool(data.get("logs_request"))
    # Hard structural guard, not just a prompt instruction -- even if the
    # classifier mis-flags both true on some future phrasing, logs_request
    # always wins. This is the actual fix for the live failure: "research
    # why is this happening" (an internal job failure) got search_needed
    # instead, and Sandy hallucinated from random web results about
    # "unrestricted API keys" instead of reading her own real logs.
    search_needed = bool(data.get("search_needed")) and not logs_req

    return {
        "config_change": cfg,
        "selfmod": selfmod_req,
        "mastery": mastery_req,
        "mastery_explore": mastery_explore,
        "mastery_control": mastery_control,
        "hermes_job_edit": hermes_job_edit,
        "native_job_edit": native_job_edit,
        "codebase_analysis": bool(data.get("codebase_analysis")),
        "logs_request": logs_req,
        "search_needed": search_needed,
        "llm_override": override,
        "complexity": complexity,
        "job_status_request": bool(data.get("job_status_request")),
        "push_history_request": bool(data.get("push_history_request")),
        "capabilities_request": bool(data.get("capabilities_request")),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    try:
        resp = _handle_chat(req, background_tasks)
    except CapExceeded as e:
        # ponytail: one guard around the whole flow, not one per LLM call-site —
        # every path through this function calls an LLM somewhere.
        resp = ChatResponse(
            reply=f"Can't do that right now — {e}. Tell me to raise the cap or try again later."
        )
    except Exception as e:
        # ponytail: last-resort net -- an unexpected error anywhere in this
        # flow (broken provider config, etc) should never surface as a raw
        # 500 to Ruk. Logged here so it's still visible in the Space logs.
        log(f"[/chat] unhandled error: {e!r}")
        resp = ChatResponse(
            reply="Kuch gadbad ho gayi mere end pe, Ruk — Space logs check kar, koi provider/config galat lag raha hai."
        )

    # Step 2 -- Proactive Reporting. One single choke point (not wrapped
    # around every one of _handle_chat's many internal return points) --
    # any real failure Sandy hasn't SHOWN Ruk in chat yet gets prepended
    # here, regardless of what he actually asked this turn. This is the
    # channel that can never silently fail to reach him the way an
    # unconfigured WhatsApp secret can (confirmed from his own logs it
    # currently does) -- chat is guaranteed, WhatsApp is best-effort on
    # top of it, not instead of it.
    try:
        unannounced = healing.list_unannounced_fixes()
        if unannounced:
            blocks = []
            for f in unannounced:
                block = f"⚠️ Ruk, ek real problem hai -- '{f['job_name']}' ({f['job_ref']}): {f['root_cause']}."
                if f.get("research_note"):
                    block += f"\n{f['research_note']}"
                block += f" PROPOSED FIX: {f['updates']} -- 'haan'/'fix it' bolo." if f["updates"] else " Iska koi safe auto-fix nahi hai -- khud dekhna padega."
                blocks.append(block)
                healing.mark_announced(f["job_ref"])
            resp.reply = "\n\n".join(blocks) + "\n\n---\n\n" + resp.reply
    except Exception as e:
        log(f"[healing] chat-alert prepend failed, continuing: {e!r}")

    return resp


def _handle_chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    # Opportunistic failure check -- runs on every real message, not just
    # a fixed timer. Given HF Spaces sleeps the container when idle, a
    # periodic-only check could sleep through a failure for hours; a real
    # incoming message means the container is awake RIGHT NOW, which is
    # the most reliable moment to actually check. Best-effort -- must
    # never block or break a normal chat turn.
    try:
        new_alerts = healing.run_check_and_alert()
        if new_alerts:
            log(f"[healing] {len(new_alerts)} new failure(s) detected and alerted")
    except Exception as e:
        log(f"[healing] opportunistic check failed, continuing chat normally: {e!r}")

    # Step 1 -- Self-Heal Intercept. A short affirmative ("haan, fix it" /
    # "yes" / "kar do") with a real pending fix waiting resolves it
    # directly -- bypasses classify_message and the LLM chat pipeline
    # entirely, so there is no window for a hallucinated "done" reply.
    # scode: real bug found live from Ruk's own log -- the old version
    # used a rigid phrase-enumeration regex that required an EXACT match
    # against one whole alternative, so "haan, fix it" (comma-joined)
    # matched NONE of them and fell through to the classifier, which is
    # exactly the failure this feature exists to prevent. Token-based
    # matching (every word individually in a small whitelist) closes
    # this whole class of near-miss instead of patching one phrase.
    if _is_short_affirmative(req.message):
        pending = healing.list_announced_pending_fixes()
        if len(pending) == 1:
            fix = pending[0]
            _remember(background_tasks, req.message, role="user")
            if not fix["updates"]:
                # Honest path -- some failures (skill_missing/rate_limit/auth/
                # timeout/native/hermes-internal-bug) have no safe auto-fix.
                # Saying "haan" to one of these must NOT pretend to apply
                # something that doesn't exist -- that's the exact
                # hallucination this whole feature is meant to prevent.
                reply = f"Ruk, '{fix['job_name']}' ({fix['job_ref']}) ka koi safe auto-fix nahi hai ({fix['root_cause']}) -- khud dekhna padega, ya bata kya karna hai."
                if fix.get("research_note"):
                    reply += f"\n{fix['research_note']}"
            else:
                # Real fix and its cleanup bookkeeping are deliberately
                # separated -- a real bug found live: pop_pending_fix's
                # own crash (Postgres NOT NULL violation, now fixed
                # separately) got caught by a shared except block and
                # reported to Ruk as "the fix failed", when the actual
                # edit_mastery_job() call had already succeeded. Sandy
                # must never report a real success as a failure just
                # because unrelated bookkeeping afterward had a problem.
                try:
                    reply = mastery.edit_mastery_job(fix["job_ref"], fix["updates"])
                    trigger_reply = mastery.trigger_mastery_job_now(fix["job_ref"])
                    reply += f"\n\n{trigger_reply}"
                except Exception as e:
                    log(f"[healing] real fix apply failed: {e!r}")
                    reply = f"Ruk, fix apply karte waqt real error aa gaya: {e}"
                else:
                    try:
                        healing.pop_pending_fix(fix["job_ref"])
                    except Exception as e:
                        log(f"[healing] fix succeeded but cleanup bookkeeping failed (non-fatal, cosmetic only): {e!r}")
            _remember(background_tasks, reply, role="assistant")
            return ChatResponse(reply=reply)
        elif len(pending) > 1:
            _remember(background_tasks, req.message, role="user")
            reply = "Ruk, ek se zyada pending fixes hain -- exact job bata kaunsa apply karna hai: " + ", ".join(f"{f['job_ref']} ({f['updates'] or 'no auto-fix'})" for f in pending)
            _remember(background_tasks, reply, role="assistant")
            return ChatResponse(reply=reply)
        # else: no pending fix at all -- fall through, this is just an ordinary "yes"/chat message

    # /masterystart is a shorthand trigger, NOT a bypass -- strip the
    # prefix and let the rest flow through the exact same classification
    # + explain-first pipeline as typing it in plain English. This
    # guarantees "same flow" (explain understanding/vision before
    # starting) regardless of how Ruk phrases the request.
    if req.message.strip().lower().startswith("/masterystart"):
        req.message = req.message.strip()[len("/masterystart"):].strip() or "start a mastery job"

    # If Ruk already has a pending self-edit proposal for this session and
    # just approved it, apply it directly -- don't classify at all, since
    # re-running any classifier on the approval turn could in principle
    # disagree with what was actually shown and approved. Checked first,
    # before spending a classify_message() call on it.
    if req.approved and req.session_id in selfmod._pending:
        _remember(background_tasks, req.message, role="user")
        try:
            reply = selfmod.apply_pending(req.session_id)
        except selfmod.GitOpError as e:
            reply = f"Ruk, edit push nahi ho paya: {e}"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    # A pending mastery plan takes priority over classification -- a
    # feedback message like "add more research to this" has no skill/
    # days/hours in it, so classify_message would never recognize it as
    # a mastery request on its own. Route straight to approve-or-revise.
    #
    # scode: real bug found live -- when engine="both" proposes a Hermes
    # AND a native plan together, they used to be handled by two SEPARATE
    # checks in sequence; whichever ran first (pending_plan/Hermes) would
    # return early, leaving the native plan orphaned -- a "confirm" would
    # only ever confirm the Hermes side. Now handled together in one
    # block whenever both are actually pending for this session.
    pending_plan = mastery.get_pending_plan(req.session_id)
    pending_native = native_mastery.get_pending_native_plan(req.session_id)
    if pending_plan and pending_native:
        _remember(background_tasks, req.message, role="user")
        if req.approved:
            try:
                reply = mastery.confirm_plan(req.session_id)
            except Exception as e:
                log(f"[confirm_plan] real failure, job was NOT created: {e!r}")
                reply = f"Ruk, Hermes job create nahi hui -- real error: {e}."
            try:
                native_reply, _run_id = native_mastery.confirm_native_plan(req.session_id, background_tasks=background_tasks)
                reply += "\n\n---\n\n" + native_reply
            except Exception as e:
                reply += f"\n\n---\n\nRuk, native run start nahi hui -- real error: {e}."
        else:
            hermes_plan = mastery.propose_plan(
                req.session_id, pending_plan["skill"], pending_plan["days"], pending_plan["hours_per_day"],
                feedback=req.message, context=_recall_context(pending_plan["skill"], skill=pending_plan["skill"]),
            )
            mode, weights, caps = native_mastery.parse_directives(req.message)
            native_job = native_mastery.propose(
                req.session_id, pending_native["skill"], mode, weights, caps,
                feedback=req.message, prior=pending_native, context=_recall_context(pending_native["skill"], skill=pending_native["skill"]),
            )
            reply = f"**Hermes path:**\n\n{hermes_plan}\n\n---\n\n**Native path:**\n\n{native_job['plan']}\n\nTheek hai ab dono? Confirm karo ya aur changes bolo."
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply, needs_approval=not req.approved)

    if pending_plan:
        _remember(background_tasks, req.message, role="user")
        if req.approved:
            try:
                reply = mastery.confirm_plan(req.session_id)
            except Exception as e:
                # scode: real gap closed here -- confirm_plan() had zero
                # exception handling at any call site, so a genuine
                # create_job() failure either 500'd raw or (worse) risked
                # a misleading response further up. Now it's impossible
                # for Ruk to be told a job is registered when it isn't.
                log(f"[confirm_plan] real failure, job was NOT created: {e!r}")
                reply = f"Ruk, job actually create nahi hui -- real error: {e}. Job registered NAHI hai, dobara try karo ya check karo kya galat hai."
        else:
            plan = mastery.propose_plan(
                req.session_id, pending_plan["skill"], pending_plan["days"], pending_plan["hours_per_day"],
                feedback=req.message, context=_recall_context(pending_plan["skill"], skill=pending_plan["skill"]),
            )
            reply = f"{plan}\n\nTheek hai ab? Confirm karo ya aur changes bolo."
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply, needs_approval=not req.approved)

    if pending_native:
        _remember(background_tasks, req.message, role="user")
        if req.approved:
            try:
                reply, run_id = native_mastery.confirm_native_plan(req.session_id, background_tasks=background_tasks)
            except Exception as e:
                log(f"[confirm_native_plan] real failure: {e!r}")
                reply = f"Ruk, native run start nahi hui -- real error: {e}."
        else:
            mode, weights, caps = native_mastery.parse_directives(req.message)
            native_job = native_mastery.propose(
                req.session_id, pending_native["skill"], mode, weights, caps,
                feedback=req.message, prior=pending_native, context=_recall_context(pending_native["skill"], skill=pending_native["skill"]),
            )
            reply = f"{native_job['plan']}\n\nTheek hai ab? Confirm karo ya aur changes bolo."
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply, needs_approval=not req.approved)

    yt_id = youtube.extract_video_id(req.message)
    if yt_id:
        _remember(background_tasks, req.message, role="user")
        try:
            text = youtube.transcript(yt_id)
        except RuntimeError as e:
            reply = f"Ruk, transcript nahi mil paya: {e}"
        else:
            reply = call_llm_with_fallback(
                "gemini",
                [
                    {"role": "system", "content": SANDY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Ruk's message: {req.message}\n\nReal transcript of the video "
                        f"(video ID {yt_id}):\n{text[:15000]}\n\n"
                        "Answer what he actually asked, using ONLY what's really in this transcript "
                        "-- don't add technique/detail that isn't actually said in it.",
                    },
                ],
            )
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if _is_trivial(req.message):
        # scode: real, verified inefficiency -- classify_message()'s full
        # ~15-field JSON schema call was firing on EVERY message, including
        # "hi"/"thanks" -- the exact "tries to do everything every time,
        # even when she doesn't need to" pattern Ruk described. _is_trivial
        # already existed (used to gate Mem0 extraction) but was never used
        # to skip the classifier itself. A trivial greeting needs one cheap
        # reply, not a full intent-extraction pass.
        _remember(background_tasks, req.message, role="user")
        reply = call_llm_with_fallback("groq", [{"role": "system", "content": SANDY_SYSTEM_PROMPT}, {"role": "user", "content": req.message}])
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    pending_skill = mastery.get_pending_explore(req.session_id)
    # scode: real near-bug caught here -- if Sandy is mid-flow waiting on
    # "how many days/hours?", Ruk's reply might just be "3 days, 4 hours
    # a day" with no mastery/skill/job keyword in it at all. The fast
    # pre-filter would wrongly skip straight past classify_message()'s
    # own belt-and-suspenders regex fallback that exists specifically for
    # this reply. Only allow the fast path when nothing is pending.
    cls = _maybe_skip_classifier(req.message) if not pending_skill else None
    if cls is None:
        cls = classify_message(req.message, pending_skill=pending_skill)
    if cls["mastery"]:
        mastery.pop_pending_explore(req.session_id)  # resolved -- clear it so it can't leak into a later unrelated message

    cfg_change = cls["config_change"]
    if cfg_change:
        _remember(background_tasks, req.message, role="user")
        if cfg_change["key"] == "caps":
            value = cfg_change["value"] if isinstance(cfg_change["value"], dict) else {}
            # scode: real bug found live -- an ambiguous multi-provider message (mastery plan
            # weights, not an actual cap-change request) got the classifier to invent a
            # placeholder key like "provider_name" instead of a real provider, and it went
            # straight into Supabase unvalidated ("Done, updated caps to {'provider_name': 100}").
            # Guard: only accept keys that are real known providers -- anything else gets refused
            # with a clear question instead of silently writing garbage config.
            bad_keys = [k for k in value if k not in MODELS]
            if bad_keys or not value:
                reply = (
                    f"Ruk, mujhe clear nahi hua kis REAL provider ka cap change karna hai "
                    f"(mila: {list(value.keys()) or 'kuch nahi'}) -- in me se ek bolo: {', '.join(MODELS)}."
                )
                _remember(background_tasks, reply, role="assistant")
                return ChatResponse(reply=reply)
            current = config.get_config("caps") or {}
            current.update(value)
            config.set_config("caps", current)
        else:
            config.set_config(cfg_change["key"], cfg_change["value"])
        reply = f"Done, Ruk — updated {cfg_change['key']} to {cfg_change['value']}. 🎯"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    mctl = cls["mastery_control"]
    if mctl:
        _remember(background_tasks, req.message, role="user")
        action, engine, job_ref = mctl["action"], mctl["engine"], mctl["job_ref"]
        try:
            if engine == "native":
                ambiguous_remove_reply = None
                if not job_ref:
                    candidates = [j for j in native_mastery.list_jobs() if j.get("session_id") == req.session_id]
                    if action == "pause":
                        candidates = [j for j in candidates if j["state"] in ("running", "scheduled_waiting")]
                    elif action == "resume":
                        candidates = [j for j in candidates if j["state"] == "paused"]
                    elif action == "continue":
                        candidates = [j for j in candidates if j["state"] == "scheduled_waiting"]
                    # remove: no state filter, but ambiguity is handled explicitly below --
                    # a destructive action shouldn't silently guess between real candidates
                    if action == "remove" and len(candidates) > 1:
                        ambiguous_remove_reply = f"Ruk, {len(candidates)} native jobs hain -- exact job ID bolo kisko remove karna hai: " + ", ".join(f"{j['id']} ({j['skill']})" for j in candidates)
                    else:
                        job_ref = candidates[0]["id"] if candidates else None
                if ambiguous_remove_reply:
                    reply = ambiguous_remove_reply
                elif not job_ref:
                    reply = f"Ruk, {action} karne layak koi native mastery job nahi mila is session ke liye."
                elif action == "pause":
                    reply = native_mastery.pause(job_ref)
                elif action == "resume":
                    reply = native_mastery.resume(job_ref, background_tasks=background_tasks)
                elif action == "continue":
                    reply = native_mastery.continue_now(job_ref, background_tasks=background_tasks)
                else:  # remove
                    reply = native_mastery.remove(job_ref)
            else:  # hermes
                if not job_ref:
                    jobs = mastery.list_mastery_jobs()
                    if len(jobs) == 1:
                        job_ref = jobs[0]["id"]
                    elif len(jobs) > 1:
                        reply = f"Ruk, {len(jobs)} Hermes jobs hain -- exact naam/ID bolo kisko {action} karna hai: " + ", ".join(j["name"] for j in jobs)
                        job_ref = None
                    else:
                        reply = "Ruk, koi Hermes mastery job hi nahi hai abhi."
                        job_ref = None
                if job_ref:
                    if action == "pause":
                        reply = mastery.pause_mastery_job(job_ref)
                    elif action == "resume":
                        reply = mastery.resume_mastery_job(job_ref)
                    elif action == "continue":
                        reply = mastery.trigger_mastery_job_now(job_ref)
                    else:  # remove
                        reply = mastery.remove_mastery_job(job_ref)
        except Exception as e:
            log(f"[mastery_control] real failure: {e!r}")
            reply = f"Ruk, {action} karte waqt real error aa gaya: {e}"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    hje = cls["hermes_job_edit"]
    if hje:
        # Real spend-adjacent change (provider/model pin) -- confirm first,
        # same pattern as selfmod/mastery proposals, not applied silently
        # like a cap-number tweak. This whole branch exists because this
        # exact request used to hit NO handler at all (mastery_req=None,
        # mastery_explore=None -- confirmed from Ruk's own log) and fell
        # into generic chat, which then narrated a fake CLI command and a
        # fake "done" instead of touching anything real.
        if not req.approved:
            _remember(background_tasks, req.message, role="user")
            reply = f"Ruk, confirm karo -- job '{hje['job_ref']}' ko {hje['updates']} se update kar du?"
            return ChatResponse(reply=reply, needs_approval=True)
        _remember(background_tasks, req.message, role="user")
        try:
            reply = mastery.edit_mastery_job(hje["job_ref"], hje["updates"])
        except Exception as e:
            log(f"[hermes_job_edit] real failure: {e!r}")
            reply = f"Ruk, update karte waqt real error aa gaya: {e}"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    nje = cls["native_job_edit"]
    if nje:
        # Mirrors hje right above -- same reasoning: a real spend-adjacent
        # change (weights/caps) confirms first, doesn't apply silently.
        # Only reachable now that native_mastery.edit_native_job() exists
        # -- previously identity.py claimed this capability but there was
        # no function behind it at all for an already-confirmed job.
        if not req.approved:
            _remember(background_tasks, req.message, role="user")
            reply = f"Ruk, confirm karo -- native job '{nje['job_ref']}' ko {nje['updates']} se update kar du?"
            return ChatResponse(reply=reply, needs_approval=True)
        _remember(background_tasks, req.message, role="user")
        try:
            reply = native_mastery.edit_native_job(
                nje["job_ref"],
                weights=nje["updates"].get("weights"),
                caps=nje["updates"].get("caps"),
                mode=nje["updates"].get("mode"),
            )
        except Exception as e:
            log(f"[native_job_edit] real failure: {e!r}")
            reply = f"Ruk, update karte waqt real error aa gaya: {e}"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    selfmod_req = cls["selfmod"]
    if selfmod_req:
        action = selfmod_req["action"]

        if action == "history":
            try:
                reply = selfmod.recent_history()
            except selfmod.GitOpError as e:
                reply = f"Ruk, history nahi mil payi: {e}"
            return ChatResponse(reply=reply)

        if action == "edit":
            try:
                reply = selfmod.propose_edit(
                    req.session_id, selfmod_req["file_path"], selfmod_req["instruction"]
                )
            except selfmod.GitOpError as e:
                reply = f"Ruk, proposal nahi bana paayi: {e}"
                return ChatResponse(reply=reply)
            needs_approval = req.session_id in selfmod._pending
            return ChatResponse(reply=reply, needs_approval=needs_approval)

        if action == "rollback":
            if not req.approved:
                return ChatResponse(
                    reply=(
                        f"Ruk, confirm karo — commit {selfmod_req['commit_hash']} "
                        "revert karke push kar du?"
                    ),
                    needs_approval=True,
                )
            _remember(background_tasks, req.message, role="user")
            try:
                reply = selfmod.rollback_to(selfmod_req["commit_hash"])
            except selfmod.GitOpError as e:
                reply = f"Ruk, rollback nahi ho paaya: {e}"
            _remember(background_tasks, reply, role="assistant")
            return ChatResponse(reply=reply)

    mastery_req = cls["mastery"]
    log(f"[classify] mastery_req={mastery_req!r} mastery_explore={cls['mastery_explore']!r} (msg={req.message[:80]!r})")
    if mastery_req:
        # Kicking off a multi-day autonomous job spends real API credits
        # over days unattended -- always confirm first, same as risky
        # tasks, regardless of what the generic risk-classifier thinks.
        engine = mastery_req.get("engine", "hermes")
        skill = mastery_req["skill"]

        if engine == "native":
            # Native path doesn't use hermes's day/hour scheduling shape --
            # mode (continuous/scheduled) + provider weights come from the
            # message itself instead.
            if not req.approved:
                _remember(background_tasks, req.message, role="user")
                mode, weights, caps = native_mastery.parse_directives(req.message)
                native_job = native_mastery.propose(
                    req.session_id, skill, mode, weights, caps, context=_recall_context(skill, skill=skill),
                )
                reply = f"{native_job['plan']}\n\nRuk, ye plan theek hai? Confirm karo, ya jo change karna hai bol do."
                _remember(background_tasks, reply, role="assistant")
                return ChatResponse(reply=reply, needs_approval=True)
            _remember(background_tasks, req.message, role="user")
            try:
                reply, _run_id = native_mastery.confirm_native_plan(req.session_id, background_tasks=background_tasks)
            except Exception as e:
                log(f"[confirm_native_plan] real failure: {e!r}")
                reply = f"Ruk, native run start nahi hui -- real error: {e}."
            _remember(background_tasks, reply, role="assistant")
            return ChatResponse(reply=reply)

        # engine == "hermes" or "both" -- hermes side always proposed/confirmed;
        # "both" additionally proposes/confirms the native side alongside it.
        if not req.approved:
            _remember(background_tasks, req.message, role="user")
            plan = mastery.propose_plan(
                req.session_id, skill, mastery_req["days"], mastery_req["hours_per_day"],
                context=_recall_context(skill, skill=skill),
            )
            parts = [plan]
            if engine == "both":
                mode, weights, caps = native_mastery.parse_directives(req.message)
                native_job = native_mastery.propose(
                    req.session_id, skill, mode, weights, caps, context=_recall_context(skill, skill=skill),
                )
                parts = [f"**Hermes path:**\n\n{plan}", f"**Native (my orchestration) path:**\n\n{native_job['plan']}"]
            reply = "\n\n---\n\n".join(parts) + "\n\nRuk, plan(s) theek hain? Confirm karo, ya jo change karna hai bol do."
            _remember(background_tasks, reply, role="assistant")
            return ChatResponse(reply=reply, needs_approval=True)
        _remember(background_tasks, req.message, role="user")
        try:
            reply = mastery.confirm_plan(req.session_id)
        except Exception as e:
            log(f"[confirm_plan] real failure, job was NOT created: {e!r}")
            reply = f"Ruk, job actually create nahi hui -- real error: {e}. Job registered NAHI hai, dobara try karo ya check karo kya galat hai."
        if engine == "both":
            try:
                native_reply, _run_id = native_mastery.confirm_native_plan(req.session_id, background_tasks=background_tasks)
                reply += "\n\n---\n\n" + native_reply
            except Exception as e:
                reply += f"\n\n---\n\nRuk, native run start nahi hui -- real error: {e}."
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["mastery_explore"] or cls["push_history_request"] or cls["capabilities_request"]:
        _remember(background_tasks, req.message, role="user")
        parts = []

        if cls["push_history_request"] or cls["capabilities_request"]:
            grounding = []
            if cls["push_history_request"]:
                try:
                    grounding.append("Real recent push history (git log):\n" + selfmod.recent_history(limit=5))
                except Exception as e:
                    grounding.append(f"(Couldn't read real push history: {e})")
            if cls["capabilities_request"]:
                grounding.append(CAPABILITIES)
            try:
                parts.append(call_llm_with_fallback(
                    "gemini",
                    [
                        {"role": "system", "content": SANDY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Ruk's question: {req.message}\n\n" + "\n\n".join(grounding)
                            + "\n\nAnswer his ACTUAL question above in Hinglish, using ONLY the real "
                            "facts given -- don't invent anything not listed here.",
                        },
                    ],
                ))
            except Exception as e:
                parts.append(f"Ruk, ye check karte waqt error aa gaya: {e}")

        if cls["mastery_explore"]:
            skill = cls["mastery_explore"]["skill"]
            engine = cls["mastery_explore"].get("engine", "hermes")
            mastery.set_pending_explore(req.session_id, skill)  # so a bare "3 days, 4 hours" reply resolves to THIS skill
            try:
                mastery.save_skill_notes(skill, req.message)  # verbatim -- see mastery.py docstring for why Mem0 recall alone isn't enough here
            except Exception as e:
                log(f"[mastery_explore] save_skill_notes failed: {e!r}")

            # scode: real bug found live -- mastery_explore only ever called explain_*_flow(),
            # which never stores a real pending plan. Native/both engine conversations have NO
            # OTHER path to a real proposed plan (mastery_req's schema requires days+hours_per_day,
            # which native mode doesn't use -- it uses mode+weights instead), so they looped in
            # "explain" mode forever. A later message giving weights ("80% gemini... 10% other")
            # then had nothing pending to attach to, and got misclassified as a config_change
            # instead (see the cfg_change validation guard above -- same root incident).
            # Fix: once native/both engine AND real mode/weights info is actually given, treat
            # that as ready and propose a real, confirmable native plan instead of explaining again.
            mode, weights, caps = native_mastery.parse_directives(req.message)
            native_ready = engine in ("native", "both") and (
                weights or re.search(r"\bcontinuous\b|\bschedul", req.message, re.I)
            )
            if native_ready:
                try:
                    native_job = native_mastery.propose(req.session_id, skill, mode, weights, caps, context=_recall_context(skill, skill=skill))
                    parts.append(f"**Native (my orchestration) path -- real plan, ready to confirm:**\n\n{native_job['plan']}")
                    if engine == "both":
                        parts.append("Hermes side still needs a real time commitment (days + hours/day) before I can propose that plan too -- give me that and I'll lock both in together.")
                except Exception as e:
                    parts.append(f"Ruk, native plan propose karte waqt error aa gaya: {e}")
                reply = "\n\n---\n\n".join(parts) + "\n\nRuk, native plan confirm karo ya jo change karna hai bol do."
                _remember(background_tasks, reply, role="assistant")
                return ChatResponse(reply=reply, needs_approval=True)

            try:
                if engine == "native":
                    parts.append(native_mastery.explain_native_flow(skill, req.message, context=_recall_context(skill, skill=skill)))
                elif engine == "both":
                    parts.append("**Hermes path:**\n\n" + mastery.explain_flow(skill, req.message, context=_recall_context(skill, skill=skill)))
                    parts.append("**Native (my orchestration) path:**\n\n" + native_mastery.explain_native_flow(skill, req.message, context=_recall_context(skill, skill=skill)))
                else:
                    parts.append(mastery.explain_flow(skill, req.message, context=_recall_context(skill, skill=skill)))
            except Exception as e:
                parts.append(f"Ruk, mastery flow explain karte waqt error aa gaya: {e}")

        reply = "\n\n---\n\n".join(parts)
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["job_status_request"]:
        _remember(background_tasks, req.message, role="user")
        try:
            jobs = mastery.list_mastery_jobs()
            for j in jobs:
                j.setdefault("engine", "hermes")
            jobs += native_mastery.status_shape()
        except Exception as e:
            reply = f"Ruk, job list check karne mein error aa gaya: {e}. Space logs check kar."
        else:
            if not jobs:
                reply = "Ruk, abhi koi mastery job registered nahi hai — koi bhi active nahi chal raha."
            else:
                # scode: real gap found live -- "is there a fault" and a plain status
                # check used to get the EXACT same generic name/state/next-run line,
                # even when a real last_error or a dead scheduler ticker existed.
                # Full per-job diagnostics (state, real error, output directory) plus
                # the scheduler's own heartbeat (separate from any one job -- a dead
                # ticker explains "next run unknown" for EVERY job, not a per-job bug)
                # replace the old three-field summary.
                blocks = []
                if any(j.get("engine") != "native" for j in jobs):
                    blocks.append(mastery.scheduler_health())
                for j in jobs:
                    if j.get("engine") == "native":
                        blocks.append(native_mastery.job_diagnostics(j["id"]))
                    else:
                        blocks.append(mastery.job_diagnostics(j["id"]))
                reply = "Ruk, real status (Hermes + native se seedha):\n\n" + "\n\n".join(blocks)
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["codebase_analysis"]:
        _remember(background_tasks, req.message, role="user")
        reply = codebase.analyze(req.message)
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["logs_request"]:
        _remember(background_tasks, req.message, role="user")
        # Enhanced with real Hermes gateway logs + env key presence +
        # git state -- previously this only ever read Sandy's OWN
        # process log, completely blind to the separate gateway
        # subprocess where things like the groq-auth 401 traceback
        # actually lived. This is also the concrete answer to "why
        # couldn't Sandy figure this out" -- she now has a real way to.
        logs = diagnostics.inspect_system_health()
        git_info = diagnostics.git_push_summary()
        reply = call_llm_with_fallback(
            "gemini",
            [
                {"role": "system", "content": SANDY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Ruk's question: {req.message}\n\nYour real internal diagnostics this turn:\n{logs}\n\n"
                    f"Latest real deployment state:\n{git_info}\n\n"
                    "Answer in Hinglish, grounded ONLY in what's actually in the diagnostics/logs above -- "
                    "if the real cause isn't visible in them, say so plainly instead of guessing. Never invent "
                    "an explanation an external web search might produce for an internal error like this.",
                },
            ],
        )
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    _remember(background_tasks, req.message, role="user")
    recalled = memory.recall(req.message)
    context = (
        "Background facts Sandy remembers about Ruk (may or may not be relevant "
        "to this specific question -- use only what actually applies):\n" + "\n".join(recalled)
    ) if recalled else ""

    date_range = chatlog.extract_date_range(req.message)
    if date_range:
        start, end = date_range
        dated_msgs = chatlog.get_history_in_range(start, end)
        if dated_msgs:
            dated_block = "\n".join(f"[{m['created_at']}] {m['role']}: {m['message']}" for m in dated_msgs)
            context += f"\n\nReal messages from {start} (the period Ruk is asking about):\n{dated_block}"
        else:
            context += f"\n\n(Ruk is asking about {start}, but no messages were found for that date -- tell him plainly, don't guess.)"

    # scode: complexity now comes straight from the single classify_message
    # call above (the "complexity" axis added there) instead of a second,
    # separate Groq call to brain.classify_complexity() -- that was the
    # actual unmerged duplicate call flagged last session. brain.answer()
    # below is always given this tier now, not just on search_needed
    # messages, so it never re-classifies internally either.
    tier = cls["complexity"]
    if cls["search_needed"]:
        provider = req.search_provider or _extract_search_provider(req.message)
        wants_all = _wants_all_providers(req.message)
        try:
            if wants_all:
                all_results = search.search_all(req.message)
                blocks = []
                for name, r in all_results.items():
                    if isinstance(r, str):  # that provider failed -- say so, don't paper over it
                        blocks.append(f"=== {name} ===\n(this provider FAILED: {r})")
                    else:
                        block = "\n\n".join(f"[{x['title']}]({x['url']})\n{x['content']}" for x in r) or "(no results)"
                        blocks.append(f"=== {name} ===\n{block}")
                context += (
                    "\n\nReal, separate results from all three search engines -- "
                    "report on EXACTLY these, attributed to the correct engine. "
                    "Do NOT invent results for a provider, and do NOT add a "
                    "confidence score or any section not shown here:\n" + "\n\n".join(blocks)
                )
            else:
                results = search.search(req.message, provider=provider, complexity=tier)
                search_block = "\n\n".join(f"[{r['title']}]({r['url']})\n{r['content']}" for r in results)
                used = provider or search._COMPLEXITY_DEFAULT.get(tier, "tavily")
                context += f"\n\nWeb search results (via {used} -- this is the ONLY provider actually used, don't claim to have checked others):\n{search_block}"
        except Exception as e:
            log(f"[/chat] search failed: {e!r}")
            context += (
                "\n\n(Web search was attempted but FAILED with a real error -- "
                "tell Ruk plainly that the search failed, don't soften it to "
                "'slow', and suggest he check the Space logs. Then answer "
                "from your own knowledge if you can.)"
            )

    override = req.override_llms or cls["llm_override"]
    recent = chatlog.get_history(limit=30)
    history = [{"role": m["role"], "content": m["message"]} for m in recent]
    reply = brain.answer(req.message, context=context, override=override, history=history, tier=tier)
    _remember(background_tasks, reply, role="assistant")
    return ChatResponse(reply=reply)


@app.post("/new_chat")
def new_chat():
    """'New Chat' button: summarizes everything since the last reset
    (what got done, what's still open) and marks a fresh boundary in
    chat_log. Purely a visual reset for Ruk's Home -- the real chat_log
    history and Mem0's permanent facts are untouched, so Sandy's actual
    memory never shrinks. Deliberately NOT a real separate session (see
    the handoff doc for why: chat_log has no session concept, and adding
    one is a bigger schema change than this feature needs)."""
    recent = chatlog.get_history(limit=200)
    since_last = []
    for m in reversed(recent):
        if m["message"].startswith("[SESSION SUMMARY]"):
            break
        since_last.append(m)
    since_last.reverse()

    if not since_last:
        summary = "Koi naya kaam nahi hua pichhle reset ke baad, Ruk."
    else:
        convo = "\n".join(f"{m['role']}: {m['message']}" for m in since_last)
        summary = call_llm_with_fallback(
            "gemini",
            [{
                "role": "user",
                "content": (
                    "Summarize this conversation between Ruk and Sandy in Hinglish, "
                    "short and concrete: what got done, what's still open/remaining. "
                    "A few lines, no filler.\n\n" + convo
                ),
            }],
        )
    chatlog.log(f"[SESSION SUMMARY] {summary}", role="assistant")
    return {"summary": summary}


@app.get("/health")
def health():
    """Real dependency checks -- confirmed bug: this endpoint used to
    unconditionally return {"status": "ok"} the ENTIRE time the
    native_mastery_jobs/healing_ledger schema mismatch was breaking every
    job-related feature. Cheap on purpose (one lightweight query per
    table, not a heavy diagnostic dump) -- for the full picture on
    demand, that's inspect_system_health() via the logs_request path.

    Checks REAL COLUMNS per table, not just table existence -- a plain
    select("*") never references a specific column, so it would NOT have
    caught the exact live bug that shipped (healing_ledger existed, but
    was missing announced_in_chat; "*" happily returns whatever columns
    ARE there and never notices one is absent). O(1) per table, 4 tables
    -- trivial at this scale, not worth a heavier migration-tracking
    system for a schema this small that changes this rarely.

    Each table check is independently wrapped -- a real, deliberate
    isolation boundary (one table's schema problem must not prevent
    reporting on the other three), not a lazy catch-all."""
    checks = {}
    overall_ok = True

    # The exact columns each table's real code paths actually reference --
    # kept here, next to the check, so it's obvious when this needs
    # updating alongside a real schema change.
    _EXPECTED_COLUMNS = {
        "sandy_config": "key,value",
        "mastery_events": "run_id,agent,round,event_type,provider,summary,detail,parent_event_id",
        "native_mastery_jobs": "id,session_id,skill,mode,weights,caps,usage,state,plan,result,round,created_at,updated_at",
        "healing_ledger": "job_ref,job_name,engine,root_cause,proposed_updates,is_resolved,announced_in_chat,created_at,resolved_at,research_note",
    }

    try:
        client = config.get_client()
        for table, columns in _EXPECTED_COLUMNS.items():
            try:
                client.table(table).select(columns).limit(1).execute()
                checks[table] = "ok"
            except Exception as e:
                checks[table] = f"MISSING table or column mismatch: {e}"
                overall_ok = False
    except Exception as e:
        checks["supabase"] = f"client init failed: {e}"
        overall_ok = False

    try:
        from cron.jobs import get_ticker_heartbeat_age
        hb = get_ticker_heartbeat_age()
        if hb is None:
            checks["hermes_gateway"] = "no heartbeat file -- ticker may never have run"
            overall_ok = False
        elif hb > 180:
            checks["hermes_gateway"] = f"heartbeat {int(hb)}s old -- likely dead"
            overall_ok = False
        else:
            checks["hermes_gateway"] = "ok"
    except Exception as e:
        checks["hermes_gateway"] = f"check failed: {e!r}"
        overall_ok = False

    try:
        missing = [k for k, present in diagnostics.env_key_matrix().items() if not present]
        # Informational only -- some keys (WhatsApp) are known-optional,
        # so a missing one shouldn't flip the whole app to "degraded".
        checks["env_keys"] = "ok" if not missing else f"missing (may be optional): {missing}"
    except Exception as e:
        checks["env_keys"] = f"check failed: {e!r}"

    return {"status": "ok" if overall_ok else "degraded", "checks": checks}


@app.get("/history")
def history(limit: int = 200):
    return {"messages": chatlog.get_history(limit=limit)}


@app.get("/api/job-output/{job_id}")
def job_output(job_id: str):
    """Real per-job output for Ruk's Home's Agents view -- reads straight
    from ~/.hermes/cron/output/{job_id}/, no narration."""
    try:
        outputs = mastery.job_output(job_id)
        try:
            mastery.backfill_events_from_output(job_id)  # best-effort, feeds the Hermes-side graph
        except Exception as e:
            log(f"[/api/job-output] event backfill failed (non-fatal): {e!r}")
        return {"outputs": outputs}
    except Exception as e:
        log(f"[/api/job-output] failed for {job_id}: {e!r}")
        return {"outputs": [], "error": str(e)}


@app.get("/api/mastery-runs")
def mastery_runs():
    """Every real mastery run across BOTH agents (sandy native + hermes),
    for Ruk's Home's run picker -- derived straight from the real event
    log, not a separate thing that could drift out of sync."""
    try:
        return {"runs": events.list_runs()}
    except Exception as e:
        log(f"[/api/mastery-runs] failed: {e!r}")
        return {"runs": [], "error": str(e)}


@app.get("/api/mastery-graph/{run_id}")
def mastery_graph(run_id: str):
    """Every real event for one run, oldest first -- exactly what the
    orb graph renders client-side. No server-side graph layout here;
    the frontend builds the force simulation from these rows."""
    try:
        return {"events": events.get_events(run_id)}
    except Exception as e:
        log(f"[/api/mastery-graph] failed for {run_id}: {e!r}")
        return {"events": [], "error": str(e)}


@app.post("/api/mastery-graph/{run_id}/explain")
async def mastery_graph_explain(run_id: str, req: Request):
    """Graph-chat integration -- Ruk asks about a node/cluster/round,
    Sandy answers grounded ONLY in that run's real logged events, never
    freeform narration. Mirrors the same real-data-in-prompt discipline
    used everywhere else (logs_request, job_status_request, etc.).

    scode: real event-loop-blocking bug fixed here -- this was the ONE
    async def route in the whole file that called a sync function
    (call_llm_with_fallback, and events.get_events's sync Supabase call)
    directly, unwrapped. Sandy runs single-process uvicorn (confirmed:
    no --workers flag in entrypoint.sh) -- an unwrapped sync call inside
    async def freezes the ENTIRE event loop, including every other
    in-flight /chat request, for the full duration of that LLM call.
    Only `body = await req.json()` needs this route to be async at all;
    everything else is wrapped in run_in_threadpool so it runs off the
    event loop, same as every other real route in this file gets for
    free by being plain `def`."""
    body = await req.json()
    question = body.get("question", "")
    run_events = await run_in_threadpool(events.get_events, run_id)
    if not run_events:
        return {"answer": "Ruk, is run_id ke liye koi real event mila hi nahi -- shayad abhi tak kuch hua nahi hai ya run_id galat hai."}
    event_lines = "\n".join(
        f"[id={e['id']} round={e['round']} type={e['event_type']} provider={e.get('provider')}] {e['summary']}"
        for e in run_events
    )
    try:
        answer = await run_in_threadpool(
            call_llm_with_fallback,
            "gemini",
            [
                {"role": "system", "content": SANDY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Ruk's question about this mastery run's graph: {question}\n\n"
                    f"Real logged events for this run (id={run_id}):\n{event_lines}\n\n"
                    "Answer using ONLY these real events -- reference specific event ids/rounds "
                    "where relevant, don't invent detail not present above.",
                },
            ],
        )
    except Exception as e:
        answer = f"Ruk, explain karte waqt error aa gaya: {e}"
    return {"answer": answer}


@app.get("/status")
def status():
    """Real data for Ruk's Home's dashboard views (Command Center, Memory,
    Workflows) -- no mocked numbers. Each source is independently
    try/except'd so one failing piece (e.g. Mem0 hiccup) doesn't blank
    out the other two."""
    try:
        caps = config.get_all_config()
    except Exception as e:
        log(f"[/status] config read failed: {e!r}")
        caps = None
    try:
        usage = config.get_config("usage")
    except Exception as e:
        log(f"[/status] usage read failed: {e!r}")
        usage = None
    try:
        facts = memory.get_all_facts(limit=30)
    except Exception as e:
        log(f"[/status] memory read failed: {e!r}")
        facts = None
    try:
        jobs = mastery.list_mastery_jobs()
        for j in jobs:
            j.setdefault("engine", "hermes")
        jobs += native_mastery.status_shape()
    except Exception as e:
        log(f"[/status] job list read failed: {e!r}")
        jobs = None
    try:
        graph = config.get_config("last_orchestration_graph")
    except Exception as e:
        log(f"[/status] orchestration graph read failed: {e!r}")
        graph = None
    return {"config": caps, "usage": usage, "memory_facts": facts, "jobs": jobs, "models": MODELS, "orchestration_graph": graph}


@app.get("/api/debug-cron")
def debug_cron(key: str = ""):
    """TEMPORARY -- remove after debugging. Gated behind DEBUG_KEY so it's
    not wide open on a public Space. Runs real diagnostic checks and
    returns raw output, no narration, no fabrication.

    scode: rebuilt after a previous session's regex edit deleted this
    AND app.mount below it in the same accidental block-delete (zero
    additions in that commit -- confirmed via git log -p). Two real bugs
    fixed from that version: 'hermes cron runs' isn't a real subcommand
    (the actual ones are list/create/add/edit/pause/resume/run/remove/rm/
    delete/status/tick -- confirmed from Hermes's own CLI error output);
    and supervisorctl needs -c pointing at this repo's actual config path
    (/app/supervisord.conf, not the default it was silently failing to find)."""
    import subprocess, os
    if key != os.environ.get("DEBUG_KEY"):
        return {"error": "unauthorized"}

    def run(cmd):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
            return {"stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:], "returncode": r.returncode}
        except Exception as e:
            return {"error": str(e)}

    return {
        "jobs_json": run("cat ~/.hermes/cron/jobs.json"),
        "cron_list": run("hermes cron list"),
        "output_dir": run("ls -la ~/.hermes/cron/output/"),
        "gateway_stderr_tail": run("supervisorctl -c /app/supervisord.conf tail gateway stderr"),
        "gateway_status": run("supervisorctl -c /app/supervisord.conf status gateway"),
        "hermes_home_listing": run("find ~/.hermes -maxdepth 2"),
    }


# Ruk's Home -- served as a plain static PWA from this same backend, no
# separate hosting/build step. service-worker.js and manifest.json are
# served from root (not /static) so the PWA's scope covers the whole app.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def ruks_home():
    return FileResponse("static/index.html")


@app.get("/manifest.json")
def manifest_json():
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/service-worker.js")
def service_worker():
    return FileResponse("static/service-worker.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
