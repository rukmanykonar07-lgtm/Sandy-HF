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

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chatlog
import codebase
import config
import search
from identity import SANDY_SYSTEM_PROMPT
import search
import memory
import brain
import mastery
import selfmod
from llm import call_llm, call_llm_with_fallback, CapExceeded, MODELS, strip_json_fence, log

app = FastAPI()


def _remember(background_tasks: BackgroundTasks, text: str, role: str) -> None:
    """Every message goes through here instead of calling memory.remember()
    directly -- saves the extracted-facts memory (Mem0) AND the verbatim
    chat_log (for Ruk's Home's history-on-refresh), always together.
    Scheduled as background tasks: this runs AFTER the reply is already
    sent back, so Ruk isn't waiting on Mem0's LLM-based fact extraction
    just to see the message he already got."""
    background_tasks.add_task(memory.remember, text, role=role)
    background_tasks.add_task(chatlog.log, text, role=role)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: tighten to the real Ruk's Home domain once frontend is deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def classify_message(message: str) -> dict:
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
        '  "mastery": {"skill": "...", "days": 3, "hours_per_day": 4} or null,\n'
        '  "codebase_analysis": false,\n'
        '  "logs_request": false,\n'
        '  "search_needed": false,\n'
        '  "llm_override": ["provider_name"] or null\n'
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
        "- codebase_analysis: true only for READ-ONLY review/scan/analyze of Sandy's own "
        "source code -- not asking to change/fix/edit anything (that's selfmod's job).\n"
        "- logs_request: true only if asking about Sandy's recent RUNTIME logs/errors/what "
        "went wrong while running -- not her source code.\n"
        "- search_needed: true if answering well requires current/external info from the web "
        "(current events, specific facts, research, competitor info) rather than reasoning/"
        "writing from existing knowledge.\n"
        "- llm_override: set ONLY if the message explicitly names WHICH model to use for the "
        "task (not what the task is). Valid: " + ", ".join(providers) + ", or 'orchestrator' "
        "for full multi-round mode. If no model is named, use null.\n\n"
        f'Message: "{message}"'
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
    else:
        mastery_req = None

    override = data.get("llm_override")
    if not isinstance(override, list) or not override:
        override = None
    else:
        valid = set(providers) | {"orchestrator"}
        if not all(p in valid for p in override):
            override = None  # hallucinated provider name -> fail safe

    return {
        "config_change": cfg,
        "selfmod": selfmod_req,
        "mastery": mastery_req,
        "codebase_analysis": bool(data.get("codebase_analysis")),
        "logs_request": bool(data.get("logs_request")),
        "search_needed": bool(data.get("search_needed")),
        "llm_override": override,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    try:
        return _handle_chat(req, background_tasks)
    except CapExceeded as e:
        # ponytail: one guard around the whole flow, not one per LLM call-site —
        # every path through this function calls an LLM somewhere.
        return ChatResponse(
            reply=f"Can't do that right now — {e}. Tell me to raise the cap or try again later."
        )
    except Exception as e:
        # ponytail: last-resort net -- an unexpected error anywhere in this
        # flow (broken provider config, etc) should never surface as a raw
        # 500 to Ruk. Logged here so it's still visible in the Space logs.
        log(f"[/chat] unhandled error: {e!r}")
        return ChatResponse(
            reply="Kuch gadbad ho gayi mere end pe, Ruk — Space logs check kar, koi provider/config galat lag raha hai."
        )


def _handle_chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
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

    cls = classify_message(req.message)

    cfg_change = cls["config_change"]
    if cfg_change:
        _remember(background_tasks, req.message, role="user")
        if cfg_change["key"] == "caps":
            current = config.get_config("caps") or {}
            current.update(cfg_change["value"])
            config.set_config("caps", current)
        else:
            config.set_config(cfg_change["key"], cfg_change["value"])
        reply = f"Done, Ruk — updated {cfg_change['key']} to {cfg_change['value']}. 🎯"
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
    if mastery_req:
        # Kicking off a multi-day autonomous job spends real API credits
        # over days unattended -- always confirm first, same as risky
        # tasks, regardless of what the generic risk-classifier thinks.
        if not req.approved:
            understanding = mastery.explain_understanding(
                mastery_req["skill"], mastery_req["days"], mastery_req["hours_per_day"]
            )
            return ChatResponse(
                reply=(
                    f"{understanding}\n\n"
                    f"Confirm kar do — \"{mastery_req['skill']}\" mein master "
                    f"banne ka mission, {mastery_req['days']} din, roz "
                    f"~{mastery_req['hours_per_day']}h — shuru karu?"
                ),
                needs_approval=True,
            )
        _remember(background_tasks, req.message, role="user")
        reply = mastery.start_mastery(
            mastery_req["skill"], mastery_req["days"], mastery_req["hours_per_day"]
        )
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["codebase_analysis"]:
        _remember(background_tasks, req.message, role="user")
        reply = codebase.analyze(req.message)
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if cls["logs_request"]:
        _remember(background_tasks, req.message, role="user")
        logs = codebase.read_recent_logs()
        reply = call_llm(
            "gemini",
            [
                {"role": "system", "content": SANDY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Ruk's question: {req.message}\n\nYour recent runtime log:\n{logs}\n\n"
                    "Answer in Hinglish, referencing what's actually in the log above.",
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

    # Computed at most once, here -- brain.answer() below reuses this
    # instead of re-classifying complexity a second time internally.
    # ponytail: this used to be classify_complexity() called twice per
    # search-needing message (once here, once again inside brain.answer),
    # two separate Groq calls classifying the exact same message.
    tier = None
    if cls["search_needed"]:
        tier = brain.classify_complexity(req.message)
        provider = req.search_provider or _extract_search_provider(req.message)
        try:
            results = search.search(req.message, provider=provider, complexity=tier)
            search_block = "\n\n".join(f"[{r['title']}]({r['url']})\n{r['content']}" for r in results)
            context += f"\n\nWeb search results:\n{search_block}"
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def history(limit: int = 200):
    return {"messages": chatlog.get_history(limit=limit)}


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
        facts = memory.get_all_facts(limit=30)
    except Exception as e:
        log(f"[/status] memory read failed: {e!r}")
        facts = None
    try:
        jobs = mastery.list_mastery_jobs()
    except Exception as e:
        log(f"[/status] job list read failed: {e!r}")
        jobs = None
    return {"config": caps, "memory_facts": facts, "jobs": jobs}


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
