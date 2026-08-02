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


def _is_config_change(message: str) -> dict | None:
    """Cheap check: does this message ask to change a cap/preference?
    Returns the parsed {key, value} or None if it's a normal task."""
    prompt = (
        "Does this message ask to change an LLM credit cap? This does NOT "
        "include requests to edit Sandy's own code/files, her identity/"
        "personality, or how she talks/behaves -- those are handled "
        "elsewhere, always answer false for those, even if they sound "
        "like a 'preference'. "
        f'Message: "{message}"\n'
        'If yes, reply JSON: {"is_config": true, "key": "caps", "value": {"gemini": 100}} '
        '(value is a dict like this, with the provider name and new cap number). '
        'If no, reply exactly: {"is_config": false}'
    )
    raw = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(strip_json_fence(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not parsed.get("is_config"):
        return None
    if "key" not in parsed or "value" not in parsed:
        return None  # malformed -> fail safe, treat as a normal task instead
    return parsed


def _extract_llm_override(message: str) -> list[str] | None:
    """Does this message explicitly tell Sandy WHICH model(s) to use for
    the task (not what the task itself is)? Returns a provider list for
    brain.answer()'s `override`, or None to let auto-classification run
    as usual. Provider names come from llm.MODELS, so adding a new
    provider there (e.g. claude, kimi) is picked up here automatically."""
    providers = list(MODELS)
    prompt = (
        "Does this message explicitly tell Sandy which LLM/model to use "
        f"for this task? Valid providers: {', '.join(providers)}, or the "
        "word 'orchestrator' for full multi-round mode. This is about "
        "WHICH MODEL to use, not what the task is -- if the message is "
        "just a normal task/question with no model mentioned, answer no.\n"
        f'Message: "{message}"\n'
        'If yes, reply JSON only: {"override": ["provider_name"]} '
        '(or {"override": ["orchestrator"]}). '
        'If no, reply exactly: {"override": null}'
    )
    raw = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(strip_json_fence(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    override = parsed.get("override")
    if not isinstance(override, list) or not override:
        return None
    valid = set(providers) | {"orchestrator"}
    if not all(p in valid for p in override):
        return None  # malformed/hallucinated provider name -> fail safe
    return override


def _is_logs_request(message: str) -> bool:
    """Does this message ask Sandy to check her own recent runtime
    logs/errors (what went wrong, what broke), as opposed to her source
    code (that's _is_codebase_analysis_request)?"""
    prompt = (
        "Does this message ask Sandy to check her own recent runtime "
        "logs, errors, or what went wrong while running (NOT her source "
        "code files)? Answer with exactly one word: yes or no.\n"
        f'Message: "{message}"'
    )
    result = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}]).strip().lower()
    return result.startswith("yes")


def _is_search_request(message: str) -> bool:
    """Does answering this well need current/external info from the web
    (current events, specific facts, research on a topic), as opposed to
    something Sandy can reason/write from her own knowledge?"""
    prompt = (
        "Does answering this well require looking something up on the web "
        "(current events, specific facts, research on a topic, company/"
        "competitor info, etc) rather than just reasoning or writing? "
        "Answer with exactly one word: yes or no.\n"
        f'Message: "{message}"'
    )
    result = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}]).strip().lower()
    return result.startswith("yes")


def _extract_search_provider(message: str) -> str | None:
    """Cheap keyword check (no LLM call needed) for an explicit provider
    name in the message -- 'use exa for this' etc."""
    lower = message.lower()
    for p in ("tavily", "exa", "linkup"):
        if p in lower:
            return p
    return None


def _is_codebase_analysis_request(message: str) -> bool:
    """Does this message ask Sandy to look at / review / scan / analyze
    her own actual code (read-only understanding) -- NOT asking her to
    change/fix/edit anything, that's selfmod's job."""
    prompt = (
        "Does this message ask Sandy to look at, review, scan, or analyze "
        "her own codebase/code/files in a read-only way (NOT asking her to "
        "change, edit, or fix anything -- edits are handled elsewhere)? "
        "Answer with exactly one word: yes or no.\n"
        f'Message: "{message}"'
    )
    result = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}]).strip().lower()
    return result.startswith("yes")


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
    cfg_change = _is_config_change(req.message)
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

    # If Ruk already has a pending self-edit proposal for this session and
    # just approved it, apply it directly -- don't re-classify the text,
    # since re-running the LLM could in principle produce a different
    # proposal than what was actually shown and approved.
    if req.approved and req.session_id in selfmod._pending:
        _remember(background_tasks, req.message, role="user")
        try:
            reply = selfmod.apply_pending(req.session_id)
        except selfmod.GitOpError as e:
            reply = f"Ruk, edit push nahi ho paya: {e}"
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    selfmod_req = selfmod.extract_selfmod_request(req.message)
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

    mastery_req = mastery.extract_mastery_request(req.message)
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

    if _is_codebase_analysis_request(req.message):
        _remember(background_tasks, req.message, role="user")
        reply = codebase.analyze(req.message)
        _remember(background_tasks, reply, role="assistant")
        return ChatResponse(reply=reply)

    if _is_logs_request(req.message):
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

    if _is_search_request(req.message):
        provider = req.search_provider or _extract_search_provider(req.message)
        try:
            tier = brain.classify_complexity(req.message)
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

    override = req.override_llms or _extract_llm_override(req.message)
    recent = chatlog.get_history(limit=30)
    history = [{"role": m["role"], "content": m["message"]} for m in recent]
    reply = brain.answer(req.message, context=context, override=override, history=history)
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
