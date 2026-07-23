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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import memory
import brain
import mastery
import selfmod
from llm import call_llm, CapExceeded

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: tighten to the real Ruk's Home domain once frontend is deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

RISKY_KEYWORDS = [
    "delete", "remove file", "overwrite", "deploy", "hf space",
    "hugging face space", "drop table", "format", "rm -rf", "uninstall",
]


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    approved: bool = False       # frontend resends with this =true after Ruk confirms
    override_llms: list[str] | None = None  # e.g. ["gemini"] or ["orchestrator"]


class ChatResponse(BaseModel):
    reply: str
    needs_approval: bool = False


def _is_config_change(message: str) -> dict | None:
    """Cheap check: does this message ask to change a cap/preference?
    Returns the parsed {key, value} or None if it's a normal task."""
    prompt = (
        "Does this message ask to change ONLY an LLM credit cap or a "
        "structured Sandy setting (caps, approval_required_for, "
        "always_ask_approval)? This does NOT include requests to edit "
        "Sandy's own code/files, her identity/personality, or how she "
        "talks/behaves -- those are handled elsewhere, always answer "
        "false for those, even if they sound like a 'preference'. "
        f'Message: "{message}"\n'
        'If yes, reply JSON: {"is_config": true, "key": "...", "value": ...} '
        '(key is one of: caps, approval_required_for, always_ask_approval; '
        'for caps, value is a dict like {"gemini": 100}). '
        'If no, reply exactly: {"is_config": false}'
    )
    raw = call_llm("groq", [{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not parsed.get("is_config"):
        return None
    if "key" not in parsed or "value" not in parsed:
        return None  # malformed -> fail safe, treat as a normal task instead
    return parsed


def _needs_approval(message: str) -> bool:
    cfg = config.get_all_config()
    if cfg.get("always_ask_approval"):
        return True
    lower = message.lower()
    if any(kw in lower for kw in RISKY_KEYWORDS):
        return True
    return any(flag.lower() in lower for flag in cfg.get("approval_required_for", []))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return _handle_chat(req)
    except CapExceeded as e:
        # ponytail: one guard around the whole flow, not one per LLM call-site —
        # every path through this function calls an LLM somewhere.
        return ChatResponse(
            reply=f"Can't do that right now — {e}. Tell me to raise the cap or try again later."
        )


def _handle_chat(req: ChatRequest) -> ChatResponse:
    cfg_change = _is_config_change(req.message)
    if cfg_change:
        memory.remember(req.message, role="user")
        if cfg_change["key"] == "caps":
            current = config.get_config("caps") or {}
            current.update(cfg_change["value"])
            config.set_config("caps", current)
        else:
            config.set_config(cfg_change["key"], cfg_change["value"])
        reply = f"Done, Ruk — updated {cfg_change['key']} to {cfg_change['value']}. 🎯"
        memory.remember(reply, role="assistant")
        return ChatResponse(reply=reply)

    # If Ruk already has a pending self-edit proposal for this session and
    # just approved it, apply it directly -- don't re-classify the text,
    # since re-running the LLM could in principle produce a different
    # proposal than what was actually shown and approved.
    if req.approved and req.session_id in selfmod._pending:
        memory.remember(req.message, role="user")
        try:
            reply = selfmod.apply_pending(req.session_id)
        except selfmod.GitOpError as e:
            reply = f"Ruk, edit push nahi ho paya: {e}"
        memory.remember(reply, role="assistant")
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
            memory.remember(req.message, role="user")
            try:
                reply = selfmod.rollback_to(selfmod_req["commit_hash"])
            except selfmod.GitOpError as e:
                reply = f"Ruk, rollback nahi ho paaya: {e}"
            memory.remember(reply, role="assistant")
            return ChatResponse(reply=reply)

    mastery_req = mastery.extract_mastery_request(req.message)
    if mastery_req:
        # Kicking off a multi-day autonomous job spends real API credits
        # over days unattended -- always confirm first, same as risky
        # tasks, regardless of what the generic risk-classifier thinks.
        if not req.approved:
            return ChatResponse(
                reply=(
                    f"Ruk, confirm kar do — \"{mastery_req['skill']}\" mein master "
                    f"banne ka mission, {mastery_req['days']} din, roz "
                    f"~{mastery_req['hours_per_day']}h — shuru karu?"
                ),
                needs_approval=True,
            )
        memory.remember(req.message, role="user")
        reply = mastery.start_mastery(
            mastery_req["skill"], mastery_req["days"], mastery_req["hours_per_day"]
        )
        memory.remember(reply, role="assistant")
        return ChatResponse(reply=reply)

    if _needs_approval(req.message) and not req.approved:
        # ponytail: don't save to memory yet — it gets saved once below,
        # when the (approved) resend actually goes through. Saving here
        # too would double up every risky message in memory.
        return ChatResponse(
            reply=f"Ruk, ye thoda risky lag raha hai: \"{req.message}\" — confirm karoge to karti hoon.",
            needs_approval=True,
        )

    memory.remember(req.message, role="user")
    recalled = memory.recall(req.message)
    context = ("Things you remember about Ruk:\n" + "\n".join(recalled)) if recalled else ""
    reply = brain.answer(req.message, context=context, override=req.override_llms)
    memory.remember(reply, role="assistant")
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
