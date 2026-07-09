import os
import logging

from fastapi import FastAPI, Request, Response

from app.memory import get_memory
from app.llm import call_llm
from app.whatsapp import send_whatsapp_reply

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sandy")

app = FastAPI(title="Sandy")

# Lazy-init so a missing env var doesn't crash the whole container on
# startup — /health should always return 200 even if memory isn't
# configured yet, so your keep-alive ping never fails for the wrong reason.
_memory = None
def memory():
    global _memory
    if _memory is None:
        _memory = get_memory()
    return _memory


@app.get("/health")
def health():
    """
    Hit by your keep-alive ping (UptimeRobot / n8n cron) every few hours
    to stop the 48h free-tier sleep. Kept intentionally fast and dumb —
    no memory/LLM calls here, just proves the container is alive.
    """
    return {"status": "ok"}


@app.get("/webhook/whatsapp")
def verify_webhook(request: Request):
    """
    Meta Cloud API's one-time webhook verification handshake.
    Meta calls this when you first register the webhook URL.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    expected = os.environ.get("WHATSAPP_VERIFY_TOKEN")

    # TEMPORARY DEBUG LOGGING — remove once verification works.
    # Space is public, so we deliberately avoid printing the full
    # secret — just enough (length + first/last char) to spot
    # whitespace/mismatch issues in the Logs tab.
    def _peek(s):
        if not s:
            return "None"
        return f"len={len(s)} first={s[0]!r} last={s[-1]!r}"

    log.info(
        "Webhook verify attempt: received=[%s] expected=[%s] match=%s",
        _peek(token),
        _peek(expected),
        token == expected,
    )

    if mode == "subscribe" and token == expected:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)


@app.post("/webhook/whatsapp")
async def incoming_message(request: Request):
    """
    Real incoming WhatsApp messages land here. Pulls relevant memory,
    calls the LLM, saves the exchange, and sends the reply back on
    WhatsApp via Meta's Cloud API.
    """
    payload = await request.json()
    log.info("Incoming webhook payload: %s", payload)

    try:
        text = (
            payload["entry"][0]["changes"][0]["value"]
            ["messages"][0]["text"]["body"]
        )
        user_id = (
            payload["entry"][0]["changes"][0]["value"]
            ["messages"][0]["from"]
        )
    except (KeyError, IndexError):
        # Not an actual message event (e.g. a status update) — ignore safely
        return {"status": "ignored"}

    mem = memory()

    # Pull relevant memories for this user before replying
    relevant = mem.search(query=text, user_id=user_id, limit=5)
    context = "\n".join(m["memory"] for m in relevant.get("results", []))

    messages = [
        {"role": "system", "content": f"You are Sandy. Relevant memories:\n{context}"},
        {"role": "user", "content": text},
    ]
    reply = call_llm(messages)

    # Save this exchange to memory (Supabase — survives every restart)
    mem.add(
        [{"role": "user", "content": text}, {"role": "assistant", "content": reply}],
        user_id=user_id,
    )

    log.info("Reply for %s: %s", user_id, reply)

    sent = send_whatsapp_reply(to=user_id, text=reply)
    return {"status": "processed" if sent else "processed_send_failed", "reply": reply}
