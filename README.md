---
title: Sandy
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sandy — Backend

Hermes-style brain for Sandy, running on Hugging Face Spaces (free CPU Basic
tier), with permanent memory in Supabase (pgvector) so nothing is lost when
the Space sleeps/restarts.

## Setup — do these in order

### 1. Supabase (memory layer)
1. Create a free project at supabase.com if you don't have one yet.
2. Go to the SQL Editor and run the contents of the `SUPABASE_SETUP_SQL`
   string in `app/memory.py` (enables pgvector + creates the migrations
   table Mem0 needs).
3. Go to Project Settings > Database > Connection string, copy the URI
   (use the "Session pooler" one, not the direct connection).

### 2. Create the HF Space
1. Go to huggingface.co/new-space
2. SDK: **Docker**
3. Visibility: **Private** (important — keeps your tokens/logs from being
   public)
4. Hardware: leave as the free **CPU basic**

### 3. Push this code
```bash
git clone https://huggingface.co/spaces/<your-username>/sandy
cd sandy
# copy Dockerfile, requirements.txt, app/ into this folder
git add .
git commit -m "Initial Sandy backend"
git push
```

### 4. Set your Secrets (Space Settings > Variables and Secrets — NOT in code)
| Secret name | Where to get it |
|---|---|
| `SUPABASE_DB_CONNECTION_STRING` | Supabase > Project Settings > Database |
| `GROQ_API_KEY` | console.groq.com |
| `WHATSAPP_VERIFY_TOKEN` | any string you make up — you'll enter the same one in Meta's webhook config |

### 5. Confirm it's alive
Visit `https://<your-username>-sandy.hf.space/health` — should return
`{"status": "ok"}`. If it doesn't, check the Space's "Logs" tab first.

### 6. Set up the keep-alive ping (stops the 48h sleep)
Use a free UptimeRobot monitor (uptimerobot.com) pointed at your `/health`
URL, checking every few hours. This is what keeps the Space from ever
actually hitting the 48-hour inactivity threshold.

### 7. Wire up WhatsApp (Meta Cloud API)
Point Meta's webhook config at:
`https://<your-username>-sandy.hf.space/webhook/whatsapp`
using the same verify token you set in step 4.

## What's stubbed / not done yet
- Actual outgoing WhatsApp send (Meta Cloud API call) — currently the
  reply is only logged and returned in the response, not sent back
- Kokoro TTS / voice — not wired in yet
- n8n automation layer — separate piece, not part of this backend
- Multi-tier LLM routing (Gemini/Cerebras/DeepSeek fallback) — currently
  Groq-only in `app/llm.py`, extend `call_llm` when ready

## Do NOT
- Write memory/state to local disk anywhere in this project — only
  `/tmp` is writable on HF Spaces and nothing outside Supabase survives
  a restart
- Make this Space public — your tokens live in Secrets, but logs and
  code would still be world-visible
