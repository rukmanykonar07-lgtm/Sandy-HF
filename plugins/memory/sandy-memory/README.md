# sandy-memory

Hermes memory provider that reuses Sandy's existing `memory.py` (Groq
extraction + Gemini embeddings + Supabase pgvector via mem0ai) instead of
Hermes' bundled Mem0 plugin, which only allows `openai`/`ollama` for the
LLM/embedder (confirmed by reading `plugins/memory/mem0/_oss_providers.py`
in the installed package).

## Install (in the Docker image)

Copy this whole folder to `$HERMES_HOME/plugins/sandy-memory/` — note it
goes directly under `plugins/`, not `plugins/memory/`. Verified by reading
the real loader (`plugins/memory/__init__.py`): user-installed providers
are scanned from `$HERMES_HOME/plugins/<name>/`, not a memory/ subfolder.

```
mkdir -p $HERMES_HOME/plugins
cp -r plugins/memory/sandy-memory $HERMES_HOME/plugins/sandy-memory
```

## Activate

In `config.yaml`:

```yaml
memory:
  provider: sandy-memory
```

## Requirements

- `memory.py` must be importable — the project root (`/app`) needs to be
  on `PYTHONPATH` inside the container.
- Env vars: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `SUPABASE_DB_CONNECTION_STRING`
  (same ones the FastAPI app already uses — no new secrets needed).

## Notes

- Single global user (`RUK`) — matches `memory.py`'s design. Hermes' own
  `session_id` is accepted but not used for scoping.
- `sync_turn()` runs on a daemon thread since `remember()` calls out to
  Groq + Supabase. `shutdown()` joins that thread with a 5s timeout.
- Skips writes unless `agent_context == "primary"`, so cron/subagent turns
  never pollute Ruk's real memory.
- No `register()` function — Hermes' loader falls back to scanning for a
  `MemoryProvider` subclass and instantiating it directly when there's no
  working `register(ctx)`. Verified via the actual loader, not assumed.
