"""
Sandy Memory Provider — wraps the project's existing memory.py (mem0ai:
Groq extraction + Gemini embeddings + Supabase pgvector) so Hermes uses
Sandy's real memory instead of Hermes' bundled Mem0 plugin (which hard-locks
LLM/embedder choice to openai/ollama and can't be pointed at Groq+Gemini —
confirmed by reading plugins/memory/mem0/_oss_providers.py directly).

Thin adapter only — no memory logic duplicated. All extraction, merging,
contradiction-resolution stays in memory.py / mem0 exactly as it runs today
for the FastAPI /chat endpoint.

Requires the project root (where memory.py lives) to be importable —
PYTHONPATH must include /app in the container (see Dockerfile).
"""

import logging
import os
import threading

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

REQUIRED_ENV = ("GROQ_API_KEY", "GOOGLE_API_KEY", "SUPABASE_DB_CONNECTION_STRING")

# agent_context values where writes should be skipped — cron/subagent turns
# would pollute Ruk's actual memory (per MemoryProvider ABC's own docstring).
WRITE_CONTEXTS = {"primary"}


class SandyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "sandy-memory"

    def is_available(self) -> bool:
        missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
        if missing:
            logger.warning("sandy-memory unavailable, missing env: %s", missing)
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._sync_thread = None

        import memory as _sandy_memory
        self._memory = _sandy_memory

    def get_tool_schemas(self):
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        try:
            hits = self._memory.recall(query)
        except Exception as e:
            logger.warning("sandy-memory prefetch failed: %s", e)
            return ""
        if not hits:
            return ""
        return "\n".join(hits)

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if self._agent_context not in WRITE_CONTEXTS:
            return

        def _sync():
            try:
                self._memory.remember(user_content, role="user")
                self._memory.remember(assistant_content, role="assistant")
            except Exception as e:
                logger.warning("sandy-memory sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True)
        self._sync_thread.start()

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

# No register() needed — Hermes' loader (plugins/memory/__init__.py) falls
# back to scanning for a MemoryProvider subclass and instantiating it
# directly when there's no working register(ctx). Verified working via
# that fallback (see build notes / conversation).
