"""
Sandy's permanent memory — everything Ruk says/does gets remembered
automatically. Powered by Mem0 (extracts facts, merges, resolves
contradictions) on top of Supabase pgvector.

ponytail: Mem0's own Supabase/pgvector integration handles the vector
store config — we don't hand-roll embeddings or similarity search.
"""
import os

from mem0 import Memory

from llm import log, CapExceeded, _check_and_bump_cap

RUK = "ruk"  # single user for now — multi-user is a config value away, not a rewrite

# scode: root-cause fix for the "Groq dies in 5-6 messages" bug. This used
# to be provider="groq" -- Mem0's OWN internal fact-extraction call, on the
# SAME GROQ_API_KEY as Sandy's main brain, on a code path that completely
# bypassed Sandy's own cap system (config.py). HF Space logs showed single
# extraction calls burning 9k-13k tokens, exhausting Groq's real 100k/day
# account-wide limit in ~8-10 messages regardless of how many classifier
# calls main.py/brain.py made. Moved to Gemini, which nothing else in the
# hot path leans on as hard, and gated below with Sandy's real cap check
# (_check_and_bump_cap) so it now actually respects caps set via chat --
# "set gemini cap to X" now genuinely limits this call too, not just the
# visible chat replies.
_MEM_PROVIDER = "gemini"
_config = {
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ["SUPABASE_DB_CONNECTION_STRING"],  # matches your existing HF secret name
            "collection_name": "sandy_memories",
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-3.5-flash",
            "api_key": os.environ["GOOGLE_API_KEY"],
        },
    },
    # ponytail: reuse the Google key you already have as GOOGLE_API_KEY
    # (that's the standard name litellm/mem0 both look for) instead of
    # adding a differently-named duplicate secret.
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ["GOOGLE_API_KEY"],
            "embedding_dims": 768,
        },
    },
}

_memory: Memory | None = None


def _m() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory.from_config(_config)
    return _memory


def remember(message: str, role: str = "user") -> None:
    """Store a message so Sandy can recall it later. Mem0 auto-extracts
    the actual facts worth keeping -- we don't decide what's important.
    Best-effort: if this fails (e.g. Groq's TPD cap, which Mem0's own
    internal LLM call doesn't respect Sandy's cap system for), the raw
    message is still safe in chat_log -- losing one fact extraction is
    not worth crashing the background task over, which is exactly what
    was happening before this try/except existed."""
    try:
        _check_and_bump_cap(_MEM_PROVIDER)  # counts as one call against the same cap Ruk sets via chat
    except CapExceeded as e:
        log(f"[memory.remember] {_MEM_PROVIDER} capped, skipping extraction this message: {e}")
        return
    try:
        _m().add(message, user_id=RUK, metadata={"role": role})
    except Exception as e:
        log(f"[memory.remember] Mem0 extraction failed, message stays in chat_log only: {e!r}")


def recall(query: str, limit: int = 5) -> list[str]:
    """Pull the memories most relevant to the current message."""
    results = _m().search(query, filters={"user_id": RUK}, top_k=limit)
    return [r["memory"] for r in results["results"]]


def get_all_facts(limit: int = 50) -> list[str]:
    """Every fact Mem0 has extracted and stored for Ruk, most recent
    first -- for Ruk's Home's Memory view. Separate from recall(), which
    is a semantic search against a specific query."""
    results = _m().get_all(filters={"user_id": RUK}, top_k=limit)
    return [r["memory"] for r in results["results"]]


if __name__ == "__main__":
    # ponytail self-check: needs live Groq key + Supabase Postgres conn string.
    # Not a mock test — Mem0's value IS the real extraction behavior,
    # a mocked version would test nothing meaningful.
    remember("Ruk's laptop is an Acer Ryzen 3 7320U with 8GB RAM")
    hits = recall("what laptop does Ruk have")
    assert any("Acer" in h or "Ryzen" in h for h in hits), f"memory didn't recall the fact: {hits}"
    print("memory.py: remember + recall OK ->", hits)
