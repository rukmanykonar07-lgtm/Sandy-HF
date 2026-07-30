"""
Sandy's permanent memory — everything Ruk says/does gets remembered
automatically. Powered by Mem0 (extracts facts, merges, resolves
contradictions) on top of Supabase pgvector.

ponytail: Mem0's own Supabase/pgvector integration handles the vector
store config — we don't hand-roll embeddings or similarity search.
"""
import os

from mem0 import Memory

RUK = "ruk"  # single user for now — multi-user is a config value away, not a rewrite

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
        "provider": "groq",
        "config": {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.environ["GROQ_API_KEY"],
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
    the actual facts worth keeping — we don't decide what's important."""
    _m().add(message, user_id=RUK, metadata={"role": role})


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
