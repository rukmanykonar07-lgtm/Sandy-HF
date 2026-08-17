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
#
# Then live logs (17 Aug 2026) showed a DIFFERENT real problem: Gemini's
# free tier caps generate_content_free_tier_requests at 20/DAY total for
# gemini-3.5-flash -- with remember() firing on every single message,
# that's exhausted almost immediately, and every extraction after that
# silently fails (message still lands in chat_log, but nothing gets
# permanently learned via Mem0 -- a real, meaningful degradation of "Sandy
# remembers everything," not cosmetic).
#
# Switching straight back to Groq would UN-fix the original bug (same
# shared 100k/day account cap chat already leans on hard). Real fix: a
# genuine fallback, not a single point of failure -- gemini primary (still
# useful for the first ~20 msgs/day), Cerebras as fallback via Mem0's
# "litellm" passthrough provider. Cerebras isn't in Mem0's own native
# provider list, but litellm already knows it (same MODELS["cerebras"]
# string used everywhere else in this codebase) -- and it's not already
# under heavy contention the way Groq (chat) or Gemini (its own tiny cap)
# both are.
_MEM_PROVIDER = "gemini"
_MEM_FALLBACK_PROVIDER = "cerebras"
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

# Same vector_store/embedder (embeddings weren't the thing hitting quota --
# only gemini's LLM extraction call was) -- only the LLM leg changes,
# routed through litellm so it can use the exact same MODELS["cerebras"]
# model string call_llm_with_fallback already relies on elsewhere.
_fallback_config = {
    **_config,
    "llm": {
        "provider": "litellm",
        "config": {"model": "cerebras/gpt-oss-120b"},
    },
}

_memory: Memory | None = None
_memory_fallback: Memory | None = None


def _m() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory.from_config(_config)
    return _memory


def _m_fallback() -> Memory:
    global _memory_fallback
    if _memory_fallback is None:
        _memory_fallback = Memory.from_config(_fallback_config)
    return _memory_fallback


def remember(message: str, role: str = "user") -> None:
    """Store a message so Sandy can recall it later. Mem0 auto-extracts
    the actual facts worth keeping -- we don't decide what's important.

    Real 2-tier fallback, not a single point of failure: gemini first
    (cap-checked for real), and ONLY on gemini actually failing (its tiny
    20/day free-tier cap, or any other real error) does this fall back to
    Cerebras (also cap-checked, separately, for real) -- not Groq, which
    would silently reintroduce the exact contention bug that got Mem0
    moved off Groq in the first place. If BOTH fail, the raw message is
    still safe in chat_log -- losing one fact extraction is not worth
    crashing the background task over."""
    try:
        _check_and_bump_cap(_MEM_PROVIDER)
        _m().add(message, user_id=RUK, metadata={"role": role})
        return
    except CapExceeded as e:
        log(f"[memory.remember] {_MEM_PROVIDER} capped, trying fallback: {e}")
    except Exception as e:
        log(f"[memory.remember] {_MEM_PROVIDER} extraction failed ({e!r}), trying fallback")

    try:
        _check_and_bump_cap(_MEM_FALLBACK_PROVIDER)
        _m_fallback().add(message, user_id=RUK, metadata={"role": role})
    except CapExceeded as e:
        log(f"[memory.remember] fallback {_MEM_FALLBACK_PROVIDER} also capped, message stays in chat_log only: {e}")
    except Exception as e:
        log(f"[memory.remember] fallback {_MEM_FALLBACK_PROVIDER} also failed, message stays in chat_log only: {e!r}")


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
