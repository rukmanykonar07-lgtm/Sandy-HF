"""
Sandy's memory layer.

IMPORTANT: this is the ONLY place memory should be written.
Never write memory/state to local disk in this project — HF Spaces
wipes everything outside /tmp on every restart, and even /tmp doesn't
survive a restart. Supabase lives outside HF entirely, so this is what
makes Sandy's memory permanent.

NOTE: Mem0's dedicated "supabase" vector store provider isn't in the
current PyPI release yet (it's on GitHub main but unreleased). Since
Supabase is just PostgreSQL + pgvector under the hood, we connect to it
using Mem0's "pgvector" provider instead — same database, same effect,
already supported in the installed version.
"""

import os
from urllib.parse import urlparse, unquote
from mem0 import Memory


def _parse_connection_string(conn_string: str) -> dict:
    """
    Splits a postgres connection string like:
      postgresql://user:password@host:port/dbname
    into the individual fields mem0's pgvector provider expects.
    """
    parsed = urlparse(conn_string)
    return {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") or "postgres",
    }


def get_memory() -> Memory:
    """
    Returns a configured Mem0 instance backed by Supabase's Postgres
    database via the pgvector provider.

    Required env vars (set these as SECRETS in the HF Space settings,
    never hardcode them):
      SUPABASE_DB_CONNECTION_STRING  -> postgres connection string from
                                         Supabase > Project Settings > Database
      GROQ_API_KEY                   -> used by Mem0 for its own
                                         extraction/summarization calls
      GOOGLE_API_KEY                 -> used by Mem0's embedder (Gemini
                                         text-embedding-004, free tier)

    NOTE on embedding_model_dims: Gemini's text-embedding-004 outputs
    768-dim vectors. Mem0 defaults to 1536 (OpenAI's dimension) if this
    isn't set explicitly — that mismatch doesn't error, it just makes
    every future search() silently return nothing. Keep this in sync if
    you ever change the embedder model.
    """
    db = _parse_connection_string(os.environ["SUPABASE_DB_CONNECTION_STRING"])

    config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "user": db["user"],
                "password": db["password"],
                "host": db["host"],
                "port": db["port"],
                "dbname": db["dbname"],
                "collection_name": "sandy_memories",
                "embedding_model_dims": 768,
            }
        },
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.3-70b-versatile",
                "api_key": os.environ["GROQ_API_KEY"],
            }
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/text-embedding-004",
                "api_key": os.environ["GOOGLE_API_KEY"],
            }
        },
    }
    return Memory.from_config(config)


# --- One-time Supabase setup (run this SQL once in the Supabase SQL editor,
# NOT in this Python file, before Sandy ever tries to write a memory) ---
SUPABASE_SETUP_SQL = """
-- Enable the pgvector extension
create extension if not exists vector;
"""
