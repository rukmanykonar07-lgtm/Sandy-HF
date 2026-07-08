"""
Sandy's memory layer.

IMPORTANT: this is the ONLY place memory should be written.
Never write memory/state to local disk in this project — HF Spaces
wipes everything outside /tmp on every restart, and even /tmp doesn't
survive a restart. Supabase lives outside HF entirely, so this is what
makes Sandy's memory permanent.
"""

import os
from mem0 import Memory

def get_memory() -> Memory:
    """
    Returns a configured Mem0 instance backed by Supabase (pgvector).

    Required env vars (set these as SECRETS in the HF Space settings,
    never hardcode them):
      SUPABASE_DB_CONNECTION_STRING  -> postgres connection string from
                                         Supabase > Project Settings > Database
      GROQ_API_KEY                   -> used by Mem0 for its own
                                         extraction/summarization calls
    """
    config = {
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": os.environ["SUPABASE_DB_CONNECTION_STRING"],
                "collection_name": "sandy_memories",
                "index_method": "hnsw",
                "index_measure": "cosine_distance",
            }
        },
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.3-70b-versatile",
                "api_key": os.environ["GROQ_API_KEY"],
            }
        },
    }
    return Memory.from_config(config)


# --- One-time Supabase setup (run this SQL once in the Supabase SQL editor,
# NOT in this Python file, before Sandy ever tries to write a memory) ---
SUPABASE_SETUP_SQL = """
-- Enable the pgvector extension
create extension if not exists vector;

-- Mem0's migrations table (prevents the common
-- "could not find table memory_migrations" error)
create table if not exists memory_migrations (
    id serial primary key,
    version text not null,
    applied_at timestamptz default now()
);

-- Mem0 will create its own 'sandy_memories' collection/table
-- automatically on first write once the extension above exists.
"""
