"""
LLM routing for Sandy's brain.

Starts simple: Groq only, since that's enough to get the whole
pipeline working end to end. Add Gemini / Cerebras / DeepSeek as
additional tiers later by extending `call_llm` — the rest of the
app never needs to change.
"""

import os
from groq import Groq

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def call_llm(messages: list[dict], model: str = "llama-3.3-70b-versatile") -> str:
    """
    messages: standard OpenAI-style [{"role": "user", "content": "..."}] list
    Returns the assistant's reply as plain text.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content
