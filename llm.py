"""
One function to call any of Sandy's LLMs. litellm gives every provider
the same interface â€” no hand-written HTTP client per provider.

Cap enforcement lives here (checked before every call) so it's
impossible for a router/orchestrator to accidentally bypass it.
"""
import datetime

from litellm import completion

import config

MODELS = {
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "gemini/gemini-3.5-flash",
    "cerebras": "cerebras/llama-3.3-70b",
}

# ponytail: call counts kept in the same sandy_config table via a
# "usage" key, not a new table â€” one less thing to provision.


class CapExceeded(Exception):
    pass


def _today() -> str:
    return datetime.date.today().isoformat()


def _check_and_bump_cap(provider: str) -> None:
    caps = config.get_config("caps") or {}
    cap = caps.get(provider)
    if cap is None:
        return  # no cap set for this provider

    usage = config.get_config("usage") or {}
    if usage.get("date") != _today():
        usage = {"date": _today()}  # new day -> counts reset automatically

    used = usage.get(provider, 0)
    if used >= cap:
        raise CapExceeded(f"{provider} hit its cap of {cap} calls today")
    usage[provider] = used + 1
    config.set_config("usage", usage)


def call_llm(provider: str, messages: list[dict], **kwargs) -> str:
    """provider: 'groq' | 'gemini' | 'cerebras'. messages: standard
    OpenAI-style [{role, content}] list."""
    if provider not in MODELS:
        raise ValueError(f"unknown provider: {provider}")
    _check_and_bump_cap(provider)
    response = completion(model=MODELS[provider], messages=messages, **kwargs)
    return response.choices[0].message.content


if __name__ == "__main__":
    # ponytail self-check: real call to Groq (fastest/free), confirms
    # the model string + key actually work end to end.
    reply = call_llm("groq", [{"role": "user", "content": "reply with exactly: pong"}])
    assert "pong" in reply.lower(), f"unexpected reply: {reply}"
    print("llm.py: groq call OK ->", reply)
