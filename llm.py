"""
One function to call any of Sandy's LLMs. litellm gives every provider
the same interface â€” no hand-written HTTP client per provider.

Cap enforcement lives here (checked before every call) so it's
impossible for a router/orchestrator to accidentally bypass it.
"""
import datetime
import logging
import re

from litellm import completion

import config

logging.basicConfig(
    filename="/tmp/sandy.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


def log(msg: str) -> None:
    """Every diagnostic message goes through here instead of a bare
    print(): still prints (unchanged, visible in HF's live log viewer),
    AND writes to /tmp/sandy.log so Sandy can read her own recent logs
    when Ruk asks what went wrong."""
    print(msg)
    logging.info(msg)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_fence(raw: str) -> str:
    """LLMs asked for 'raw content only, no fences' commonly wrap the
    response in ```language ... ``` fences anyway. Strip that before
    using the content for anything -- otherwise the literal fence
    markers end up written into whatever the content becomes (a file,
    a parsed JSON blob, etc)."""
    m = _FENCE_RE.match(raw.strip())
    return m.group(1) if m else raw


def strip_json_fence(raw: str) -> str:
    """Same as strip_fence() -- kept as a separate name at JSON call
    sites so it's clear what's being extracted there."""
    return strip_fence(raw)

MODELS = {
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "gemini/gemini-3.5-flash",
    "cerebras": "cerebras/gpt-oss-120b",
}

# scode: free-tier context ceilings, confirmed against each provider's own
# docs (Cerebras's 8,192 is real and is why it was dying in 1-2 messages --
# not a credits problem, every provider was getting the same ~2.5-3k token
# history/identity blob and Cerebras just can't hold that much). Unlisted
# provider -> generous fallback, no truncation.
CONTEXT_LIMITS = {"cerebras": 8_192}
_DEFAULT_CONTEXT_LIMIT = 128_000
_RESPONSE_RESERVE = 1_500  # leave room for the actual reply, not just the prompt


def _estimate_tokens(text: str) -> int:
    return len(text) // 4  # standard rough heuristic -- good enough without a live tokenizer


def fit_to_budget(messages: list[dict], provider: str) -> list[dict]:
    """Trims oldest history first until the message list fits the
    provider's real context window. Always keeps the first message
    (system/identity prompt) and the last (the actual current task) --
    those are never dropped, only what's in between."""
    budget = CONTEXT_LIMITS.get(provider, _DEFAULT_CONTEXT_LIMIT) - _RESPONSE_RESERVE
    if len(messages) <= 2 or budget <= 0:
        return messages
    head, tail = messages[0], messages[-1]
    used = _estimate_tokens(head["content"]) + _estimate_tokens(tail["content"])
    kept = []
    for m in reversed(messages[1:-1]):  # most recent history first
        t = _estimate_tokens(m["content"])
        if used + t > budget:
            break
        kept.append(m)
        used += t
    kept.reverse()
    return [head] + kept + [tail]

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
    messages = fit_to_budget(messages, provider)
    response = completion(model=MODELS[provider], messages=messages, **kwargs)
    return response.choices[0].message.content


def call_llm_with_fallback(provider: str, messages: list[dict], **kwargs) -> str:
    """Same as call_llm, but if `provider` is capped OR its API call fails
    for any reason (outage, rate limit, timeout), tries the other
    providers in turn instead of failing the whole request. Use this for
    single-provider call sites (_judge, orchestrator) that have no other
    provider to fall back on already — _run_tier's multi-provider loop
    doesn't need this, it already gathers from several providers."""
    order = [provider] + [p for p in MODELS if p != provider]
    last_err = None
    for p in order:
        try:
            return call_llm(p, messages, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


if __name__ == "__main__":
    # ponytail self-check: real call to Groq (fastest/free), confirms
    # the model string + key actually work end to end.
    reply = call_llm("groq", [{"role": "user", "content": "reply with exactly: pong"}])
    assert "pong" in reply.lower(), f"unexpected reply: {reply}"
    print("llm.py: groq call OK ->", reply)
