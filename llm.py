"""
One function to call any of Sandy's LLMs. litellm gives every provider
the same interface â€” no hand-written HTTP client per provider.

Cap enforcement lives here (checked before every call) so it's
impossible for a router/orchestrator to accidentally bypass it.
"""
import datetime
import logging
import os
import re
import time

from litellm import completion

import config
import observability

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
    # scode: added Aug 2026 -- verified against litellm's real provider docs
    # (docs.litellm.ai/docs/providers/<name>) one by one, not assumed, same
    # discipline as the original three. Each is added to MODELS (so
    # llm_override / orchestrator can reach it) but deliberately NOT added
    # to brain.TIERS yet -- that's a second, separate decision (which tier
    # each belongs in) that needs Ruk's real-world testing first, not a
    # guess baked in here.
    "deepseek": "deepseek/deepseek-chat",
    "mistral": "mistral/mistral-large-latest",
    "cohere": "cohere/command-r-plus",
    "moonshot": "moonshot/kimi-k2-0711-preview",
    "zhipu": "zai/glm-4.7",
    "cloudflare": "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "nvidia": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "novita": "novita/deepseek/deepseek-r1",
    "deepinfra": "deepinfra/meta-llama/Llama-3.3-70B-Instruct",
    "siliconflow": "siliconflow/deepseek-ai/DeepSeek-V3",
    "openrouter": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    # scode: added Aug 2026 (phase A) -- GitHub Models, verified against
    # docs.litellm.ai/docs/providers/github: prefix is "github/", all
    # GitHub-hosted models supported by name. litellm's default env var
    # is GITHUB_API_KEY but this deployment's secret is GITHUB_TOKEN --
    # mapped explicitly in _API_KEY_ENV below so call_llm injects it as
    # api_key= directly instead of renaming the secret on HF's side.
    # Model picked for free-tier reliability + speed (not the biggest):
    # GPT-4.1 mini tier class; swap the string here if Ruk wants another.
    "github": "github/gpt-4.1-mini",
    # NOT added -- couldn't verify a real litellm provider for these,
    # which is exactly the mistake this project has been burned by
    # before. Flagging honestly instead of inventing:
    #   - "byteplus" (BYTEPLUS_API_KEY) -- re-checked Aug 2026 against
    #     litellm's full provider index: NO byteplus entry exists
    #     (closest name is "Bytez", a different service). If Ruk ever
    #     needs it, it would be a manual openai-compatible base_url
    #     setup, not a provider prefix.
}

# scode: litellm resolves each provider's API key from a DEFAULT env var
# name per provider (e.g. DEEPSEEK_API_KEY, MISTRAL_API_KEY) -- most of
# your secrets already match those defaults. Two don't (Zhipu's real
# litellm provider is "zai" expecting ZAI_API_KEY, not ZHIPU_API_KEY; NVIDIA
# NIM expects NVIDIA_NIM_API_KEY, not NVIDIA_API_KEY) -- mapped explicitly
# below instead of renaming/duplicating secrets on HF's side.
_API_KEY_ENV = {
    "zhipu": "ZHIPU_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "github": "GITHUB_TOKEN",  # litellm default is GITHUB_API_KEY; Ruk's real secret is GITHUB_TOKEN
}
_NVIDIA_NIM_BASE = "https://integrate.api.nvidia.com/v1"  # NVIDIA's standard NIM cloud endpoint

# Part 6 stall safety: litellm.completion() has NO default HTTP timeout --
# a wedged provider (TCP black hole, dead gateway) would hang a worker
# thread forever and hold the orchestrator's hard barrier hostage. This
# bounds every completion() call; it must stay WELL under brain.py's
# orchestrator stall threshold (default 180s) so the watchdog there is a
# rare backstop, not the normal unblock path. Env-overridable.
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "90"))

# scode: complete provider -> real env var name reference, for anything that
# needs to just KNOW the name (diagnostics.py's presence checks) without
# affecting call_llm's actual behavior. Deliberately separate from
# _API_KEY_ENV above (which only holds the 2 cases call_llm must explicitly
# override) -- this is a superset that also documents the defaults litellm
# already resolves on its own, so there's exactly one place this mapping
# lives instead of a second hand-maintained copy that can go stale (which
# is exactly how diagnostics.py's old _KNOWN_KEYS ended up checking
# TAVILY_API_KEY/EXA_API_KEY/LINKUP_API_KEY -- names that were never real --
# instead of the actual TAVILY/EXA/LINKUP secrets in search.py).
# "gemini": GOOGLE_API_KEY confirmed against this deployment's own real
# secret name (memory.py reads os.environ["GOOGLE_API_KEY"] directly for
# the exact same Gemini account) -- not a guess.
PROVIDER_API_KEY_ENV = {
    "groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "cerebras": "CEREBRAS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY", "mistral": "MISTRAL_API_KEY", "cohere": "COHERE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY", "cloudflare": "CLOUDFLARE_API_KEY",
    "novita": "NOVITA_API_KEY", "deepinfra": "DEEPINFRA_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    **_API_KEY_ENV,  # zhipu/nvidia/github overrides -- kept in sync automatically, not copy-pasted
}

# scode: free-tier context ceilings, confirmed against each provider's own
# docs (Cerebras's 8,192 is real and is why it was dying in 1-2 messages --
# not a credits problem, every provider was getting the same ~2.5-3k token
# history/identity blob and Cerebras just can't hold that much). Unlisted
# provider -> generous fallback, no truncation.
#
# Aug 2026 expansion (per-provider max-power work): documented real windows
# for the rest of the pool, each verified against the provider's own docs /
# model card at write time, same discipline as MODELS. These are the
# PROVIDER-level safe ceilings for the default models in MODELS above --
# fit_to_budget() uses them so a mid-chat switch refits instead of erroring
# (groq -> cerebras shrinks to what cerebras can actually hold).
CONTEXT_LIMITS = {
    "cerebras": 8_192,
    "groq": 128_000,        # llama-3.3-70b-versatile on Groq free tier
    "gemini": 1_000_000,    # gemini-3.5-flash -- the long-doc/research king
    "deepseek": 64_000,     # deepseek-chat V3
    "mistral": 128_000,     # mistral-large-latest
    "cohere": 128_000,      # command-r-plus
    "moonshot": 131_072,    # kimi-k2
    "zhipu": 128_000,       # glm-4.7
    "cloudflare": 24_000,   # workers-ai llama-3.3-70b -- small window, treat carefully
    "nvidia": 128_000,      # NIM llama-3.3-70b
    "novita": 64_000,       # deepseek-r1 (reasoning models reserve output room)
    "deepinfra": 64_000,
    "siliconflow": 64_000,  # DeepSeek-V3 hosting tier
    "openrouter": 128_000,  # llama-3.3-70b :free route
    "github": 128_000,      # GitHub Models gpt-4.1-mini class window
}
_DEFAULT_CONTEXT_LIMIT = 128_000

# scode: provider strengths profile -- the "tailor everything per provider"
# map. Deterministic facts only (no LLM calls to use it): what each free
# tier is actually GOOD at, how fast it answers, and which orchestration
# roles it may be picked for. brain._orchestrate consults this when it
# needs an extra/replacement worker beyond the core TIERS pool: candidates
# come from here, filtered by key-present + breaker-closed, ordered by
# strength match then cap headroom. Roles are conservative on purpose --
# a provider earns "judge"/"research" only if its default model genuinely
# suits that job; unknown providers get no roles rather than optimistic ones.
PROFILES = {
    "groq":        {"latency": "ultra-fast", "strength": "generalist",
                    "roles": ["classify", "simple", "worker"]},
    "gemini":      {"latency": "fast", "strength": "long-context research",
                    "roles": ["research", "judge", "worker", "orchestrator"]},
    "cerebras":    {"latency": "ultra-fast", "strength": "reasoning",
                    "roles": ["worker"], "caveat": "8k context -- short subtasks only"},
    "deepseek":    {"latency": "medium", "strength": "deep reasoning + code",
                    "roles": ["worker", "judge"]},
    "mistral":     {"latency": "fast", "strength": "generalist EU, strong instruction following",
                    "roles": ["worker"]},
    "cohere":      {"latency": "fast", "strength": "RAG / grounded synthesis",
                    "roles": ["worker", "judge"]},
    "moonshot":    {"latency": "medium", "strength": "long-context agentic",
                    "roles": ["worker", "research"]},
    "zhipu":       {"latency": "fast", "strength": "generalist + tool use",
                    "roles": ["worker"]},
    "cloudflare":  {"latency": "fast", "strength": "edge fallback generalist",
                    "roles": ["worker"], "caveat": "24k context -- trim history hard"},
    "nvidia":      {"latency": "fast", "strength": "solid llama generalist",
                    "roles": ["worker"]},
    "novita":      {"latency": "slow", "strength": "deepseek-r1 chain-of-thought",
                    "roles": ["worker"], "caveat": "reasoning model -- slow, burns output tokens"},
    "deepinfra":   {"latency": "fast", "strength": "cheap reliable llama",
                    "roles": ["worker"]},
    "siliconflow": {"latency": "fast", "strength": "DeepSeek-V3 generalist",
                    "roles": ["worker"]},
    "openrouter":  {"latency": "variable", "strength": "free llama fallback route",
                    "roles": ["worker"], "caveat": ":free routes rate-limited unpredictably"},
    "github":      {"latency": "fast", "strength": "gpt-4.1-mini class generalist",
                    "roles": ["worker"]},
}

# scode: free-tier RATE LIMITS -- Part 10's self-awareness map, same
# verified-against-provider-docs discipline as CONTEXT_LIMITS/PROFILES.
# Shape per provider:
#   rpm  -- requests/minute ceiling on the default model (None = unknown)
#   tpm  -- tokens/minute ceiling where the tier documents one
#   daily -- requests/day ceiling (most free tiers express this instead
#           of RPM; None = unlimited/not documented)
# These are PROVIDER-level planning numbers for the default models in
# MODELS, NOT hard gates -- check_limit() and the caps config remain the
# enforcement path (fail-open invariant untouched). Consumers:
#   - /api/usage/summary surfaces them next to live burn data
#   - extended_pool() uses `daily` to skip providers projected to run
#     out before they'd be useful (deterministic, no LLM calls)
# Sources checked Aug 2026 at write time: groq docs rate-limits page,
# google ai studio pricing page, cerebras inference docs, deepseek
# platform docs, mistral console tiers, cohere docs, moonshot platform,
# zai open platform, cloudflare workers-ai limits page, nvidia NIM docs,
# novita/deepinfra/siliconflow dashboards, openrouter :free route notes,
# github models docs. Numbers drift -- update when a provider changes
# its tier, don't guess.
RATE_LIMITS = {
    "groq":        {"rpm": 30,    "tpm": None,   "daily": 14_400},
    "gemini":      {"rpm": 15,    "tpm": 250_000, "daily": None},
    "cerebras":    {"rpm": 30,    "tpm": 60_000,  "daily": None},
    "deepseek":    {"rpm": None,  "tpm": None,    "daily": None},
    "mistral":     {"rpm": 1,     "tpm": None,    "daily": None},
    "cohere":      {"rpm": 20,    "tpm": 40_000,  "daily": 1_000},
    "moonshot":    {"rpm": 3,     "tpm": 32_000,  "daily": None},
    "zhipu":       {"rpm": 5,     "tpm": None,    "daily": None},
    "cloudflare":  {"rpm": 300,   "tpm": None,    "daily": 10_000},   # neurons/day budget
    "nvidia":      {"rpm": 40,    "tpm": None,    "daily": None},
    "novita":      {"rpm": None,  "tpm": None,    "daily": None},
    "deepinfra":   {"rpm": None,  "tpm": None,    "daily": None},
    "siliconflow": {"rpm": None,  "tpm": None,    "daily": None},
    "openrouter":  {"rpm": 20,    "tpm": None,    "daily": 50},       # :free route is stingy per-day
    "github":      {"rpm": 15,    "tpm": None,    "daily": 150},      # models tier low-rate
}


def key_audit() -> dict[str, bool]:
    """One deterministic pass over os.environ: which configured LLM
    providers currently have their key present. No network calls. Used by
    /api/usage/summary and diagnostics so the panel can show red/green per
    provider instead of discovering a missing key via a failed call."""
    return {p: bool(os.environ.get(env)) for p, env in PROVIDER_API_KEY_ENV.items()}


def _provider_healthy(provider: str) -> bool:
    """Key present AND circuit not open -- the two deterministic gates a
    provider must pass before brain considers it as extended-pool worker.
    Cap state is NOT checked here (caps are soft, fail-open); callers that
    care about headroom order by it separately."""
    if not key_audit().get(provider):
        return False
    c = _CIRCUITS.get(provider)
    return bool(c is None or c.get("state") in ("closed", "half_open"))


def _cap_headroom(provider: str) -> int:
    """Remaining daily-cap calls for a provider, or a large sentinel if
    uncapped -- used ONLY to order candidates, never to gate them (caps
    stay soft/fail-open; the pre-network cap check still owns rejection)."""
    big = 10**9
    try:
        caps = config.get_config("caps") or {}
        cap = caps.get(provider)
        if cap is None:
            return big
        usage = config.get_config("usage") or {}
        used = usage.get(provider, 0) if usage.get("date") == datetime.date.today().isoformat() else 0
        return max(0, cap - used)
    except Exception:
        return big  # ordering is best-effort; never let it break selection


_EXHAUST_FRACTION = 0.8   # >=80% of the documented daily cap already burned
_EXHAUST_WINDOW_S = 600   # ...and projected to hit 0 within ~10 minutes


def _projected_exhaustion(provider: str) -> bool:
    """Part 10 deterministic skip: True only when the provider has BOTH a
    documented daily rate limit (RATE_LIMITS.daily), >=80% of it consumed,
    AND a live burn rate projecting exhaustion inside the window. Any
    missing input (no entry, no usage yet, no rate samples) -> False --
    fail-open like every other cap signal here. Pure math on existing
    counters; never raises."""
    try:
        rl = (RATE_LIMITS.get(provider) or {}).get("daily")
        if not rl:
            return False
        import config as _config
        usage = _config.get_config("usage") or {}
        used = usage.get(provider, 0) if usage.get("date") == datetime.date.today().isoformat() else 0
        if used < rl * _EXHAUST_FRACTION:
            return False
        import observability as _obs
        rate = _obs.burn_rate(provider)  # calls/sec EMA
        if rate <= 0:
            return False
        remaining_calls = rl - used
        projected_s = remaining_calls / rate
        return projected_s <= _EXHAUST_WINDOW_S
    except Exception:
        return False


def extended_pool(need: str | None = None) -> list[str]:
    """Ordered candidate list for gap-round / replan worker picks from the
    FULL provider set (not just core-3): healthy providers whose PROFILES
    roles match `need` first, then other healthy ones; each group sorted by
    remaining daily-cap headroom (most room first, uncapped on top).
    Deterministic within equal headroom = MODELS declaration order. Empty
    list means even the extended pool has nobody usable -> caller falls
    back to its existing orchestrator-fallback path.

    Part 10: providers with a known RATE_LIMITS daily cap that are BOTH
    >=80% consumed AND burning fast enough to hit 0 within ~10 minutes are
    skipped entirely (projected exhaustion) -- deterministic math on live
    counters, no LLM calls, no behavior change for capped-but-calm or
    uncapped providers."""
    matched, rest = [], []
    for p in MODELS:
        if not _provider_healthy(p) or _projected_exhaustion(p):
            continue
        roles = PROFILES.get(p, {}).get("roles", [])
        if need and need in roles:
            matched.append(p)
        else:
            rest.append(p)
    stable = lambda seq: [p for p in MODELS if p in set(seq)]  # restore declaration order
    matched.sort(key=_cap_headroom, reverse=True)
    rest.sort(key=_cap_headroom, reverse=True)
    return stable(matched) + stable(rest)
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
    """Daily cap policy lives here (llm.py's domain); the actual atomic
    read-modify-write mechanics live in config.atomic_update, which
    also means this now costs zero extra Supabase round-trips in the
    common case (cached) instead of the 2 reads + 1 write this used to
    do on every single call_llm invocation. Adapted from a second Claude
    session's own OmniRoute-derived fix (its enforce.ts documents this
    explicitly as design principle B16), verified sound before porting.

    Fail-open: if the cap check ITSELF can't run -- Supabase is down,
    the network hiccups, anything infra-side -- this now lets the call
    through instead of raising. Before this fix, a transient Supabase
    outage would silently block EVERY LLM call across EVERY provider,
    even ones with plenty of quota room, because config.get_config()/
    atomic_update() raising wasn't caught here at all. A cap is a soft
    spend guardrail Ruk set for himself, not a safety-critical gate --
    failing closed on infra trouble is strictly worse than the rare
    case of slightly overrunning a self-imposed cap during an outage."""
    try:
        caps = config.get_config("caps") or {}
        cap = caps.get(provider)
        if cap is None:
            return  # no cap set for this provider

        def _bump(usage):
            usage = usage or {}
            if usage.get("date") != _today():
                usage = {"date": _today()}  # new day -> counts reset automatically
            used = usage.get(provider, 0)
            if used >= cap:
                raise CapExceeded(f"{provider} hit its cap of {cap} calls today")
            usage[provider] = used + 1
            return usage

        config.atomic_update("usage", _bump)
    except CapExceeded:
        raise  # a real, intentional cap block -- not an infra failure, must propagate
    except Exception as e:
        log(f"[_check_and_bump_cap] cap check itself failed for {provider} (infra issue, not a real cap) -- failing OPEN: {e!r}")
        return


# --- circuit breaker, adapted from OmniRoute's real adaptiveCircuit.ts +
# failureClassification.ts (github.com/diegosouzapw/OmniRoute) -- studied
# their actual source, not just the README, before writing this. Plain
# in-memory state is correct here (not Supabase): Sandy runs single-
# process (confirmed: no --workers in entrypoint.sh, no multi-instance
# to keep in sync), and "closed" is the exact same safe default a
# container restart already resets to, matching OmniRoute's own
# createAdaptiveCircuit(). This is short-lived operational memory, not
# a record Ruk needs to review later the way healing_ledger is.
_CIRCUITS: dict[str, dict] = {}


def _classify_failure(exc: Exception) -> tuple[str, bool]:
    """Real failure type + whether retrying soon could plausibly help.
    Adapted from OmniRoute's classifyProviderFailure() -- an auth error
    or an exhausted daily cap won't fix itself by retrying in 60
    seconds the way a timeout, rate limit, or 5xx often does."""
    if isinstance(exc, CapExceeded):
        return "quota_exhausted", False
    msg = str(exc).lower()
    if "401" in msg or "unauthorized" in msg or ("invalid" in msg and "key" in msg):
        return "authentication_error", False
    if "403" in msg or "permission" in msg or "forbidden" in msg:
        return "permission_error", False
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return "rate_limit", True
    if "timeout" in msg or "timed out" in msg:
        return "timeout", True
    if any(code in msg for code in ("500", "502", "503", "504")):
        return "provider_5xx", True
    return "unknown", True


def _circuit_allows(provider: str) -> bool:
    """False means: skip this provider right now, its circuit is open."""
    c = _CIRCUITS.get(provider)
    if not c or c["state"] == "closed":
        return True
    if c["state"] == "open":
        if time.time() >= c["next_probe_at"]:
            c["state"] = "half_open"  # exactly one probe call allowed through
            return True
        return False
    return True  # half_open: the one probe call is already in flight


def _circuit_observe(provider: str, success: bool, retryable: bool = True) -> None:
    """Real circuit-breaker state machine: closed -> open on repeated
    (or one non-retryable) failure, half_open probes after a cooldown,
    back to closed on the first real success. Mirrors OmniRoute's
    observeCircuit() shape exactly."""
    c = _CIRCUITS.setdefault(provider, {"state": "closed", "failure_count": 0})
    if success:
        c["state"] = "closed"
        c["failure_count"] = 0
        c.pop("next_probe_at", None)
        return
    c["failure_count"] += 1
    # Non-retryable (bad key, cap already hit today) opens immediately
    # with a long cooldown -- retrying in a minute won't fix either one.
    # Retryable (timeout/rate-limit/5xx) uses a real 3-strike threshold
    # with a short cooldown, since those often self-resolve fast.
    threshold = 1 if not retryable else 3
    cooldown = 600 if not retryable else 60
    if c["state"] == "half_open" or c["failure_count"] >= threshold:
        c["state"] = "open"
        c["next_probe_at"] = time.time() + cooldown


def call_llm(provider: str, messages: list[dict], caller: str = "unknown", **kwargs) -> str:
    """provider: 'groq' | 'gemini' | 'cerebras'. messages: standard
    OpenAI-style [{role, content}] list. caller: short tag for WHO is
    asking (e.g. "brain.simple", "classify_message", "codebase.analyze")
    -- this is what makes observability.today_summary() able to say
    which code path spent the tokens, not just which provider. Always
    pass a real tag at call sites; "unknown" is a fallback for stray
    direct callers, not something to leave as-is."""
    if provider not in MODELS:
        raise ValueError(f"unknown provider: {provider}")
    _check_and_bump_cap(provider)
    messages = fit_to_budget(messages, provider)
    env_name = _API_KEY_ENV.get(provider)
    if env_name:
        kwargs.setdefault("api_key", os.environ.get(env_name))
    if provider == "nvidia":
        kwargs.setdefault("api_base", _NVIDIA_NIM_BASE)
    kwargs.setdefault("timeout", LLM_TIMEOUT_S)  # never hang forever on a dead provider
    response = completion(model=MODELS[provider], messages=messages, **kwargs)
    text = response.choices[0].message.content
    try:
        observability.record_call(provider, caller, messages, text)
    except Exception as e:  # observability must never be able to take down a real LLM call
        log(f"observability.record_call failed (non-fatal): {e}")
    return text


def call_llm_with_fallback(provider: str, messages: list[dict], caller: str = "unknown", **kwargs) -> str:
    """Same as call_llm, but if `provider` is capped OR its API call fails
    for any reason (outage, rate limit, timeout), tries the other
    providers in turn instead of failing the whole request. Use this for
    single-provider call sites (_judge, orchestrator) that have no other
    provider to fall back on already — _run_tier's multi-provider loop
    doesn't need this, it already gathers from several providers.

    scode: real gap this closes, learned from studying OmniRoute's real
    adaptiveCircuit.ts + failureClassification.ts (an open-source AI
    gateway Ruk pointed me at) -- this used to have NO memory of a
    provider's recent failures. If gemini's key was bad or its daily cap
    was already hit, EVERY single call still wasted a full network round
    trip hitting gemini first before falling through to groq/cerebras --
    on the orchestrator's ~8 calls per round, that's up to 8 wasted
    round trips to a provider already known broken for this whole
    window. A real circuit breaker (closed -> open -> half_open ->
    closed, same shape as OmniRoute's) now remembers this and skips
    straight past a provider that's currently down, only probing it
    again after a cooldown."""
    # ponytail: fallback cascade stays limited to the 3 battle-tested core
    # providers, not all 14 -- classify_message/mastery-extraction call this
    # on every message, so a broken new provider shouldn't add 10+ slow
    # failing attempts before falling through. New providers are reachable
    # via explicit llm_override or the orchestrator, not blind cascade.
    _core = ("groq", "gemini", "cerebras")
    order = [provider] + [p for p in _core if p != provider]
    last_err = None
    attempted = False
    for p in order:
        if not _circuit_allows(p):
            continue
        attempted = True
        try:
            result = call_llm(p, messages, caller=caller, **kwargs)
            _circuit_observe(p, success=True)
            return result
        except Exception as e:
            _, retryable = _classify_failure(e)
            _circuit_observe(p, success=False, retryable=retryable)
            last_err = e
    if not attempted:
        # scode: real edge case, must not skip silently -- every core
        # provider's circuit happened to be open at the same moment (all
        # recently failed). `last_err` would still be None here, and
        # `raise None` is a confusing crash unrelated to the real
        # problem. A real error from one real attempt beats a fake one,
        # so force exactly one real call through instead of giving up.
        return call_llm(provider, messages, caller=caller, **kwargs)
    raise last_err


if __name__ == "__main__":
    # scode self-checks: circuit breaker state machine, pure logic, no
    # network needed for these -- verifies the exact behavior the
    # call_llm_with_fallback fix above depends on.
    _CIRCUITS.clear()
    assert _circuit_allows("gemini"), "a provider with no history must be allowed"

    # 3 retryable failures should open the circuit; the 4th call is skipped
    for _ in range(3):
        _circuit_observe("gemini", success=False, retryable=True)
    assert not _circuit_allows("gemini"), "circuit should be OPEN after 3 retryable failures"
    print("llm.py: circuit opens after 3 retryable failures -> OK")

    # one non-retryable failure (bad key) must open it immediately, not
    # wait for 3 strikes
    _CIRCUITS.clear()
    _circuit_observe("gemini", success=False, retryable=False)
    assert not _circuit_allows("gemini"), "one non-retryable failure must open the circuit immediately"
    print("llm.py: circuit opens on a single non-retryable failure -> OK")

    # after the cooldown elapses, exactly one probe call is let through
    _CIRCUITS["gemini"]["next_probe_at"] = time.time() - 1  # simulate cooldown already passed
    assert _circuit_allows("gemini"), "circuit should allow exactly one probe after cooldown"
    assert _CIRCUITS["gemini"]["state"] == "half_open"
    print("llm.py: circuit half-opens for one probe after cooldown -> OK")

    # a real success on that probe must close the circuit again
    _circuit_observe("gemini", success=True)
    assert _circuit_allows("gemini") and _CIRCUITS["gemini"]["state"] == "closed"
    print("llm.py: circuit closes again after a successful probe -> OK")

    # classification: CapExceeded and an auth-style message must be
    # flagged non-retryable; a timeout must be flagged retryable
    assert _classify_failure(CapExceeded("groq hit its cap")) == ("quota_exhausted", False)
    assert _classify_failure(Exception("401 Unauthorized: invalid api key"))[1] is False
    assert _classify_failure(Exception("Request timed out"))[1] is True
    print("llm.py: failure classification (retryable vs not) -> OK")

    # real call to Groq (fastest/free), confirms the model string + key
    # actually work end to end.
    reply = call_llm("groq", [{"role": "user", "content": "reply with exactly: pong"}])
    assert "pong" in reply.lower(), f"unexpected reply: {reply}"
    print("llm.py: groq call OK ->", reply)
