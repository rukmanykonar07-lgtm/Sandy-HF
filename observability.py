"""Per-call LLM observability -- the missing piece that made "credit
burn" impossible to actually diagnose from code-reading alone.

Before this file, the cap system counted CALLS per provider per day.
Most providers' real quotas (definitely Gemini's free tier) are
token/request-based, not call-count-based -- so two calls could look
identical to the old system ("1 call to gemini") while one sends 200
tokens and the other sends 55,000 (codebase.analyze() dumping the whole
repo, before that was fixed too). The call-count cap was structurally
blind to the thing actually burning quota.

This module doesn't replace the cap system (config.py still owns "is
this provider allowed to run at all right now") -- it sits alongside
it as the thing that answers "where did today's tokens actually go,
broken down by which code path asked for them", which is the question
that's been impossible to answer without reading every file by hand.

ponytail: structured logging + a simple in-process rollup, not a real
metrics stack (Prometheus/Datadog) -- Sandy is a single-process
personal assistant, not a fleet. Revisit if that ever changes.
"""
import datetime
import logging
import threading
import time

import config

log = logging.getLogger("sandy.cost")

_lock = threading.Lock()
# {date_iso: {provider: {"calls": int, "tokens_in": int, "tokens_out": int,
#                         "callers": {caller_tag: {"calls": int, "tokens_in": int, "tokens_out": int}}}}}
_DAILY: dict = {}

# --- persistence into Supabase `sandy_usage_daily` (plan Part 4) ---------
# The in-memory rollup dies with the process -- and HF Spaces sleeps/restarts
# constantly. So: a single flusher thread owns ALL writes (scenario #6: no
# two threads ever race an upsert), flushing at most once per 30s and only
# when something changed. Worst case a crash loses <=30s of rollups --
# acceptable for a dashboard feed. Boot does the reverse: load_today()
# rehydrates today's rows so a mid-day restart doesn't reset the counters.
#
# One-time setup (run once in Supabase SQL editor):
#
#     create table sandy_usage_daily (
#         date text not null,
#         provider text not null,
#         caller text not null,
#         calls int not null default 0,
#         tokens_in bigint not null default 0,
#         tokens_out bigint not null default 0,
#         updated_at timestamptz default now(),
#         primary key (date, provider, caller)
#     );
_FLUSH_INTERVAL = 30          # seconds between flush attempts
_dirty = threading.Event()
_flusher_started = False


def _flush_loop():
    while True:
        time.sleep(_FLUSH_INTERVAL)
        if _dirty.is_set():
            result = flush()
            if result.get("ok"):
                _dirty.clear()


def start_flusher() -> None:
    """Idempotent boot hook -- called once from main's lifespan."""
    global _flusher_started
    if _flusher_started:
        return
    _flusher_started = True
    threading.Thread(target=_flush_loop, daemon=True, name="usage-flusher").start()


def flush() -> dict:
    """Upsert today's rollup into sandy_usage_daily. One row per
    (date, provider, caller); totals are rebuilt by summing callers.
    Fail-open like every Supabase touch: an outage logs and retries
    next tick, never blocks the chat path."""
    try:
        client = config.get_client()
        date = _today()
        with _lock:
            day = _DAILY.get(date, {})
            rows = []
            for provider, p in day.items():
                for caller, c in p["callers"].items():
                    rows.append({"date": date, "provider": provider, "caller": caller,
                                 "calls": c["calls"], "tokens_in": c["tokens_in"],
                                 "tokens_out": c["tokens_out"]})
        if not rows:
            return {"ok": True, "rows": 0}
        client.table("sandy_usage_daily").upsert(rows).execute()
        return {"ok": True, "rows": len(rows)}
    except Exception as e:
        log.warning("usage flush failed (will retry next tick): %r", e)
        return {"ok": False, "error": str(e)}


def load_today() -> int:
    """Boot rehydration: pull today's persisted rows back into _DAILY so
    a Space wake-up doesn't zero the morning's burn. Returns row count;
    any failure leaves _DAILY empty but raises nothing."""
    try:
        client = config.get_client()
        date = _today()
        res = client.table("sandy_usage_daily").select("*").eq("date", date).execute()
        rows = res.data or []
        with _lock:
            day = _DAILY.setdefault(date, {})
            for r in rows:
                p = day.setdefault(r["provider"],
                                   {"calls": 0, "tokens_in": 0, "tokens_out": 0, "callers": {}})
                c = p["callers"].setdefault(r["caller"],
                                            {"calls": 0, "tokens_in": 0, "tokens_out": 0})
                c["calls"] += r["calls"]
                c["tokens_in"] += r["tokens_in"]
                c["tokens_out"] += r["tokens_out"]
                # rebuild the provider totals from the caller rows we just loaded
                p["calls"] += r["calls"]
                p["tokens_in"] += r["tokens_in"]
                p["tokens_out"] += r["tokens_out"]
        return len(rows)
    except Exception as e:
        log.warning("usage load_today failed (starting empty): %r", e)
        return 0

# --- burn-rate tracking (added, ported from OmniRoute's burnRate.ts EMA
# algorithm -- studied its src/lib/quota/burnRate.ts directly, not
# guessed. Same idea, translated: OmniRoute tracks $/token burn across a
# multi-tenant pool; Sandy only has one user and one process, so this is
# a much smaller in-memory version, not a port of its Redis-backed
# store) -----------------------------------------------------------------
_EMA_ALPHA = 0.3
_MAX_SAMPLES = 30  # per provider -- a rolling window is enough for "how fast right now", not a permanent history
_samples: dict[str, list[tuple[float, int]]] = {}  # provider -> [(epoch_seconds, cumulative_calls_today), ...]


def _record_sample(provider: str) -> None:
    with _lock:
        today = _today()
        calls_today = _DAILY.get(today, {}).get(provider, {}).get("calls", 0)
        s = _samples.setdefault(provider, [])
        s.append((datetime.datetime.now().timestamp(), calls_today))
        if len(s) > _MAX_SAMPLES:
            del s[: len(s) - _MAX_SAMPLES]


def burn_rate(provider: str) -> float:
    """Calls/second, as an EMA over recent deltas -- same alpha=0.3
    OmniRoute uses. Needs at least 2 samples; returns 0.0 until then
    (a fresh day/provider genuinely has no rate yet, not a bug)."""
    with _lock:
        history = list(_samples.get(provider, []))
    if len(history) < 2:
        return 0.0
    ema = None
    for i in range(1, len(history)):
        dt = history[i][0] - history[i - 1][0]
        if dt <= 0:
            continue
        rate = (history[i][1] - history[i - 1][1]) / dt
        ema = rate if ema is None else _EMA_ALPHA * rate + (1 - _EMA_ALPHA) * ema
    return max(0.0, ema or 0.0)


def time_to_cap_exhaustion(provider: str) -> float | None:
    """Seconds until `provider` hits its daily cap at the CURRENT burn
    rate, or None if there's no cap, no rate yet, or it's already
    exhausted (0 -- "now", not None, so a caller can tell "exhausted"
    apart from "no data"). This is what lets Sandy answer "at this
    rate, when does gemini run out" instead of Ruk finding out only
    once a call actually fails."""
    caps = config.get_config("caps") or {}
    cap = caps.get(provider)
    if cap is None:
        return None
    usage = config.get_config("usage") or {}
    if usage.get("date") != _today():
        used = 0
    else:
        used = usage.get(provider, 0)
    remaining = max(0, cap - used)
    if remaining == 0:
        return 0.0
    rate = burn_rate(provider)
    if rate <= 0:
        return None
    return remaining / rate


_WARN_THRESHOLD = 0.8  # 80% of cap -- same default OmniRoute's budgetGuard.ts uses


def cap_status(provider: str) -> dict:
    """3-state read on a provider's cap -- "ok" / "warn" / "exhausted" --
    instead of the old binary allowed/CapExceeded. Ported from
    OmniRoute's evaluateBudget() (src/lib/usage/budgetGuard.ts): the
    real value here is the WARN band -- Sandy's cap system used to only
    ever find out about a capped provider the moment a call actually
    failed. Now `main.py` can proactively mention "gemini's at 85% for
    today" before that happens. No cap set -> always "ok" (uncapped is
    not the same as healthy, but there's nothing to warn about)."""
    caps = config.get_config("caps") or {}
    cap = caps.get(provider)
    if cap is None:
        return {"state": "ok", "used": None, "cap": None, "fraction": None}
    usage = config.get_config("usage") or {}
    used = usage.get(provider, 0) if usage.get("date") == _today() else 0
    fraction = (used / cap) if cap else 0.0
    if used >= cap:
        state = "exhausted"
    elif fraction >= _WARN_THRESHOLD:
        state = "warn"
    else:
        state = "ok"
    return {"state": state, "used": used, "cap": cap, "fraction": round(fraction, 3)}


def _today() -> str:
    return datetime.date.today().isoformat()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4  # same rough heuristic llm.py already used elsewhere -- kept consistent, not a second guess


def record_call(provider: str, caller: str, messages_in: list[dict], response_text: str) -> dict:
    """Call this right after every real LLM call. `caller` is a short
    tag identifying WHICH code path made the call (e.g.
    "classify_message", "brain.simple", "brain.orchestrate.research",
    "codebase.analyze") -- this is the part that was missing everywhere
    else: knowing a call happened to gemini tells you nothing about
    whether it was a 300-token classify or a 55,000-token repo dump.

    Returns the per-call record (also logged + rolled into the daily
    total) so callers that want to react to a single expensive call
    (e.g. warn Ruk inline) can do so without a second lookup.
    """
    tokens_in = sum(_estimate_tokens(m.get("content", "")) for m in messages_in)
    tokens_out = _estimate_tokens(response_text)
    record = {
        "provider": provider,
        "caller": caller,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
    }

    log.info(
        "llm_call provider=%s caller=%s tokens_in=%d tokens_out=%d",
        provider, caller, tokens_in, tokens_out,
    )

    today = _today()
    with _lock:
        day = _DAILY.setdefault(today, {})
        p = day.setdefault(provider, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "callers": {}})
        p["calls"] += 1
        p["tokens_in"] += tokens_in
        p["tokens_out"] += tokens_out
        c = p["callers"].setdefault(caller, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        c["calls"] += 1
        c["tokens_in"] += tokens_in
        c["tokens_out"] += tokens_out
        # keep only today + yesterday in memory -- this is a live dashboard
        # feed, not a permanent ledger; long-term history should read the
        # structured log lines above (or a real table) if that's ever needed
        if len(_DAILY) > 2:
            for old_key in sorted(_DAILY)[:-2]:
                del _DAILY[old_key]

    _record_sample(provider)
    _dirty.set()  # wake the flusher -- it owns ALL sandy_usage_daily writes
    return record


def today_summary(provider: str | None = None) -> dict:
    """What /status (or Sandy answering 'where did today's credits go')
    should read. Without a provider filter: totals per provider +
    top callers overall. With one: that provider's caller breakdown,
    sorted by tokens spent, most expensive first."""
    with _lock:
        day = _DAILY.get(_today(), {})
        if provider:
            p = day.get(provider, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "callers": {}})
            callers = sorted(p["callers"].items(), key=lambda kv: -kv[1]["tokens_in"] - kv[1]["tokens_out"])
            return {"provider": provider, "calls": p["calls"], "tokens_in": p["tokens_in"],
                    "tokens_out": p["tokens_out"], "top_callers": callers[:10]}
        return {
            provider: {"calls": p["calls"], "tokens_in": p["tokens_in"], "tokens_out": p["tokens_out"]}
            for provider, p in day.items()
        }
