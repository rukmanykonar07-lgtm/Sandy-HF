"""
Decides HOW to answer a task:
  simple       -> 1 LLM
  medium       -> 2 LLMs, judged
  complex      -> 3 LLMs, judged
  very_complex -> orchestrator mode (loop engineering)

Ruk can always override with plain language ("use only gemini",
"start orchestrator mode") — that's parsed in main.py and passed in
here as `override`, which always wins over auto-classification.
"""
import ast
import concurrent.futures
import json
import re
import time

import mastery
import search
from llm import call_llm, call_llm_with_fallback, CapExceeded, MODELS, strip_fence, strip_json_fence, log
from identity import SANDY_SYSTEM_PROMPT

_IDENTITY_MSG = {"role": "system", "content": SANDY_SYSTEM_PROMPT}

_HISTORY_FRAME = {
    "role": "system",
    "content": (
        "The messages below (if any) are RECENT CONVERSATION HISTORY, for "
        "background only. The actual question to answer is the LAST "
        "message, sent just now. Do not confuse an old topic in this "
        "history with the current question -- if unsure what's being "
        "asked, ask Ruk to clarify rather than guessing or answering "
        "something from earlier in the history."
    ),
}


HISTORY_WINDOW = 10  # last N messages (~5 user/assistant turns), not the raw 30
# main.py fetches. scode: real, measured problem -- 30 raw messages were
# getting attached to EVERY call in a tier (2-3x for medium/complex, and
# for _orchestrate specifically -- plan call + every worker + every gap-
# round worker, sometimes 8-15+ real calls for one user turn), each one
# repeating the identical block. Confirmed by current (2026) production
# guidance too: "the orchestrator accumulates context from every worker
# -- at 4+ workers this frequently exceeds window limits." 5 turns is
# enough for real immediate context (a follow-up referencing 2-3
# messages back still works); anything older than that is what Mem0
# (long-term recall) already exists to handle -- keeping a huge window
# "just in case" would just reintroduce the exact bloat this fixes.
# Scoped ONLY to what gets sent to the LLM -- main.py's own
# chatlog.get_history(limit=30) for the /history UI endpoint is
# untouched, Ruk still sees his full real chat log either way.


def _with_history(history: list[dict] | None) -> list[dict]:
    """Wraps history with clear framing so it can't get mistaken for the
    current question. Empty list if there's no history to frame. Only
    the last HISTORY_WINDOW messages -- see the module-level comment
    above for the real, measured reason."""
    if not history:
        return []
    trimmed = history[-HISTORY_WINDOW:]
    return [_HISTORY_FRAME] + trimmed

TIERS = {
    "simple": ["groq"],
    "medium": ["groq", "gemini"],
    "complex": ["groq", "gemini", "cerebras"],
}
MAX_ORCHESTRATOR_ROUNDS = 3  # ponytail: hard stop so a bad loop can't burn the whole day's cap
MAX_RESEARCH_QUERIES = 5  # cap on Gemini's own multi-angle research pass -- "deep research"
                          # must not mean "silently burn the whole day's quota on one message"
MAX_WORKER_RESEARCH = 1   # each worker gets at most one extra targeted search of its own,
                          # on top of whatever research Gemini already handed it
RESEARCH_CACHE_TTL = 3600  # 1 hour -- long enough for a follow-up question in the same
                           # running session, short enough to never serve stale info. Plain
                           # in-memory dict, not Supabase: this doesn't need to survive a
                           # rebuild, it just needs to save a repeat search minutes apart --
                           # a DB round trip on every single query to maybe save one repeat
                           # occasionally isn't worth the added latency on every call.
_research_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}
_CONFIDENCE_RE = re.compile(r"\n?CONFIDENCE:\s*(\d{1,2})\s*/\s*10\s*\(?([^)\n]*)\)?\s*$", re.IGNORECASE)
_CONFIDENCE_INSTRUCTION = (
    "\n\nEnd your answer with a new final line, exactly: "
    "CONFIDENCE: X/10 (one short reason) -- your own honest rating of how "
    "sure you are this is correct/complete."
)


def classify_complexity(task: str) -> str:
    prompt = (
        "Classify this task's difficulty as exactly one word: "
        "simple, medium, complex, or very_complex.\n"
        f"Task: {task}\nAnswer with one word only."
    )
    result = call_llm_with_fallback("groq", [{"role": "user", "content": prompt}]).strip().lower()
    return result if result in {"simple", "medium", "complex", "very_complex"} else "medium"


def _cached_search(query: str, provider: str | None = None) -> list[dict]:
    """Same search.search(), but skips a repeat network call if the
    exact same (query, provider) was searched within the last
    RESEARCH_CACHE_TTL seconds -- real savings when Ruk asks a follow-up
    close to something already researched."""
    key = (query.lower().strip(), provider or "")
    cached = _research_cache.get(key)
    if cached and time.time() - cached[0] < RESEARCH_CACHE_TTL:
        return cached[1]
    results = search.search(query, provider=provider) if provider else search.search(query)
    _research_cache[key] = (time.time(), results)
    return results


_SKILL_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "build", "make", "create", "add", "fix", "sandy", "this", "that", "it",
}


def _find_relevant_skill(task: str) -> str:
    """Checks Sandy's own mastery jobs (real Hermes cron jobs, not a new
    system) for anything relevant to this task -- if she's already spent
    real time mastering something in this territory, that's a head
    start worth using instead of researching from zero again, same
    instinct as the original vision's 'second time is faster because
    she already learned it.' Keyword overlap only, no LLM call -- this
    is a cheap pre-check, not another classifier. Best-effort: any
    failure here just means no head start, never blocks research."""
    try:
        jobs = mastery.list_mastery_jobs()
    except Exception as e:
        log(f"[brain._find_relevant_skill] job list failed, skipping: {e!r}")
        return ""
    task_words = set(re.findall(r"\w+", task.lower())) - _SKILL_STOPWORDS
    best, best_overlap = None, 0
    for job in jobs:
        job_text = (job.get("name", "") + " " + job.get("prompt", "")).lower()
        job_words = set(re.findall(r"\w+", job_text)) - _SKILL_STOPWORDS
        overlap = len(task_words & job_words)
        if overlap > best_overlap:
            best, best_overlap = job, overlap
    if best and best_overlap >= 2:  # require real overlap, not one common word matching
        return (
            f"Sandy already has a mastery skill in progress/completed: "
            f"'{best.get('name', 'unnamed')}' (state: {best.get('state', 'unknown')}). "
            "Treat this as a head start -- don't re-research territory she's already covered."
        )
    return ""


def _extract_confidence(text: str) -> tuple[str, int | None, str]:
    """Pulls a trailing 'CONFIDENCE: X/10 (reason)' line off an answer,
    returns (clean_answer, confidence_or_None, reason). Confidence is
    optional -- if a provider ignores the instruction and doesn't
    include one, this just returns None, never breaks anything."""
    m = _CONFIDENCE_RE.search(text.strip())
    if not m:
        return text, None, ""
    clean = text[: m.start()].rstrip()
    try:
        conf = max(0, min(10, int(m.group(1))))
    except ValueError:
        return text, None, ""
    return clean, conf, m.group(2).strip()


def _extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)


def _self_check_output(worker: str, sub_task: str, output: str) -> str:
    """Best-effort correctness pass before a worker's output is trusted.
    Python code blocks get a real ast.parse() syntax check -- same
    instinct as selfmod.py's pre-push validation, generalized here to
    anything the orchestrator builds, not just self-edits. Anything else
    gets one review pass asking specifically for bugs/logic errors.
    Never blocks or loses work on failure -- if the check itself breaks,
    the original output comes back unchanged."""
    code_blocks = _extract_code_blocks(output)
    if code_blocks:
        for block in code_blocks:
            try:
                ast.parse(block)
            except SyntaxError as e:
                fix_prompt = (
                    f"This Python code has a syntax error (line {e.lineno}: {e.msg}):\n\n"
                    f"{block}\n\nFix it. Return ONLY the corrected code, no explanation, no fences."
                )
                try:
                    fixed = call_llm_with_fallback(worker, [{"role": "user", "content": fix_prompt}])
                    output = output.replace(block, strip_fence(fixed))
                except Exception as e2:
                    log(f"[brain._self_check_output] fix attempt failed, returning as-is: {e2!r}")
        return output
    # No fenced code found -- one general bug/logic-error review pass.
    try:
        review_prompt = (
            f"Sub-task: {sub_task}\n\nOutput:\n{output}\n\n"
            "Does this have any obvious bugs, logic errors, or mistakes? If "
            "yes, return the corrected version. If no, return it EXACTLY "
            "unchanged. Output ONLY the (possibly corrected) content, no "
            "commentary, no preamble."
        )
        return call_llm_with_fallback(worker, [{"role": "user", "content": review_prompt}])
    except Exception as e:
        log(f"[brain._self_check_output] review failed, returning original: {e!r}")
        return output


def _research_log_summary(research_log: list[dict]) -> str:
    """Turns the raw list of every search attempt this run into a short
    report for the review step: which tool actually worked vs failed,
    and whether the same (or near-same) query got run more than once by
    different sources -- wasted calls, not real extra research."""
    if not research_log:
        return ""
    lines = [f"- \"{r['query']}\" via {r['provider']} ({'ok' if r['ok'] else 'FAILED'}, by {r['source']})" for r in research_log]
    seen: dict[str, list[str]] = {}
    for r in research_log:
        key = r["query"].strip().lower()
        seen.setdefault(key, []).append(r["source"])
    dupes = [f"\"{q}\" searched by both {', '.join(sources)}" for q, sources in seen.items() if len(sources) > 1]
    summary = "\n\nResearch tools used this run:\n" + "\n".join(lines)
    if dupes:
        summary += "\n\nPossible duplicate research (wasted calls, not extra insight): " + "; ".join(dupes)
    return summary


def _judge(task: str, answers: dict[str, str], confidences: dict[str, tuple[int | None, str]] | None = None) -> str:
    """One LLM picks/merges the best answer out of several. If self-rated
    confidence scores came through, the merge is told about them
    explicitly instead of treating every answer as equally trustworthy."""
    joined = "\n\n".join(f"[{name}]: {ans}" for name, ans in answers.items())
    conf_note = ""
    if confidences:
        conf_lines = [
            f"- {name}: {c}/10 ({reason})" if c is not None else f"- {name}: no confidence given"
            for name, (c, reason) in confidences.items()
        ]
        conf_note = "\n\nSelf-rated confidence per model:\n" + "\n".join(conf_lines) + (
            "\n\nWeigh the merge toward higher-confidence answers where they conflict, "
            "but don't ignore a low-confidence answer if it's still clearly correct."
        )
    prompt = (
        f"Task: {task}\n\nHere are answers from different models:\n{joined}{conf_note}\n\n"
        "Write the single best final answer, merging the strongest parts. "
        "Output only the final answer, no commentary."
    )
    return call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])


def _run_tier(task: str, providers: list[str], context: str, history: list[dict] | None = None) -> str:
    task_content = (f"{context}\n\nTask: {task}" if context else task) + _CONFIDENCE_INSTRUCTION
    messages = [_IDENTITY_MSG] + _with_history(history) + [{"role": "user", "content": task_content}]
    answers = {}
    confidences = {}
    last_error = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        future_to_provider = {pool.submit(call_llm, p, messages): p for p in providers}
        for future in concurrent.futures.as_completed(future_to_provider):
            p = future_to_provider[future]
            try:
                raw = future.result()
                clean, conf, reason = _extract_confidence(raw)
                answers[p] = clean
                confidences[p] = (conf, reason)
            except CapExceeded:
                continue  # ponytail: skip a capped provider, don't fail the whole task
            except Exception as e:
                # ponytail: a single broken/misconfigured provider (bad model
                # name, outage, whatever) shouldn't sink the whole task if the
                # others in this tier can still answer -- skip it, keep going.
                log(f"[brain._run_tier] provider '{p}' failed, skipping: {e!r}")
                last_error = e
                continue
    if not answers:
        for p in [m for m in MODELS if m not in providers]:  # last resort: try what this tier never had
            try:
                raw = call_llm(p, messages)
                clean, _, _ = _extract_confidence(raw)
                return clean
            except Exception as e:
                log(f"[brain._run_tier] last-resort provider '{p}' also failed: {e!r}")
                last_error = e
        raise CapExceeded(str(last_error) if last_error else "all providers for this tier are capped")
    if len(answers) == 1:
        return next(iter(answers.values()))
    return _judge(task, answers, confidences)


def _research_queries(topic: str, angle: str = "") -> list[str]:
    """What different angles/approaches/tools/existing solutions should
    be explored for this? Returns up to MAX_RESEARCH_QUERIES short search
    queries. Fails safe to a single direct query if this call itself
    fails or returns something unusable -- research is a quality boost,
    never something that should block orchestration from proceeding."""
    prompt = (
        f"For this task: {topic}\n"
        + (f"Specifically for this part: {angle}\n" if angle else "")
        + f"List up to {MAX_RESEARCH_QUERIES} short, genuinely DIFFERENT web "
        "search queries that would help find the best ways to build/answer "
        "this -- different tools, approaches, existing solutions/repos, "
        "angles. Not near-duplicates of each other. "
        'Reply JSON only: {"queries": ["...", "..."]}'
    )
    try:
        raw = call_llm_with_fallback("gemini", [{"role": "user", "content": prompt}])
        queries = json.loads(strip_json_fence(raw)).get("queries")
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries][:MAX_RESEARCH_QUERIES]
    except Exception as e:
        log(f"[brain._research_queries] failed, falling back to one direct query: {e!r}")
    return [topic]


def _research(topic: str, angle: str = "", log_list: list[dict] | None = None) -> str:
    """Runs up to MAX_RESEARCH_QUERIES real searches across different
    angles on a topic, returns a combined findings block. Deliberately
    rotates across Tavily/Exa/Linkup instead of defaulting every query
    to Tavily-first (search.search()'s own fallback chain) -- Ruk pays
    for all three and they're genuinely different tools (Tavily fast/
    cheap, Exa neural/conceptual, Linkup deep/structured), so actual
    variety across a multi-angle research pass beats always reaching
    for the same one and treating the other two as pure insurance.
    Any individual search failing is just skipped, never raised.
    log_list: if given, every attempt (query, provider, ok/fail) gets
    appended -- lets the review step later check whether a tool was
    actually working and whether work was duplicated, not just trust
    that research silently happened correctly."""
    providers = ["linkup", "exa", "tavily"]  # deep-understanding query first, then conceptual, then fast
    findings = []
    for i, q in enumerate(_research_queries(topic, angle)):
        provider = providers[i % len(providers)]
        try:
            results = _cached_search(q, provider)
            block = "\n".join(f"- {r['title']}: {r['content'][:300]}" for r in results[:3])
            if block:
                findings.append(f"[{q} via {provider}]\n{block}")
            if log_list is not None:
                log_list.append({"query": q, "provider": provider, "source": "gemini-research", "ok": True})
        except Exception as e:
            log(f"[brain._research] search '{q}' via {provider} failed, skipping: {e!r}")
            if log_list is not None:
                log_list.append({"query": q, "provider": provider, "source": "gemini-research", "ok": False})
    return "\n\n".join(findings)


def _orchestrate(
    task: str, context: str, history: list[dict] | None = None, on_event=None,
    workers: list[str] | None = None, extra_workers: list[str] | None = None,
    provider_guard=None, should_continue=None,
) -> str:
    """v3 (built after Ruk's refinement): Gemini researches multiple
    angles FIRST (existing tools, approaches, repos), plans 2-4
    sub-tasks with that research attached, workers each execute (and can
    ask for one extra targeted search of their own on top of what
    Gemini gave them) -- then instead of immediately handing out new
    sub-tasks, Gemini reviews ALL worker output together, specifically
    checking for CONFLICTS between workers (this is where real bugs
    come from in multi-agent builds -- two workers independently
    assuming different things about the same piece, not any one worker
    being wrong), researches again only if something's genuinely
    unresolved, and only then replans holistically for the next round.
    Every step degrades gracefully -- a failure anywhere falls back to
    the best available partial result, never an unhandled crash.

    on_event(event_type, summary, round, provider=None, detail=None) --
    optional hook, called at each real transition (research/planning/
    worker-done/conflict/replan/synthesis). Used by native_mastery.py to
    log real events for the orb graph. Default None -- normal chat use
    (brain.answer) never sets this, so behavior/cost here is completely
    unchanged for every existing caller.

    workers -- optional custom round-robin worker list (e.g. weighted by
    Ruk's per-run provider percentages for a native mastery run). Default
    None keeps the original ["groq", "cerebras"] behavior exactly as
    before -- existing chat callers are unaffected.

    extra_workers -- optional models NOT in the normal rotation that only
    get pulled in once a round needed real replanning (i.e. the task
    turned out genuinely hard) -- native_mastery.py's "add more models if
    the task is hard" behavior. Default None = never used, unchanged for
    every existing caller.

    provider_guard(provider) -> bool -- optional per-call check, used by
    native_mastery.py to enforce Ruk's own per-JOB provider caps (on top
    of, not instead of, llm.py's global daily cap). A provider failing
    this check is skipped for that call, same as a capped/failed provider
    already is. Default None = no extra check, unchanged for every
    existing caller.

    should_continue() -> bool -- optional pause check, polled once per
    replan round (the only real checkpoint this loop has -- there's no
    finer mid-round pause). Returns False -> stop and return the best
    result so far instead of continuing to replan. Default None = never
    stops early, unchanged for every existing caller."""
    def _emit(event_type, summary, round=0, provider=None, detail=None):
        if on_event:
            try:
                on_event(event_type, summary, round, provider, detail)
            except Exception as e:
                log(f"[brain._orchestrate] on_event hook failed, continuing: {e!r}")

    orchestrator = "gemini"
    workers = workers or ["groq", "cerebras"]
    extra_workers = extra_workers or []

    def _call_guarded(worker: str, msgs: list[dict]):
        """call_llm, but skips straight to the orchestrator fallback if
        provider_guard rejects `worker` -- same shape as a real provider
        failure, so callers don't need a separate code path for it."""
        if provider_guard is not None:
            try:
                if not provider_guard(worker):
                    raise CapExceeded(f"{worker}: job-level cap reached")
            except CapExceeded:
                raise
            except Exception as e:
                log(f"[brain._orchestrate] provider_guard errored, treating as OK: {e!r}")
        return call_llm(worker, msgs)
    research_log: list[dict] = []  # every search this run makes: query, provider, source, ok/fail --
                                    # lets the review step check tools are actually working and
                                    # workers aren't quietly duplicating each other's searches

    skill_context = _find_relevant_skill(task)
    research = _research(task, log_list=research_log)
    if skill_context:
        research = f"{skill_context}\n\n{research}" if research else skill_context
    _emit("planning", f"Research done ({len(research_log)} queries) -- planning sub-tasks", provider=orchestrator)
    plan_prompt = (
        (f"{context}\n\n" if context else "")
        + f"Task: {task}\n\nResearch findings:\n{research}\n\n"
        "Using this research, break the task into 2-4 concrete sub-tasks "
        "that, done well, complete it. For each sub-task, include the "
        "specific slice of research it actually needs (not everything). "
        'Return JSON only: {"subtasks": [{"task": "...", "notes": "..."}]}'
    )
    try:
        plan_raw = call_llm_with_fallback(orchestrator, [{"role": "user", "content": plan_prompt}])
        subtasks = json.loads(strip_json_fence(plan_raw))["subtasks"]
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError
        subtasks = [s if isinstance(s, dict) else {"task": str(s), "notes": ""} for s in subtasks]
    except Exception as e:
        log(f"[brain._orchestrate] planning failed, treating as single task: {e!r}")
        subtasks = [{"task": task, "notes": research}]
    _emit("planning", f"{len(subtasks)} sub-task(s) planned", detail=str(subtasks))

    def _worker_own_research(worker: str, sub_task: str, notes: str) -> str:
        """One shot for the worker to ask for ONE more targeted search on
        top of what Gemini already gave it -- filling a gap specific to
        its own piece, not redoing Gemini's broader research. The worker
        also picks which tool actually fits its need (fast fact-check vs
        conceptual vs deep/structured), not just whatever the default
        happens to be -- real agency over the tool, matching how a
        person would actually pick a search engine for the question."""
        try:
            need_prompt = (
                f"Sub-task: {sub_task}\nGiven research: {notes}\n\n"
                "Do you need ONE more specific web search to do this well "
                "(exact syntax, a specific tool's docs, etc)? If yes, also "
                "pick whichever tool actually fits: 'tavily' (fast/quick "
                "facts), 'exa' (conceptual/similar approaches), 'linkup' "
                "(deep/structured). "
                'Reply JSON only: {"query": "...", "provider": "tavily"} or {"query": null}'
            )
            need_raw = call_llm_with_fallback(worker, [{"role": "user", "content": need_prompt}])
            need = json.loads(strip_json_fence(need_raw))
            if need.get("query"):
                provider = need.get("provider") if need.get("provider") in ("tavily", "exa", "linkup") else None
                try:
                    results = _cached_search(need["query"], provider)
                    research_log.append({"query": need["query"], "provider": provider or "default", "source": worker, "ok": True})
                    return "\n".join(f"- {r['title']}: {r['content'][:300]}" for r in results[:3])
                except Exception as e:
                    research_log.append({"query": need["query"], "provider": provider or "default", "source": worker, "ok": False})
                    raise
        except Exception as e:
            log(f"[brain._orchestrate] worker '{worker}' own-research skipped: {e!r}")
        return ""

    def _run_subtask(i_sub):
        i, sub = i_sub
        worker = workers[i % len(workers)]
        sub_task, notes = sub["task"], sub.get("notes", "")
        extra = _worker_own_research(worker, sub_task, notes)
        sub_with_context = (
            (f"{context}\n\n" if context else "")
            + f"Sub-task: {sub_task}\nResearch: {notes}"
            + (f"\nAdditional research: {extra}" if extra else "")
            + _CONFIDENCE_INSTRUCTION
        )
        msgs = [_IDENTITY_MSG] + _with_history(history) + [{"role": "user", "content": sub_with_context}]
        try:
            raw = _call_guarded(worker, msgs)
        except Exception as e:
            log(f"[brain._orchestrate] worker '{worker}' failed, trying orchestrator fallback: {e!r}")
            try:
                raw = call_llm_with_fallback(orchestrator, msgs)
            except Exception as e2:
                log(f"[brain._orchestrate] worker '{worker}' fallback also failed: {e2!r}")
                return f"(this sub-task could not be completed: {sub_task} -- all providers failed)", (None, "")
        clean, conf, reason = _extract_confidence(raw)
        checked = _self_check_output(worker, sub_task, clean)
        _emit("worker_call", f"{worker} finished sub-task: {sub_task[:80]}", provider=worker, detail=checked[:500])
        return checked, (conf, reason)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        subtask_out = list(pool.map(_run_subtask, enumerate(subtasks)))
    results = [r for r, _ in subtask_out]
    confidences = {f"worker_{i}": c for i, (_, c) in enumerate(subtask_out)}

    def _run_gap(sub_text: str, worker: str) -> str:
        msgs = [_IDENTITY_MSG] + _with_history(history) + [{"role": "user", "content": sub_text}]
        try:
            raw = _call_guarded(worker, msgs)
        except Exception as e:
            log(f"[brain._orchestrate] gap worker '{worker}' failed, trying orchestrator fallback: {e!r}")
            try:
                raw = call_llm_with_fallback(orchestrator, msgs)
            except Exception as e2:
                log(f"[brain._orchestrate] gap worker '{worker}' fallback also failed: {e2!r}")
                return f"(gap sub-task could not be completed: {sub_text})"
        return _self_check_output(worker, sub_text, raw)

    for _round in range(MAX_ORCHESTRATOR_ROUNDS):
        if should_continue is not None:
            try:
                if not should_continue():
                    _emit("obstacle", "Paused by Ruk -- stopping here, best result so far kept", round=_round, provider=orchestrator)
                    return results[-1]
            except Exception as e:
                log(f"[brain._orchestrate] should_continue check errored, continuing normally: {e!r}")
        conf_note = ""
        if confidences:
            conf_lines = [
                f"- {name}: {c}/10 ({reason})" if c is not None else f"- {name}: no confidence given"
                for name, (c, reason) in confidences.items()
            ]
            conf_note = "\n\nSelf-rated confidence per worker:\n" + "\n".join(conf_lines)
        review_prompt = (
            (f"{context}\n\n" if context else "")
            + f"Original task: {task}\n\nWorker outputs:\n"
            + "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
            + conf_note
            + _research_log_summary(research_log)
            + "\n\nReview this. Check specifically: (a) is this enough to "
            "fully answer the original task, (b) do any worker outputs "
            "CONFLICT with each other -- different assumptions, mismatched "
            "approaches, inconsistent naming/structure between pieces that "
            "are supposed to fit together. That's the most common real "
            "source of bugs when separate workers build separate pieces. "
            "(c) did any search tool actually fail this run -- if so, "
            "consider whether the missing info matters enough to retry with "
            "a different tool. (d) was research duplicated across workers "
            "-- if so, note it as wasted effort, not a real problem to fix. "
            "Low-confidence outputs deserve extra scrutiny here. "
            'Reply JSON only: {"done": true, "answer": "..."} or '
            '{"done": false, "conflicts": "...", "missing": "...", '
            '"research_query": "..." or null, "research_provider": "tavily" or null}'
        )
        try:
            verdict_raw = call_llm_with_fallback(
                orchestrator, [_IDENTITY_MSG] + _with_history(history) + [{"role": "user", "content": review_prompt}]
            )
        except Exception as e:
            log(f"[brain._orchestrate] review failed, returning raw results: {e!r}")
            return "\n\n".join(results)
        try:
            verdict = json.loads(strip_json_fence(verdict_raw))
        except json.JSONDecodeError:
            return verdict_raw  # orchestrator didn't return JSON -> just use its text
        if not isinstance(verdict, dict):
            return verdict_raw
        if verdict.get("done"):
            _emit("synthesis", "Review found the work complete -- synthesizing final answer", round=_round, provider=orchestrator)
            return verdict.get("answer") or results[-1]
        if verdict.get("conflicts") and verdict["conflicts"].lower() not in ("none", "no", ""):
            _emit("conflict", f"Conflict found between worker outputs: {verdict['conflicts'][:200]}", round=_round, provider=orchestrator, detail=verdict.get("conflicts"))

        # Only research again if the review step actually asked for it --
        # informed by what the workers produced, not a blind re-search.
        # Uses whichever tool Gemini itself picked (research_provider),
        # since by this point it's seen which tools worked/failed above.
        extra_research = ""
        if verdict.get("research_query"):
            provider = verdict.get("research_provider") if verdict.get("research_provider") in ("tavily", "exa", "linkup") else None
            try:
                r = _cached_search(verdict["research_query"], provider)
                extra_research = "\n".join(f"- {x['title']}: {x['content'][:300]}" for x in r[:3])
                research_log.append({"query": verdict["research_query"], "provider": provider or "default", "source": "gemini-review", "ok": True})
            except Exception as e:
                log(f"[brain._orchestrate] round-{_round} research failed, skipping: {e!r}")
                research_log.append({"query": verdict["research_query"], "provider": provider or "default", "source": "gemini-review", "ok": False})

        replan_prompt = (
            f"Original task: {task}\nWorker outputs so far:\n"
            + "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
            + f"\n\nConflicts found: {verdict.get('conflicts', 'none')}"
            + f"\nStill missing: {verdict.get('missing', '')}"
            + (f"\nNew research: {extra_research}" if extra_research else "")
            + "\n\nGive the next concrete step(s) to fix/complete this, "
            'holistically using everything above. JSON only: {"subtasks": ["...", "..."]}'
        )
        try:
            replan_raw = call_llm_with_fallback(orchestrator, [{"role": "user", "content": replan_prompt}])
            next_subtasks = json.loads(strip_json_fence(replan_raw))["subtasks"]
            if not isinstance(next_subtasks, list) or not next_subtasks:
                raise ValueError
        except Exception as e:
            log(f"[brain._orchestrate] replan failed, stopping loop early: {e!r}")
            return results[-1]
        # scode: task needed a genuine replan -- it's harder than the original
        # worker set assumed. Pull in any extra_workers (Ruk's job-scoped
        # "use more models if the task is hard") for THIS gap round only --
        # normal chat callers never pass extra_workers, so this is a no-op
        # for them (gap_workers == workers, unchanged behavior).
        gap_workers = workers + [w for w in extra_workers if w not in workers]
        if extra_workers:
            _emit("planning", f"Task needs more help -- adding {extra_workers} to the rotation this round", round=_round + 1, provider=orchestrator)
        _emit("planning", f"Round {_round+1}: replanned with {len(next_subtasks)} gap sub-task(s)", round=_round + 1, provider=orchestrator)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(next_subtasks)) as pool:
            gap_results = list(
                pool.map(lambda i_s: _run_gap(str(i_s[1]), gap_workers[i_s[0] % len(gap_workers)]), enumerate(next_subtasks))
            )
        results.extend(gap_results)

    _emit("synthesis", "Ran out of rounds -- returning best-effort last result", provider=orchestrator)
    return results[-1]  # ran out of rounds -> best-effort last result


def answer(
    task: str,
    context: str = "",
    override: list[str] | None = None,
    history: list[dict] | None = None,
    tier: str | None = None,
) -> str:
    """override: explicit provider list from Ruk (e.g. ["gemini"]), or
    ["orchestrator"] to force orchestrator mode. None = auto-classify.
    history: recent conversation turns (from chatlog), so every call
    actually has short-term memory, not just long-term Mem0 facts.
    tier: pass this in if the caller already ran classify_complexity()
    for another reason (e.g. picking a search provider) -- skips a
    second, redundant classification call for the same message."""
    if override == ["orchestrator"]:
        return _orchestrate(task, context, history)
    if override:
        return _run_tier(task, override, context, history)

    tier = tier or classify_complexity(task)
    if tier == "very_complex":
        return _orchestrate(task, context, history)
    return _run_tier(task, TIERS[tier], context, history)


if __name__ == "__main__":
    tier = classify_complexity("what's 2+2")
    assert tier == "simple", f"expected simple, got {tier}"
    print("brain.py: classify OK ->", tier)
