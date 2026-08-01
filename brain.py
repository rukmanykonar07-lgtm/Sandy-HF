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
import concurrent.futures
import json

from llm import call_llm, call_llm_with_fallback, CapExceeded, strip_json_fence, log
from identity import SANDY_SYSTEM_PROMPT

_IDENTITY_MSG = {"role": "system", "content": SANDY_SYSTEM_PROMPT}

TIERS = {
    "simple": ["groq"],
    "medium": ["groq", "gemini"],
    "complex": ["groq", "gemini", "cerebras"],
}
MAX_ORCHESTRATOR_ROUNDS = 3  # ponytail: hard stop so a bad loop can't burn the whole day's cap


def classify_complexity(task: str) -> str:
    prompt = (
        "Classify this task's difficulty as exactly one word: "
        "simple, medium, complex, or very_complex.\n"
        f"Task: {task}\nAnswer with one word only."
    )
    result = call_llm("groq", [{"role": "user", "content": prompt}]).strip().lower()
    return result if result in {"simple", "medium", "complex", "very_complex"} else "medium"


def _judge(task: str, answers: dict[str, str]) -> str:
    """One LLM picks/merges the best answer out of several."""
    joined = "\n\n".join(f"[{name}]: {ans}" for name, ans in answers.items())
    prompt = (
        f"Task: {task}\n\nHere are answers from different models:\n{joined}\n\n"
        "Write the single best final answer, merging the strongest parts. "
        "Output only the final answer, no commentary."
    )
    return call_llm_with_fallback("gemini", [_IDENTITY_MSG, {"role": "user", "content": prompt}])


def _run_tier(task: str, providers: list[str], context: str, history: list[dict] | None = None) -> str:
    messages = [_IDENTITY_MSG] + (history or []) + [{"role": "user", "content": f"{context}\n\nTask: {task}" if context else task}]
    answers = {}
    last_error = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        future_to_provider = {pool.submit(call_llm, p, messages): p for p in providers}
        for future in concurrent.futures.as_completed(future_to_provider):
            p = future_to_provider[future]
            try:
                answers[p] = future.result()
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
        raise CapExceeded(str(last_error) if last_error else "all providers for this tier are capped")
    if len(answers) == 1:
        return next(iter(answers.values()))
    return _judge(task, answers)


def _orchestrate(task: str, context: str, history: list[dict] | None = None) -> str:
    """Best-fit LLM plans subtasks, delegates, loops until satisfied.
    Every LLM call here is wrapped so a provider outage/rate-limit can
    degrade gracefully at each step -- this function should never raise
    an unhandled exception, only ever return its best available answer."""
    orchestrator = "gemini"  # highest quality per Ruk's stack ranking
    workers = ["groq", "cerebras"]
    plan_prompt = (
        f"{context}\n\nTask: {task}\n\n"
        "Break this into 2-4 concrete sub-tasks that, done well, complete the task. "
        'Return JSON only: {"subtasks": ["...", "..."]}'
    )
    try:
        plan_raw = call_llm_with_fallback(orchestrator, [{"role": "user", "content": plan_prompt}])
        subtasks = json.loads(strip_json_fence(plan_raw))["subtasks"]
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError
    except Exception as e:
        # ponytail: bad/malformed plan OR every provider failed on the
        # planning call itself -> fall back to treating it as one task,
        # same degradation either way.
        log(f"[brain._orchestrate] planning failed, treating as single task: {e!r}")
        subtasks = [task]

    def _run_subtask(i_sub):
        i, sub = i_sub
        worker = workers[i % len(workers)]
        sub_with_context = f"{context}\n\nSub-task: {sub}" if context else sub
        msgs = [_IDENTITY_MSG] + (history or []) + [{"role": "user", "content": sub_with_context}]
        try:
            return call_llm(worker, msgs)
        except Exception as e:
            log(f"[brain._orchestrate] worker '{worker}' failed, trying orchestrator fallback: {e!r}")
            try:
                return call_llm_with_fallback(orchestrator, msgs)
            except Exception as e2:
                # ponytail: worker AND its fallback both failed (e.g. every
                # provider rate-limited at once) -- don't crash the whole
                # orchestration over one sub-task, return a clear
                # placeholder so the synthesis step can work with what's left.
                log(f"[brain._orchestrate] worker '{worker}' fallback also failed: {e2!r}")
                return f"(this sub-task could not be completed: {sub} -- all providers failed)"

    # Sub-tasks run concurrently -- "orchestration" wasn't actually
    # parallel before despite the name; each worker call is independent
    # so there's no reason to wait for one before starting the next.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        results = list(pool.map(_run_subtask, enumerate(subtasks)))

    for _round in range(MAX_ORCHESTRATOR_ROUNDS):
        synth_prompt = (
            (f"{context}\n\n" if context else "")
            + f"Original task: {task}\n\nSub-task results:\n"
            + "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
            + "\n\nIs this enough to fully answer the original task? "
            'Reply JSON: {"done": true, "answer": "..."} or {"done": false, "missing": "..."}'
        )
        try:
            verdict_raw = call_llm_with_fallback(orchestrator, [_IDENTITY_MSG] + (history or []) + [{"role": "user", "content": synth_prompt}])
        except Exception as e:
            # ponytail: every provider failed on synthesis -- the raw
            # worker results are still real, useful answers even
            # unsynthesized. Better than crashing.
            log(f"[brain._orchestrate] synthesis failed, returning raw results: {e!r}")
            return "\n\n".join(results)
        try:
            verdict = json.loads(strip_json_fence(verdict_raw))
        except json.JSONDecodeError:
            return verdict_raw  # ponytail: orchestrator didn't return JSON -> just use its text as the answer
        if not isinstance(verdict, dict):
            return verdict_raw
        if verdict.get("done"):
            return verdict.get("answer") or results[-1]  # malformed but done -> best-effort fallback
        try:
            gap_answer = call_llm_with_fallback(orchestrator, [_IDENTITY_MSG, {"role": "user", "content": verdict.get("missing", task)}])
        except Exception as e:
            log(f"[brain._orchestrate] gap-fill failed, stopping loop early: {e!r}")
            return results[-1]  # best-effort -- can't fill the gap, return what we have
        results.append(gap_answer)

    return results[-1]  # ran out of rounds -> best-effort last result


def answer(task: str, context: str = "", override: list[str] | None = None, history: list[dict] | None = None) -> str:
    """override: explicit provider list from Ruk (e.g. ["gemini"]), or
    ["orchestrator"] to force orchestrator mode. None = auto-classify.
    history: recent conversation turns (from chatlog), so every call
    actually has short-term memory, not just long-term Mem0 facts."""
    if override == ["orchestrator"]:
        return _orchestrate(task, context, history)
    if override:
        return _run_tier(task, override, context, history)

    tier = classify_complexity(task)
    if tier == "very_complex":
        return _orchestrate(task, context, history)
    return _run_tier(task, TIERS[tier], context, history)


if __name__ == "__main__":
    tier = classify_complexity("what's 2+2")
    assert tier == "simple", f"expected simple, got {tier}"
    print("brain.py: classify OK ->", tier)
