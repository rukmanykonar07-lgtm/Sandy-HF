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

from llm import call_llm, call_llm_with_fallback, CapExceeded
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


def _run_tier(task: str, providers: list[str], context: str) -> str:
    messages = [_IDENTITY_MSG, {"role": "user", "content": f"{context}\n\nTask: {task}" if context else task}]
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
                print(f"[brain._run_tier] provider '{p}' failed, skipping: {e!r}")
                last_error = e
                continue
    if not answers:
        raise CapExceeded(str(last_error) if last_error else "all providers for this tier are capped")
    if len(answers) == 1:
        return next(iter(answers.values()))
    return _judge(task, answers)


def _orchestrate(task: str, context: str) -> str:
    """Best-fit LLM plans subtasks, delegates, loops until satisfied."""
    orchestrator = "gemini"  # highest quality per Ruk's stack ranking
    workers = ["groq", "cerebras"]
    plan_prompt = (
        f"{context}\n\nTask: {task}\n\n"
        "Break this into 2-4 concrete sub-tasks that, done well, complete the task. "
        'Return JSON only: {"subtasks": ["...", "..."]}'
    )
    plan_raw = call_llm_with_fallback(orchestrator, [{"role": "user", "content": plan_prompt}])
    try:
        subtasks = json.loads(plan_raw)["subtasks"]
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        subtasks = [task]  # ponytail: bad/malformed plan -> fall back to treating it as one task

    results = []
    for i, sub in enumerate(subtasks):
        worker = workers[i % len(workers)]
        try:
            results.append(call_llm(worker, [_IDENTITY_MSG, {"role": "user", "content": sub}]))
        except CapExceeded:
            results.append(call_llm_with_fallback(orchestrator, [_IDENTITY_MSG, {"role": "user", "content": sub}]))
        except Exception as e:
            print(f"[brain._orchestrate] worker '{worker}' failed, falling back to orchestrator: {e!r}")
            results.append(call_llm_with_fallback(orchestrator, [_IDENTITY_MSG, {"role": "user", "content": sub}]))

    for _round in range(MAX_ORCHESTRATOR_ROUNDS):
        synth_prompt = (
            f"Original task: {task}\n\nSub-task results:\n"
            + "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
            + "\n\nIs this enough to fully answer the original task? "
            'Reply JSON: {"done": true, "answer": "..."} or {"done": false, "missing": "..."}'
        )
        verdict_raw = call_llm_with_fallback(orchestrator, [{"role": "user", "content": synth_prompt}])
        try:
            verdict = json.loads(verdict_raw)
        except json.JSONDecodeError:
            return verdict_raw  # ponytail: orchestrator didn't return JSON -> just use its text as the answer
        if not isinstance(verdict, dict):
            return verdict_raw
        if verdict.get("done"):
            return verdict.get("answer") or results[-1]  # malformed but done -> best-effort fallback
        gap_answer = call_llm_with_fallback(orchestrator, [_IDENTITY_MSG, {"role": "user", "content": verdict.get("missing", task)}])
        results.append(gap_answer)

    return results[-1]  # ran out of rounds -> best-effort last result


def answer(task: str, context: str = "", override: list[str] | None = None) -> str:
    """override: explicit provider list from Ruk (e.g. ["gemini"]), or
    ["orchestrator"] to force orchestrator mode. None = auto-classify."""
    if override == ["orchestrator"]:
        return _orchestrate(task, context)
    if override:
        return _run_tier(task, override, context)

    tier = classify_complexity(task)
    if tier == "very_complex":
        return _orchestrate(task, context)
    return _run_tier(task, TIERS[tier], context)


if __name__ == "__main__":
    tier = classify_complexity("what's 2+2")
    assert tier == "simple", f"expected simple, got {tier}"
    print("brain.py: classify OK ->", tier)
