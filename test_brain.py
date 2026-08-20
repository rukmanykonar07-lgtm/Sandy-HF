"""brain.py: _with_history's window slicing -- the highest-leverage fix
from the context-rot review (confirmed by both our own measured token
math and outside 2026 production guidance: "the orchestrator accumulates
context from every worker -- at 4+ workers this frequently exceeds
window limits"). Every call path (chat, complex-tier cross-checking, the
full orchestrator, and native mastery runs) goes through this one
function, so a bug here would be a bug everywhere at once.
"""
import brain


def _msg(i: int) -> dict:
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}


def test_long_history_is_trimmed_to_the_window():
    history = [_msg(i) for i in range(30)]
    result = brain._with_history(history)
    real_messages = result[1:]  # index 0 is the framing message
    assert len(real_messages) == brain.HISTORY_WINDOW


def test_trimmed_history_keeps_the_most_recent_messages_in_order():
    history = [_msg(i) for i in range(30)]
    result = brain._with_history(history)
    real_messages = result[1:]
    assert real_messages[0]["content"] == f"msg-{30 - brain.HISTORY_WINDOW}"
    assert real_messages[-1]["content"] == "msg-29"


def test_short_history_is_not_altered():
    history = [_msg(i) for i in range(3)]
    result = brain._with_history(history)
    assert len(result) == 4  # framing + all 3, untouched


def test_empty_or_missing_history_returns_empty_no_crash():
    assert brain._with_history(None) == []
    assert brain._with_history([]) == []
