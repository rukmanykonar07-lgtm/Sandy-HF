"""healing.py: error classification, fix proposals, and the real
healing_ledger CRUD. Covers a real bug found via actual testing (not
assumption) this session: list_pending_fixes() was originally indexed
off state only the poller touched, so a failure alerted directly by
native_mastery.py's in-process handler was written but permanently
undiscoverable. Fixed by making alert_and_store the single source of
truth for its own index.
"""
import pytest

import healing


def test_classify_error_recognizes_the_real_drift_error():
    text = (
        "RuntimeError: Skipped to prevent unintended spend: global inference "
        "config drifted since this job was created (provider 'openrouter' -> "
        "'gemini'), and this job is unpinned."
    )
    diag = healing.classify_error(text, entity="test-job")
    assert diag["fix_kind"] == "pin_provider"


def test_classify_error_recognizes_the_real_deprecated_model_error():
    text = "This model models/gemini-2.5-flash is no longer available to new users."
    diag = healing.classify_error(text, entity="test-job")
    assert diag["fix_kind"] == "model_fallback"


def test_classify_error_recognizes_hermes_internal_bug_vs_real_error():
    """The httpx ResponseNotRead traceback is a bug inside Hermes's OWN
    error-summarization code, not a real failure of Ruk's job -- must be
    classified distinctly, not lumped in with 'unknown'."""
    text = "ResponseNotRead: Attempted to access streaming response content, without having called read()."
    diag = healing.classify_error(text, entity="test-job")
    assert diag["fix_kind"] == "hermes_internal_bug"


def test_classify_error_never_invents_a_root_cause_for_unrecognized_text():
    diag = healing.classify_error("some totally novel error nobody has seen", entity="test-job")
    assert diag["fix_kind"] == "unknown"
    assert "needs a human look" in diag["root_cause"]


def test_propose_fix_uses_sandys_own_maintained_model_string_not_a_guess():
    """propose_fix must never invent a model name -- it can only ever
    return MODELS[provider], a value already maintained elsewhere in
    the codebase for a real reason."""
    from llm import MODELS

    job = {"id": "job-1", "engine": "hermes", "provider": "gemini", "model": "gemini-2.5-flash"}
    diag = {"fix_kind": "model_fallback", "root_cause": "deprecated"}
    fix = healing.propose_fix(diag, job=job)
    assert fix["updates"]["model"] == MODELS["gemini"].split("/", 1)[-1]


def test_propose_fix_returns_none_for_native_jobs():
    """Native jobs have no single pinned provider/model to re-pin the
    way a Hermes job does -- propose_fix must not fabricate one."""
    job = {"id": "job-1", "engine": "native"}
    diag = {"fix_kind": "model_fallback", "root_cause": "x"}
    assert healing.propose_fix(diag, job=job) is None


def test_alert_and_store_is_discoverable_from_any_caller(fake_db, monkeypatch):
    """Real bug found writing this test: alert_and_store used to be
    reachable via check_for_new_failures() (the poller) OR called
    directly (native_mastery.py's live in-process alert) -- but the
    pending-fix index was only ever updated by the poller's own
    bookkeeping. A directly-alerted failure was written but permanently
    invisible to list_pending_fixes(). Fixed by making alert_and_store
    maintain its own index regardless of caller."""
    monkeypatch.setattr(healing, "send_whatsapp", lambda msg: False)
    job = {"id": "native-job-1", "name": "native: test-skill", "engine": "native"}
    diag = healing.classify_error("some real exception", entity=job["name"])
    healing.alert_and_store([{"job": job, "diag": diag, "fix": None}])

    pending = healing.list_pending_fixes()
    assert len(pending) == 1
    assert pending[0]["job_ref"] == "native-job-1"


def test_announce_then_resolve_lifecycle(fake_db, monkeypatch):
    monkeypatch.setattr(healing, "send_whatsapp", lambda msg: False)
    job = {"id": "job-1", "name": "test-job", "engine": "hermes"}
    diag = {"root_cause": "x", "fix_kind": "pin_provider"}
    fix = {"job_ref": "job-1", "updates": {"provider": "gemini"}}
    healing.alert_and_store([{"job": job, "diag": diag, "fix": fix}])

    assert len(healing.list_unannounced_fixes()) == 1
    assert len(healing.list_announced_pending_fixes()) == 0

    healing.mark_announced("job-1")
    assert len(healing.list_unannounced_fixes()) == 0
    assert len(healing.list_announced_pending_fixes()) == 1

    popped = healing.pop_pending_fix("job-1")
    assert popped["job_ref"] == "job-1"
    assert healing.list_pending_fixes() == []


def test_self_heal_never_targets_an_unannounced_fix(fake_db, monkeypatch):
    """Real edge case caught by deliberate scenario-walking, not by
    accident: a failure detected the SAME turn Ruk happens to send a
    coincidental short affirmative must NOT get auto-applied -- he
    hasn't actually seen it yet. list_announced_pending_fixes() is the
    only list Step 1 (self-heal intercept) is allowed to read from."""
    monkeypatch.setattr(healing, "send_whatsapp", lambda msg: False)
    job = {"id": "job-1", "name": "test-job", "engine": "hermes"}
    diag = {"root_cause": "x", "fix_kind": "pin_provider"}
    fix = {"job_ref": "job-1", "updates": {"provider": "gemini"}}
    healing.alert_and_store([{"job": job, "diag": diag, "fix": fix}])

    # Never announced -- self-heal must see nothing eligible.
    assert healing.list_announced_pending_fixes() == []
