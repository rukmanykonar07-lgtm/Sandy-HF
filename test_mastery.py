"""mastery.py: validating a Hermes job's provider against Hermes's REAL
provider registry -- the actual fix for the live "groq isn't a real
Hermes provider" 401 bug. Tests against the real installed hermes-agent
package's provider list, not a hand-maintained fake one, since a fake
list could silently drift from reality the exact way the original bug
happened in the first place.
"""
import mastery


def test_groq_is_confirmed_not_a_real_hermes_provider():
    """This is the actual root cause of a real production incident:
    Ruk pinned a job to provider='groq' (real for Sandy's OWN litellm
    setup, NOT real for Hermes's separate provider registry), and it
    silently fell into an unrelated fallback-credential path that
    failed with a confusing 401 two turns later."""
    providers = mastery.real_hermes_providers()
    assert "groq" not in providers


def test_gemini_is_a_real_hermes_provider():
    providers = mastery.real_hermes_providers()
    assert "gemini" in providers


def test_edit_mastery_job_refuses_an_invalid_provider_before_touching_hermes():
    """The exact request that broke live, refused up front now instead
    of silently accepted and failing two turns later."""
    result = mastery.edit_mastery_job("some-job-id", {"provider": "groq", "model": "llama-3.3-70b-versatile"})
    assert "real provider nahi hai" in result
    assert "gemini" in result  # a real, valid alternative is offered, not just a refusal


def test_edit_mastery_job_lists_actual_registry_options_not_a_stale_hardcoded_list():
    """The offered alternatives must come from the SAME real registry
    check, not a separately hand-maintained string that could drift."""
    result = mastery.edit_mastery_job("some-job-id", {"provider": "not-a-real-provider-xyz", "model": "x"})
    real_providers = mastery.real_hermes_providers()
    for p in real_providers - {"custom"}:
        assert p in result
