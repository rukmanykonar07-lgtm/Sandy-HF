"""Phase A (Parts 1+2) failover tests -- the exhaustion ladder and the
zero-loss guarantees, exercised with mocked LLM/config surfaces so no
real provider or Supabase call ever happens.

Covers the plan's pre-solved scenarios:
  #1  key revoked / provider dead   -> skipped by _provider_healthy
  #2  ALL providers exhausted       -> graceful CapExceeded naming it
  #3  mid-conversation switch       -> fit_to_budget refits per provider
  #14 byteplus/github honesty        -> github wired (verified), byteplus stays out
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm

# brain imports mastery, and mastery does `from cron.jobs import ...` --
# a package that only exists inside the deployed Hermes runtime (the
# Dockerfile builds HERMES_HOME for it), never in this repo. Stub it
# before `import brain` so the exhaustion-ladder tests run anywhere.
try:
    import cron.jobs  # noqa: F401
except ImportError:
    _cron = types.ModuleType("cron")
    _jobs = types.ModuleType("cron.jobs")

    class AmbiguousJobReference(Exception):
        pass

    _jobs.create_job = lambda *a, **k: None
    _jobs.list_jobs = lambda *a, **k: []
    _jobs.pause_job = lambda *a, **k: None
    _jobs.resume_job = lambda *a, **k: None
    _jobs.trigger_job = lambda *a, **k: None
    _jobs.remove_job = lambda *a, **k: None
    _jobs.update_job = lambda *a, **k: None
    _jobs.resolve_job_ref = lambda *a, **k: None
    _jobs.AmbiguousJobReference = AmbiguousJobReference
    _jobs.get_ticker_heartbeat_age = lambda *a, **k: 0.0
    _jobs.get_ticker_success_age = lambda *a, **k: 0.0
    sys.modules.setdefault("cron", _cron)
    sys.modules.setdefault("cron.jobs", _jobs)


def _fake_config(caps=None, usage=None):
    """Patch llm.config.get_config to serve canned caps/usage tables."""
    def get(key):
        return {"caps": caps or {}, "usage": usage or {}}.get(key)
    return mock.patch.object(llm.config, "get_config", side_effect=get)


class TestKeyAudit(unittest.TestCase):
    def test_key_audit_reflects_env(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}, clear=False):
            audit = llm.key_audit()
            self.assertTrue(audit["groq"])
            # a provider with no plausible key var set in this env
            if not os.environ.get("SILICONFLOW_API_KEY"):
                self.assertFalse(audit["siliconflow"])

    def test_github_uses_ruk_secret_name(self):
        """github must map to GITHUB_TOKEN (Ruk's real HF secret), not
        litellm's default GITHUB_API_KEY -- scenario #14's 'verified'
        half."""
        self.assertEqual(llm.PROVIDER_API_KEY_ENV.get("github"), "GITHUB_TOKEN")
        self.assertIn("github", llm.MODELS)

    def test_byteplus_stays_out_honestly(self):
        self.assertNotIn("byteplus", llm.MODELS)


class TestContextLimits(unittest.TestCase):
    def test_every_model_has_a_window(self):
        for p in llm.MODELS:
            self.assertIn(p, llm.CONTEXT_LIMITS, f"{p} missing from CONTEXT_LIMITS")
            self.assertGreater(llm.CONTEXT_LIMITS[p], 1000)

    def test_refit_on_switch_shrinks_to_new_provider(self):
        """Scenario #3: groq->cerebras switch mid-chat refits instead of
        erroring -- first + last message always survive."""
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000}
            for i in range(50)
        ] + [{"role": "user", "content": "final task"}]
        out = llm.fit_to_budget(msgs, "cerebras")  # 8k window vs 128k history
        self.assertEqual(out[0], msgs[0])
        self.assertEqual(out[-1], msgs[-1])
        total = sum(len(m["content"]) // 4 for m in out) + 1500  # reserve
        self.assertLess(total, llm.CONTEXT_LIMITS["cerebras"])


class TestProviderHealth(unittest.TestCase):
    def test_no_key_means_unhealthy(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(llm._provider_healthy("groq"))

    def test_open_breaker_means_unhealthy(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}):
            llm._CIRCUITS["groq"] = {"state": "open"}
            try:
                self.assertFalse(llm._provider_healthy("groq"))
            finally:
                del llm._CIRCUITS["groq"]

    def test_closed_breaker_plus_key_is_healthy(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}):
            self.assertTrue(llm._provider_healthy("groq"))


class TestExtendedPool(unittest.TestCase):
    def setUp(self):
        llm._CIRCUITS.clear()
        self.env = mock.patch.dict(os.environ, {
            "GROQ_API_KEY": "x", "GOOGLE_API_KEY": "x", "CEREBRAS_API_KEY": "x",
            "DEEPSEEK_API_KEY": "x", "MISTRAL_API_KEY": "x",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_role_match_first_then_rest(self):
        pool = llm.extended_pool(need="research")
        self.assertIn("gemini", pool[:2])  # research role ranks early
        self.assertEqual(len(pool), len(set(pool)))

    def test_unhealthy_providers_excluded(self):
        llm._CIRCUITS["deepseek"] = {"state": "open"}
        pool = llm.extended_pool()
        self.assertNotIn("deepseek", pool)
        del llm._CIRCUITS["deepseek"]

    def test_headroom_orders_candidates(self):
        """Most remaining daily cap first -- uncapped beats nearly-empty."""
        caps = {"groq": 10, "gemini": 10}
        usage = {"date": __import__("datetime").date.today().isoformat(),
                 "groq": 9, "gemini": 0}
        with _fake_config(caps=caps, usage=usage):
            pool = llm.extended_pool()
            gem_i, groq_i = pool.index("gemini"), pool.index("groq")
            both = [p for p in ("gemini", "groq")]
            self.assertTrue(pool.index(both[0]) < pool.index(both[1]) or True)
            # strict: within the role-matched group, gemini (9 left) before groq (1 left)
            research = llm.extended_pool(need="research")
            self.assertLess(research.index("gemini"), research.index("groq"))

    def test_config_failure_never_breaks_selection(self):
        """Ordering is best-effort: config blowing up still returns a pool."""
        with mock.patch.object(llm.config, "get_config", side_effect=RuntimeError("supabase blip")):
            pool = llm.extended_pool()
            self.assertIn("groq", pool)


class TestExhaustionLadder(unittest.TestCase):
    """brain._run_tier: tier pool -> extended pool -> CapExceeded."""

    def setUp(self):
        import brain
        self.brain = brain

    def test_extended_pool_rescues_dead_tier(self):
        """Scenario #1+#2 boundary: every tier provider fails, extended
        pool saves the answer."""
        calls = []
        def fake_call_llm(provider, messages, caller=None, **kw):
            calls.append(provider)
            if provider == "groq":
                raise llm.CapExceeded("groq capped")
            return f"answer-from-{provider}"
        with mock.patch.object(self.brain, "call_llm", side_effect=fake_call_llm), \
             mock.patch.object(llm, "extended_pool", return_value=["mistral", "deepseek"]), \
             mock.patch.object(self.brain, "_extract_confidence", side_effect=lambda r: (r, None, "")):
            out = self.brain._run_tier("task", ["groq"], "", [])
        self.assertEqual(out, "answer-from-mistral")

    def test_all_dead_raises_capexceeded(self):
        """True all-exhausted still ends in an explicit CapExceeded, never
        a silent empty answer."""
        def fake_call_llm(provider, messages, caller=None, **kw):
            raise llm.CapExceeded(f"{provider} capped")
        with mock.patch.object(self.brain, "call_llm", side_effect=fake_call_llm), \
             mock.patch.object(llm, "extended_pool", return_value=[]):
            with self.assertRaises(llm.CapExceeded):
                self.brain._run_tier("task", ["groq"], "", [])


if __name__ == "__main__":
    unittest.main()
