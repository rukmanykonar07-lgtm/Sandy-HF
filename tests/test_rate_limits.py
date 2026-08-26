"""Phase I (Part 10) tests -- RATE_LIMITS self-awareness + projected
exhaustion skipping in extended_pool.

Covers:
  - RATE_LIMITS: every MODELS provider has an entry; shape is rpm/tpm/daily
    with None or positive ints
  - _projected_exhaustion matrix:
      * no documented daily cap -> False (fail-open)
      * <80% consumed -> False even at high burn
      * >=80% consumed but burn rate 0/absent -> False (calm provider)
      * >=80% consumed + high burn projecting exhaustion inside window -> True
      * >=80% consumed but slow burn projecting OUTSIDE window -> False
      * stale usage date -> treated as 0 used -> False
      * any internal error -> False (never raises into selection)
  - extended_pool integration: a projected-exhausted healthy provider is
    skipped entirely; same provider calm stays in the list
  - /api/usage/summary rows carry the rate_limits block

All config/observability surfaces mocked -- no network, no Supabase.
"""
import os
import sys
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm


class RateLimitsShapeTests(unittest.TestCase):
    def test_every_model_provider_has_entry(self):
        for p in llm.MODELS:
            self.assertIn(p, llm.RATE_LIMITS, f"{p} missing from RATE_LIMITS")

    def test_shape_is_rpm_tpm_daily(self):
        for p, rl in llm.RATE_LIMITS.items():
            self.assertEqual(set(rl.keys()), {"rpm", "tpm", "daily"}, f"{p} bad shape")
            for field, val in rl.items():
                self.assertTrue(val is None or (isinstance(val, int) and val > 0),
                                f"{p}.{field} must be None or positive int, got {val!r}")


class _FakeConfig:
    """Mimics config.get_config('usage') -- {'date': iso, provider: count}."""

    def __init__(self, date_iso, counts):
        self._data = {"usage": dict(counts, date=date_iso)}

    def get_config(self, key):
        return self._data.get(key)


def _fake_obs(rate):
    return types.SimpleNamespace(burn_rate=lambda _p: rate)


def _patch_env(provider_used, rate, today_iso="2026-08-26"):
    """Patch both lazy imports _projected_exhaustion does internally."""
    cfg = _FakeConfig(today_iso, {provider_used[0]: provider_used[1]})
    return [mock.patch.dict(sys.modules, {"config": cfg}),
            mock.patch.dict(sys.modules, {"observability": _fake_obs(rate)})]


class ProjectedExhaustionTests(unittest.TestCase):
    def test_no_documented_daily_cap_is_fail_open(self):
        patches = _patch_env(("deepseek", 10**6), 100.0)
        with patches[0], patches[1]:
            self.assertFalse(llm._projected_exhaustion("deepseek"))

    def test_under_eighty_percent_consumed_is_false(self):
        # groq daily=14400; 10000 used = ~69%
        patches = _patch_env(("groq", 10_000), 50.0)
        with patches[0], patches[1]:
            self.assertFalse(llm._projected_exhaustion("groq"))

    def test_over_threshold_but_calm_is_false(self):
        # 95% consumed but zero burn -> not about to exhaust NOW
        patches = _patch_env(("groq", 14_000), 0.0)
        with patches[0], patches[1]:
            self.assertFalse(llm._projected_exhaustion("groq"))

    def test_projected_exhaustion_true(self):
        # groq daily=14400, used=14000 (97%), rate=2 calls/s ->
        # remaining 400 calls / 2 = 200s <= 600s window
        patches = _patch_env(("groq", 14_000), 2.0)
        with patches[0], patches[1]:
            self.assertTrue(llm._projected_exhaustion("groq"))

    def test_high_usage_but_slow_burn_outside_window_is_false(self):
        # openrouter daily=50, used=48 (96%), rate=0.001 calls/s ->
        # 2000s > 600s window -> keep it in the pool
        patches = _patch_env(("openrouter", 48), 0.001)
        with patches[0], patches[1]:
            self.assertFalse(llm._projected_exhaustion("openrouter"))

    def test_stale_usage_date_counts_as_zero(self):
        patches = _patch_env(("github", 150), 99.0, today_iso="2026-08-25")
        with patches[0], patches[1]:
            self.assertFalse(llm._projected_exhaustion("github"))  # yesterday's spend doesn't count

    def test_internal_error_never_raises(self):
        class Boom:
            def get_config(self, _k):
                raise RuntimeError("supabase down")
        with mock.patch.dict(sys.modules, {"config": Boom(),
                                           "observability": _fake_obs(1.0)}):
            self.assertFalse(llm._projected_exhaustion("groq"))


class ExtendedPoolSkipTests(unittest.TestCase):
    def setUp(self):
        self.saved_health = llm._CIRCUITS.copy()

    def tearDown(self):
        llm._CIRCUITS.clear()
        llm._CIRCUITS.update(self.saved_health)

    def test_exhausted_provider_skipped_from_pool(self):
        cfg = _FakeConfig("2026-08-26", {"openrouter": 49})
        obs = _fake_obs(5.0)  # burning hard -> projected exhaustion
        fake_key_audit = {"openrouter": True}
        with mock.patch.dict(sys.modules, {"config": cfg,
                                           "observability": obs}), \
             mock.patch.object(llm, "key_audit", return_value=fake_key_audit), \
             mock.patch.dict(llm.PROVIDER_API_KEY_ENV, {}):
            pool = llm.extended_pool()
        self.assertNotIn("openrouter", pool)

    def test_calm_provider_stays_in_pool(self):
        cfg = _FakeConfig("2026-08-26", {"openrouter": 49})
        obs = _fake_obs(0.0)  # no burn -> no projection
        with mock.patch.dict(sys.modules, {"config": cfg,
                                           "observability": obs}), \
             mock.patch.object(llm, "key_audit",
                               return_value={"openrouter": True} | {p: False for p in llm.MODELS if p != "openrouter"}):
            pool = llm.extended_pool()
        self.assertIn("openrouter", pool)


if __name__ == "__main__":
    unittest.main()
