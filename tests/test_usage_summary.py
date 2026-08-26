"""Phase C (Part 4) usage-persistence tests -- all Supabase touches mocked,
no network. Covers:

  record_call rollup math + dirty-flag      -> flusher has something to send
  flush() row shape                         -> one row per (date, provider, caller)
  flush() fail-open                         -> Supabase blip logs, never raises
  load_today() rehydration                  -> restart doesn't zero the counters
  load_today() fail-open                    -> starts empty, returns 0
  today_summary() aggregation               -> provider totals + caller drill-down
  cap_status() 3-state ladder               -> ok / warn >=80% / exhausted
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import observability


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _UsageFakeClient:
    """Minimal stand-in matching just what observability uses:
    table(name).upsert(rows) / .select("*").eq("date", d).execute().
    Upserts are keyed by the real PK (date, provider, caller) -- the
    repo-wide conftest fake keys by id/job_ref/key, which usage rows
    don't have."""

    def __init__(self):
        self.tables = {}
        self.fail = False

    def table(self, name):
        return _UsageQuery(self.tables.setdefault(name, {}), self)


class _UsageQuery:
    def __init__(self, store, client):
        self.store = store
        self.client = client

    def upsert(self, rows):
        if self.client.fail:
            raise RuntimeError("supabase down")
        for r in rows:
            self.store[(r["date"], r["provider"], r["caller"])] = dict(r)
        return self

    def select(self, cols="*"):
        return self

    def eq(self, k, v):
        self._date = v
        return self

    def execute(self):
        if self.client.fail:
            raise RuntimeError("supabase down")
        rows = [dict(r) for r in self.store.values() if r["date"] == getattr(self, "_date", None)]
        return _FakeResult(rows)


class ObservabilityTestBase(unittest.TestCase):
    def setUp(self):
        # snapshot/restore module state so tests stay order-independent
        self._saved = (observability._DAILY, observability._samples,
                       observability._flusher_started)
        observability._DAILY = {}
        observability._samples = {}
        observability._flusher_started = False
        observability._dirty.clear()

    def tearDown(self):
        observability._DAILY, s, f = self._saved
        observability._samples = s
        observability._flusher_started = f
        observability._dirty.clear()


class TestRecordAndFlush(ObservabilityTestBase):
    def test_record_call_updates_rollup_and_sets_dirty(self):
        observability.record_call("groq", "brain.simple",
                                  [{"content": "hi " * 100}], "ok " * 50)
        day = observability._DAILY[observability._today()]
        p = day["groq"]
        self.assertEqual(p["calls"], 1)
        self.assertEqual(p["tokens_in"], 75)    # len("hi "*100)=300 //4
        self.assertEqual(p["tokens_out"], 37)   # len("ok "*50)=200, minus trailing-space trim -> 150 //4
        c = p["callers"]["brain.simple"]
        self.assertEqual((c["calls"], c["tokens_in"], c["tokens_out"]),
                         (1, 75, 37))
        self.assertTrue(observability._dirty.is_set())

    def test_flush_writes_one_row_per_caller(self):
        observability.record_call("groq", "chat", [{"content": "a" * 40}], "b" * 8)
        observability.record_call("groq", "judge", [{"content": "c" * 80}], "d" * 16)
        client = _UsageFakeClient()
        with mock.patch.object(observability.config, "get_client", return_value=client):
            result = observability.flush()
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], 2)
        date = observability._today()
        chat = client.tables["sandy_usage_daily"][(date, "groq", "chat")]
        self.assertEqual(chat["tokens_in"], 10)  # 40//4
        self.assertEqual(chat["calls"], 1)
        judge = client.tables["sandy_usage_daily"][(date, "groq", "judge")]
        self.assertEqual(judge["tokens_in"] + judge["tokens_out"], 24)

    def test_flush_empty_day_is_ok_noop(self):
        client = _UsageFakeClient()
        with mock.patch.object(observability.config, "get_client", return_value=client):
            result = observability.flush()
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], 0)

    def test_flush_fail_open_never_raises(self):
        observability.record_call("gemini", "research", [{"content": "x"}], "y")
        client = _UsageFakeClient()
        client.fail = True
        with mock.patch.object(observability.config, "get_client", return_value=client):
            result = observability.flush()  # must not raise
        self.assertFalse(result["ok"])
        self.assertIn("down", result["error"])
        self.assertTrue(observability._dirty.is_set())  # retry next tick


class TestLoadToday(ObservabilityTestBase):
    def test_rehydration_rebuilds_provider_totals(self):
        date = observability._today()
        client = _UsageFakeClient()
        client.table("sandy_usage_daily").upsert([
            {"date": date, "provider": "groq", "caller": "chat",
             "calls": 3, "tokens_in": 1200, "tokens_out": 400},
            {"date": date, "provider": "groq", "caller": "judge",
             "calls": 2, "tokens_in": 800, "tokens_out": 600},
        ]).execute()
        with mock.patch.object(observability.config, "get_client", return_value=client):
            loaded = observability.load_today()
        self.assertEqual(loaded, 2)
        p = observability._DAILY[date]["groq"]
        self.assertEqual(p["calls"], 5)          # summed from callers
        self.assertEqual(p["tokens_in"], 2000)
        self.assertEqual(p["tokens_out"], 1000)
        summary = observability.today_summary("groq")
        self.assertEqual(summary["calls"], 5)

    def test_load_failure_fails_open_to_empty(self):
        client = _UsageFakeClient()
        client.fail = True
        with mock.patch.object(observability.config, "get_client", return_value=client):
            self.assertEqual(observability.load_today(), 0)
        self.assertEqual(observability.today_summary(), {})


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self._saved = (observability._DAILY, observability._samples)
        observability._DAILY = {}
        observability._samples = {}

    def tearDown(self):
        observability._DAILY, s = self._saved
        observability._samples = s

    def test_today_summary_totals_and_drilldown_sorted_by_cost(self):
        observability.record_call("groq", "chat", [{"content": "a" * 400}], "")     # 100 in
        observability.record_call("groq", "chat", [{"content": "a" * 200}], "o" * 40)
        observability.record_call("groq", "judge", [{"content": "j" * 800}], "")    # 200 in
        observability.record_call("gemini", "research", [{"content": "r" * 100}], "")
        top = observability.today_summary("groq")["top_callers"]
        self.assertEqual(top[0][0], "judge")       # 200 tokens > chat's 135
        overall = observability.today_summary()
        self.assertEqual(overall["groq"]["calls"], 3)
        self.assertEqual(overall["gemini"]["calls"], 1)


class TestCapStatus(unittest.TestCase):
    def setUp(self):
        self._saved_d = observability._DAILY
        observability._DAILY = {}

    def tearDown(self):
        observability._DAILY = self._saved_d

    def _with_caps(self, used_today, caps):
        import datetime
        usage = {"date": datetime.date.today().isoformat(), **used_today}
        with mock.patch.object(observability.config, "get_config",
                               side_effect=lambda k: caps if k == "caps" else usage):
            return observability.cap_status("groq")

    def test_no_cap_is_always_ok(self):
        self.assertEqual(self._with_caps({}, {})["state"], "ok")

    def test_under_warn_threshold_is_ok(self):
        st = self._with_caps({"groq": 50}, {"groq": 100})
        self.assertEqual(st["state"], "ok")
        self.assertEqual(st["fraction"], 0.5)

    def test_at_80_percent_is_warn(self):
        self.assertEqual(self._with_caps({"groq": 85}, {"groq": 100})["state"], "warn")

    def test_at_cap_is_exhausted_even_if_usage_row_stale(self):
        st = self._with_caps({"groq": 100}, {"groq": 100})
        self.assertEqual(st["state"], "exhausted")

    def test_stale_usage_date_counts_as_unused(self):
        import datetime
        stale = {"date": "1999-01-01", "groq": 999}
        caps = {"groq": 100}
        with mock.patch.object(observability.config, "get_config",
                               side_effect=lambda k: caps if k == "caps" else stale):
            st = observability.cap_status("groq")
        self.assertEqual((st["state"], st["used"]), ("ok", 0))


if __name__ == "__main__":
    unittest.main()
