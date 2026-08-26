"""Phase F (Part 6) tests -- orchestration shape + stall watchdog.

Covers the plan's Part 6 contract:
  - orch_barrier_map preserves pool.map's order and results on success
  - a stalled round is released with best-so-far (finished results kept,
    stuck slots get an explicit watchdog note), never hangs forever
  - exactly ONE critical alert fires per release, and it goes through
    notify.alert (never raises into the caller even if notify explodes)
  - the progress heartbeat registry: idle reads 0, start stamps fresh,
    every LLM return inside _orchestrate counts as progress

All LLM/notify/config surfaces are mocked -- no network, no Supabase.
"""
import os
import sys
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# brain imports mastery -> `from cron.jobs import ...`, a package that only
# exists in the deployed Hermes runtime (see tests/test_failover.py).
try:
    import cron.jobs  # noqa: F401
except ImportError:
    _cron = types.ModuleType("cron")
    _jobs = types.ModuleType("cron.jobs")
    for _name in ("create_job", "pause_job", "resume_job", "trigger_job",
                  "remove_job", "update_job", "resolve_job_ref"):
        setattr(_jobs, _name, lambda *a, **k: None)
    _jobs.list_jobs = lambda *a, **k: []

    class AmbiguousJobReference(Exception):
        pass

    _jobs.AmbiguousJobReference = AmbiguousJobReference
    _jobs.get_ticker_heartbeat_age = lambda *a, **k: 0.0
    _jobs.get_ticker_success_age = lambda *a, **k: 0.0
    sys.modules.setdefault("cron", _cron)
    sys.modules.setdefault("cron.jobs", _jobs)

import brain


def _wait_workers():
    """notify.alert spawns fire-and-forget notify-* threads; join them."""
    import threading

    for t in threading.enumerate():
        if t.name.startswith("notify-"):
            t.join(timeout=5)


class StallRegistryTests(unittest.TestCase):
    def setUp(self):
        with brain._orch_lock:
            brain._ORCH.update(task=None, started_at=0.0, last_progress_at=0.0)

    def tearDown(self):
        self.setUp()

    def test_idle_reads_zero(self):
        self.assertEqual(brain.orch_stalled_for(), 0.0)

    def test_start_stamps_fresh_clock(self):
        brain._orch_note_start("task A")
        time.sleep(0.05)
        first = brain.orch_stalled_for()
        brain._orch_note_start("task B")  # crashed predecessor can't poison next run
        self.assertLess(brain.orch_stalled_for(), first)

    def test_heartbeat_resets_clock(self):
        brain._orch_note_start("task")
        time.sleep(0.05)
        self.assertGreater(brain.orch_stalled_for(), 0.04)
        brain._orch_note_progress()
        self.assertLess(brain.orch_stalled_for(), 0.01)  # clock re-stamped

    def test_stall_seconds_env_override(self):
        with mock.patch.dict(os.environ, {"ORCH_STALL_SECONDS": "7"}):
            self.assertEqual(brain._stall_seconds(), 7.0)
        with mock.patch.dict(os.environ, {"ORCH_STALL_SECONDS": "banana"}):
            self.assertEqual(brain._stall_seconds(), 180.0)  # garbage falls back


class BarrierMapSuccessTests(unittest.TestCase):
    def _fresh_run(self):
        brain._orch_note_start("barrier test")

    def tearDown(self):
        with brain._orch_lock:
            brain._ORCH.update(task=None, started_at=0.0, last_progress_at=0.0)

    def test_order_preserved_despite_variable_latency(self):
        self._fresh_run()

        def fn(item):
            i = item[0]
            time.sleep(0.06 if i == 0 else 0.005)  # first item slowest
            return f"r{i}"

        with ThreadPoolExecutor(max_workers=3) as pool:
            out = brain.orch_barrier_map(pool, fn, list(enumerate(range(3))), what="t")
        self.assertEqual(out, ["r0", "r1", "r2"])

    def test_worker_exception_propagates_like_pool_map(self):
        self._fresh_run()

        def fn(item):
            if item[0] == 1:
                raise ValueError("worker blew up")
            return "ok"

        with ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaises(ValueError):
                brain.orch_barrier_map(pool, fn, list(enumerate(range(2))), what="t")


class BarrierStallReleaseTests(unittest.TestCase):
    """Stuck workers block on an Event so the executor's __exit__ join is
    instant once we set it -- keeps each stall test under ~2s wall time."""

    def setUp(self):
        brain._orch_note_start("stall test")
        self._unblock = []

    def tearDown(self):
        for ev in self._unblock:
            ev.set()
        with brain._orch_lock:
            brain._ORCH.update(task=None, started_at=0.0, last_progress_at=0.0)
        _wait_workers()

    def _wedged(self):
        ev = __import__("threading").Event()
        self._unblock.append(ev)

        def fn(_item):
            ev.wait(timeout=30)  # simulates a provider black hole
            return "too-late"

        return fn

    def _drain(self):
        """Un-wedge workers BEFORE the executor's __exit__ join, else shutdown
        blocks on each ev.wait(30) (~30s/test)."""
        for ev in self._unblock:
            ev.set()

    def test_stuck_slots_get_watchdog_note_finished_results_kept(self):
        fn = self._wedged()
        with mock.patch.object(brain, "_stall_seconds", return_value=0.15), \
             mock.patch("notify.alert"):
            with ThreadPoolExecutor(max_workers=3) as pool:
                t0 = time.time()
                out = brain.orch_barrier_map(
                    pool,
                    lambda item: fn(item) if item[0] == 1 else f"done{item[0]}",
                    list(enumerate(range(3))),
                    what="dispatch",
                )
                self._drain()
                elapsed = time.time() - t0  # measure BEFORE __exit__ join
        self.assertEqual(out[0], "done0")
        self.assertEqual(out[2], "done2")
        self.assertIn("cut loose", str(out[1]))
        self.assertLess(elapsed, 3)  # released at ~0.15s+poll, not wedged forever

    def test_one_critical_alert_per_release_via_notify(self):
        fn = self._wedged()
        with mock.patch.object(brain, "_stall_seconds", return_value=0.15), \
             mock.patch("notify.alert"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                out = brain.orch_barrier_map(pool, fn, [(0,)], what="t")
                self._drain()
        self.assertIn("cut loose", str(out[0]))

    def test_alert_receives_critical_severity_and_context(self):
        alerts = []
        fn = self._wedged()
        with mock.patch.object(brain, "_stall_seconds", return_value=0.15), \
             mock.patch("notify.alert", side_effect=lambda t, b, **k: alerts.append((t, k.get("severity")))):
            with ThreadPoolExecutor(max_workers=1) as pool:
                brain.orch_barrier_map(pool, fn, [(0,)], what="gap round")
                self._drain()
        self.assertEqual(len(alerts), 1)
        title, severity = alerts[0]
        self.assertIn("stalled", title.lower())
        self.assertEqual(severity, "critical")

    def test_notify_explosion_never_raises_into_caller(self):
        fn = self._wedged()
        with mock.patch.object(brain, "_stall_seconds", return_value=0.15), \
             mock.patch("notify.alert", side_effect=RuntimeError("alert infra dead")):
            with ThreadPoolExecutor(max_workers=1) as pool:
                out = brain.orch_barrier_map(pool, fn, [(0,)], what="t")
                self._drain()
        self.assertIn("cut loose", str(out[0]))

    def test_on_event_hook_called_with_obstacle(self):
        events = []
        fn = self._wedged()
        with mock.patch.object(brain, "_stall_seconds", return_value=0.15), \
             mock.patch("notify.alert"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                brain.orch_barrier_map(
                    pool, fn, [(0,)], what="t",
                    on_event=lambda etype, summary, rnd, prov, det: events.append(etype),
                )
                self._drain()
        self.assertEqual(events, ["obstacle"])


if __name__ == "__main__":
    unittest.main()
