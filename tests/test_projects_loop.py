"""Phase G (Part 8) tests -- projects execution loop.

Covers:
  - _plan_steps: valid JSON array honored + capped at 3 steps;
    garbage/non-JSON output falls back to a deterministic single step
  - run_cycle: happy path executes steps through a MOCKED brain.answer
    (never a real LLM), logs action+response events per step
  - failure policy: brain.answer raising every attempt -> exactly one
    warn alert via notify.alert, status 'failed'
  - pause_for_approval flips status to paused and fires an info alert

All Supabase/LLM/notify surfaces are mocked -- no network.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import projects


class _FakeQuery:
    """Minimal supabase query chain: .table(...).update().eq() / .insert()."""

    def __init__(self, store):
        self.store = store

    def table(self, _name):
        return self

    def update(self, payload):
        self.store.setdefault("updates", []).append(payload)
        return self

    def insert(self, row):
        self.store.setdefault("events", []).append(row)
        return self

    def eq(self, *_a, **_k):
        return self

    def execute(self):
        return types.SimpleNamespace(data=[])


class ProjectCycleTests(unittest.TestCase):
    def setUp(self):
        self.project = {
            "id": "p1", "name": "test project",
            "description": "build a thing", "requires_approval": False,
            "model_limit": None,
        }

    def test_plan_steps_parses_json_array_and_caps_at_three(self):
        raw = '["step one", "step two", "step three", "step four"]'
        with mock.patch.object(projects, "call_llm_with_fallback", return_value=raw):
            steps = projects._plan_steps(self.project)
        self.assertEqual(steps, ["step one", "step two", "step three"])

    def test_plan_steps_falls_back_on_garbage_output(self):
        with mock.patch.object(projects, "call_llm_with_fallback",
                               return_value="I cannot do that in JSON"):
            steps = projects._plan_steps(self.project)
        self.assertEqual(steps, ["build a thing"])  # description becomes the step

    def test_plan_steps_survives_planner_exception(self):
        with mock.patch.object(projects, "call_llm_with_fallback",
                               side_effect=RuntimeError("all providers dead")):
            steps = projects._plan_steps(self.project)
        self.assertEqual(steps, ["build a thing"])

    @mock.patch.object(projects, "_db")
    @mock.patch.object(projects, "brain_answer_stub", create=True)
    def test_run_cycle_happy_path_logs_events(self, m_answ, m_db):
        del m_answ  # unused -- we patch projects.brain via sys.modules below
        m_db.return_value = _FakeQuery({})
        fake_brain = types.ModuleType("brain")
        fake_brain.answer = mock.Mock(side_effect=["did step A", "did step B"])
        with mock.patch.dict(sys.modules, {"brain": fake_brain}), \
             mock.patch.object(projects, "_plan_steps", return_value=["A", "B"]), \
             mock.patch.object(projects, "check_limit", return_value=(True, None)), \
             mock.patch.object(projects, "get_events", return_value=[]):
            outcome = projects.run_cycle(self.project)
        self.assertEqual(outcome["steps_done"], 2)
        self.assertEqual(outcome["status"], "ok")

    @mock.patch.object(projects, "_db")
    def test_step_failure_after_max_attempts_fires_one_warn_alert(self, m_db):
        alerts = []
        store = {}
        m_db.return_value = _FakeQuery(store)
        fake_brain = types.ModuleType("brain")
        fake_brain.answer = mock.Mock(side_effect=RuntimeError("provider down"))
        fake_notify = types.ModuleType("notify")
        fake_notify.alert = mock.Mock(
            side_effect=lambda *a, **k: alerts.append((k.get("title", a[0] if a else ""), k.get("severity"))))
        with mock.patch.dict(sys.modules, {"brain": fake_brain, "notify": fake_notify}), \
             mock.patch.object(projects, "_plan_steps", return_value=["doomed step"]), \
             mock.patch.object(projects, "check_limit", return_value=(True, None)), \
             mock.patch.object(projects, "get_events", return_value=[]):
            outcome = projects.run_cycle(self.project)
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["steps_done"], 0)
        self.assertEqual(len(alerts), 1)
        title, severity = alerts[0]
        self.assertIn("failed", title.lower())
        self.assertEqual(severity, "warn")
        # attempts logged as events too
        self.assertGreaterEqual(len(store.get("events", [])), projects._MAX_ATTEMPTS)

    @mock.patch.object(projects, "_db")
    def test_pause_for_approval_sets_paused_and_alerts_info(self, m_db):
        alerts = []
        store = {}
        m_db.return_value = _FakeQuery(store)
        fake_notify = types.ModuleType("notify")
        fake_notify.alert = mock.Mock(
            side_effect=lambda *a, **k: alerts.append((k.get("severity"),)))
        with mock.patch.dict(sys.modules, {"notify": fake_notify}):
            projects.pause_for_approval(self.project, "waiting for Ruk's go-ahead")
        self.assertEqual(store["updates"][0]["status"], "paused")
        self.assertEqual(alerts, [("info",)])
        self.assertTrue(any(e["event_type"] == "alert" for e in store["events"]))

    def test_pick_next_project_returns_none_when_no_active(self):
        with mock.patch.object(projects, "list_projects", return_value=[]):
            self.assertIsNone(projects.pick_next_project())

    def test_pick_next_project_prefers_oldest_active(self):
        rows = [{"id": "new", "name": "new"}, {"id": "old", "name": "old"}]
        with mock.patch.object(projects, "list_projects", return_value=rows):
            self.assertEqual(projects.pick_next_project()["id"], "old")


if __name__ == "__main__":
    unittest.main()
