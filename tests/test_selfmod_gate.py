"""Phase H (Part 9) tests -- selfmod gate hardening.

Covers:
  - propose_edit: oversize diff (> _MAX_DIFF_LINES) rejected, nothing
    stored in _pending, no mirror write
  - propose_edit: valid proposal stores it + persists JSON mirror via
    config.set_config
  - apply_pending success path clears the mirror via config.delete_config
  - _risk_label: risky-API markers -> high; >100 lines -> high; small
    clean diff -> low (informational only)
  - _load_persisted_pending: restores into empty _pending at boot;
    no-op when _pending already populated
  - ast.parse gate still rejects syntactically broken generated content
  - propose_edit on a file outside the repo is refused

All LLM/config surfaces are mocked -- no network.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import selfmod


class SelfmodGateTestCase(unittest.TestCase):
    """Isolate the module-level _pending dict per test."""

    def setUp(self):
        self._saved = dict(selfmod._pending)
        selfmod._pending.clear()

    def tearDown(self):
        selfmod._pending.clear()
        selfmod._pending.update(self._saved)


class RiskLabelTests(unittest.TestCase):
    def test_subprocess_marker_is_high(self):
        self.assertEqual(selfmod._risk_label("import subprocess\n+do_thing()"), "high")

    def test_run_git_marker_is_high(self):
        self.assertEqual(selfmod._risk_label("+self._run_git('push')"), "high")

    def test_eval_and_exec_markers_are_high(self):
        self.assertEqual(selfmod._risk_label("+x = eval(expr)"), "high")
        self.assertEqual(selfmod._risk_label("+exec(code)"), "high")

    def test_hf_write_token_marker_is_high(self):
        self.assertEqual(selfmod._risk_label("+os.environ['HF_WRITE_TOKEN']"), "high")

    def test_over_hundred_lines_is_high_even_if_clean(self):
        big = "\n".join(f"+line {i}" for i in range(101))
        self.assertEqual(selfmod._risk_label(big), "high")

    def test_exactly_hundred_clean_lines_is_low(self):
        edge = "\n".join(f"+line {i}" for i in range(100))
        self.assertEqual(selfmod._risk_label(edge), "low")

    def test_small_clean_diff_is_low(self):
        self.assertEqual(selfmod._risk_label("+print('hi')"), "low")


class ProposeGateTests(SelfmodGateTestCase):
    def setUp(self):
        super().setUp()
        # A tiny real repo file for propose/apply flows to chew on.
        self.tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_gate_target.py")
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            f.write("value = 1\n")

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        super().tearDown()

    @mock.patch.dict(os.environ, {"SANDY_REPO_DIR": ""})
    def _patch_repo_dir(self):
        return mock.patch.object(selfmod, "REPO_DIR", os.path.dirname(self.tmp_path))

    def test_oversize_diff_rejected_nothing_pending(self):
        huge_new = "\n".join(f"line_{i} = {i}" for i in range(500))
        rel = os.path.basename(self.tmp_path)
        with self._patch_repo_dir(), \
             mock.patch.object(selfmod, "call_llm_with_fallback",
                               return_value=huge_new):
            reply = selfmod.propose_edit("s1", rel, "make it huge")
        self.assertIn("400 lines se bada", reply)
        self.assertNotIn("s1", selfmod._pending)

    def test_valid_proposal_stores_and_persists(self):
        rel = os.path.basename(self.tmp_path)
        with self._patch_repo_dir(), \
             mock.patch.object(selfmod, "call_llm_with_fallback",
                               return_value="value = 2\n"):
            reply = selfmod.propose_edit("s2", rel, "bump value")
        self.assertIn("Proposed change to", reply)
        self.assertEqual(selfmod._pending["s2"]["file_path"], rel)
        self.assertEqual(selfmod._pending["s2"]["new_content"], "value = 2\n")
        # risk note present and low for this clean one-liner
        self.assertIn("(risk: low)", reply)

    def test_ast_broken_content_rejected_by_existing_gate(self):
        rel = os.path.basename(self.tmp_path)
        with self._patch_repo_dir(), \
             mock.patch.object(selfmod, "call_llm_with_fallback",
                               return_value="def broken(:\n"):
            reply = selfmod.propose_edit("s3", rel, "break it")
        self.assertIn("syntax error", reply)
        self.assertNotIn("s3", selfmod._pending)

    def test_escape_path_refused(self):
        with self._patch_repo_dir():
            reply = selfmod.propose_edit("s4", "../outside.py", "escape")
        self.assertIn("repo ke bahar", reply)
        self.assertNotIn("s4", selfmod._pending)


class PersistMirrorTests(SelfmodGateTestCase):
    def test_propose_calls_set_config_with_json_dump(self):
        calls = {}
        fake_config = types.SimpleNamespace(
            set_config=lambda k, v: calls.__setitem__(k, v),
            get_config=lambda _k: None,
            delete_config=lambda _k: None,
        )
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_mirror_target.py")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("x = 0\n")
        try:
            with mock.patch.object(selfmod, "REPO_DIR", os.path.dirname(tmp)), \
                 mock.patch.dict(sys.modules, {"config": fake_config}), \
                 mock.patch.object(selfmod, "call_llm_with_fallback", return_value="x = 1\n"):
                selfmod.propose_edit("s5", os.path.basename(tmp), "bump x")
        finally:
            os.remove(tmp)
        self.assertIn(selfmod._PERSIST_KEY, calls)
        import json as _json
        restored = _json.loads(calls[selfmod._PERSIST_KEY])
        self.assertIn("s5", restored)
        self.assertEqual(restored["s5"]["new_content"], "x = 1\n")

    def test_apply_success_clears_mirror(self):
        fake_config = types.SimpleNamespace(
            set_config=lambda *_a: None,
            get_config=lambda _k: None,
            delete_config=lambda k: clears.append(k),
        )
        clears = []
        selfmod._pending["s6"] = {
            "file_path": "ok.txt",
            "new_content": "fine\n",
            "commit_message": "test commit",
        }
        tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_apply_repo")
        os.makedirs(tmpdir, exist_ok=True)
        target = os.path.join(tmpdir, "ok.txt")
        try:
            with mock.patch.object(selfmod, "REPO_DIR", tmpdir), \
                 mock.patch.dict(sys.modules, {"config": fake_config}), \
                 mock.patch.object(selfmod, "ensure_git_ready"), \
                 mock.patch.object(selfmod, "_assert_up_to_date"), \
                 mock.patch.object(selfmod, "_contained_path", return_value=target), \
                 mock.patch.object(selfmod, "_run_git") as git_mock:
                reply = selfmod.apply_pending("s6")
            self.assertIn("Done, Ruk", reply)
            git_mock.assert_any_call("push", "origin", "main")
            self.assertEqual(clears, [selfmod._PERSIST_KEY])
            self.assertNotIn("s6", selfmod._pending)
        finally:
            if os.path.exists(target):
                os.remove(target)
            os.rmdir(tmpdir)

    def test_load_restores_when_empty(self):
        import json as _json
        saved = {"s7": {"file_path": "a.py", "new_content": "a=1\n", "commit_message": "m"}}
        fake_config = types.SimpleNamespace(
            set_config=lambda *_a: None,
            get_config=lambda _k: _json.dumps(saved),
            delete_config=lambda _k: None,
        )
        with mock.patch.dict(sys.modules, {"config": fake_config}):
            selfmod._load_persisted_pending()
        self.assertIn("s7", selfmod._pending)
        self.assertEqual(selfmod._pending["s7"]["new_content"], "a=1\n")

    def test_load_noop_when_already_populated(self):
        selfmod._pending["live"] = {"file_path": "b.py", "new_content": "b=1\n", "commit_message": "n"}
        raise_called = False
        def boom(_k):
            nonlocal raise_called
            raise_called = True
            raise RuntimeError("should not be read")
        fake_config = types.SimpleNamespace(set_config=lambda *_a: None,
                                            get_config=boom, delete_config=lambda _k: None)
        with mock.patch.dict(sys.modules, {"config": fake_config}):
            selfmod._load_persisted_pending()
        self.assertFalse(raise_called)
        self.assertIn("live", selfmod._pending)


if __name__ == "__main__":
    unittest.main()
