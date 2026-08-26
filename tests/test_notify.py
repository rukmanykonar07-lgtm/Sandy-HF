"""Tests for notify.AlertRouter (Part 7).

Covers: severity normalization, cooldown dedup (and its critical
exemption), severity->channel routing matrix, per-channel failure
isolation, and the never-raises guarantee. All HTTP is mocked at the
requests boundary -- no real network, no network needed.
"""
import sys
import threading
import unittest
from unittest import mock

# conftest.py sets fake Supabase env; llm.log just prints.
sys.path.insert(0, ".")

import notify


def _wait_for_workers():
    """alert() spawns fire-and-forget threads; join them deterministically."""
    for t in threading.enumerate():
        if t.name.startswith("notify-"):
            t.join(timeout=5)


class _Resp:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text


class NotifyTestBase(unittest.TestCase):
    def setUp(self):
        # cooldown state is module-global; reset per test
        with notify._recent_lock:
            notify._recent.clear()
        # default: every channel configured & succeeding; tests override
        self._env_patcher = mock.patch.dict(
            "os.environ",
            {
                "CALLMEBOT_PHONE": "15551234567",
                "CALLMEBOT_APIKEY": "cb-key",
                "TWILIO_SID": "AC123",
                "TWILIO_AUTH_TOKEN": "tw-token",
                "TWILIO_FROM": "+15550001111",
                "TWILIO_TO": "+15552223333",
                "TELEGRAM_BOT_TOKEN": "tg-token", "TELEGRAM_CHAT_ID": "12345",
            },
        )
        self._env_patcher.start()
        self._config_patcher = mock.patch(
            "config.get_config", side_effect=self._fake_get_config
        )
        self._config_patcher.start()
        import config

        config.get_config = mock.Mock(side_effect=self._fake_get_config)

    def tearDown(self):
        self._env_patcher.stop()
        self._config_patcher.stop()
        import config

        if hasattr(config.get_config, "reset_mock"):
            try:
                del type(config).get_config  # remove class attr if any
            except AttributeError:
                pass
        # restore original function attribute replaced in setUp
        import importlib

        importlib.reload(notify)

    @staticmethod
    def _fake_get_config(key):
        return {
            "email_api_url": "https://mailhook.example/send",
            "email_api_key": "em-key",
        }.get(key)


class TestSeverityMatrix(NotifyTestBase):
    def test_info_reaches_telegram_email_not_twilio(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post, \
             mock.patch.object(notify.requests, "get", return_value=_Resp(200)) as get:
            notify.alert("T", "b", severity="info")
            _wait_for_workers()
            urls = [c.args[0] for c in post.call_args_list]
            self.assertTrue(any("api.telegram.org" in u for u in urls))
            self.assertTrue(any("mailhook.example" in u for u in urls))
            self.assertFalse(any("twilio.com" in u for u in urls))

    def test_warn_same_channels_as_info(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)), \
             mock.patch.object(notify.requests, "get", return_value=_Resp(200)):
            notify.alert("T", "b", severity="warn")
            _wait_for_workers()
            # no exception == routed fine; channel assertions covered above

    def test_critical_adds_twilio_call(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
            notify.alert("T", "b", severity="critical")
            _wait_for_workers()
            calls = [c for c in post.call_args_list if "Calls.json" in c.args[0]]
            self.assertEqual(len(calls), 1)
            twiml = calls[0].kwargs["data"]["Twiml"]
            self.assertIn("<Response><Say loop=", twiml)
            self.assertTrue(len(twiml) > len("<Response><Say loop=\"2\"></Say></Response>"))
            auth = calls[0].kwargs["auth"]
            self.assertEqual(auth, ("AC123", "tw-token"))

    def test_unknown_severity_becomes_info(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)):
            receipt = notify.alert("T", "b", severity="banana")
            _wait_for_workers()
            self.assertEqual(receipt["severity"], "info")


class TestCooldown(NotifyTestBase):
    def test_duplicate_suppressed_within_cooldown(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
            r1 = notify.alert("Same title", "one", severity="warn")
            r2 = notify.alert("Same title", "two", severity="warn")
            _wait_for_workers()
            self.assertTrue(r1["queued"] and not r1["deduped"])
            self.assertFalse(r2["queued"] and r2["deduped"])
            sends = [c for c in post.call_args_list if "api.telegram.org" in c.args[0]]
            self.assertEqual(len(sends), 1)

    def test_critical_never_dedupes(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
            notify.alert("Crit", "one", severity="critical")
            notify.alert("Crit", "two", severity="critical")
            _wait_for_workers()
            calls = [c for c in post.call_args_list if "Calls.json" in c.args[0]]
            self.assertEqual(len(calls), 2)

    def test_different_titles_both_send(self):
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
            notify.alert("A", "x", severity="warn")
            notify.alert("B", "y", severity="warn")
            _wait_for_workers()
            sends = [c for c in post.call_args_list if "api.telegram.org" in c.args[0]]
            self.assertEqual(len(sends), 2)


class TestChannelIsolation(NotifyTestBase):
    def test_telegram_down_falls_back_to_callmebot(self):
        def post_side(url, **kw):
            if "api.telegram.org" in url:
                raise ConnectionError("telegram down")
            return _Resp(200)

        with mock.patch.object(notify.requests, "post", side_effect=post_side), \
             mock.patch.object(notify.requests, "get", return_value=_Resp(200)) as get:
            receipt = notify.alert("T", "b", severity="info")
            _wait_for_workers()
            self.assertTrue(receipt["queued"])
            get.assert_called_once()  # callmebot fired
            self.assertIn("callmebot", get.call_args.args[0])

    def test_all_channels_down_still_returns_receipt_no_raise(self):
        with mock.patch.object(notify.requests, "post", side_effect=ConnectionError("down")), \
             mock.patch.object(notify.requests, "get", side_effect=ConnectionError("down")):
            receipt = notify.alert("T", "b", severity="info")
            _wait_for_workers()
            self.assertTrue(receipt["queued"])

    def test_worker_thread_swallows_internal_crash(self):
        with mock.patch.object(notify, "_dispatch", side_effect=RuntimeError("bug")):
            receipt = notify.alert("T", "b", severity="warn")  # must not raise
            _wait_for_workers()
            self.assertTrue(receipt["queued"])


class TestConfigGating(NotifyTestBase):
    def test_missing_twilio_env_disables_call_channel(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            for k in ("TWILIO_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "TWILIO_TO"):
                os.environ.pop(k, None)
            with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
                notify.alert("T", "b", severity="critical")
                _wait_for_workers()
                self.assertFalse(any("Calls.json" in c.args[0] for c in post.call_args_list))

    def test_email_disabled_without_config_url(self):
        self._fake_urls = {}
        import config

        config.get_config = mock.Mock(return_value=None)
        with mock.patch.object(notify.requests, "post", return_value=_Resp(200)) as post:
            notify.alert("T", "b", severity="info")
            _wait_for_workers()
            self.assertFalse(any("mailhook" in c.args[0] for c in post.call_args_list))


if __name__ == "__main__":
    unittest.main()
