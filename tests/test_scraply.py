"""Phase B (Part 3) scraply tests -- everything mocked, no network.

Covers the plan's pre-solved scenarios:
  #10 site blocked (403/CF)      -> structured failure, never raises
  #11 huge page dump (1MB)       -> markdown truncated before prompts
  import-failure degrade         -> research proceeds, ok=False result
  stealth opt-in gate            -> disabled by default via sandy_config
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraply


class _FakePage:
    def __init__(self, html='<html><title>Hi</title><body>hello</body></html>',
                 status=200):
        self._html = html
        self.status = status

    def css_first(self, sel):
        if "title" in sel:
            return "Hi"
        return None

    def markdown(self):
        return "# Hi\n\nhello world content"


def _fake_session(page=None):
    """Patch scraply.FetcherSession so session.get returns the fake page."""
    page = page or _FakePage()
    entered = mock.MagicMock()
    entered.get.return_value = page
    session_cls = mock.MagicMock()
    session_cls.return_value.__enter__.return_value = entered
    return session_cls


class TestFastFetch(unittest.TestCase):
    def setUp(self):
        self.sess = _fake_session()
        p = mock.patch.object(scraply, "FetcherSession", self.sess)
        p.start()
        self.addCleanup(p.stop)

    def test_ok_shape(self):
        r = scraply.fetch("https://example.com")
        self.assertTrue(r["ok"])
        self.assertEqual(r["url"], "https://example.com")
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["title"], "Hi")
        self.assertIn("hello", r["markdown"])

    def test_network_failure_is_structured(self):
        self.sess.return_value.__enter__.return_value.get.side_effect = \
            RuntimeError("connection reset")
        r = scraply.fetch("https://blocked.example")
        self.assertFalse(r["ok"])
        self.assertIn("reset", r["error"])

    def test_huge_page_truncated(self):
        big = _FakePage()
        big.markdown = lambda: "x" * 900_000
        self.sess.return_value.__enter__.return_value.get.return_value = big
        r = scraply.fetch("https://huge.example")
        self.assertTrue(r["ok"])
        self.assertLess(len(r["markdown"]), 20_000)
        self.assertTrue(r["markdown"].endswith("[truncated by scraply]"))


class TestStealthGate(unittest.TestCase):
    def test_stealth_disabled_by_config_default(self):
        with mock.patch.object(scraply.config, "get_config", side_effect=lambda k: None):
            r = scraply.fetch("https://cf.example", mode="stealth")
            self.assertFalse(r["ok"])
            self.assertIn("disabled", r["error"])

    def test_stealth_opt_in_uses_stealthy_fetcher(self):
        sf = mock.MagicMock()
        sf.fetch.return_value = _FakePage()
        with mock.patch.object(scraply.config, "get_config",
                               side_effect=lambda k: True if k == "scrapling_stealth" else None), \
             mock.patch.object(scraply, "StealthyFetcher", sf):
            r = scraply.fetch("https://cf.example", mode="stealth")
            self.assertTrue(r["ok"])
            self.assertEqual(r["mode"], "stealth")

    def test_stealth_config_blip_fails_open_to_disabled(self):
        with mock.patch.object(scraply.config, "get_config",
                               side_effect=RuntimeError("supabase blip")):
            self.assertFalse(scraply.stealth_enabled())


class TestImportDegrade(unittest.TestCase):
    def test_import_failure_never_raises(self):
        with mock.patch.object(scraply, "_IMPORT_OK", False), \
             mock.patch.object(scraply, "_IMPORT_ERROR", RuntimeError("bad rebuild")):
            r = scraply.fetch("https://example.com")
            self.assertFalse(r["ok"])
            self.assertIn("not installed", r["error"])


class TestFetchTop(unittest.TestCase):
    def test_follow_up_scrape_drops_failures(self):
        good = {"url": "https://a.example", "title": "A"}
        bad = {"url": None}                       # skipped: no url
        with mock.patch.object(scraply, "fetch",
                               side_effect=[{"ok": True, "url": "https://a.example",
                                             "markdown": "md", "status": 200,
                                             "title": ""}, {"ok": False, "url": "x"}]):
            pages = scraply.fetch_top([good, bad], n=2)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["title"], "A")   # backfilled from the snippet


if __name__ == "__main__":
    unittest.main()
