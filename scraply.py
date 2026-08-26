"""scraply -- Sandy's scraping engine, wrapping Scrapling (D4Vinci/Scrapling).

Two modes:
  fast    -- plain HTTP via FetcherSession with TLS impersonation, no
             browser. Default for everything: cheap, quick, HF-free-tier
             friendly.
  stealth -- StealthyFetcher (real browser, solves Cloudflare). Opt-in
             only, gated behind sandy_config["scrapling_stealth"]=true,
             because the browser fetchers need Playwright/Camoufox
             downloads that don't belong on the free tier by default.

Contract (per plan Part 3):
  - returns a structured dict {ok, url, title, markdown, status} or
    {ok: False, url, error}; NEVER raises into chat/research paths.
  - size cap + markdown truncation before anything reaches a prompt,
    so token discipline holds even on a 1MB page dump.
  - if the scrapling import itself fails at runtime (bad rebuild), the
    module degrades to snippet-only callers by returning ok=False --
    logged once, non-fatal.
"""
import config
from llm import log

try:
    from scrapling.fetchers import FetcherSession, StealthyFetcher
    _IMPORT_OK = True
    _IMPORT_ERROR = None
except Exception as _import_err:  # pragma: no cover - depends on deploy env
    _IMPORT_OK = False
    _IMPORT_ERROR = _import_err

MAX_BYTES = 512 * 1024          # hard cap on raw page body we keep
MAX_MARKDOWN_CHARS = 12_000     # cap before the result enters any prompt
FETCH_TIMEOUT = 20              # seconds


def stealth_enabled() -> bool:
    """Stealth mode is opt-in via sandy_config; fail-open to False on any
    config blip (stealth is an upgrade, never a requirement)."""
    try:
        return bool(config.get_config("scrapling_stealth"))
    except Exception:
        return False


def fetch(url: str, mode: str = "fast") -> dict:
    """Fetch a URL and return LLM-ready markdown. Structured errors only."""
    if not _IMPORT_OK:
        log(f"[scraply] scrapling unavailable ({_IMPORT_ERROR!r}) -- degrading to failure")
        return {"ok": False, "url": url, "error": "scrapling not installed"}

    try:
        if mode == "stealth":
            if not stealth_enabled():
                return {"ok": False, "url": url,
                        "error": "stealth mode disabled (set sandy_config[scrapling_stealth]=true)"}
            page = StealthyFetcher.fetch(url, headless=True, network_idle=True,
                                         disable_resources=True, timeout=FETCH_TIMEOUT * 1000)
        else:
            mode = "fast"
            with FetcherSession(impersonate="chrome") as session:
                page = session.get(url, timeout=FETCH_TIMEOUT, stealthy_headers=True)
        return _pack(page, url, mode)
    except Exception as e:
        log(f"[scraply] {mode} fetch failed for {url}: {e!r}")
        return {"ok": False, "url": url, "error": str(e)}


def fetch_top(results: list[dict], n: int = 2, mode: str = "fast") -> list[dict]:
    """Follow-up scrape of the top-N search results -- gives workers real
    page content instead of 300-char snippets. Failures are dropped
    silently here (the caller already has the snippets as fallback)."""
    pages = []
    for r in results[:n]:
        url = r.get("url")
        if not url:
            continue
        page = fetch(url, mode=mode)
        if page.get("ok"):
            page["title"] = page.get("title") or r.get("title", "")
            pages.append(page)
    return pages


def _pack(page, url: str, mode: str) -> dict:
    """Normalize a Scrapling page object into the structured dict, with
    the size caps applied. Any accessor difference across scrapling
    versions is absorbed here."""
    try:
        status = getattr(page, "status", None)
        title = ""
        try:
            title = (page.css_first("title::text") or "").strip()
        except Exception:
            pass
        md = page.markdown() or ""
        if len(md.encode("utf-8")) > MAX_BYTES:
            md = md[:MAX_MARKDOWN_CHARS] + "\n\n[truncated by scraply]"
        elif len(md) > MAX_MARKDOWN_CHARS:
            md = md[:MAX_MARKDOWN_CHARS] + "\n\n[truncated by scraply]"
        return {"ok": True, "url": url, "mode": mode, "status": status,
                "title": title, "markdown": md}
    except Exception as e:
        log(f"[scraply] packing page for {url} failed: {e!r}")
        return {"ok": False, "url": url, "error": f"extract failed: {e}"}
