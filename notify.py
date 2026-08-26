"""Part 7 -- AlertRouter: Sandy's single outbound-notification chokepoint.

Replaces the old direct Meta Cloud API send_whatsapp() that lived in
projects.py (and which healing.py imported). The WHATSAPP_* env vars are
DEPRECATED -- WhatsApp now flows through the local Baileys sidecar
(node-service/server.js, bound to 127.0.0.1:3001) that Ruk pairs exactly
once by scanning a QR printed to the container logs; session credentials
persist in sandy_config[baileys_creds] so rebuilds reconnect silently.

Severity matrix (master plan Part 7):
    info/warn  -> WhatsApp (+ email if configured, + CallMeBot fallback
                  if WhatsApp fails and CallMeBot is configured)
    critical   -> all of the above PLUS a Twilio voice call reading the
                  message aloud twice (<Say loop="2">)

Guarantees (these are the whole point of this module):
    - alert() NEVER raises and NEVER blocks the caller: real dispatch
      happens on a fire-and-forget daemon thread.
    - Dedup: identical (title, severity) alerts are suppressed for
      ALERT_COOLDOWN_S (default 900s) EXCEPT critical, which always
      goes through -- Ruk always picks up a call.
    - Each channel fails independently and open: a dead channel costs a
      log line, never an exception and never blocks the other channels.
"""
import os
import threading
import time
from xml.sax.saxutils import escape as xml_escape

import requests

from llm import log

SEVERITIES = ("info", "warn", "critical")

_ALERT_COOLDOWN_S = int(os.environ.get("ALERT_COOLDOWN_S", "900"))
_BAILEYS_URL = os.environ.get("BAILEYS_URL", "http://127.0.0.1:3001")
_HTTP_TIMEOUT = 6

# (title, severity) -> monotonic timestamp of last dispatch attempt
_recent: dict[tuple[str, str], float] = {}
_recent_lock = threading.Lock()


def _normalize(severity: str | None) -> str:
    return severity if severity in SEVERITIES else "info"


def _recently_sent(title: str, severity: str) -> bool:
    with _recent_lock:
        ts = _recent.get((title, severity))
        return ts is not None and (time.monotonic() - ts) < _ALERT_COOLDOWN_S


def _mark_sent(title: str, severity: str) -> None:
    with _recent_lock:
        # opportunistic GC so the dict can't grow forever
        cutoff = time.monotonic() - max(_ALERT_COOLDOWN_S * 4, 3600)
        for k in [k for k, v in _recent.items() if v < cutoff]:
            del _recent[k]
        _recent[(title, severity)] = time.monotonic()


# ---------------------------------------------------------------- channels
# Each returns {"ok": bool, "detail": str}. "disabled" in detail means the
# channel isn't configured (expected on day one) -- that's not an error.

def _send_whatsapp(text: str) -> dict:
    """POST to the Baileys sidecar. Unreachable sidecar is a normal,
    logged condition (e.g. during boot or if Node died) -- never fatal."""
    try:
        r = requests.post(
            f"{_BAILEYS_URL}/send", json={"text": text}, timeout=_HTTP_TIMEOUT
        )
        if r.status_code == 200:
            return {"ok": True, "detail": "sent"}
        return {"ok": False, "detail": f"sidecar HTTP {r.status_code}: {r.text[:120]}"}
    except Exception as e:
        return {"ok": False, "detail": f"sidecar unreachable: {e!r}"}


def _send_callmebot(text: str) -> dict:
    """Optional last-resort WhatsApp path (no sidecar needed). Off unless
    CALLMEBOT_PHONE + CALLMEBOT_APIKEY are set."""
    phone = os.environ.get("CALLMEBOT_PHONE")
    apikey = os.environ.get("CALLMEBOT_APIKEY")
    if not phone or not apikey:
        return {"ok": False, "detail": "disabled"}
    try:
        r = requests.get(
            "https://api.callmebot.com/whatsapp.php",
            params={"phone": phone, "text": text, "apikey": apikey},
            timeout=_HTTP_TIMEOUT,
        )
        return {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "detail": f"{e!r}"}


def _send_email(subject: str, body: str) -> dict:
    """Stub per plan: POSTs {subject, body} to a generic webhook. Reads
    its endpoint from sandy_config so Ruk can wire it later WITHOUT a
    redeploy. Missing url => channel off."""
    import config

    try:
        url = config.get_config("email_api_url")
    except Exception as e:
        return {"ok": False, "detail": f"config read failed: {e!r}"}
    if not url:
        return {"ok": False, "detail": "disabled"}
    api_key = config.get_config("email_api_key") or ""
    try:
        r = requests.post(
            url, json={"subject": subject, "body": body, "key": api_key},
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=_HTTP_TIMEOUT,
        )
        return {"ok": r.status_code < 300, "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "detail": f"{e!r}"}


def _place_call(message: str) -> dict:
    """Critical-only Twilio voice call. Raw REST (no SDK -- free-tier
    discipline: zero extra deps). Any missing TWILIO_* env => silently
    off. Message is XML-escaped and hard-capped at 300 chars."""
    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_ = os.environ.get("TWILIO_FROM")
    to = os.environ.get("TWILIO_TO")
    if not (sid and token and from_ and to):
        return {"ok": False, "detail": "disabled"}
    said = xml_escape(f"{message}")[:300]
    twiml = f'<Response><Say loop="2">{said}</Say></Response>'
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
            data={"To": to, "From": from_, "Twiml": twiml},
            auth=(sid, token),
            timeout=_HTTP_TIMEOUT,
        )
        return {"ok": r.status_code < 300, "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "detail": f"{e!r}"}


# ---------------------------------------------------------------- dispatch

def _dispatch(title: str, body: str, severity: str) -> dict:
    """Runs ON THE WORKER THREAD. Sends through every channel this
    severity warrants; returns the per-channel report."""
    text = f"{title}\n{body}"
    channels: dict[str, dict] = {}

    channels["whatsapp"] = _send_whatsapp(text)
    if not channels["whatsapp"]["ok"]:
        channels["callmebot_fallback"] = _send_callmebot(text)

    channels["email"] = _send_email(subject=title, body=body)

    if severity == "critical":
        channels["twilio_call"] = _place_call(f"{title}. {body}")

    dispatched = any(c["ok"] for c in channels.values())
    return {"severity": severity, "channels": channels, "dispatched": dispatched}


def _safe_dispatch(title: str, body: str, severity: str) -> None:
    """Top-level worker-thread guard: even a bug above must only cost a
    log line."""
    try:
        report = _dispatch(title, body, severity)
        bad = {k: v["detail"] for k, v in report["channels"].items() if not v["ok"]}
        if report["dispatched"]:
            log(f"[notify] '{title}' ({severity}) sent{'; failed: ' + str(bad) if bad else ''}")
        else:
            log(f"[notify] '{title}' ({severity}) reached NO channel: {bad}")
    except Exception as e:
        log(f"[notify] dispatch crashed for '{title}': {e!r}")


def alert(title: str, body: str, severity: str = "info", meta: dict | None = None) -> dict:
    """The one entrypoint everything else calls. Returns fast with a
    receipt; the actual sending continues on a daemon thread.

    meta, if given, is appended as readable key: value lines (job refs,
    provider names etc.) -- kept out of `body` so callers stay terse."""
    severity = _normalize(severity)
    if severity != "critical" and _recently_sent(title, severity):
        return {"queued": False, "deduped": True, "severity": severity}
    _mark_sent(title, severity)

    if meta:
        meta_lines = "\n".join(f"- {k}: {v}" for k, v in meta.items())
        body = f"{body}\n{meta_lines}"

    threading.Thread(
        target=_safe_dispatch, args=(title, body, severity),
        name=f"notify-{title[:24]}", daemon=True,
    ).start()
    return {"queued": True, "deduped": False, "severity": severity}
