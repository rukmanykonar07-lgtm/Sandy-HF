"""
Sends outgoing WhatsApp messages via Meta's Cloud API.

Required env vars (set as SECRETS in HF Space settings):
  WHATSAPP_TOKEN            -> permanent System User access token
  WHATSAPP_PHONE_NUMBER_ID  -> the "Phone Number ID" from API Setup
                                (not the phone number itself)
"""

import os
import logging
import httpx

log = logging.getLogger("sandy")

GRAPH_API_VERSION = "v21.0"


def send_whatsapp_reply(to: str, text: str) -> bool:
    """
    Sends a plain text WhatsApp message to `to` (the sender's WhatsApp ID,
    e.g. "919876543210" — no plus sign, exactly as Meta sends it in the
    incoming webhook payload's "from" field).

    Returns True on success, False on failure (never raises — a failed
    reply shouldn't crash the webhook handler, just gets logged).
    """
    phone_number_id = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
    token = os.environ["WHATSAPP_TOKEN"]

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=15.0)
        response.raise_for_status()
        log.info("WhatsApp reply sent to %s", to)
        return True
    except httpx.HTTPStatusError as e:
        # Meta's error body usually explains exactly what went wrong
        # (expired token, unregistered test number, etc.) — log it in full.
        log.error("WhatsApp send failed (%s): %s", e.response.status_code, e.response.text)
        return False
    except httpx.RequestError as e:
        log.error("WhatsApp send request error: %s", e)
        return False
