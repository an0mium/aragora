"""Slack sender for debate origin result routing.

Supports token rotation: when ``SLACK_REFRESH_TOKEN``, ``SLACK_CLIENT_ID``,
and ``SLACK_CLIENT_SECRET`` are set, the sender automatically exchanges the
refresh token for a short-lived access token before each batch of API calls.
Falls back to the static ``SLACK_BOT_TOKEN`` when rotation env vars are absent.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

from ..formatting import _format_result_message
from ..models import DebateOrigin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token management with rotation support
# ---------------------------------------------------------------------------

_token_cache: dict[str, Any] = {}
_token_lock = threading.Lock()


async def _get_slack_token() -> str:
    """Get a valid Slack bot token, exchanging the refresh token if needed.

    Priority:
    1. Cached access token (if not expired)
    2. Token rotation via refresh token exchange
    3. Static SLACK_BOT_TOKEN from environment
    """
    with _token_lock:
        # Check cache
        cached = _token_cache.get("access_token", "")
        expires_at = _token_cache.get("expires_at", 0)
        if cached and time.time() < expires_at - 300:  # 5 min buffer
            return cached

    # Try token rotation
    refresh_token = os.environ.get("SLACK_REFRESH_TOKEN", "")
    client_id = os.environ.get("SLACK_CLIENT_ID", "")
    client_secret = os.environ.get("SLACK_CLIENT_SECRET", "")

    if refresh_token and client_id and client_secret:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                )
                data = resp.json()
                if data.get("ok"):
                    access_token = data["access_token"]
                    new_refresh = data.get("refresh_token", "")
                    expires_in = int(data.get("expires_in", 43200))

                    with _token_lock:
                        _token_cache["access_token"] = access_token
                        _token_cache["expires_at"] = time.time() + expires_in

                    # Update the refresh token in env for the next rotation
                    if new_refresh:
                        os.environ["SLACK_REFRESH_TOKEN"] = new_refresh
                        logger.debug("Slack token rotated, expires in %ds", expires_in)

                    return access_token
                else:
                    logger.warning("Slack token refresh failed: %s", data.get("error"))
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            logger.warning("Slack token refresh error: %s", exc)

    # Fallback to static token
    return os.environ.get("SLACK_BOT_TOKEN", "")


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _truncate_fallback_text(text: str, limit: int = 3000) -> str:
    """Return a Slack-safe fallback text string."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_slack_payload(
    origin: DebateOrigin,
    message: str | dict[str, Any],
    *,
    fallback_text: str,
) -> dict[str, Any]:
    """Build a Slack chat.postMessage payload from string or Block Kit content."""
    payload: dict[str, Any] = {"channel": origin.channel_id}

    if isinstance(message, dict):
        payload.update(message)
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            payload["text"] = _truncate_fallback_text(fallback_text)
    else:
        payload["text"] = message
        payload["mrkdwn"] = True

    if origin.thread_id:
        payload["thread_ts"] = origin.thread_id

    return payload


# ---------------------------------------------------------------------------
# Senders
# ---------------------------------------------------------------------------


async def _send_slack_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Slack."""
    token = await _get_slack_token()
    if not token:
        logger.debug("No Slack token available (set SLACK_BOT_TOKEN or SLACK_REFRESH_TOKEN)")
        return False

    channel = origin.channel_id
    message = _format_result_message(result, origin, markdown=True)
    fallback_text = str(result.get("final_answer") or result.get("task") or "Debate Complete")

    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = _build_slack_payload(origin, message, fallback_text=fallback_text)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                resp_data = response.json()
                if resp_data.get("ok"):
                    logger.info("Slack result sent to %s", channel)
                    return True
                else:
                    logger.warning("Slack API error: %s", resp_data.get("error"))
                    return False
            else:
                logger.warning("Slack send failed: %s", response.status_code)
                return False

    except (httpx.HTTPError, OSError, ValueError, TypeError) as e:
        logger.error("Slack result send error: %s", e)
        return False


async def _send_slack_receipt(origin: DebateOrigin, summary: str, receipt_url: str) -> bool:
    """Post receipt to Slack with button to view full receipt."""
    token = await _get_slack_token()
    if not token:
        return False

    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "channel": origin.channel_id,
            "text": summary,
            "mrkdwn": True,
            "attachments": [
                {
                    "fallback": "View Receipt",
                    "actions": [
                        {
                            "type": "button",
                            "text": "View Full Receipt",
                            "url": receipt_url,
                            "style": "primary",
                        }
                    ],
                }
            ],
        }

        if origin.thread_id:
            data["thread_ts"] = origin.thread_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                resp_data = response.json()
                if resp_data.get("ok"):
                    logger.info("Slack receipt posted to %s", origin.channel_id)
                    return True
            return False

    except (httpx.HTTPError, OSError, ValueError, TypeError) as e:
        logger.error("Slack receipt post error: %s", e)
        return False


async def _send_slack_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Slack."""
    token = await _get_slack_token()
    if not token:
        return False

    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "channel": origin.channel_id,
            "text": message,
            "mrkdwn": True,
        }

        if origin.thread_id:
            data["thread_ts"] = origin.thread_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            return response.is_success and response.json().get("ok", False)

    except (httpx.HTTPError, OSError, ValueError, TypeError) as e:
        logger.error("Slack error send failed: %s", e)
        return False
