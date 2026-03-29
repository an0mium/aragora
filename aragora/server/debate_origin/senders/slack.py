"""Slack sender for debate origin result routing."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..models import DebateOrigin
from ..formatting import _format_result_message

logger = logging.getLogger(__name__)


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


async def _send_slack_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Slack."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.debug("SLACK_BOT_TOKEN not configured")
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
    token = os.environ.get("SLACK_BOT_TOKEN", "")
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
    token = os.environ.get("SLACK_BOT_TOKEN", "")
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
