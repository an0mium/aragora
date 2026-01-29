"""Slack sender for debate origin result routing."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message

logger = logging.getLogger(__name__)


async def _send_slack_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Slack."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.warning("SLACK_BOT_TOKEN not configured")
        return False

    channel = origin.channel_id
    message = _format_result_message(result, origin, markdown=True)

    try:
        import httpx

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "channel": channel,
            "text": message,
            "mrkdwn": True,
        }

        # Reply in thread if we have thread_ts
        if origin.thread_id:
            data["thread_ts"] = origin.thread_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                resp_data = response.json()
                if resp_data.get("ok"):
                    logger.info(f"Slack result sent to {channel}")
                    return True
                else:
                    logger.warning(f"Slack API error: {resp_data.get('error')}")
                    return False
            else:
                logger.warning(f"Slack send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Slack result send error: {e}")
        return False


async def _send_slack_receipt(origin: DebateOrigin, summary: str, receipt_url: str) -> bool:
    """Post receipt to Slack with button to view full receipt."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

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
                    logger.info(f"Slack receipt posted to {origin.channel_id}")
                    return True
            return False

    except Exception as e:
        logger.error(f"Slack receipt post error: {e}")
        return False


async def _send_slack_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Slack."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

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

    except Exception as e:
        logger.error(f"Slack error send failed: {e}")
        return False
