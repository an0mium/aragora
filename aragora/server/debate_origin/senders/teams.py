"""Microsoft Teams sender for debate origin result routing."""

from __future__ import annotations

import logging
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message

logger = logging.getLogger(__name__)


async def _send_teams_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Microsoft Teams."""
    # Teams uses webhook URLs stored in metadata
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        logger.warning("Teams webhook URL not in origin metadata")
        return False

    message = _format_result_message(result, origin, markdown=False)

    try:
        import httpx

        # Teams Adaptive Card format
        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Aragora Debate Complete",
                                "weight": "Bolder",
                                "size": "Large",
                            },
                            {"type": "TextBlock", "text": message, "wrap": True},
                        ],
                    },
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=card)
            if response.is_success:
                logger.info("Teams result sent via webhook")
                return True
            else:
                logger.warning(f"Teams send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Teams result send error: {e}")
        return False


async def _send_teams_receipt(origin: DebateOrigin, summary: str, receipt_url: str) -> bool:
    """Post receipt to Teams with link button."""
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        return False

    try:
        import httpx

        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Decision Receipt",
                                "weight": "Bolder",
                                "size": "Large",
                            },
                            {"type": "TextBlock", "text": summary, "wrap": True},
                        ],
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "title": "View Full Receipt",
                                "url": receipt_url,
                            }
                        ],
                    },
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=card)
            if response.is_success:
                logger.info("Teams receipt posted via webhook")
                return True
            return False

    except Exception as e:
        logger.error(f"Teams receipt post error: {e}")
        return False


async def _send_teams_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Teams."""
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        return False

    try:
        import httpx

        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Aragora Notice",
                                "weight": "Bolder",
                                "color": "Attention",
                            },
                            {"type": "TextBlock", "text": message, "wrap": True},
                        ],
                    },
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=card)
            return response.is_success

    except Exception as e:
        logger.error(f"Teams error send failed: {e}")
        return False
