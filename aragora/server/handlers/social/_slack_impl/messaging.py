# mypy: ignore-errors
"""
Slack messaging utilities.

Response helpers and async message posting for Slack Web API and response URLs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import (
    HandlerResult,
    SLACK_BOT_TOKEN,
    _validate_slack_url,
    json_response,
)

logger = logging.getLogger(__name__)


class MessagingMixin:
    """Mixin providing Slack message posting and response formatting."""

    def _slack_response(
        self,
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a simple Slack response."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
            }
        )

    def _slack_blocks_response(
        self,
        blocks: list[dict[str, Any]],
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a Slack response with blocks."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
                "blocks": blocks,
            }
        )

    async def _post_to_response_url(self, url: str, payload: dict[str, Any]) -> None:
        """POST a message to Slack's response_url.

        Includes SSRF protection by validating the URL is a Slack endpoint.
        """
        # Validate URL to prevent SSRF attacks
        if not _validate_slack_url(url):
            logger.warning(f"Invalid Slack response_url blocked (SSRF protection): {url[:50]}")
            return

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.warning(
                            f"Slack response_url POST failed: {response.status} - {text[:100]}"
                        )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting to Slack response_url: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error posting to Slack response_url: {e}")

    async def _post_message_async(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        """Post a message to Slack using the Web API.

        Args:
            channel: Channel ID to post to
            text: Message text
            thread_ts: Optional thread timestamp to reply to
            blocks: Optional Block Kit blocks for rich formatting

        Returns:
            Message timestamp (ts) if successful, None otherwise
        """
        import aiohttp

        if not SLACK_BOT_TOKEN:
            logger.warning("Cannot post message: SLACK_BOT_TOKEN not configured")
            return None

        try:
            payload: dict[str, Any] = {
                "channel": channel,
                "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts
            if blocks:
                payload["blocks"] = blocks

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(f"Slack API error: {result.get('error')}")
                        return None
                    # Return message timestamp for thread tracking
                    return result.get("ts")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting Slack message: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error posting Slack message: {e}")
            return None
