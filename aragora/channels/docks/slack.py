"""
Slack Dock - Channel dock implementation for Slack.

Handles message delivery to Slack workspaces via the Slack API.

Example:
    from aragora.channels.docks.slack import SlackDock

    dock = SlackDock({"token": "xoxb-..."})
    await dock.initialize()
    await dock.send_message(channel_id, message)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult

if TYPE_CHECKING:
    from aragora.channels.normalized import NormalizedMessage

logger = logging.getLogger(__name__)

__all__ = ["SlackDock"]


class SlackDock(ChannelDock):
    """
    Slack platform dock.

    Supports rich text formatting, buttons, threads, file uploads,
    and reactions via the Slack API.
    """

    PLATFORM = "slack"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT
        | ChannelCapability.BUTTONS
        | ChannelCapability.THREADS
        | ChannelCapability.FILES
        | ChannelCapability.REACTIONS
    )

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize Slack dock.

        Config options:
            token: Slack bot token (or use SLACK_BOT_TOKEN env var)
        """
        super().__init__(config)
        self._token: Optional[str] = None

    async def initialize(self) -> bool:
        """Initialize the Slack dock."""
        self._token = self.config.get("token") or os.environ.get("SLACK_BOT_TOKEN", "")
        if not self._token:
            logger.warning("Slack token not configured")
            return False

        self._initialized = True
        return True

    async def send_message(
        self,
        channel_id: str,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a message to Slack.

        Args:
            channel_id: Slack channel ID
            message: The normalized message to send
            **kwargs: Additional options (thread_ts, etc.)

        Returns:
            SendResult indicating success or failure
        """
        if not self._token:
            return SendResult.fail(
                error="Slack token not configured",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        try:
            import httpx

            # Build Slack message payload
            payload = self._build_payload(channel_id, message, **kwargs)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        return SendResult.ok(
                            message_id=data.get("ts"),
                            platform=self.PLATFORM,
                            channel_id=channel_id,
                            thread_ts=data.get("ts"),
                        )
                    else:
                        return SendResult.fail(
                            error=data.get("error", "Unknown Slack error"),
                            platform=self.PLATFORM,
                            channel_id=channel_id,
                        )
                else:
                    return SendResult.fail(
                        error=f"HTTP {response.status_code}",
                        platform=self.PLATFORM,
                        channel_id=channel_id,
                    )

        except Exception as e:
            logger.error(f"Slack send error: {e}")
            return SendResult.fail(
                error=str(e),
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

    def _build_payload(
        self,
        channel_id: str,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build Slack API payload from normalized message."""
        from aragora.channels.normalized import MessageFormat

        payload: dict[str, Any] = {
            "channel": channel_id,
        }

        # Handle thread replies
        thread_ts = kwargs.get("thread_ts") or message.thread_id
        if thread_ts:
            payload["thread_ts"] = thread_ts

        # Build blocks for rich formatting
        blocks = []

        # Add title if present
        if message.title:
            blocks.append(
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": message.title[:150]},
                }
            )

        # Add main content
        if message.content:
            # Slack uses mrkdwn format
            text = message.content
            if message.format == MessageFormat.PLAIN:
                # Plain text doesn't need transformation
                pass
            elif message.format == MessageFormat.MARKDOWN:
                # Slack mrkdwn is slightly different from standard markdown
                # Bold: **text** -> *text*
                text = text.replace("**", "*")

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text[:3000]},
                }
            )

            # Also set plain text for notifications
            payload["text"] = message.to_plain_text()[:3000]

        # Add buttons if present
        if message.has_buttons():
            button_elements = []
            for button in message.buttons[:5]:  # Slack limits to 5 buttons
                if isinstance(button, dict):
                    label = button.get("label", "Click")
                    action = button.get("action", "")
                else:
                    label = getattr(button, "label", "Click")
                    action = getattr(button, "action", "")

                if action.startswith("http"):
                    button_elements.append(
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": label[:75]},
                            "url": action,
                        }
                    )
                else:
                    button_elements.append(
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": label[:75]},
                            "action_id": action,
                        }
                    )

            if button_elements:
                blocks.append(
                    {
                        "type": "actions",
                        "elements": button_elements,
                    }
                )

        if blocks:
            payload["blocks"] = blocks

        return payload
