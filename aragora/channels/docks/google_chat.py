"""
Google Chat Dock - Channel dock implementation for Google Chat.

Handles message delivery to Google Chat via the Chat API.

Example:
    from aragora.channels.docks.google_chat import GoogleChatDock

    dock = GoogleChatDock()
    await dock.initialize()
    await dock.send_message(space_name, message)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult

if TYPE_CHECKING:
    from aragora.channels.normalized import NormalizedMessage

logger = logging.getLogger(__name__)

__all__ = ["GoogleChatDock"]


class GoogleChatDock(ChannelDock):
    """
    Google Chat platform dock.

    Supports Card v2 format, buttons, threads, and file uploads
    via the Google Chat API.
    """

    PLATFORM = "google_chat"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT
        | ChannelCapability.BUTTONS
        | ChannelCapability.THREADS
        | ChannelCapability.FILES
        | ChannelCapability.CARDS
    )

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize Google Chat dock.

        Config options:
            credentials_file: Path to service account credentials
        """
        super().__init__(config)
        self._connector: Any = None

    async def initialize(self) -> bool:
        """Initialize the Google Chat dock."""
        try:
            from aragora.server.handlers.bots.google_chat import get_google_chat_connector

            self._connector = get_google_chat_connector()
            if self._connector:
                self._initialized = True
                return True
            else:
                logger.warning("Google Chat connector not configured")
                return False
        except ImportError as e:
            logger.warning(f"Google Chat connector not available: {e}")
            return False

    async def send_message(
        self,
        channel_id: str,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a message to Google Chat.

        Args:
            channel_id: Google Chat space name (format: "spaces/XXXXX")
            message: The normalized message to send
            **kwargs: Additional options (thread_name, etc.)

        Returns:
            SendResult indicating success or failure
        """
        if not self._connector:
            return SendResult.fail(
                error="Google Chat connector not configured",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        try:
            space_name = channel_id
            thread_name = kwargs.get("thread_name") or kwargs.get("thread_id") or message.thread_id

            # Build card sections
            sections = self._build_card_sections(message)

            # Send via connector
            text_fallback = message.to_plain_text()[:500]
            response = await self._connector.send_message(
                space_name,
                text_fallback,
                blocks=sections,
                thread_id=thread_name,
            )

            if response.success:
                return SendResult.ok(
                    message_id=getattr(response, "message_id", None),
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )
            else:
                return SendResult.fail(
                    error=getattr(response, "error", "Unknown error"),
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )

        except Exception as e:
            logger.error(f"Google Chat send error: {e}")
            return SendResult.fail(
                error=str(e),
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

    def _build_card_sections(self, message: "NormalizedMessage") -> list[dict[str, Any]]:
        """Build Google Chat Card v2 sections from normalized message."""
        sections: list[dict[str, Any]] = []

        # Add header if we have a title
        if message.title:
            sections.append({"header": message.title[:100]})

        # Add main content
        if message.content:
            text = message.to_plain_text()[:4000]
            sections.append({"widgets": [{"textParagraph": {"text": text}}]})

        # Add buttons
        if message.has_buttons():
            buttons = []
            for button in message.buttons[:5]:  # Google Chat limits
                if isinstance(button, dict):
                    label = button.get("label", "Click")
                    action = button.get("action", "")
                else:
                    label = getattr(button, "label", "Click")
                    action = getattr(button, "action", "")

                if action.startswith("http"):
                    buttons.append(
                        {
                            "text": label[:50],
                            "onClick": {"openLink": {"url": action}},
                        }
                    )
                else:
                    buttons.append(
                        {
                            "text": label[:50],
                            "onClick": {
                                "action": {
                                    "function": action[:64],
                                }
                            },
                        }
                    )

            if buttons:
                sections.append({"widgets": [{"buttonList": {"buttons": buttons}}]})

        return sections

    async def send_result(
        self,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a debate result to Google Chat with Card v2 formatting."""
        if not self._connector:
            return SendResult.fail(
                error="Google Chat connector not configured",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        try:
            space_name = channel_id
            thread_name = thread_id or kwargs.get("thread_name")

            # Format result data
            consensus = result.get("consensus_reached", False)
            answer = result.get("final_answer", "No conclusion reached.")
            confidence = result.get("confidence", 0)
            topic = result.get("task", kwargs.get("topic", "Unknown topic"))
            debate_id = result.get("id", kwargs.get("debate_id", ""))

            # Truncate long answers
            if len(answer) > 500:
                answer = answer[:500] + "..."

            # Build Card v2 sections
            consensus_emoji = "✅" if consensus else "❌"
            confidence_bar = "█" * int(confidence * 5) + "░" * (5 - int(confidence * 5))

            sections = [
                {"header": f"{consensus_emoji} Debate Complete"},
                {"widgets": [{"textParagraph": {"text": f"<b>Topic:</b> {topic[:200]}"}}]},
                {
                    "widgets": [
                        {
                            "decoratedText": {
                                "topLabel": "Consensus",
                                "text": "Yes" if consensus else "No",
                            }
                        },
                        {
                            "decoratedText": {
                                "topLabel": "Confidence",
                                "text": f"{confidence_bar} {confidence:.0%}",
                            }
                        },
                    ]
                },
                {"widgets": [{"textParagraph": {"text": f"<b>Conclusion:</b>\n{answer}"}}]},
            ]

            # Add vote buttons
            sections.append(
                {
                    "widgets": [
                        {
                            "buttonList": {
                                "buttons": [
                                    {
                                        "text": "👍 Agree",
                                        "onClick": {
                                            "action": {
                                                "function": "vote_agree",
                                                "parameters": [
                                                    {"key": "debate_id", "value": debate_id}
                                                ],
                                            }
                                        },
                                    },
                                    {
                                        "text": "👎 Disagree",
                                        "onClick": {
                                            "action": {
                                                "function": "vote_disagree",
                                                "parameters": [
                                                    {"key": "debate_id", "value": debate_id}
                                                ],
                                            }
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                }
            )

            # Send message with card
            response = await self._connector.send_message(
                space_name,
                f"Debate complete: {topic[:50]}...",
                blocks=sections,
                thread_id=thread_name,
            )

            if response.success:
                return SendResult.ok(
                    message_id=getattr(response, "message_id", None),
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )
            else:
                return SendResult.fail(
                    error=getattr(response, "error", "Unknown error"),
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )

        except Exception as e:
            logger.error(f"Google Chat result send error: {e}")
            return SendResult.fail(
                error=str(e),
                platform=self.PLATFORM,
                channel_id=channel_id,
            )
