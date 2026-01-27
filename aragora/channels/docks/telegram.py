"""
Telegram Dock - Channel dock implementation for Telegram.

Handles message delivery to Telegram chats via the Telegram Bot API.

Example:
    from aragora.channels.docks.telegram import TelegramDock

    dock = TelegramDock({"token": "123456:ABC..."})
    await dock.initialize()
    await dock.send_message(chat_id, message)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult

if TYPE_CHECKING:
    from aragora.channels.normalized import NormalizedMessage

logger = logging.getLogger(__name__)

__all__ = ["TelegramDock"]


class TelegramDock(ChannelDock):
    """
    Telegram platform dock.

    Supports markdown formatting, inline buttons, voice messages,
    and file uploads via the Telegram Bot API.
    """

    PLATFORM = "telegram"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT
        | ChannelCapability.BUTTONS
        | ChannelCapability.VOICE
        | ChannelCapability.FILES
        | ChannelCapability.INLINE_IMAGES
    )

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize Telegram dock.

        Config options:
            token: Telegram bot token (or use TELEGRAM_BOT_TOKEN env var)
        """
        super().__init__(config)
        self._token: Optional[str] = None

    async def initialize(self) -> bool:
        """Initialize the Telegram dock."""
        self._token = self.config.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not self._token:
            logger.warning("Telegram token not configured")
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
        Send a message to Telegram.

        Args:
            channel_id: Telegram chat ID
            message: The normalized message to send
            **kwargs: Additional options (reply_to_message_id, etc.)

        Returns:
            SendResult indicating success or failure
        """
        if not self._token:
            return SendResult.fail(
                error="Telegram token not configured",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        # Check for voice message
        audio = message.get_audio_attachment()
        if audio:
            return await self._send_voice(channel_id, audio, message, **kwargs)

        try:
            import httpx

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = self._build_payload(channel_id, message, **kwargs)

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        result_msg = data.get("result", {})
                        return SendResult.ok(
                            message_id=str(result_msg.get("message_id", "")),
                            platform=self.PLATFORM,
                            channel_id=channel_id,
                        )
                    else:
                        return SendResult.fail(
                            error=data.get("description", "Unknown Telegram error"),
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
            logger.error(f"Telegram send error: {e}")
            return SendResult.fail(
                error=str(e),
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

    async def _send_voice(
        self,
        channel_id: str,
        audio: Any,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> SendResult:
        """Send a voice message to Telegram."""
        try:
            import httpx

            url = f"https://api.telegram.org/bot{self._token}/sendVoice"

            # Get audio data
            from aragora.channels.normalized import MessageAttachment

            audio_data = audio.data if isinstance(audio, MessageAttachment) else audio.get("data")

            if not audio_data:
                return SendResult.fail(
                    error="No audio data provided",
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )

            async with httpx.AsyncClient() as client:
                files = {"voice": ("voice.ogg", audio_data, "audio/ogg")}
                data = {"chat_id": channel_id}

                if message.content:
                    data["caption"] = message.content[:1024]

                if message.reply_to or kwargs.get("reply_to_message_id"):
                    data["reply_to_message_id"] = message.reply_to or kwargs.get(
                        "reply_to_message_id"
                    )

                response = await client.post(url, data=data, files=files, timeout=60.0)

                if response.status_code == 200:
                    data_resp = response.json()
                    if data_resp.get("ok"):
                        return SendResult.ok(
                            message_id=str(data_resp.get("result", {}).get("message_id", "")),
                            platform=self.PLATFORM,
                            channel_id=channel_id,
                        )

                return SendResult.fail(
                    error=f"HTTP {response.status_code}",
                    platform=self.PLATFORM,
                    channel_id=channel_id,
                )

        except Exception as e:
            logger.error(f"Telegram voice send error: {e}")
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
        """Build Telegram API payload from normalized message."""
        from aragora.channels.normalized import MessageFormat

        payload: dict[str, Any] = {
            "chat_id": channel_id,
        }

        # Build text content
        text_parts = []
        if message.title:
            text_parts.append(f"*{message.title}*")
        if message.content:
            text_parts.append(message.content)

        text = "\n\n".join(text_parts)

        # Telegram supports MarkdownV2 or HTML
        if message.format in (MessageFormat.MARKDOWN, MessageFormat.PLAIN):
            payload["text"] = text[:4096]
            payload["parse_mode"] = "Markdown"
        elif message.format == MessageFormat.HTML:
            payload["text"] = text[:4096]
            payload["parse_mode"] = "HTML"
        else:
            payload["text"] = message.to_plain_text()[:4096]

        # Handle reply
        reply_to = kwargs.get("reply_to_message_id") or message.reply_to
        if reply_to:
            payload["reply_to_message_id"] = reply_to

        # Add inline keyboard for buttons
        if message.has_buttons():
            keyboard = []
            for button in message.buttons[:10]:  # Telegram limits
                btn = button if hasattr(button, "label") else type("B", (), button)()
                label = (
                    getattr(btn, "label", btn.get("label", "Click"))
                    if isinstance(button, dict)
                    else btn.label
                )
                action = (
                    getattr(btn, "action", btn.get("action", ""))
                    if isinstance(button, dict)
                    else btn.action
                )

                if action.startswith("http"):
                    keyboard.append([{"text": label, "url": action}])
                else:
                    keyboard.append([{"text": label, "callback_data": action[:64]}])

            if keyboard:
                payload["reply_markup"] = {"inline_keyboard": keyboard}

        return payload
