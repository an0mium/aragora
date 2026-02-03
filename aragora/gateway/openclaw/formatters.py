"""
Channel formatters for OpenClaw message formatting.

Provides channel-specific message formatting for WhatsApp, Telegram,
Slack, and Discord. Custom formatters can be registered at runtime.
"""

from __future__ import annotations

from typing import Any

from .models import OpenClawMessage, OpenClawSession


class ChannelFormatter:
    """Base class for channel-specific message formatting."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for sending to channel."""
        return message.to_dict()

    def parse_incoming(
        self,
        raw_message: dict[str, Any],
        session: OpenClawSession,
    ) -> OpenClawMessage:
        """Parse incoming channel message to OpenClaw format."""
        return OpenClawMessage.from_dict(raw_message)


class WhatsAppFormatter(ChannelFormatter):
    """Formatter for WhatsApp messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for WhatsApp."""
        result: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": session.metadata.get("phone_number"),
            "type": self._map_message_type(message.type),
        }

        if message.type == "text":
            result["text"] = {"body": message.content}
        elif message.type == "image":
            result["image"] = {"link": message.content}
        elif message.type == "audio":
            result["audio"] = {"link": message.content}
        elif message.type == "video":
            result["video"] = {"link": message.content}
        elif message.type == "file":
            result["document"] = {"link": message.content}

        return result

    def _map_message_type(self, msg_type: str) -> str:
        """Map OpenClaw message type to WhatsApp type."""
        type_map = {
            "text": "text",
            "image": "image",
            "audio": "audio",
            "video": "video",
            "file": "document",
        }
        return type_map.get(msg_type, "text")


class TelegramFormatter(ChannelFormatter):
    """Formatter for Telegram messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Telegram."""
        chat_id = session.metadata.get("chat_id")
        result: dict[str, Any] = {"chat_id": chat_id}

        if message.type == "text":
            result["text"] = message.content
            result["parse_mode"] = "HTML"
        elif message.type == "image":
            result["photo"] = message.content
        elif message.type == "audio":
            result["audio"] = message.content
        elif message.type == "video":
            result["video"] = message.content
        elif message.type == "file":
            result["document"] = message.content

        if message.reply_to:
            result["reply_to_message_id"] = message.reply_to

        return result


class SlackFormatter(ChannelFormatter):
    """Formatter for Slack messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Slack."""
        result: dict[str, Any] = {
            "channel": session.metadata.get("channel_id"),
        }

        if message.type == "text":
            result["text"] = message.content
            # Support Slack blocks for rich formatting
            if message.metadata.get("blocks"):
                result["blocks"] = message.metadata["blocks"]
        elif message.type in ("image", "file"):
            result["text"] = message.metadata.get("alt_text", "")
            result["attachments"] = [
                {
                    "fallback": message.metadata.get("alt_text", "attachment"),
                    "image_url" if message.type == "image" else "title_link": message.content,
                }
            ]

        if message.thread_id:
            result["thread_ts"] = message.thread_id

        return result


class DiscordFormatter(ChannelFormatter):
    """Formatter for Discord messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Discord."""
        result: dict[str, Any] = {}

        if message.type == "text":
            result["content"] = message.content
            if message.metadata.get("embeds"):
                result["embeds"] = message.metadata["embeds"]
        elif message.type == "image":
            result["embeds"] = [{"image": {"url": message.content}}]
        elif message.type == "file":
            result["content"] = message.metadata.get("description", "")
            result["attachments"] = [{"url": message.content}]

        return result
