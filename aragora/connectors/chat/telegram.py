"""
Telegram Bot Connector.

Implements ChatPlatformConnector for Telegram using the Bot API.

Environment Variables:
- TELEGRAM_BOT_TOKEN: Bot API token from @BotFather
- TELEGRAM_WEBHOOK_URL: Webhook URL for receiving updates
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import ChatPlatformConnector
from .models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    MessageType,
    SendMessageResponse,
    UserInteraction,
    VoiceMessage,
    WebhookEvent,
)

# Environment configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL", "")

# Telegram API
TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramConnector(ChatPlatformConnector):
    """
    Telegram connector using Bot API.

    Supports:
    - Sending messages with Markdown/HTML formatting
    - Inline keyboards (buttons)
    - File uploads (documents, photos, voice)
    - Reply messages (threads)
    - Callback queries (button interactions)
    - Webhook and long-polling
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        webhook_url: Optional[str] = None,
        parse_mode: str = "MarkdownV2",
        **config: Any,
    ):
        """
        Initialize Telegram connector.

        Args:
            bot_token: Bot API token from @BotFather
            webhook_url: Webhook URL for receiving updates
            parse_mode: Default parse mode (Markdown, MarkdownV2, HTML)
            **config: Additional configuration
        """
        super().__init__(
            bot_token=bot_token or TELEGRAM_BOT_TOKEN,
            webhook_url=webhook_url or TELEGRAM_WEBHOOK_URL,
            **config,
        )
        self.parse_mode = parse_mode
        self._api_base = f"{TELEGRAM_API_BASE}{self.bot_token}"

    @property
    def platform_name(self) -> str:
        return "telegram"

    @property
    def platform_display_name(self) -> str:
        return "Telegram"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a message to a Telegram chat."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        payload: dict[str, Any] = {
            "chat_id": channel_id,
            "text": self._escape_markdown(text) if self.parse_mode.startswith("Markdown") else text,
            "parse_mode": self.parse_mode,
        }

        # Reply to specific message (thread)
        if thread_id:
            payload["reply_to_message_id"] = int(thread_id)

        # Convert blocks to inline keyboard
        if blocks:
            keyboard = self._blocks_to_keyboard(blocks)
            if keyboard:
                payload["reply_markup"] = json.dumps(keyboard)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._api_base}/sendMessage",
                json=payload,
            )
            data = response.json()

            if not data.get("ok"):
                logger.error(f"Telegram send failed: {data.get('description')}")
                return SendMessageResponse(
                    success=False,
                    error=data.get("description", "Unknown error"),
                )

            result = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(result.get("message_id")),
                channel_id=str(result.get("chat", {}).get("id")),
                timestamp=datetime.fromtimestamp(result.get("date", 0)),
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Edit an existing message."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        payload: dict[str, Any] = {
            "chat_id": channel_id,
            "message_id": int(message_id),
            "text": self._escape_markdown(text) if self.parse_mode.startswith("Markdown") else text,
            "parse_mode": self.parse_mode,
        }

        if blocks:
            keyboard = self._blocks_to_keyboard(blocks)
            if keyboard:
                payload["reply_markup"] = json.dumps(keyboard)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._api_base}/editMessageText",
                json=payload,
            )
            data = response.json()

            if not data.get("ok"):
                return SendMessageResponse(
                    success=False,
                    error=data.get("description", "Unknown error"),
                )

            return SendMessageResponse(
                success=True,
                message_id=message_id,
                channel_id=channel_id,
            )

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a message."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._api_base}/deleteMessage",
                json={
                    "chat_id": channel_id,
                    "message_id": int(message_id),
                },
            )
            data = response.json()
            return data.get("ok", False)

    async def upload_file(
        self,
        channel_id: str,
        file_path: str,
        filename: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload a file as a document."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                files = {"document": (filename or file_path.split("/")[-1], f)}
                data = {"chat_id": channel_id}
                if comment:
                    data["caption"] = comment

                response = await client.post(
                    f"{self._api_base}/sendDocument",
                    data=data,
                    files=files,
                )
                result = response.json()

                if not result.get("ok"):
                    raise RuntimeError(result.get("description", "Upload failed"))

                doc = result.get("result", {}).get("document", {})
                return FileAttachment(
                    file_id=doc.get("file_id", ""),
                    filename=doc.get("file_name", filename or ""),
                    size_bytes=doc.get("file_size", 0),
                    mime_type=doc.get("mime_type", "application/octet-stream"),
                )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> bytes:
        """Download a file by file_id."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get file path
            response = await client.get(
                f"{self._api_base}/getFile",
                params={"file_id": file_id},
            )
            data = response.json()

            if not data.get("ok"):
                raise RuntimeError(data.get("description", "Failed to get file"))

            file_path = data.get("result", {}).get("file_path")
            if not file_path:
                raise RuntimeError("No file path returned")

            # Download file
            download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            response = await client.get(download_url)
            return response.content

    async def handle_webhook(
        self,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> WebhookEvent:
        """Process incoming Telegram webhook update."""
        # Handle message updates
        if "message" in payload:
            msg = payload["message"]
            return WebhookEvent(
                event_type="message",
                platform="telegram",
                channel_id=str(msg.get("chat", {}).get("id")),
                user_id=str(msg.get("from", {}).get("id")),
                message_id=str(msg.get("message_id")),
                timestamp=datetime.fromtimestamp(msg.get("date", 0)),
                raw_payload=payload,
            )

        # Handle callback queries (button clicks)
        if "callback_query" in payload:
            query = payload["callback_query"]
            return WebhookEvent(
                event_type="callback_query",
                platform="telegram",
                channel_id=str(query.get("message", {}).get("chat", {}).get("id")),
                user_id=str(query.get("from", {}).get("id")),
                message_id=str(query.get("message", {}).get("message_id")),
                timestamp=datetime.now(),
                raw_payload=payload,
            )

        return WebhookEvent(
            event_type="unknown",
            platform="telegram",
            raw_payload=payload,
        )

    async def parse_message(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> ChatMessage:
        """Parse a Telegram message into ChatMessage."""
        msg = payload.get("message", payload)

        user_data = msg.get("from", {})
        chat_data = msg.get("chat", {})

        # Determine message type
        msg_type = MessageType.TEXT
        if "voice" in msg:
            msg_type = MessageType.VOICE
        elif "document" in msg:
            msg_type = MessageType.FILE
        elif "photo" in msg:
            msg_type = MessageType.FILE  # Images treated as files

        # Build proper ChatChannel and ChatUser objects
        channel = ChatChannel(
            id=str(chat_data.get("id")),
            platform="telegram",
            name=chat_data.get("title") or chat_data.get("username"),
            is_private=chat_data.get("type") == "private",
            is_dm=chat_data.get("type") == "private",
        )
        author = ChatUser(
            id=str(user_data.get("id")),
            platform="telegram",
            username=user_data.get("username", ""),
            display_name=f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
        )

        return ChatMessage(
            id=str(msg.get("message_id")),
            platform="telegram",
            channel=channel,
            author=author,
            content=msg.get("text", msg.get("caption", "")),
            message_type=msg_type,
            timestamp=datetime.fromtimestamp(msg.get("date", 0)),
            thread_id=str(msg.get("reply_to_message", {}).get("message_id")) if msg.get("reply_to_message") else None,
            metadata={"raw_data": msg},
        )

    async def parse_command(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[BotCommand]:
        """Parse a Telegram command (e.g., /start, /help)."""
        msg = payload.get("message", payload)
        text = msg.get("text", "")

        if not text.startswith("/"):
            return None

        # Parse command and args
        parts = text.split()
        command_name = parts[0][1:]  # Remove leading /

        # Handle @bot suffix (e.g., /help@mybot)
        if "@" in command_name:
            command_name = command_name.split("@")[0]

        args = parts[1:] if len(parts) > 1 else []

        # Build user and channel objects
        user_data = msg.get("from", {})
        chat_data = msg.get("chat", {})
        user = ChatUser(
            id=str(user_data.get("id")),
            platform="telegram",
            username=user_data.get("username", ""),
            display_name=f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
        )
        channel = ChatChannel(
            id=str(chat_data.get("id")),
            platform="telegram",
            name=chat_data.get("title") or chat_data.get("username"),
        )

        return BotCommand(
            name=command_name,
            text=text,
            args=args,
            user=user,
            channel=channel,
            platform="telegram",
            metadata={"message_id": str(msg.get("message_id")), "raw_data": msg},
        )

    async def handle_interaction(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> UserInteraction:
        """Handle callback query (button click)."""
        query = payload.get("callback_query", {})
        user_data = query.get("from", {})
        msg = query.get("message", {})

        return UserInteraction(
            interaction_type=InteractionType.BUTTON_CLICK,
            user_id=str(user_data.get("id")),
            channel_id=str(msg.get("chat", {}).get("id")),
            message_id=str(msg.get("message_id")),
            action_id=query.get("data", ""),
            action_value=query.get("data", ""),
            platform="telegram",
            raw_data=query,
        )

    async def send_voice_message(
        self,
        channel_id: str,
        audio_data: bytes,
        duration: Optional[int] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"voice": ("voice.ogg", audio_data, "audio/ogg")}
            data = {"chat_id": channel_id}
            if duration:
                data["duration"] = str(duration)

            response = await client.post(
                f"{self._api_base}/sendVoice",
                data=data,
                files=files,
            )
            result = response.json()

            if not result.get("ok"):
                return SendMessageResponse(
                    success=False,
                    error=result.get("description", "Voice send failed"),
                )

            msg = result.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=channel_id,
                timestamp=datetime.fromtimestamp(msg.get("date", 0)),
            )

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message."""
        return await self.download_file(voice_message.file_id)

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> ChatChannel:
        """Get information about a chat."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._api_base}/getChat",
                params={"chat_id": channel_id},
            )
            data = response.json()

            if not data.get("ok"):
                raise RuntimeError(data.get("description", "Failed to get chat"))

            chat = data.get("result", {})
            return ChatChannel(
                channel_id=str(chat.get("id")),
                name=chat.get("title") or chat.get("username") or "",
                channel_type=chat.get("type", "private"),
                platform="telegram",
                member_count=chat.get("member_count"),
            )

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser:
        """Get information about a user (limited in Telegram)."""
        # Telegram doesn't have a direct getUser API
        # User info typically comes from message updates
        return ChatUser(
            user_id=user_id,
            platform="telegram",
        )

    async def extract_evidence(
        self,
        message: ChatMessage,
        **kwargs: Any,
    ) -> ChatEvidence:
        """Extract evidence from a message for debate."""
        return ChatEvidence(
            content=message.text,
            source_url=f"https://t.me/c/{message.channel_id}/{message.message_id}",
            author=message.user.display_name or message.user.username,
            timestamp=message.timestamp,
            platform="telegram",
            channel_id=message.channel_id,
            message_id=message.message_id,
            metadata={"raw": message.raw_data},
        )

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Answer a callback query (acknowledge button click)."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._api_base}/answerCallbackQuery",
                json=payload,
            )
            return response.json().get("ok", False)

    def _blocks_to_keyboard(self, blocks: list[dict]) -> Optional[dict]:
        """Convert generic blocks to Telegram inline keyboard."""
        buttons = []

        for block in blocks:
            if block.get("type") == "button":
                buttons.append({
                    "text": block.get("text", ""),
                    "callback_data": block.get("action_id", block.get("value", "")),
                })
            elif block.get("type") == "url_button":
                buttons.append({
                    "text": block.get("text", ""),
                    "url": block.get("url", ""),
                })

        if not buttons:
            return None

        # Group buttons into rows (max 8 buttons per row)
        rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
        return {"inline_keyboard": rows}

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for MarkdownV2."""
        # Characters that need escaping in MarkdownV2
        special_chars = "_*[]()~`>#+-=|{}.!"
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text


__all__ = ["TelegramConnector"]
