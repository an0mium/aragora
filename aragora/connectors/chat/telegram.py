"""
Telegram Bot Connector.

Implements ChatPlatformConnector for Telegram using the Bot API.
Includes circuit breaker protection for fault tolerance.

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

    All HTTP operations include circuit breaker protection for fault tolerance.
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
        """Send a message to a Telegram chat.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error,
            )

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

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{self._api_base}/sendMessage",
                    json=payload,
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    return SendMessageResponse(
                        success=False,
                        error=data.get("description", "Rate limit exceeded"),
                    )

                if not data.get("ok"):
                    logger.error(f"Telegram send failed: {data.get('description')}")
                    self._record_failure(Exception(data.get("description", "Unknown error")))
                    return SendMessageResponse(
                        success=False,
                        error=data.get("description", "Unknown error"),
                    )

                self._record_success()
                result = data.get("result", {})
                return SendMessageResponse(
                    success=True,
                    message_id=str(result.get("message_id")),
                    channel_id=str(result.get("chat", {}).get("id")),
                    timestamp=datetime.fromtimestamp(result.get("date", 0)).isoformat(),
                )
        except Exception as e:
            self._record_failure(e)
            raise

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Edit an existing message.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error,
            )

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

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{self._api_base}/editMessageText",
                    json=payload,
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    return SendMessageResponse(
                        success=False,
                        error=data.get("description", "Rate limit exceeded"),
                    )

                if not data.get("ok"):
                    self._record_failure(Exception(data.get("description", "Unknown error")))
                    return SendMessageResponse(
                        success=False,
                        error=data.get("description", "Unknown error"),
                    )

                self._record_success()
                return SendMessageResponse(
                    success=True,
                    message_id=message_id,
                    channel_id=channel_id,
                )
        except Exception as e:
            self._record_failure(e)
            raise

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a message.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.warning(f"Circuit breaker open: {cb_error}")
            return False

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{self._api_base}/deleteMessage",
                    json={
                        "chat_id": channel_id,
                        "message_id": int(message_id),
                    },
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    return False

                if data.get("ok"):
                    self._record_success()
                    return True
                else:
                    self._record_failure(Exception(data.get("description", "Delete failed")))
                    return False
        except Exception as e:
            self._record_failure(e)
            raise

    async def upload_file(  # type: ignore[override]
        self,
        channel_id: str,
        file_path: str,
        filename: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload a file as a document.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error)

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout * 2) as client:
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

                    # Check for rate limit (429)
                    if result.get("error_code") == 429:
                        self._record_failure(Exception("Rate limit exceeded (429)"))
                        raise RuntimeError(result.get("description", "Rate limit exceeded"))

                    if not result.get("ok"):
                        self._record_failure(Exception(result.get("description", "Upload failed")))
                        raise RuntimeError(result.get("description", "Upload failed"))

                    self._record_success()
                    doc = result.get("result", {}).get("document", {})
                    return FileAttachment(
                        id=doc.get("file_id", ""),
                        filename=doc.get("file_name", filename or ""),
                        size=doc.get("file_size", 0),
                        content_type=doc.get("mime_type", "application/octet-stream"),
                    )
        except Exception as e:
            self._record_failure(e)
            raise

    async def download_file(  # type: ignore[override]
        self,
        file_id: str,
        **kwargs: Any,
    ) -> bytes:
        """Download a file by file_id.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error)

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout * 2) as client:
                # Get file path
                response = await client.get(
                    f"{self._api_base}/getFile",
                    params={"file_id": file_id},
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    raise RuntimeError(data.get("description", "Rate limit exceeded"))

                if not data.get("ok"):
                    self._record_failure(Exception(data.get("description", "Failed to get file")))
                    raise RuntimeError(data.get("description", "Failed to get file"))

                file_path = data.get("result", {}).get("file_path")
                if not file_path:
                    self._record_failure(Exception("No file path returned"))
                    raise RuntimeError("No file path returned")

                # Download file
                download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
                response = await client.get(download_url)
                self._record_success()
                return response.content
        except Exception as e:
            self._record_failure(e)
            raise

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
                timestamp=datetime.fromtimestamp(msg.get("date", 0)),
                raw_payload=payload,
                metadata={
                    "channel_id": str(msg.get("chat", {}).get("id")),
                    "user_id": str(msg.get("from", {}).get("id")),
                    "message_id": str(msg.get("message_id")),
                },
            )

        # Handle callback queries (button clicks)
        if "callback_query" in payload:
            query = payload["callback_query"]
            return WebhookEvent(
                event_type="callback_query",
                platform="telegram",
                timestamp=datetime.now(),
                raw_payload=payload,
                metadata={
                    "channel_id": str(query.get("message", {}).get("chat", {}).get("id")),
                    "user_id": str(query.get("from", {}).get("id")),
                    "message_id": str(query.get("message", {}).get("message_id")),
                },
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
            thread_id=(
                str(msg.get("reply_to_message", {}).get("message_id"))
                if msg.get("reply_to_message")
                else None
            ),
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
        chat_data = msg.get("chat", {})

        # Build user and channel objects
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

        return UserInteraction(
            id=query.get("id", ""),
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id=query.get("data", ""),
            value=query.get("data", ""),
            user=user,
            channel=channel,
            message_id=str(msg.get("message_id")),
            platform="telegram",
            metadata={"raw_data": query},
        )

    async def send_voice_message(  # type: ignore[override]
        self,
        channel_id: str,
        audio_data: bytes,
        duration: Optional[int] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error,
            )

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout * 2) as client:
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

                # Check for rate limit (429)
                if result.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    return SendMessageResponse(
                        success=False,
                        error=result.get("description", "Rate limit exceeded"),
                    )

                if not result.get("ok"):
                    self._record_failure(Exception(result.get("description", "Voice send failed")))
                    return SendMessageResponse(
                        success=False,
                        error=result.get("description", "Voice send failed"),
                    )

                self._record_success()
                msg = result.get("result", {})
                return SendMessageResponse(
                    success=True,
                    message_id=str(msg.get("message_id")),
                    channel_id=channel_id,
                    timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
                )
        except Exception as e:
            self._record_failure(e)
            raise

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message."""
        return await self.download_file(voice_message.file.id)

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> ChatChannel:
        """Get information about a chat.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error)

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.get(
                    f"{self._api_base}/getChat",
                    params={"chat_id": channel_id},
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    raise RuntimeError(data.get("description", "Rate limit exceeded"))

                if not data.get("ok"):
                    self._record_failure(Exception(data.get("description", "Failed to get chat")))
                    raise RuntimeError(data.get("description", "Failed to get chat"))

                self._record_success()
                chat = data.get("result", {})
                chat_type = chat.get("type", "private")
                return ChatChannel(
                    id=str(chat.get("id")),
                    platform="telegram",
                    name=chat.get("title") or chat.get("username") or "",
                    is_private=chat_type == "private",
                    is_dm=chat_type == "private",
                    metadata={"type": chat_type, "member_count": chat.get("member_count")},
                )
        except Exception as e:
            self._record_failure(e)
            raise

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser:
        """Get information about a user (limited in Telegram)."""
        # Telegram doesn't have a direct getUser API
        # User info typically comes from message updates
        return ChatUser(
            id=user_id,
            platform="telegram",
        )

    async def extract_evidence(
        self,
        message: ChatMessage,
        **kwargs: Any,
    ) -> ChatEvidence:
        """Extract evidence from a message for debate."""
        import hashlib

        # Generate unique ID
        evidence_id = hashlib.sha256(
            f"telegram:{message.channel.id}:{message.id}".encode()
        ).hexdigest()[:16]

        return ChatEvidence(
            id=evidence_id,
            source_type="chat",
            source_id=message.id,
            platform="telegram",
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            content=message.content,
            title=message.content[:100] if message.content else "",
            author_id=message.author.id,
            author_name=message.author.display_name or message.author.username,
            timestamp=message.timestamp,
            source_message=message,
            metadata={
                "raw": message.metadata,
                "permalink": f"https://t.me/c/{message.channel.id}/{message.id}",
            },
        )

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Answer a callback query (acknowledge button click).

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.warning(f"Circuit breaker open: {cb_error}")
            return False

        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{self._api_base}/answerCallbackQuery",
                    json=payload,
                )
                data = response.json()

                # Check for rate limit (429)
                if data.get("error_code") == 429:
                    self._record_failure(Exception("Rate limit exceeded (429)"))
                    return False

                if data.get("ok"):
                    self._record_success()
                    return True
                else:
                    self._record_failure(
                        Exception(data.get("description", "Failed to answer callback query"))
                    )
                    return False
        except Exception as e:
            self._record_failure(e)
            raise

    def _blocks_to_keyboard(self, blocks: list[dict]) -> Optional[dict]:
        """Convert generic blocks to Telegram inline keyboard."""
        buttons = []

        for block in blocks:
            if block.get("type") == "button":
                buttons.append(
                    {
                        "text": block.get("text", ""),
                        "callback_data": block.get("action_id", block.get("value", "")),
                    }
                )
            elif block.get("type") == "url_button":
                buttons.append(
                    {
                        "text": block.get("text", ""),
                        "url": block.get("url", ""),
                    }
                )

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

    # =========================================================================
    # Required Abstract Method Implementations
    # =========================================================================

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        buttons: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Format content as Telegram-compatible blocks.

        Telegram uses inline keyboards for interactive elements.
        This method converts generic block structure to Telegram format.
        """
        result = []

        # Telegram doesn't have rich text blocks like Slack
        # We use inline keyboard buttons for interactivity
        if buttons:
            result.extend(buttons)

        return result

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Format a button for Telegram inline keyboard."""
        if url:
            return {
                "type": "url_button",
                "text": text,
                "url": url,
            }
        return {
            "type": "button",
            "text": text,
            "action_id": action_id,
            "value": value or action_id,
        }

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Telegram webhook request.

        Telegram uses a secret_token query parameter for verification,
        which is set when configuring the webhook URL. Since this is
        URL-based verification, this method always returns True.

        For enhanced security, the webhook URL itself should contain
        a secret token that the server validates.
        """
        # Telegram webhook verification is done via secret_token in URL
        # This is different from signature-based verification used by Slack
        return True

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse a Telegram webhook payload into a WebhookEvent."""
        payload = json.loads(body) if body else {}

        # Handle message updates
        if "message" in payload:
            msg = payload["message"]
            return WebhookEvent(
                event_type="message",
                platform="telegram",
                timestamp=datetime.fromtimestamp(msg.get("date", 0)),
                raw_payload=payload,
                metadata={
                    "channel_id": str(msg.get("chat", {}).get("id")),
                    "user_id": str(msg.get("from", {}).get("id")),
                    "message_id": str(msg.get("message_id")),
                },
            )

        # Handle edited messages
        if "edited_message" in payload:
            msg = payload["edited_message"]
            return WebhookEvent(
                event_type="message_edited",
                platform="telegram",
                timestamp=datetime.fromtimestamp(msg.get("edit_date", msg.get("date", 0))),
                raw_payload=payload,
                metadata={
                    "channel_id": str(msg.get("chat", {}).get("id")),
                    "user_id": str(msg.get("from", {}).get("id")),
                    "message_id": str(msg.get("message_id")),
                },
            )

        # Handle callback queries (button clicks)
        if "callback_query" in payload:
            query = payload["callback_query"]
            return WebhookEvent(
                event_type="callback_query",
                platform="telegram",
                timestamp=datetime.now(),
                raw_payload=payload,
                metadata={
                    "channel_id": str(query.get("message", {}).get("chat", {}).get("id")),
                    "user_id": str(query.get("from", {}).get("id")),
                    "message_id": str(query.get("message", {}).get("message_id")),
                    "callback_data": query.get("data"),
                },
            )

        # Handle channel posts
        if "channel_post" in payload:
            post = payload["channel_post"]
            return WebhookEvent(
                event_type="channel_post",
                platform="telegram",
                timestamp=datetime.fromtimestamp(post.get("date", 0)),
                raw_payload=payload,
                metadata={
                    "channel_id": str(post.get("chat", {}).get("id")),
                    "message_id": str(post.get("message_id")),
                },
            )

        return WebhookEvent(
            event_type="unknown",
            platform="telegram",
            raw_payload=payload,
        )

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a bot command.

        Telegram doesn't support ephemeral messages, so the ephemeral
        parameter is ignored. Includes circuit breaker protection via send_message.
        """
        channel_id = command.channel.id if command.channel else kwargs.get("channel_id")
        if not channel_id:
            return SendMessageResponse(
                success=False,
                error="No channel ID available for command response",
            )

        # Get reply_to message_id from command metadata
        reply_to = kwargs.get("reply_to") or command.metadata.get("message_id")

        return await self.send_message(
            channel_id=channel_id,
            text=text,
            blocks=blocks,
            thread_id=reply_to,
        )

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Respond to a user interaction (button click).

        If replace_original is True, edits the original message.
        Otherwise, answers the callback query with a notification.
        Includes circuit breaker protection via answer_callback_query and update_message.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Telegram connector")

        # First, acknowledge the callback query
        await self.answer_callback_query(
            interaction.id, text=text if not replace_original else None
        )

        if replace_original and interaction.message_id:
            # Edit the original message
            result = await self.update_message(
                channel_id=interaction.channel.id,
                message_id=interaction.message_id,
                text=text,
                blocks=blocks,
            )
            return result.success

        return True


__all__ = ["TelegramConnector"]
