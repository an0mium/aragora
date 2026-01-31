"""
Telegram Bot Connector.

Implements ChatPlatformConnector for Telegram using the Bot API.
Includes circuit breaker protection for fault tolerance.

Environment Variables:
- TELEGRAM_BOT_TOKEN: Bot API token from @BotFather
- TELEGRAM_WEBHOOK_URL: Webhook URL for receiving updates
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from aragora.connectors.exceptions import (
    classify_connector_error,
)

# Distributed tracing support
try:
    from aragora.observability.tracing import build_trace_headers

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def build_trace_headers() -> dict[str, str]:
        return {}


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

    All HTTP operations include circuit breaker protection for fault tolerance.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        webhook_url: str | None = None,
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

    async def _telegram_api_request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        operation: str = "api_call",
        *,
        method: str = "POST",
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """
        Make a Telegram API request with circuit breaker, retry, and timeout.

        Centralizes the resilience pattern for all Telegram API calls.

        Args:
            endpoint: API endpoint (e.g., "sendMessage", "getMe")
            payload: JSON payload to send
            operation: Operation name for logging
            method: HTTP method - "GET" or "POST" (default: "POST")
            files: File data for multipart uploads
            timeout: Optional timeout override
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (success, response_data, error_message)
        """
        import asyncio
        import random

        if not HTTPX_AVAILABLE:
            return False, None, "httpx not available"

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return False, None, cb_error

        last_error: str | None = None
        url = f"{self._api_base}/{endpoint}"
        request_timeout = timeout if timeout is not None else self._request_timeout

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    if files:
                        # Multipart form data for file uploads
                        response = await client.post(
                            url,
                            data=payload,
                            files=files,
                            headers=build_trace_headers(),
                        )
                    elif method.upper() == "GET":
                        response = await client.get(
                            url,
                            params=payload,
                            headers=build_trace_headers(),
                        )
                    else:
                        response = await client.post(
                            url,
                            json=payload,
                            headers=build_trace_headers(),
                        )

                    data = response.json()

                    # Check for rate limit (429)
                    if data.get("error_code") == 429:
                        retry_after = data.get("parameters", {}).get("retry_after", 60)
                        error_desc = data.get("description", "Rate limit exceeded")
                        last_error = error_desc

                        if attempt < max_retries - 1:
                            logger.warning(
                                f"[telegram] {operation} rate limited, "
                                f"retry in {retry_after}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(min(retry_after, 60))
                            continue

                        classified = classify_connector_error(
                            error_desc, "telegram", status_code=429, retry_after=retry_after
                        )
                        self._record_failure(classified)
                        return False, None, error_desc

                    if not data.get("ok"):
                        error_desc = data.get("description", "Unknown error")
                        error_code = data.get("error_code")
                        last_error = error_desc

                        # Check if retryable
                        if error_code in {500, 502, 503, 504} and attempt < max_retries - 1:
                            delay = min(1.0 * (2**attempt), 30.0)
                            jitter = random.uniform(0, delay * 0.1)
                            logger.warning(
                                f"[telegram] {operation} server error {error_code} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay + jitter)
                            continue

                        classified = classify_connector_error(
                            error_desc, "telegram", status_code=error_code
                        )
                        logger.error(
                            f"[telegram] {operation} failed [{type(classified).__name__}]: {error_desc}"
                        )
                        self._record_failure(classified)
                        return False, data, error_desc

                    self._record_success()
                    return True, data, None

            except httpx.TimeoutException:
                last_error = f"Request timeout after {request_timeout}s"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[telegram] {operation} timeout (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.error(f"[telegram] {operation} timeout after {max_retries} attempts")

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[telegram] {operation} network error (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.error(
                    f"[telegram] {operation} network error after {max_retries} attempts: {e}"
                )

            except (httpx.RequestError, OSError, ValueError, RuntimeError, TypeError) as e:
                # Unexpected error - don't retry
                last_error = f"Unexpected error: {e}"
                classified = classify_connector_error(last_error, "telegram")
                logger.exception(
                    f"[telegram] {operation} unexpected error [{type(classified).__name__}]: {e}"
                )
                break

        classified = classify_connector_error(last_error or "Unknown error", "telegram")
        self._record_failure(classified)
        return False, None, last_error or "Unknown error"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a message to a Telegram chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        """
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

        success, data, error = await self._telegram_api_request(
            "sendMessage",
            payload=payload,
            operation="send_message",
        )

        if success and data:
            result = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(result.get("message_id")),
                channel_id=str(result.get("chat", {}).get("id")),
                timestamp=datetime.fromtimestamp(result.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Edit an existing message.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        """
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

        success, _, error = await self._telegram_api_request(
            "editMessageText",
            payload=payload,
            operation="update_message",
        )

        if success:
            return SendMessageResponse(
                success=True,
                message_id=message_id,
                channel_id=channel_id,
            )

        return SendMessageResponse(success=False, error=error)

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a message.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        """
        success, _, error = await self._telegram_api_request(
            "deleteMessage",
            payload={
                "chat_id": channel_id,
                "message_id": int(message_id),
            },
            operation="delete_message",
        )

        if not success and error:
            logger.warning(f"Delete message failed: {error}")

        return success

    async def send_typing_indicator(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> bool:
        """Send typing indicator (chat action) to a Telegram chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        Typing indicators expire after ~5 seconds.
        """
        success, _, error = await self._telegram_api_request(
            "sendChatAction",
            payload={
                "chat_id": channel_id,
                "action": "typing",
            },
            operation="send_typing_indicator",
            max_retries=1,  # Don't retry typing indicators
        )

        if not success:
            logger.debug(f"Telegram typing indicator failed: {error}")

        return success

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload a file as a document.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target channel
            content: File content as bytes
            filename: Name of the file
            content_type: MIME type
            title: Optional display title (used as caption)
            thread_id: Optional thread for the file (reply_to_message_id)
            **kwargs: Additional options (comment alias for title)
        """
        actual_filename = filename
        file_content = content
        # Support legacy 'comment' kwarg as alias for title
        comment = kwargs.pop("comment", None) or title

        files = {"document": (actual_filename, file_content, content_type)}
        payload: dict[str, Any] = {"chat_id": channel_id}
        if comment:
            payload["caption"] = comment
        if thread_id:
            payload["reply_to_message_id"] = thread_id

        success, data, error = await self._telegram_api_request(
            "sendDocument",
            payload=payload,
            files=files,
            operation="upload_file",
            timeout=self._request_timeout * 2,
        )

        if success and data:
            doc = data.get("result", {}).get("document", {})
            return FileAttachment(
                id=doc.get("file_id", ""),
                filename=doc.get("file_name", actual_filename),
                size=doc.get("file_size", 0),
                content_type=doc.get("mime_type", "application/octet-stream"),
            )

        raise RuntimeError(error or "Upload failed")

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download a file by file_id.

        Uses _telegram_api_request and _http_request for circuit breaker,
        retry, and timeout handling.

        Args:
            file_id: Telegram file_id to download
            **kwargs: Additional options (url, filename for hints)

        Returns:
            FileAttachment with content populated
        """
        # Step 1: Get file path via API
        success, data, error = await self._telegram_api_request(
            "getFile",
            payload={"file_id": file_id},
            method="GET",
            operation="download_file_info",
            timeout=self._request_timeout * 2,
        )

        if not success or not data:
            raise RuntimeError(error or "Failed to get file info")

        file_info = data.get("result", {})
        file_path = file_info.get("file_path")
        if not file_path:
            raise RuntimeError("No file path returned")

        # Step 2: Download binary content
        download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        dl_success, content, dl_error = await self._http_request(
            method="GET",
            url=download_url,
            headers=build_trace_headers(),
            timeout=self._request_timeout * 2,
            return_raw=True,
            operation="download_file_content",
        )

        if not dl_success or not isinstance(content, bytes):
            raise RuntimeError(dl_error or "Failed to download file")

        # Extract filename from path or use kwargs hint
        filename = kwargs.get("filename") or file_path.split("/")[-1]

        # Guess content type from extension
        content_type = "application/octet-stream"
        if filename.endswith(".ogg") or filename.endswith(".oga"):
            content_type = "audio/ogg"
        elif filename.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        elif filename.endswith(".m4a"):
            content_type = "audio/mp4"

        return FileAttachment(
            id=file_id,
            filename=filename,
            content_type=content_type,
            size=file_info.get("file_size", len(content)),
            url=download_url,
            content=content,
            metadata={"telegram_file_path": file_path},
        )

    async def handle_webhook(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
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
    ) -> BotCommand | None:
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

    async def send_voice_message(
        self,
        channel_id: str,
        audio_content: bytes,
        filename: str = "voice_response.mp3",
        content_type: str = "audio/mpeg",
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target channel
            audio_content: Audio file content as bytes
            filename: Audio filename (default: voice_response.mp3)
            content_type: MIME type (audio/mpeg, audio/ogg, etc.)
            reply_to: Optional message ID to reply to
            **kwargs: Additional options (duration for Telegram-specific duration)
        """
        # Telegram prefers .ogg for voice, but we accept what's given
        # Map common content types to Telegram-friendly format
        actual_content_type = content_type
        actual_filename = filename
        if content_type == "audio/mpeg" and not filename.endswith(".ogg"):
            # Keep as-is, Telegram accepts various formats
            pass

        duration = kwargs.pop("duration", None)
        files = {"voice": (actual_filename, audio_content, actual_content_type)}
        payload: dict[str, Any] = {"chat_id": channel_id}
        if duration:
            payload["duration"] = str(duration)
        if reply_to:
            payload["reply_to_message_id"] = reply_to

        success, data, error = await self._telegram_api_request(
            "sendVoice",
            payload=payload,
            files=files,
            operation="send_voice_message",
            timeout=self._request_timeout * 2,
        )

        if success and data:
            msg = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=channel_id,
                timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message."""
        attachment = await self.download_file(voice_message.file.id)
        return attachment.content or b""

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> ChatChannel:
        """Get information about a chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        """
        success, data, error = await self._telegram_api_request(
            "getChat",
            payload={"chat_id": channel_id},
            method="GET",
            operation="get_channel_info",
        )

        if not success or not data:
            raise RuntimeError(error or "Failed to get chat")

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
        text: str | None = None,
        show_alert: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Answer a callback query (acknowledge button click).

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.
        """
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True

        success, _, error = await self._telegram_api_request(
            "answerCallbackQuery",
            payload=payload,
            operation="answer_callback_query",
        )

        if not success and error:
            logger.warning(f"Failed to answer callback query: {error}")

        return success

    def _blocks_to_keyboard(self, blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
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
        title: str | None = None,
        body: str | None = None,
        fields: list[tuple[str, str] | None] = None,
        actions: list[MessageButton] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format content as Telegram-compatible blocks.

        Telegram uses inline keyboards for interactive elements.
        This method converts generic block structure to Telegram format.

        Args:
            title: Section title/header (unused in Telegram)
            body: Main text content (unused in Telegram)
            fields: List of (label, value) tuples (unused in Telegram)
            actions: List of interactive buttons
            **kwargs: Platform-specific options (buttons as dict list for backwards compat)
        """
        result: list[dict[str, Any]] = []

        # Telegram doesn't have rich text blocks like Slack
        # We use inline keyboard buttons for interactivity

        # Support legacy 'buttons' kwarg for backwards compatibility
        buttons = kwargs.pop("buttons", None)
        if buttons:
            result.extend(buttons)

        # Convert MessageButton objects to dict format
        if actions:
            for action in actions:
                btn_dict = self.format_button(
                    text=action.text,
                    action_id=action.action_id,
                    value=action.value,
                    style=action.style,
                    url=action.url,
                )
                result.append(btn_dict)

        return result

    def format_button(
        self,
        text: str,
        action_id: str,
        value: str | None = None,
        style: str | None = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

        Telegram sends the X-Telegram-Bot-Api-Secret-Token header if a
        secret_token was configured when setting the webhook URL via setWebhook.

        SECURITY: Fails closed in production if TELEGRAM_WEBHOOK_SECRET is not configured.
        """
        webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        if not webhook_secret:
            env = os.environ.get("ARAGORA_ENV", "development").lower()
            is_production = env not in ("development", "dev", "local", "test")
            if is_production:
                logger.error(
                    "SECURITY: TELEGRAM_WEBHOOK_SECRET not configured in production. "
                    "Rejecting webhook to prevent signature bypass."
                )
                return False
            logger.warning(
                "TELEGRAM_WEBHOOK_SECRET not set - skipping verification. "
                "This is only acceptable in development!"
            )
            return True

        secret_header = headers.get(
            "X-Telegram-Bot-Api-Secret-Token",
            headers.get("x-telegram-bot-api-secret-token", ""),
        )
        return hmac.compare_digest(secret_header, webhook_secret)

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
        blocks: list[dict[str, Any] | None] = None,
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
        blocks: list[dict[str, Any] | None] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a user interaction (button click).

        If replace_original is True, edits the original message.
        Otherwise, answers the callback query with a notification.
        Includes circuit breaker protection via answer_callback_query and update_message.

        Args:
            interaction: The interaction event
            text: Response text
            blocks: Rich content blocks
            replace_original: If True, replace the original message
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(
                success=False,
                error="httpx is required for Telegram connector",
            )

        # First, acknowledge the callback query
        await self.answer_callback_query(
            interaction.id, text=text if not replace_original else None
        )

        if replace_original and interaction.message_id:
            # Edit the original message
            return await self.update_message(
                channel_id=interaction.channel.id,
                message_id=interaction.message_id,
                text=text,
                blocks=blocks,
            )

        # Return success response for acknowledgment-only case
        return SendMessageResponse(
            success=True,
            message_id=interaction.message_id,
            channel_id=interaction.channel.id if interaction.channel else None,
        )

    # =========================================================================
    # Rich Media Support
    # =========================================================================

    async def send_photo(
        self,
        channel_id: str,
        photo: str | bytes,
        caption: str | None = None,
        thread_id: str | None = None,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a photo to a Telegram chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID
            photo: File path, URL, or bytes of the photo
            caption: Optional caption for the photo
            thread_id: Reply to specific message
            blocks: Inline keyboard buttons
            **kwargs: Additional parameters

        Returns:
            SendMessageResponse with success status
        """
        payload: dict[str, Any] = {"chat_id": channel_id}

        if caption:
            payload["caption"] = (
                self._escape_markdown(caption)
                if self.parse_mode.startswith("Markdown")
                else caption
            )
            payload["parse_mode"] = self.parse_mode

        if thread_id:
            payload["reply_to_message_id"] = int(thread_id)

        if blocks:
            keyboard = self._blocks_to_keyboard(blocks)
            if keyboard:
                payload["reply_markup"] = json.dumps(keyboard)

        # Determine how to send the photo
        files = None
        if isinstance(photo, bytes):
            files = {"photo": ("photo.jpg", photo, "image/jpeg")}
        else:
            # URL or file_id
            payload["photo"] = photo

        success, data, error = await self._telegram_api_request(
            "sendPhoto",
            payload=payload,
            files=files,
            operation="send_photo",
        )

        if success and data:
            msg = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=str(msg.get("chat", {}).get("id")),
                timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def send_video(
        self,
        channel_id: str,
        video: str | bytes,
        caption: str | None = None,
        thread_id: str | None = None,
        duration: int | None = None,
        width: int | None = None,
        height: int | None = None,
        supports_streaming: bool = True,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a video to a Telegram chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID
            video: File path, URL, or bytes of the video
            caption: Optional caption for the video
            thread_id: Reply to specific message
            duration: Video duration in seconds
            width: Video width
            height: Video height
            supports_streaming: Whether the video supports streaming
            blocks: Inline keyboard buttons
            **kwargs: Additional parameters

        Returns:
            SendMessageResponse with success status
        """
        payload: dict[str, Any] = {
            "chat_id": channel_id,
            "supports_streaming": supports_streaming,
        }

        if caption:
            payload["caption"] = (
                self._escape_markdown(caption)
                if self.parse_mode.startswith("Markdown")
                else caption
            )
            payload["parse_mode"] = self.parse_mode

        if thread_id:
            payload["reply_to_message_id"] = int(thread_id)
        if duration:
            payload["duration"] = duration
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height

        if blocks:
            keyboard = self._blocks_to_keyboard(blocks)
            if keyboard:
                payload["reply_markup"] = json.dumps(keyboard)

        # Determine how to send the video
        files = None
        if isinstance(video, bytes):
            files = {"video": ("video.mp4", video, "video/mp4")}
        else:
            # URL or file_id
            payload["video"] = video

        success, data, error = await self._telegram_api_request(
            "sendVideo",
            payload=payload,
            files=files,
            operation="send_video",
        )

        if success and data:
            msg = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=str(msg.get("chat", {}).get("id")),
                timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def send_animation(
        self,
        channel_id: str,
        animation: str | bytes,
        caption: str | None = None,
        thread_id: str | None = None,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send an animation (GIF) to a Telegram chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID
            animation: File path, URL, or bytes of the animation
            caption: Optional caption
            thread_id: Reply to specific message
            blocks: Inline keyboard buttons
            **kwargs: Additional parameters

        Returns:
            SendMessageResponse with success status
        """
        payload: dict[str, Any] = {"chat_id": channel_id}

        if caption:
            payload["caption"] = (
                self._escape_markdown(caption)
                if self.parse_mode.startswith("Markdown")
                else caption
            )
            payload["parse_mode"] = self.parse_mode

        if thread_id:
            payload["reply_to_message_id"] = int(thread_id)

        if blocks:
            keyboard = self._blocks_to_keyboard(blocks)
            if keyboard:
                payload["reply_markup"] = json.dumps(keyboard)

        # Handle bytes input (file upload) vs URL/file_id
        files: dict[str, Any] | None = None
        if isinstance(animation, bytes):
            files = {"animation": ("animation.gif", animation, "image/gif")}
        else:
            payload["animation"] = animation

        success, data, error = await self._telegram_api_request(
            "sendAnimation",
            payload=payload,
            files=files,
            operation="send_animation",
        )

        if success and data:
            msg = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=str(msg.get("chat", {}).get("id")),
                timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def send_media_group(
        self,
        channel_id: str,
        media: list[dict[str, Any]],
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> list[SendMessageResponse]:
        """Send a group of photos or videos as an album.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID
            media: List of media items, each with:
                - type: "photo" or "video"
                - media: URL or file_id
                - caption: Optional caption (only first item shown)
            thread_id: Reply to specific message
            **kwargs: Additional parameters

        Returns:
            List of SendMessageResponse for each sent message
        """
        # Format media for API
        formatted_media = []
        for item in media:
            media_item = {
                "type": item.get("type", "photo"),
                "media": item.get("media"),
            }
            if "caption" in item:
                caption = item["caption"]
                media_item["caption"] = (
                    self._escape_markdown(caption)
                    if self.parse_mode.startswith("Markdown")
                    else caption
                )
                media_item["parse_mode"] = self.parse_mode
            formatted_media.append(media_item)

        payload: dict[str, Any] = {
            "chat_id": channel_id,
            "media": json.dumps(formatted_media),
        }

        if thread_id:
            payload["reply_to_message_id"] = int(thread_id)

        success, data, error = await self._telegram_api_request(
            "sendMediaGroup",
            payload=payload,
            operation="send_media_group",
        )

        if success and data:
            messages = data.get("result", [])
            return [
                SendMessageResponse(
                    success=True,
                    message_id=str(msg.get("message_id")),
                    channel_id=str(msg.get("chat", {}).get("id")),
                    timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
                )
                for msg in messages
            ]

        return [SendMessageResponse(success=False, error=error)]

    # =========================================================================
    # Inline Query Support
    # =========================================================================

    async def answer_inline_query(
        self,
        inline_query_id: str,
        results: list[dict[str, Any]],
        cache_time: int = 300,
        is_personal: bool = False,
        next_offset: str | None = None,
        button: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Answer an inline query with results.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            inline_query_id: Unique identifier for the inline query
            results: List of InlineQueryResult objects. Each should have:
                - type: "article", "photo", "video", etc.
                - id: Unique identifier
                - title: Title of the result
                - input_message_content: Content to send when selected
            cache_time: Time in seconds to cache results (default 300)
            is_personal: Whether results are personalized per user
            next_offset: Offset for pagination
            button: Button to show above results
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        payload: dict[str, Any] = {
            "inline_query_id": inline_query_id,
            "results": json.dumps(results),
            "cache_time": cache_time,
            "is_personal": is_personal,
        }

        if next_offset:
            payload["next_offset"] = next_offset
        if button:
            payload["button"] = json.dumps(button)

        success, data, error = await self._telegram_api_request(
            "answerInlineQuery",
            payload=payload,
            operation="answer_inline_query",
        )

        if not success:
            logger.error(f"Failed to answer inline query: {error}")

        return success

    def build_inline_article_result(
        self,
        result_id: str,
        title: str,
        message_text: str,
        description: str | None = None,
        url: str | None = None,
        thumb_url: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an InlineQueryResultArticle for use with answer_inline_query.

        Args:
            result_id: Unique identifier for this result
            title: Title of the result
            message_text: Text to send when selected
            description: Short description
            url: URL to associate with the result
            thumb_url: Thumbnail URL
            **kwargs: Additional fields

        Returns:
            Dict formatted as InlineQueryResultArticle
        """
        result: dict[str, Any] = {
            "type": "article",
            "id": result_id,
            "title": title,
            "input_message_content": {
                "message_text": message_text,
                "parse_mode": self.parse_mode,
            },
        }

        if description:
            result["description"] = description
        if url:
            result["url"] = url
        if thumb_url:
            result["thumb_url"] = thumb_url

        result.update(kwargs)
        return result

    # =========================================================================
    # Bot Management
    # =========================================================================

    async def set_my_commands(
        self,
        commands: list[dict[str, str]],
        scope: dict[str, Any] | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Set the list of the bot's commands.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            commands: List of command dicts with 'command' and 'description' keys
            scope: Scope for which commands apply (all users, specific chat, etc.)
            language_code: Language code for commands
            **kwargs: Additional parameters

        Returns:
            True if successful
        """
        payload: dict[str, Any] = {
            "commands": json.dumps(commands),
        }

        if scope:
            payload["scope"] = json.dumps(scope)
        if language_code:
            payload["language_code"] = language_code

        success, data, error = await self._telegram_api_request(
            "setMyCommands",
            payload=payload,
            operation="set_my_commands",
        )

        if not success:
            logger.error(f"Failed to set commands: {error}")

        return success

    async def get_me(self) -> dict[str, Any] | None:
        """Get basic information about the bot.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Returns:
            Bot user object with id, username, first_name, etc.
        """
        success, data, error = await self._telegram_api_request(
            "getMe",
            method="GET",
            operation="get_me",
        )

        if success and data:
            bot_info: dict[str, Any] | None = data.get("result")
            return bot_info

        logger.error(f"Failed to get bot info: {error}")
        return None

    async def get_chat_member_count(self, channel_id: str) -> int | None:
        """Get the number of members in a chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID

        Returns:
            Number of members or None if failed
        """
        success, data, error = await self._telegram_api_request(
            "getChatMemberCount",
            payload={"chat_id": channel_id},
            operation="get_chat_member_count",
        )

        if success and data:
            count: int | None = data.get("result")
            return count

        logger.error(f"Failed to get chat member count: {error}")
        return None


__all__ = ["TelegramConnector"]
