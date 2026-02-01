"""
Chat Platform Connector - Abstract base class for chat integrations.

All chat platform connectors inherit from ChatPlatformConnector and implement
standardized methods for:
- Sending and updating messages
- Handling bot commands
- Processing user interactions
- File operations (upload/download)
- Voice message handling

Includes circuit breaker support for fault tolerance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

from .models import (
    BotCommand,
    ChannelContext,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    MessageButton,
    SendMessageResponse,
    UserInteraction,
    VoiceMessage,
    WebhookEvent,
)
from .rich_context import (
    analyze_sentiment as _analyze_sentiment_impl,
    calculate_activity_patterns as _calculate_activity_patterns_impl,
    extract_topics as _extract_topics_impl,
    format_context_for_llm as _format_context_for_llm_impl,
)

logger = logging.getLogger(__name__)


class ChatPlatformConnector(ABC):
    """
    Abstract base class for chat platform integrations.

    Provides a unified interface for interacting with chat platforms
    like Slack, Microsoft Teams, Discord, and Google Chat.

    Subclasses must implement the abstract methods to handle
    platform-specific APIs and message formats.

    Includes circuit breaker support for fault tolerance in HTTP operations.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        signing_secret: str | None = None,
        webhook_url: str | None = None,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown: float = 60.0,
        request_timeout: float = 30.0,
        **config: Any,
    ):
        """
        Initialize the connector.

        Args:
            bot_token: API token for the bot
            signing_secret: Secret for webhook verification
            webhook_url: Default webhook URL for sending messages
            enable_circuit_breaker: Enable circuit breaker for fault tolerance
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_cooldown: Seconds before attempting recovery
            request_timeout: HTTP request timeout in seconds
            **config: Additional platform-specific configuration
        """
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.webhook_url = webhook_url
        self.config = config
        self._initialized = False

        # Circuit breaker settings
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_cooldown = circuit_breaker_cooldown
        self._request_timeout = request_timeout
        self._circuit_breaker: Any | None = None
        self._circuit_breaker_initialized = False

    # ==========================================================================
    # Circuit Breaker Support
    # ==========================================================================

    def _get_circuit_breaker(self) -> Any | None:
        """Get or create circuit breaker (lazy initialization)."""
        if not self._enable_circuit_breaker:
            return None

        if not self._circuit_breaker_initialized:
            try:
                from aragora.resilience import get_circuit_breaker

                self._circuit_breaker = get_circuit_breaker(
                    name=f"chat_connector_{self.platform_name}",
                    failure_threshold=self._circuit_breaker_threshold,
                    cooldown_seconds=self._circuit_breaker_cooldown,
                )
                logger.debug(f"Circuit breaker initialized for {self.platform_name}")
            except ImportError:
                logger.warning("Circuit breaker module not available")
            self._circuit_breaker_initialized = True

        return self._circuit_breaker

    def _check_circuit_breaker(self) -> tuple[bool, str | None]:
        """
        Check if circuit breaker allows the request.

        Returns:
            Tuple of (can_proceed, error_message)
        """
        cb = self._get_circuit_breaker()
        if cb is None:
            return True, None

        if not cb.can_proceed():
            remaining = cb.cooldown_remaining()
            error = f"Circuit breaker open for {self.platform_name}. Retry in {remaining:.1f}s"
            logger.warning(error)
            return False, error

        return True, None

    def _record_success(self) -> None:
        """Record a successful operation with the circuit breaker."""
        cb = self._get_circuit_breaker()
        if cb:
            cb.record_success()

    def _record_failure(self, error: Exception | None = None) -> None:
        """Record a failed operation with the circuit breaker."""
        cb = self._get_circuit_breaker()
        if cb:
            cb.record_failure()
            status = cb.get_status()
            if status == "open":
                logger.warning(
                    f"Circuit breaker OPENED for {self.platform_name} after repeated failures"
                )

    async def _with_retry(
        self,
        operation: str,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Execute an async function with exponential backoff retry and circuit breaker.

        This provides a standardized retry pattern for all connector operations.

        Args:
            operation: Name of the operation (for logging)
            func: Async function to execute
            *args: Arguments to pass to the function
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            retryable_exceptions: Tuple of exception types to retry on
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call

        Raises:
            Last exception if all retries fail
        """
        import asyncio
        import random

        # Check circuit breaker first
        can_proceed, error_msg = self._check_circuit_breaker()
        if not can_proceed:
            raise ConnectionError(error_msg)

        last_exception = None
        for attempt in range(max_retries):
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except retryable_exceptions as e:
                last_exception = e
                self._record_failure(e)

                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        f"{self.platform_name} {operation} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {total_delay:.1f}s"
                    )
                    await asyncio.sleep(total_delay)
                else:
                    logger.error(
                        f"{self.platform_name} {operation} failed after {max_retries} attempts: {e}"
                    )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"{operation} failed with no exception captured")

    def _is_retryable_status_code(self, status_code: int) -> bool:
        """
        Check if an HTTP status code indicates a retryable error.

        Args:
            status_code: HTTP status code

        Returns:
            True if the error is transient and should be retried
        """
        # 429 Too Many Requests - rate limited
        # 500 Internal Server Error - server error
        # 502 Bad Gateway - upstream error
        # 503 Service Unavailable - server overloaded
        # 504 Gateway Timeout - upstream timeout
        return status_code in {429, 500, 502, 503, 504}

    async def _http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        content: bytes | None = None,
        files: dict[str, Any] | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float | None = None,
        return_raw: bool = False,
        operation: str = "http_request",
    ) -> tuple[bool, Optional[dict[str, Any] | bytes], str | None]:
        """
        Make an HTTP request with retry, timeout, and circuit breaker support.

        This is the recommended method for all HTTP operations in chat connectors.
        Provides consistent error handling, logging, and resilience patterns.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            url: Request URL
            headers: Optional request headers
            json: Optional JSON body
            data: Optional form data (dict for form-encoded, or bytes)
            content: Optional raw bytes body (for file uploads)
            files: Optional file uploads
            max_retries: Maximum retry attempts (default 3)
            base_delay: Initial retry delay in seconds (default 1.0)
            timeout: Custom timeout in seconds (defaults to self._request_timeout)
            return_raw: If True, return raw bytes instead of JSON
            operation: Operation name for logging

        Returns:
            Tuple of (success: bool, response_data: dict|bytes | None, error: str | None)
        """
        import asyncio
        import random

        # Check circuit breaker first
        can_proceed, error_msg = self._check_circuit_breaker()
        if not can_proceed:
            return False, None, error_msg

        # Try to import httpx
        try:
            import httpx
        except ImportError:
            return False, None, "httpx not available"

        last_error: str | None = None
        request_timeout = timeout or self._request_timeout

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=json,
                        data=data,
                        content=content,
                        files=files,
                    )

                    # Check for retryable status codes
                    if self._is_retryable_status_code(response.status_code):
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        self._record_failure()

                        if attempt < max_retries - 1:
                            # Calculate delay with exponential backoff and jitter
                            delay = min(base_delay * (2**attempt), 30.0)
                            jitter = random.uniform(0, delay * 0.1)
                            total_delay = delay + jitter

                            logger.warning(
                                f"{self.platform_name} {operation} got {response.status_code} "
                                f"(attempt {attempt + 1}/{max_retries}). Retrying in {total_delay:.1f}s"
                            )
                            await asyncio.sleep(total_delay)
                            continue
                        else:
                            logger.error(
                                f"{self.platform_name} {operation} failed after {max_retries} "
                                f"attempts with status {response.status_code}"
                            )
                            return False, None, last_error

                    # Non-retryable error
                    if response.status_code >= 400:
                        self._record_failure()
                        error = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger.warning(f"{self.platform_name} {operation} failed: {error}")
                        return False, None, error

                    # Success
                    self._record_success()
                    if return_raw:
                        return True, response.content, None
                    try:
                        return True, response.json(), None
                    except ValueError:
                        # Response may not be JSON
                        return True, {"status": "ok", "text": response.text}, None

            except httpx.TimeoutException as e:
                last_error = f"Timeout after {self._request_timeout}s: {e}"
                self._record_failure()

                if attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), 30.0)
                    logger.warning(
                        f"{self.platform_name} {operation} timed out "
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{self.platform_name} {operation} timed out after {max_retries} attempts"
                    )

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                self._record_failure()

                if attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), 30.0)
                    logger.warning(
                        f"{self.platform_name} {operation} connection failed "
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{self.platform_name} {operation} connection failed after {max_retries} attempts"
                    )

            except (ValueError, TypeError, RuntimeError, OSError) as e:
                last_error = f"Unexpected error: {e}"
                self._record_failure()
                logger.error(f"{self.platform_name} {operation} unexpected error: {e}")
                # Don't retry on unexpected errors
                break

        return False, None, last_error

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier (e.g., 'slack', 'teams', 'discord')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def platform_display_name(self) -> str:
        """Return human-readable platform name (e.g., 'Microsoft Teams')."""
        raise NotImplementedError

    # ==========================================================================
    # Message Operations
    # ==========================================================================

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send a message to a channel.

        Default implementation posts to ``webhook_url`` if configured,
        otherwise returns a failure response.  Subclasses should override
        for platform-specific APIs.

        Args:
            channel_id: Target channel/conversation ID
            text: Plain text content (fallback for clients without rich support)
            blocks: Rich content blocks in platform-native format
            thread_id: Optional thread/reply ID for threaded messages
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with message ID and status
        """
        if self.webhook_url:
            payload: dict[str, Any] = {
                "channel_id": channel_id,
                "text": text,
            }
            if blocks:
                payload["blocks"] = blocks
            if thread_id:
                payload["thread_id"] = thread_id

            success, data, error = await self._http_request(
                method="POST",
                url=self.webhook_url,
                json=payload,
                operation="send_message",
            )

            if success and isinstance(data, dict):
                return SendMessageResponse(
                    success=True,
                    message_id=data.get("message_id") or data.get("id"),
                    channel_id=channel_id,
                    timestamp=data.get("timestamp"),
                )

            return SendMessageResponse(success=False, error=error or "Send failed")

        logger.warning(
            f"{self.platform_name} send_message: no webhook_url configured "
            f"and no platform-specific override provided"
        )
        return SendMessageResponse(
            success=False,
            error=f"{self.platform_name} send_message not implemented",
        )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update an existing message.

        Default implementation logs a warning and returns a failure response,
        since not all platforms support message editing.  Subclasses should
        override for platforms that do support it.

        Args:
            channel_id: Channel containing the message
            message_id: ID of message to update
            text: New text content
            blocks: New rich content blocks
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with update status
        """
        logger.warning(
            f"{self.platform_name} does not implement update_message; "
            f"message {message_id} in channel {channel_id} was not updated"
        )
        return SendMessageResponse(
            success=False,
            error=f"{self.platform_name} does not support message updates",
        )

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a message.

        Default implementation returns False, since not all platforms
        support message deletion via API.  Subclasses should override
        for platforms that do support it.

        Args:
            channel_id: Channel containing the message
            message_id: ID of message to delete
            **kwargs: Platform-specific options

        Returns:
            True if deleted successfully, False otherwise
        """
        logger.warning(
            f"{self.platform_name} does not implement delete_message; "
            f"message {message_id} in channel {channel_id} was not deleted"
        )
        return False

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send an ephemeral message visible only to one user.

        Not all platforms support this; default implementation sends regular message.

        Args:
            channel_id: Target channel
            user_id: User to show message to
            text: Message text
            blocks: Rich content blocks
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        logger.warning(f"{self.platform_name} does not support ephemeral messages")
        return await self.send_message(channel_id, text, blocks, **kwargs)

    async def send_typing_indicator(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Send a typing indicator to show the bot is processing.

        Not all platforms support this; default implementation returns False.
        Typing indicators typically expire after a few seconds.

        Args:
            channel_id: Target channel to show typing in
            **kwargs: Platform-specific options

        Returns:
            True if typing indicator was sent, False if not supported
        """
        logger.debug(f"{self.platform_name} does not support typing indicators")
        return False

    # ==========================================================================
    # Command Handling
    # ==========================================================================

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Respond to a slash command.

        Default implementation sends a regular message to the command's
        channel (ephemeral flag is ignored unless the subclass handles it).
        If the command has a ``response_url``, the default posts the
        response there.

        Args:
            command: The command that was invoked
            text: Response text
            blocks: Rich content blocks
            ephemeral: If True, only the user sees the response
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        # If the platform provides a response URL, use it
        if command.response_url:
            payload: dict[str, Any] = {"text": text}
            if blocks:
                payload["blocks"] = blocks
            if ephemeral:
                payload["response_type"] = "ephemeral"

            success, data, error = await self._http_request(
                method="POST",
                url=command.response_url,
                json=payload,
                operation="respond_to_command",
            )

            if success:
                return SendMessageResponse(success=True)
            return SendMessageResponse(success=False, error=error or "Command response failed")

        # Fall back to sending a regular channel message
        channel_id = command.channel.id if command.channel else kwargs.get("channel_id")
        if not channel_id:
            return SendMessageResponse(
                success=False,
                error="No channel ID or response_url available for command response",
            )

        return await self.send_message(
            channel_id=channel_id,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    # ==========================================================================
    # Interaction Handling
    # ==========================================================================

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Respond to a user interaction (button click, menu select, etc.).

        Default implementation sends a regular message to the interaction's
        channel.  If ``replace_original`` is True and the interaction has a
        ``message_id``, attempts to update the original message; otherwise
        falls back to sending a new message.  If the interaction has a
        ``response_url``, the default posts the response there.

        Args:
            interaction: The interaction event
            text: Response text
            blocks: Rich content blocks
            replace_original: If True, replace the original message
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        # If the platform provides a response URL, use it
        if interaction.response_url:
            payload: dict[str, Any] = {"text": text}
            if blocks:
                payload["blocks"] = blocks
            if replace_original:
                payload["replace_original"] = True

            success, data, error = await self._http_request(
                method="POST",
                url=interaction.response_url,
                json=payload,
                operation="respond_to_interaction",
            )

            if success:
                return SendMessageResponse(success=True)
            return SendMessageResponse(success=False, error=error or "Interaction response failed")

        # If replacing the original and we know the channel + message_id, update it
        channel_id = interaction.channel.id if interaction.channel else None
        if replace_original and channel_id and interaction.message_id:
            return await self.update_message(
                channel_id=channel_id,
                message_id=interaction.message_id,
                text=text,
                blocks=blocks,
                **kwargs,
            )

        # Fall back to sending a regular message
        if channel_id:
            return await self.send_message(
                channel_id=channel_id,
                text=text,
                blocks=blocks,
                **kwargs,
            )

        return SendMessageResponse(
            success=False,
            error="No channel or response_url available for interaction response",
        )

    # ==========================================================================
    # File Operations
    # ==========================================================================

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
        """
        Upload a file to a channel.

        Default implementation posts the file to ``webhook_url`` as a
        multipart upload if configured.  Subclasses should override for
        platform-specific file upload APIs.

        Args:
            channel_id: Target channel
            content: File content as bytes
            filename: Name of the file
            content_type: MIME type
            title: Optional display title
            thread_id: Optional thread for the file
            **kwargs: Platform-specific options

        Returns:
            FileAttachment with file ID and URL

        Raises:
            NotImplementedError: If no webhook_url is configured and
                the subclass does not override this method
        """
        if self.webhook_url:
            success, data, error = await self._http_request(
                method="POST",
                url=self.webhook_url,
                files={"file": (filename, content, content_type)},
                data={
                    "channel_id": channel_id,
                    **({"title": title} if title else {}),
                    **({"thread_id": thread_id} if thread_id else {}),
                },
                operation="upload_file",
            )

            if success and isinstance(data, dict):
                return FileAttachment(
                    id=data.get("file_id") or data.get("id", ""),
                    filename=data.get("filename", filename),
                    content_type=content_type,
                    size=len(content),
                    url=data.get("url"),
                )

            raise RuntimeError(error or "File upload failed")

        raise NotImplementedError(
            f"{self.platform_name} upload_file requires a platform-specific override "
            f"or a configured webhook_url"
        )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """
        Download a file by ID.

        Default implementation downloads from a ``url`` keyword argument
        if provided.  Subclasses should override for platform-specific
        file download APIs that require resolving ``file_id`` to a URL.

        Args:
            file_id: ID of the file to download
            **kwargs: Platform-specific options.  Pass ``url`` to specify
                the direct download URL, ``filename`` for a filename hint.

        Returns:
            FileAttachment with content populated

        Raises:
            NotImplementedError: If no ``url`` is provided and the
                subclass does not override this method
        """
        url = kwargs.get("url")
        if url:
            success, data, error = await self._http_request(
                method="GET",
                url=url,
                return_raw=True,
                operation="download_file",
            )

            if success and isinstance(data, bytes):
                filename = kwargs.get("filename") or url.split("/")[-1] or "file"
                return FileAttachment(
                    id=file_id,
                    filename=filename,
                    content_type=kwargs.get("content_type", "application/octet-stream"),
                    size=len(data),
                    url=url,
                    content=data,
                )

            raise RuntimeError(error or "File download failed")

        raise NotImplementedError(
            f"{self.platform_name} download_file requires a platform-specific override "
            f"or a 'url' keyword argument"
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
        """
        Send a voice/audio message to a channel.

        Default implementation uploads as a file attachment.
        Platforms may override for native voice message support.

        Args:
            channel_id: Target channel
            audio_content: Audio file content as bytes
            filename: Audio filename
            content_type: MIME type (audio/mpeg, audio/ogg, etc.)
            reply_to: Optional message ID to reply to
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        try:
            # Default implementation: upload as file
            attachment = await self.upload_file(
                channel_id=channel_id,
                content=audio_content,
                filename=filename,
                content_type=content_type,
                thread_id=reply_to,
                **kwargs,
            )
            return SendMessageResponse(
                success=True,
                message_id=attachment.id,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to send voice message on {self.platform_name}: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    # ==========================================================================
    # Rich Content Formatting
    # ==========================================================================

    def format_blocks(
        self,
        title: str | None = None,
        body: str | None = None,
        fields: list[tuple[str, str] | None] = None,
        actions: list[MessageButton] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Format content into platform-specific rich blocks.

        Default implementation produces a generic block list with
        ``section``, ``fields``, and ``actions`` entries.  Subclasses
        should override for platform-native formatting (Slack blocks,
        Discord embeds, Telegram inline keyboards, etc.).

        Args:
            title: Section title/header
            body: Main text content
            fields: List of (label, value) tuples for structured data
            actions: List of interactive buttons
            **kwargs: Platform-specific options

        Returns:
            List of platform-specific block structures
        """
        blocks: list[dict[str, Any]] = []

        if title:
            blocks.append({"type": "header", "text": title})

        if body:
            blocks.append({"type": "section", "text": body})

        if fields:
            blocks.append(
                {
                    "type": "fields",
                    "items": [
                        {"label": label, "value": value}
                        for label, value in fields
                        if label is not None and value is not None
                    ],
                }
            )

        if actions:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        self.format_button(
                            text=btn.text,
                            action_id=btn.action_id,
                            value=btn.value,
                            style=btn.style,
                            url=btn.url,
                        )
                        for btn in actions
                    ],
                }
            )

        return blocks

    def format_button(
        self,
        text: str,
        action_id: str,
        value: str | None = None,
        style: str = "default",
        url: str | None = None,
    ) -> dict[str, Any]:
        """
        Format a button element.

        Default implementation returns a generic button dict.  Subclasses
        should override for platform-native button structures.

        Args:
            text: Button label
            action_id: Unique action identifier
            value: Value to pass when clicked
            style: Button style (default, primary, danger)
            url: Optional URL for link buttons

        Returns:
            Platform-specific button structure
        """
        button: dict[str, Any] = {
            "type": "button",
            "text": text,
            "action_id": action_id,
        }
        if value is not None:
            button["value"] = value
        if style != "default":
            button["style"] = style
        if url:
            button["url"] = url
        return button

    # ==========================================================================
    # Webhook Handling
    # ==========================================================================

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """
        Verify webhook signature for security.

        Default implementation performs HMAC-SHA256 verification using
        ``signing_secret`` if configured.  Looks for the signature in
        common header locations (``X-Signature-256``, ``X-Hub-Signature-256``).
        In development mode with no ``signing_secret``, returns True with
        a warning.  In production, fails closed (returns False).

        Subclasses should override for platform-specific verification
        (e.g., Discord Ed25519, Telegram secret token header).

        Args:
            headers: HTTP headers from the webhook request
            body: Raw request body

        Returns:
            True if signature is valid
        """
        import hashlib
        import hmac as _hmac
        import os as _os

        if not self.signing_secret:
            env = _os.environ.get("ARAGORA_ENV", "development").lower()
            is_production = env not in ("development", "dev", "local", "test")
            if is_production:
                logger.error(
                    f"SECURITY: {self.platform_name} signing_secret not configured "
                    f"in production. Rejecting webhook to prevent signature bypass."
                )
                return False
            logger.warning(
                f"{self.platform_name} signing_secret not set - skipping verification. "
                f"This is only acceptable in development!"
            )
            return True

        # Look for signature in common header locations (case-insensitive)
        signature = ""
        for header_name in (
            "X-Signature-256",
            "x-signature-256",
            "X-Hub-Signature-256",
            "x-hub-signature-256",
        ):
            if header_name in headers:
                signature = headers[header_name]
                break

        if not signature:
            logger.warning(f"{self.platform_name} webhook missing signature header")
            return False

        # Strip sha256= prefix if present
        sig_value = signature
        if sig_value.startswith("sha256="):
            sig_value = sig_value[7:]

        computed = _hmac.new(
            self.signing_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        if _hmac.compare_digest(computed, sig_value):
            return True

        logger.warning(f"{self.platform_name} webhook signature mismatch")
        return False

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """
        Parse a webhook payload into a WebhookEvent.

        Default implementation parses the body as JSON and wraps it in a
        generic ``WebhookEvent``.  Subclasses should override for
        platform-specific event parsing (message types, interactions, etc.).

        Args:
            headers: HTTP headers from the request
            body: Raw request body

        Returns:
            Parsed WebhookEvent
        """
        import json

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(f"{self.platform_name} webhook body is not valid JSON")
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
                metadata={"error": "invalid_json"},
            )

        # Try to infer event type from common payload structures
        event_type = (
            payload.get("type") or payload.get("event_type") or payload.get("event", {}).get("type")
            if isinstance(payload, dict)
            else None
        ) or "unknown"

        return WebhookEvent(
            platform=self.platform_name,
            event_type=str(event_type),
            raw_payload=payload if isinstance(payload, dict) else {"data": payload},
        )

    # ==========================================================================
    # Voice Message Handling
    # ==========================================================================

    async def get_voice_message(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> VoiceMessage | None:
        """
        Retrieve a voice message for transcription.

        Args:
            file_id: ID of the voice message file
            **kwargs: Platform-specific options

        Returns:
            VoiceMessage with audio content, or None if not supported
        """
        logger.info(f"{self.platform_name} voice messages not implemented")
        return None

    # ==========================================================================
    # Channel/User Operations
    # ==========================================================================

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> ChatChannel | None:
        """
        Get information about a channel.

        Args:
            channel_id: Channel ID to look up
            **kwargs: Platform-specific options

        Returns:
            ChatChannel info or None
        """
        logger.debug(f"{self.platform_name} get_channel_info not implemented")
        return None

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser | None:
        """
        Get information about a user.

        Args:
            user_id: User ID to look up
            **kwargs: Platform-specific options

        Returns:
            ChatUser info or None
        """
        logger.debug(f"{self.platform_name} get_user_info not implemented")
        return None

    async def get_user_profile(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser | None:
        """
        Get detailed user profile information.

        This is an alias for get_user_info() for API consistency.
        Override in subclasses if the platform provides separate
        profile endpoints with more detailed information.

        Args:
            user_id: User ID to look up
            **kwargs: Platform-specific options

        Returns:
            ChatUser info or None
        """
        return await self.get_user_info(user_id, **kwargs)

    async def list_users(
        self,
        channel_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[ChatUser], str | None]:
        """
        List users in a channel or workspace.

        If channel_id is provided, lists members of that channel.
        Otherwise, lists members of the workspace (if supported).

        Args:
            channel_id: Optional channel to list members of
            limit: Maximum number of users to return (default 100)
            cursor: Pagination cursor for subsequent requests
            **kwargs: Platform-specific options

        Returns:
            Tuple of (list of ChatUser, next_cursor or None)

        Note:
            Default implementation returns empty list. Override in
            subclasses for platform-specific user enumeration.
        """
        logger.debug(f"{self.platform_name} list_users not implemented")
        return [], None

    async def create_channel(
        self,
        name: str,
        is_private: bool = False,
        description: str | None = None,
        **kwargs: Any,
    ) -> ChatChannel | None:
        """
        Create a new channel.

        Args:
            name: Name for the new channel
            is_private: Whether the channel should be private (default False)
            description: Optional channel description/topic
            **kwargs: Platform-specific options (e.g., team_id, user_ids)

        Returns:
            ChatChannel if created successfully, None otherwise

        Note:
            Default implementation returns None. Override in subclasses
            for platforms that support channel creation via API.
        """
        logger.debug(f"{self.platform_name} create_channel not implemented")
        return None

    async def send_dm(
        self,
        user_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send a direct message to a user.

        This opens or retrieves a DM channel with the user and sends
        the message. For platforms that don't distinguish between
        channels and DMs (like WhatsApp), this delegates to send_message.

        Args:
            user_id: Target user ID
            text: Message text
            blocks: Optional rich content blocks
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with message ID and status

        Note:
            Default implementation delegates to send_message with user_id
            as the channel_id. Override if the platform requires opening
            a DM channel first.
        """
        logger.debug(f"{self.platform_name} send_dm using user_id as channel_id")
        return await self.send_message(
            channel_id=user_id,
            text=text,
            blocks=blocks,
            **kwargs,
        )

    # ==========================================================================
    # Message Reactions
    # ==========================================================================

    async def react_to_message(
        self,
        channel_id: str,
        message_id: str,
        reaction: str,
        **kwargs: Any,
    ) -> bool:
        """
        Add a reaction to a message.

        Args:
            channel_id: Channel containing the message
            message_id: ID of the message to react to
            reaction: Reaction emoji or identifier (e.g., "thumbsup", ":+1:")
            **kwargs: Platform-specific options

        Returns:
            True if reaction was added successfully, False otherwise

        Note:
            Default implementation returns False. Override in subclasses
            for platforms that support message reactions.
        """
        logger.debug(f"{self.platform_name} react_to_message not implemented")
        return False

    async def remove_reaction(
        self,
        channel_id: str,
        message_id: str,
        reaction: str,
        **kwargs: Any,
    ) -> bool:
        """
        Remove a reaction from a message.

        Args:
            channel_id: Channel containing the message
            message_id: ID of the message to remove reaction from
            reaction: Reaction emoji or identifier to remove
            **kwargs: Platform-specific options

        Returns:
            True if reaction was removed successfully, False otherwise

        Note:
            Default implementation returns False. Override in subclasses
            for platforms that support message reactions.
        """
        logger.debug(f"{self.platform_name} remove_reaction not implemented")
        return False

    # ==========================================================================
    # Message Pinning
    # ==========================================================================

    async def pin_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Pin a message to a channel.

        Pinned messages are highlighted and easily accessible in most
        chat platforms, useful for important announcements or decisions.

        Args:
            channel_id: Channel containing the message
            message_id: ID of the message to pin
            **kwargs: Platform-specific options

        Returns:
            True if message was pinned successfully, False otherwise

        Note:
            Default implementation returns False. Override in subclasses
            for platforms that support message pinning.
        """
        logger.debug(f"{self.platform_name} pin_message not implemented")
        return False

    async def unpin_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Unpin a previously pinned message.

        Args:
            channel_id: Channel containing the message
            message_id: ID of the message to unpin
            **kwargs: Platform-specific options

        Returns:
            True if message was unpinned successfully, False otherwise

        Note:
            Default implementation returns False. Override in subclasses
            for platforms that support message pinning.
        """
        logger.debug(f"{self.platform_name} unpin_message not implemented")
        return False

    async def get_pinned_messages(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get all pinned messages in a channel.

        Args:
            channel_id: Channel to get pinned messages from
            **kwargs: Platform-specific options

        Returns:
            List of pinned ChatMessage objects

        Note:
            Default implementation returns empty list. Override in
            subclasses for platforms that support message pinning.
        """
        logger.debug(f"{self.platform_name} get_pinned_messages not implemented")
        return []

    # ==========================================================================
    # Threading
    # ==========================================================================

    async def create_thread(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: list[dict[str, Any] | None] = None,
        thread_name: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Create a thread reply to a message.

        This sends a reply that creates or continues a thread on the
        specified message. For platforms that don't distinguish threads
        from regular replies, this delegates to send_message with thread_id.

        Args:
            channel_id: Channel containing the parent message
            message_id: ID of the message to create thread on
            text: Thread reply text
            blocks: Optional rich content blocks
            thread_name: Optional name for the thread (if platform supports)
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with the thread reply message ID

        Note:
            Default implementation delegates to send_message with
            message_id as thread_id. Override if platform has special
            thread creation APIs.
        """
        return await self.send_message(
            channel_id=channel_id,
            text=text,
            blocks=blocks,
            thread_id=message_id,
            **kwargs,
        )

    # ==========================================================================
    # Slash Command Handling
    # ==========================================================================

    async def handle_slash_command(
        self,
        command_name: str,
        channel_id: str,
        user_id: str,
        text: str = "",
        response_url: str | None = None,
        **kwargs: Any,
    ) -> BotCommand:
        """
        Handle an incoming slash command.

        This parses and structures an incoming slash command for processing.
        Use respond_to_command() to send a response back to the user.

        Args:
            command_name: Name of the command (without leading /)
            channel_id: Channel where command was invoked
            user_id: User who invoked the command
            text: Additional text after the command
            response_url: URL for async responses (if provided by platform)
            **kwargs: Platform-specific options and raw payload data

        Returns:
            BotCommand object representing the parsed command

        Example:
            # User types: /debate Should we use React or Vue?
            cmd = await connector.handle_slash_command(
                command_name="debate",
                channel_id="C123",
                user_id="U456",
                text="Should we use React or Vue?",
            )
            # cmd.name == "debate"
            # cmd.args == ["Should", "we", "use", "React", "or", "Vue?"]
        """
        # Parse arguments from text
        args = text.split() if text else []

        # Build user and channel objects
        user = ChatUser(
            id=user_id,
            platform=self.platform_name,
        )
        channel = ChatChannel(
            id=channel_id,
            platform=self.platform_name,
        )

        return BotCommand(
            name=command_name,
            text=f"/{command_name} {text}".strip(),
            args=args,
            user=user,
            channel=channel,
            platform=self.platform_name,
            response_url=response_url,
            metadata=kwargs,
        )

    # ==========================================================================
    # Message Streaming / Receiving
    # ==========================================================================

    async def receive_messages(
        self,
        channel_id: str,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        """
        Async generator that yields incoming messages from a channel.

        This provides a streaming interface for receiving messages in
        real-time. For webhook-based platforms, this may poll or use
        long-polling. For WebSocket-based platforms, this yields messages
        as they arrive.

        Args:
            channel_id: Channel to receive messages from
            timeout: Optional timeout in seconds for the stream
            **kwargs: Platform-specific options

        Yields:
            ChatMessage objects as they are received

        Note:
            Default implementation yields nothing (not supported).
            Override in subclasses for platforms with real-time
            message streaming capabilities.

        Example:
            async for message in connector.receive_messages("C123"):
                print(f"{message.author.username}: {message.content}")
                if message.content == "/stop":
                    break
        """
        logger.debug(f"{self.platform_name} receive_messages not implemented")
        # Default implementation: empty async generator
        return
        yield  # This makes it a generator  # noqa: B901

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    async def test_connection(self) -> dict[str, Any]:
        """
        Test the connection to the platform.

        Returns:
            Dict with success status and details
        """
        return {
            "platform": self.platform_name,
            "success": bool(self.bot_token or self.webhook_url),
            "bot_token_configured": bool(self.bot_token),
            "webhook_configured": bool(self.webhook_url),
        }

    async def get_health(self) -> dict[str, Any]:
        """
        Get detailed health status for the channel connector.

        Returns comprehensive health information including circuit breaker
        state, configuration status, and connectivity details.

        Returns:
            Dict with health status and metrics
        """
        import time

        details: dict[str, Any] = {}
        circuit_breaker: dict[str, Any] | None = None
        health: dict[str, Any] = {
            "platform": self.platform_name,
            "display_name": self.platform_display_name,
            "status": "unknown",
            "configured": self.is_configured,
            "timestamp": time.time(),
            "circuit_breaker": circuit_breaker,
            "details": details,
        }

        # Check configuration
        if not self.is_configured:
            health["status"] = "unconfigured"
            details["error"] = "Missing required configuration (bot_token or webhook_url)"
            return health

        # Get circuit breaker status
        cb = self._get_circuit_breaker()
        if cb:
            cb_status = cb.get_status()
            cb_info: dict[str, Any] = {
                "state": cb_status,
                "enabled": True,
            }
            if cb_status == "open":
                cb_info["cooldown_remaining"] = cb.cooldown_remaining()
            health["circuit_breaker"] = cb_info

            # Determine health based on circuit breaker
            if cb_status == "open":
                health["status"] = "unhealthy"
                details["reason"] = "Circuit breaker is open due to repeated failures"
            elif cb_status == "half_open":
                health["status"] = "degraded"
                details["reason"] = "Circuit breaker in recovery mode"
            else:
                health["status"] = "healthy"
        else:
            health["circuit_breaker"] = {"enabled": False}
            health["status"] = "healthy"

        # Add configuration details
        details["bot_token_configured"] = bool(self.bot_token)
        details["webhook_configured"] = bool(self.webhook_url)
        details["request_timeout"] = self._request_timeout

        return health

    @property
    def is_configured(self) -> bool:
        """Check if the connector has minimum required configuration."""
        return bool(self.bot_token or self.webhook_url)

    # ==========================================================================
    # Evidence Collection
    # ==========================================================================

    async def collect_evidence(
        self,
        channel_id: str,
        query: str | None = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        **kwargs: Any,
    ) -> list["ChatEvidence"]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a channel and converts them to evidence format
        with provenance tracking and relevance scoring.

        Args:
            channel_id: Channel to collect evidence from
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include threaded replies
            min_relevance: Minimum relevance score (0-1) for inclusion
            **kwargs: Platform-specific options

        Returns:
            List of ChatEvidence objects with source tracking
        """
        # Default implementation - subclasses should override for platform-specific APIs
        logger.debug(f"{self.platform_name} collect_evidence not fully implemented")
        return []

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: str | None = None,
        latest: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get message history from a channel.

        Args:
            channel_id: Channel to get history from
            limit: Maximum number of messages
            oldest: Start timestamp (platform-specific format)
            latest: End timestamp (platform-specific format)
            **kwargs: Platform-specific options

        Returns:
            List of ChatMessage objects
        """
        logger.debug(f"{self.platform_name} get_channel_history not implemented")
        return []

    def _message_matches_query(
        self,
        message: ChatMessage,
        query: str,
    ) -> bool:
        """Check if a message matches the search query."""
        if not query:
            return True

        query_lower = query.lower()
        text_lower = (message.content or "").lower()

        # Simple keyword matching
        keywords = query_lower.split()
        return any(kw in text_lower for kw in keywords)

    def _compute_message_relevance(
        self,
        message: ChatMessage,
        query: str | None = None,
    ) -> float:
        """Compute relevance score for a message."""
        if not query:
            return 1.0

        # Simple TF-based relevance
        query_lower = query.lower()
        text_lower = (message.content or "").lower()

        keywords = query_lower.split()
        if not keywords or not text_lower:
            return 0.0

        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords)

    # ==========================================================================
    # Channel Context for Deliberation
    # ==========================================================================

    async def fetch_context(
        self,
        channel_id: str,
        lookback_minutes: int = 60,
        max_messages: int = 50,
        include_participants: bool = True,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> "ChannelContext":
        """
        Fetch recent context from a channel for deliberation.

        This method retrieves recent messages and participants from a channel
        to provide context for multi-agent vetted decisionmaking sessions. It's used by the
        orchestration handler to auto-fetch context before starting debates.

        Args:
            channel_id: Channel to fetch context from
            lookback_minutes: How far back to look for messages (default: 60)
            max_messages: Maximum messages to retrieve (default: 50)
            include_participants: Whether to extract participant info (default: True)
            thread_id: Optional thread/conversation to focus on
            **kwargs: Platform-specific options

        Returns:
            ChannelContext with messages and metadata

        Example:
            context = await slack.fetch_context("C123456", lookback_minutes=30)
            deliberation_context = context.to_context_string()
        """
        from datetime import datetime, timedelta
        from .models import ChannelContext

        warnings = []

        # Get channel info
        channel = await self.get_channel_info(channel_id)
        if not channel:
            # Create a basic channel object if lookup failed
            channel = ChatChannel(
                id=channel_id,
                platform=self.platform_name,
            )
            warnings.append(f"Could not fetch channel info for {channel_id}")

        # Calculate timestamp for lookback
        oldest_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)

        # Platform-specific oldest timestamp conversion
        oldest_str = self._format_timestamp_for_api(oldest_time)

        # Fetch messages
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=max_messages,
            oldest=oldest_str,
            **kwargs,
        )

        # Extract participants
        participants = []
        if include_participants and messages:
            seen_users: dict[str, ChatUser] = {}
            for msg in messages:
                if msg.author.id not in seen_users:
                    seen_users[msg.author.id] = msg.author
            participants = list(seen_users.values())

        # Calculate timestamps
        oldest_timestamp = None
        newest_timestamp = None
        if messages:
            oldest_timestamp = min(m.timestamp for m in messages)
            newest_timestamp = max(m.timestamp for m in messages)

        context = ChannelContext(
            channel=channel,
            messages=messages,
            participants=participants,
            oldest_timestamp=oldest_timestamp,
            newest_timestamp=newest_timestamp,
            message_count=len(messages),
            participant_count=len(participants),
            warnings=warnings,
            metadata={
                "lookback_minutes": lookback_minutes,
                "max_messages": max_messages,
                "thread_id": thread_id,
            },
        )

        logger.debug(
            f"Fetched context from {self.platform_name} channel {channel_id}: "
            f"{len(messages)} messages, {len(participants)} participants"
        )

        return context

    def _format_timestamp_for_api(self, timestamp: Any) -> str | None:
        """
        Format a datetime for the platform's API.

        Override in subclasses for platform-specific formatting.
        Default returns ISO format.
        """
        from datetime import datetime

        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp) if timestamp else None

    # ==========================================================================
    # Rich Context Injection (ClawdBot Pattern)
    # ==========================================================================

    async def fetch_rich_context(
        self,
        channel_id: str,
        lookback_minutes: int = 60,
        max_messages: int = 50,
        include_participants: bool = True,
        include_topics: bool = True,
        include_sentiment: bool = False,
        thread_id: str | None = None,
        format_for_llm: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fetch rich context from a channel for LLM prompt enrichment.

        This implements the ClawdBot pattern of context injection, providing
        comprehensive channel state for multi-agent deliberation. Returns
        structured context suitable for injection into LLM prompts.

        Args:
            channel_id: Channel to fetch context from
            lookback_minutes: How far back to look for messages (default: 60)
            max_messages: Maximum messages to retrieve (default: 50)
            include_participants: Whether to extract participant info (default: True)
            include_topics: Whether to extract discussion topics (default: True)
            include_sentiment: Whether to analyze sentiment (default: False)
            thread_id: Optional thread/conversation to focus on
            format_for_llm: Whether to include LLM-ready formatted string (default: True)
            **kwargs: Platform-specific options

        Returns:
            Dict containing:
                - channel: Channel information
                - messages: List of recent messages
                - participants: List of active participants
                - topics: Extracted discussion topics (if include_topics)
                - sentiment: Sentiment analysis (if include_sentiment)
                - statistics: Message statistics (counts, activity patterns)
                - formatted_context: LLM-ready context string (if format_for_llm)
                - metadata: Additional context metadata

        Example:
            context = await connector.fetch_rich_context(
                "C123456",
                lookback_minutes=30,
                include_topics=True,
            )

            # Use formatted context in LLM prompt
            prompt = f\"\"\"
            Channel context:
            {context['formatted_context']}

            Based on this discussion, respond to: {user_query}
            \"\"\"
        """
        from datetime import datetime

        # Fetch base context
        base_context = await self.fetch_context(
            channel_id=channel_id,
            lookback_minutes=lookback_minutes,
            max_messages=max_messages,
            include_participants=include_participants,
            thread_id=thread_id,
            **kwargs,
        )

        # Build rich context
        rich_context: dict[str, Any] = {
            "channel": {
                "id": base_context.channel.id,
                "name": base_context.channel.name,
                "platform": base_context.channel.platform,
                "type": base_context.channel.channel_type,
            },
            "messages": [
                {
                    "id": msg.id,
                    "author": msg.author.display_name or msg.author.username or msg.author.id,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "thread_id": msg.thread_id,
                }
                for msg in base_context.messages
            ],
            "participants": [
                {
                    "id": p.id,
                    "name": p.display_name or p.username or p.id,
                    "is_bot": p.is_bot,
                }
                for p in base_context.participants
            ],
            "statistics": {
                "message_count": base_context.message_count,
                "participant_count": base_context.participant_count,
                "timespan_minutes": lookback_minutes,
                "oldest_message": (
                    base_context.oldest_timestamp.isoformat()
                    if base_context.oldest_timestamp
                    else None
                ),
                "newest_message": (
                    base_context.newest_timestamp.isoformat()
                    if base_context.newest_timestamp
                    else None
                ),
            },
            "metadata": {
                "platform": self.platform_name,
                "fetched_at": datetime.utcnow().isoformat(),
                "thread_id": thread_id,
                **base_context.metadata,
            },
            "warnings": base_context.warnings,
        }

        # Extract discussion topics
        if include_topics:
            topics = self._extract_topics(base_context.messages)
            rich_context["topics"] = topics

        # Analyze sentiment (basic implementation)
        if include_sentiment:
            sentiment = self._analyze_sentiment(base_context.messages)
            rich_context["sentiment"] = sentiment

        # Calculate activity patterns
        rich_context["activity"] = self._calculate_activity_patterns(base_context.messages)

        # Format for LLM consumption
        if format_for_llm:
            rich_context["formatted_context"] = self._format_context_for_llm(rich_context)

        logger.debug(
            f"Fetched rich context from {self.platform_name} channel {channel_id}: "
            f"{len(base_context.messages)} messages, "
            f"{len(rich_context.get('topics', []))} topics extracted"
        )

        return rich_context

    def _extract_topics(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """
        Extract discussion topics from messages.

        Simple keyword extraction - can be overridden for more sophisticated
        NLP-based topic extraction.

        Args:
            messages: List of messages to analyze

        Returns:
            List of topic dicts with topic and frequency
        """
        return _extract_topics_impl(messages)

    def _analyze_sentiment(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """
        Basic sentiment analysis of messages.

        Simple keyword-based sentiment - can be overridden for more sophisticated
        analysis using NLP models.

        Args:
            messages: List of messages to analyze

        Returns:
            Dict with sentiment metrics
        """
        return _analyze_sentiment_impl(messages)

    def _calculate_activity_patterns(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """
        Calculate activity patterns from messages.

        Args:
            messages: List of messages to analyze

        Returns:
            Dict with activity metrics
        """
        return _calculate_activity_patterns_impl(messages)

    def _format_context_for_llm(self, rich_context: dict[str, Any]) -> str:
        """
        Format rich context into an LLM-ready string.

        Args:
            rich_context: The rich context dictionary

        Returns:
            Formatted string suitable for LLM prompt injection
        """
        return _format_context_for_llm_impl(rich_context)

    # ==========================================================================
    # Session Management Integration
    # ==========================================================================

    def _get_session_manager(self) -> Any:
        """
        Get the debate session manager (lazy initialization).

        Returns:
            DebateSessionManager instance, or None if unavailable
        """
        if not hasattr(self, "_session_manager"):
            try:
                from aragora.connectors.debate_session import get_debate_session_manager

                self._session_manager = get_debate_session_manager()
            except ImportError:
                logger.warning("Session manager not available")
                self._session_manager = None
        return self._session_manager

    async def get_or_create_session(
        self,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> Any | None:
        """
        Get or create a debate session for a user on this platform.

        This method integrates the chat connector with the session management
        system, enabling cross-channel session tracking and debate routing.

        Args:
            user_id: Platform-specific user identifier
            context: Optional context metadata for the session

        Returns:
            DebateSession if session manager is available, None otherwise

        Example:
            session = await slack.get_or_create_session("U123456")
            if session:
                await session_manager.link_debate(session.session_id, "debate-abc")
        """
        manager = self._get_session_manager()
        if not manager:
            return None

        session_context = {
            "platform": self.platform_name,
            **(context or {}),
        }

        return await manager.get_or_create_session(
            channel=self.platform_name,
            user_id=user_id,
            context=session_context,
        )

    async def link_debate_to_session(
        self,
        user_id: str,
        debate_id: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create or get a session and link it to a debate.

        Convenience method that combines session creation with debate linking.
        Used when starting a debate from a chat platform.

        Args:
            user_id: Platform-specific user identifier
            debate_id: Debate to link to the session
            context: Optional session context

        Returns:
            Session ID if successful, None if session manager unavailable

        Example:
            session_id = await telegram.link_debate_to_session(
                user_id="user123",
                debate_id="debate-abc",
                context={"message_id": "msg123", "chat_id": "chat456"}
            )
        """
        manager = self._get_session_manager()
        if not manager:
            return None

        session = await self.get_or_create_session(user_id, context)
        if not session:
            return None

        await manager.link_debate(session.session_id, debate_id)
        logger.debug(
            f"Linked debate {debate_id[:8]} to {self.platform_name} session for user {user_id}"
        )
        return session.session_id

    async def find_sessions_for_debate(self, debate_id: str) -> list[Any]:
        """
        Find all sessions on this platform linked to a debate.

        Used for routing debate results back to the originating channel.

        Args:
            debate_id: The debate ID to search for

        Returns:
            List of sessions for this debate on this platform
        """
        manager = self._get_session_manager()
        if not manager:
            return []

        all_sessions = await manager.find_sessions_for_debate(debate_id)
        return [s for s in all_sessions if s.channel == self.platform_name]

    async def route_debate_result(
        self,
        debate_id: str,
        result: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> list[SendMessageResponse]:
        """
        Route a debate result to all sessions on this platform.

        Finds all sessions linked to the debate and sends the result to
        each session's channel/thread.

        Args:
            debate_id: The completed debate ID
            result: The debate result/consensus text
            channel_id: Override channel ID (uses session context if not provided)
            thread_id: Override thread ID (uses session context if not provided)
            **kwargs: Additional arguments passed to send_message

        Returns:
            List of SendMessageResponse for each message sent

        Example:
            responses = await slack.route_debate_result(
                debate_id="debate-abc",
                result="The consensus is to use token bucket algorithm..."
            )
        """
        sessions = await self.find_sessions_for_debate(debate_id)
        responses = []

        for session in sessions:
            # Get channel/thread from session context
            ctx = session.context or {}
            target_channel = channel_id or ctx.get("channel_id") or ctx.get("chat_id")
            target_thread = thread_id or ctx.get("thread_id") or ctx.get("message_id")

            if not target_channel:
                logger.warning(
                    f"No channel found in session {session.session_id} context, skipping"
                )
                continue

            try:
                response = await self.send_message(
                    channel_id=target_channel,
                    text=result,
                    thread_id=target_thread,
                    **kwargs,
                )
                responses.append(response)
                logger.debug(
                    f"Routed debate {debate_id[:8]} result to {self.platform_name} "
                    f"channel {target_channel}"
                )
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(
                    f"Failed to route debate result to {self.platform_name} "
                    f"channel {target_channel}: {e}"
                )

        return responses

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform_name})"
