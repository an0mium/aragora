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
from typing import Any, Optional

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
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
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
        self._circuit_breaker: Optional[Any] = None
        self._circuit_breaker_initialized = False

    # ==========================================================================
    # Circuit Breaker Support
    # ==========================================================================

    def _get_circuit_breaker(self) -> Optional[Any]:
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

    def _check_circuit_breaker(self) -> tuple[bool, Optional[str]]:
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

    def _record_failure(self, error: Optional[Exception] = None) -> None:
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
        func,
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: tuple = (Exception,),
        **kwargs,
    ):
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
        headers: Optional[dict[str, str]] = None,
        json: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[dict[str, Any]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        operation: str = "http_request",
    ) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        """
        Make an HTTP request with retry, timeout, and circuit breaker support.

        This is the recommended method for all HTTP operations in chat connectors.
        Provides consistent error handling, logging, and resilience patterns.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            url: Request URL
            headers: Optional request headers
            json: Optional JSON body
            data: Optional form data
            files: Optional file uploads
            max_retries: Maximum retry attempts (default 3)
            base_delay: Initial retry delay in seconds (default 1.0)
            operation: Operation name for logging

        Returns:
            Tuple of (success: bool, response_json: Optional[dict], error: Optional[str])
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

        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=json,
                        data=data,
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
                    try:
                        return True, response.json(), None
                    except Exception:
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

            except Exception as e:
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

    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send a message to a channel.

        Args:
            channel_id: Target channel/conversation ID
            text: Plain text content (fallback for clients without rich support)
            blocks: Rich content blocks in platform-native format
            thread_id: Optional thread/reply ID for threaded messages
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with message ID and status
        """
        raise NotImplementedError

    @abstractmethod
    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update an existing message.

        Args:
            channel_id: Channel containing the message
            message_id: ID of message to update
            text: New text content
            blocks: New rich content blocks
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with update status
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a message.

        Args:
            channel_id: Channel containing the message
            message_id: ID of message to delete
            **kwargs: Platform-specific options

        Returns:
            True if deleted successfully
        """
        raise NotImplementedError

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
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

    @abstractmethod
    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Respond to a slash command.

        Args:
            command: The command that was invoked
            text: Response text
            blocks: Rich content blocks
            ephemeral: If True, only the user sees the response
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        raise NotImplementedError

    # ==========================================================================
    # Interaction Handling
    # ==========================================================================

    @abstractmethod
    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Respond to a user interaction (button click, menu select, etc.).

        Args:
            interaction: The interaction event
            text: Response text
            blocks: Rich content blocks
            replace_original: If True, replace the original message
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        raise NotImplementedError

    # ==========================================================================
    # File Operations
    # ==========================================================================

    @abstractmethod
    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """
        Upload a file to a channel.

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
        """
        raise NotImplementedError

    @abstractmethod
    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """
        Download a file by ID.

        Args:
            file_id: ID of the file to download
            **kwargs: Platform-specific options

        Returns:
            FileAttachment with content populated
        """
        raise NotImplementedError

    async def send_voice_message(
        self,
        channel_id: str,
        audio_content: bytes,
        filename: str = "voice_response.mp3",
        content_type: str = "audio/mpeg",
        reply_to: Optional[str] = None,
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
        except Exception as e:
            logger.error(f"Failed to send voice message on {self.platform_name}: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    # ==========================================================================
    # Rich Content Formatting
    # ==========================================================================

    @abstractmethod
    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[MessageButton]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Format content into platform-specific rich blocks.

        Args:
            title: Section title/header
            body: Main text content
            fields: List of (label, value) tuples for structured data
            actions: List of interactive buttons
            **kwargs: Platform-specific options

        Returns:
            List of platform-specific block structures
        """
        raise NotImplementedError

    @abstractmethod
    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict:
        """
        Format a button element.

        Args:
            text: Button label
            action_id: Unique action identifier
            value: Value to pass when clicked
            style: Button style (default, primary, danger)
            url: Optional URL for link buttons

        Returns:
            Platform-specific button structure
        """
        raise NotImplementedError

    # ==========================================================================
    # Webhook Handling
    # ==========================================================================

    @abstractmethod
    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """
        Verify webhook signature for security.

        Args:
            headers: HTTP headers from the webhook request
            body: Raw request body

        Returns:
            True if signature is valid
        """
        raise NotImplementedError

    @abstractmethod
    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """
        Parse a webhook payload into a WebhookEvent.

        Args:
            headers: HTTP headers from the request
            body: Raw request body

        Returns:
            Parsed WebhookEvent
        """
        raise NotImplementedError

    # ==========================================================================
    # Voice Message Handling
    # ==========================================================================

    async def get_voice_message(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> Optional[VoiceMessage]:
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
    ) -> Optional[ChatChannel]:
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
    ) -> Optional[ChatUser]:
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

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    async def test_connection(self) -> dict:
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
        query: Optional[str] = None,
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
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
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
        query: Optional[str] = None,
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
        thread_id: Optional[str] = None,
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

    def _format_timestamp_for_api(self, timestamp: Any) -> Optional[str]:
        """
        Format a datetime for the platform's API.

        Override in subclasses for platform-specific formatting.
        Default returns ISO format.
        """
        from datetime import datetime

        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp) if timestamp else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform_name})"
