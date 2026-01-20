"""
Chat Platform Connector - Abstract base class for chat integrations.

All chat platform connectors inherit from ChatPlatformConnector and implement
standardized methods for:
- Sending and updating messages
- Handling bot commands
- Processing user interactions
- File operations (upload/download)
- Voice message handling
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    MessageBlock,
    MessageButton,
    SendMessageRequest,
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
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
        **config: Any,
    ):
        """
        Initialize the connector.

        Args:
            bot_token: API token for the bot
            signing_secret: Secret for webhook verification
            webhook_url: Default webhook URL for sending messages
            **config: Additional platform-specific configuration
        """
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.webhook_url = webhook_url
        self.config = config
        self._initialized = False

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform_name})"
