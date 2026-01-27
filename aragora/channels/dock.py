"""
Channel Dock - Abstract base class for platform-specific message delivery.

Defines the interface that all platform docks must implement,
along with capability flags for feature detection.

Example:
    class SlackDock(ChannelDock):
        PLATFORM = "slack"
        CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS

        async def send_message(self, origin, message):
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.channels.normalized import NormalizedMessage

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelDock",
    "ChannelCapability",
    "MessageType",
    "SendResult",
]


class ChannelCapability(Flag):
    """Capabilities that a channel platform supports."""

    NONE = 0
    RICH_TEXT = auto()  # Markdown/HTML formatting
    BUTTONS = auto()  # Interactive buttons
    THREADS = auto()  # Threaded replies
    FILES = auto()  # File attachments
    REACTIONS = auto()  # Emoji reactions
    VOICE = auto()  # Voice messages / TTS
    CARDS = auto()  # Adaptive cards (Teams)
    INLINE_IMAGES = auto()  # Inline image support
    WEBHOOKS = auto()  # Webhook-based delivery


class MessageType:
    """Types of messages that can be sent."""

    RESULT = "result"  # Debate result
    RECEIPT = "receipt"  # Decision receipt
    ERROR = "error"  # Error notification
    VOICE = "voice"  # Voice/TTS message
    PROGRESS = "progress"  # Progress update
    NOTIFICATION = "notification"  # General notification


@dataclass
class SendResult:
    """Result of a message send operation."""

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    platform: str = ""
    channel_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        message_id: Optional[str] = None,
        platform: str = "",
        channel_id: str = "",
        **metadata: Any,
    ) -> "SendResult":
        """Create a successful result."""
        return cls(
            success=True,
            message_id=message_id,
            platform=platform,
            channel_id=channel_id,
            metadata=metadata,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        platform: str = "",
        channel_id: str = "",
        **metadata: Any,
    ) -> "SendResult":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            platform=platform,
            channel_id=channel_id,
            metadata=metadata,
        )


class ChannelDock(ABC):
    """
    Abstract base class for platform-specific message delivery.

    Each platform (Slack, Telegram, etc.) implements this interface
    to provide consistent message delivery behavior.

    Class Attributes:
        PLATFORM: The platform identifier (e.g., "slack", "telegram")
        CAPABILITIES: Flags indicating what features the platform supports
    """

    PLATFORM: str = "unknown"
    CAPABILITIES: ChannelCapability = ChannelCapability.NONE

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the dock with optional configuration.

        Args:
            config: Platform-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the dock (authenticate, connect, etc.).

        Override this method to perform platform-specific initialization.

        Returns:
            True if initialization succeeded
        """
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        """Check if the dock is initialized."""
        return self._initialized

    def supports(self, capability: ChannelCapability) -> bool:
        """
        Check if this dock supports a specific capability.

        Args:
            capability: The capability to check

        Returns:
            True if the capability is supported
        """
        return bool(self.CAPABILITIES & capability)

    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a message to a channel.

        This is the primary send method that all docks must implement.

        Args:
            channel_id: The channel/chat/room identifier
            message: The normalized message to send
            **kwargs: Platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        ...

    async def send_result(
        self,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a debate result to a channel.

        Args:
            channel_id: The channel identifier
            result: The debate result dictionary
            thread_id: Optional thread to reply in
            **kwargs: Platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        from aragora.channels.normalized import NormalizedMessage, MessageFormat

        # Create normalized message from result
        message = NormalizedMessage(
            content=self._format_result(result),
            message_type=MessageType.RESULT,
            format=MessageFormat.MARKDOWN,
            metadata={"result": result},
            thread_id=thread_id,
        )
        return await self.send_message(channel_id, message, **kwargs)

    async def send_receipt(
        self,
        channel_id: str,
        summary: str,
        receipt_url: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a decision receipt to a channel.

        Args:
            channel_id: The channel identifier
            summary: Receipt summary text
            receipt_url: Optional URL to full receipt
            thread_id: Optional thread to reply in
            **kwargs: Platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        from aragora.channels.normalized import NormalizedMessage, MessageFormat

        message = NormalizedMessage(
            content=summary,
            message_type=MessageType.RECEIPT,
            format=MessageFormat.MARKDOWN,
            metadata={"receipt_url": receipt_url} if receipt_url else {},
            thread_id=thread_id,
        )
        return await self.send_message(channel_id, message, **kwargs)

    async def send_error(
        self,
        channel_id: str,
        error_message: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Send an error message to a channel.

        Args:
            channel_id: The channel identifier
            error_message: The error message
            thread_id: Optional thread to reply in
            **kwargs: Platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        from aragora.channels.normalized import NormalizedMessage, MessageFormat

        message = NormalizedMessage(
            content=f"Error: {error_message}",
            message_type=MessageType.ERROR,
            format=MessageFormat.PLAIN,
            thread_id=thread_id,
        )
        return await self.send_message(channel_id, message, **kwargs)

    async def send_voice(
        self,
        channel_id: str,
        audio_data: bytes,
        text: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a voice message to a channel.

        Args:
            channel_id: The channel identifier
            audio_data: Audio file bytes
            text: Optional text transcript
            thread_id: Optional thread to reply in
            **kwargs: Platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        if not self.supports(ChannelCapability.VOICE):
            return SendResult.fail(
                error=f"{self.PLATFORM} does not support voice messages",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        from aragora.channels.normalized import NormalizedMessage, MessageFormat

        message = NormalizedMessage(
            content=text or "",
            message_type=MessageType.VOICE,
            format=MessageFormat.PLAIN,
            attachments=[{"type": "audio", "data": audio_data}],
            thread_id=thread_id,
        )
        return await self.send_message(channel_id, message, **kwargs)

    def _format_result(self, result: dict[str, Any]) -> str:
        """
        Format a debate result for display.

        Override for platform-specific formatting.

        Args:
            result: The debate result dictionary

        Returns:
            Formatted string for display
        """
        decision = result.get("decision", "No decision reached")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")

        lines = [
            f"**Decision:** {decision}",
            f"**Confidence:** {confidence:.0%}",
        ]
        if reasoning:
            lines.append(f"\n**Reasoning:**\n{reasoning}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__} platform={self.PLATFORM}>"
