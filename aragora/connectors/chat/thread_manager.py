"""
Thread management for chat platform connectors.

Provides abstract base class and data structures for managing conversation
threads across chat platforms like Slack and Microsoft Teams.

Each platform has different threading models:
- Slack: Uses thread_ts (timestamp of parent message)
- Teams: Uses replyToId (message ID reference)
- Discord: Uses threads as channel extensions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .models import ChatMessage


@dataclass
class ThreadInfo:
    """Thread metadata across platforms.

    Represents the common properties of a conversation thread,
    abstracting away platform-specific details.
    """

    id: str  # Platform-specific thread identifier
    channel_id: str  # Parent channel/conversation
    platform: str  # 'slack', 'teams', 'discord'

    # Thread metadata
    created_by: str  # User ID who started the thread
    created_at: datetime
    updated_at: datetime

    # Statistics
    message_count: int = 0
    participant_count: int = 0

    # Content
    title: Optional[str] = None  # Thread title or first message preview
    root_message_id: Optional[str] = None  # ID of the parent message

    # Status
    is_archived: bool = False
    is_locked: bool = False  # Thread doesn't accept new replies

    # Platform-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize thread info to dictionary."""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "platform": self.platform,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "participant_count": self.participant_count,
            "title": self.title,
            "root_message_id": self.root_message_id,
            "is_archived": self.is_archived,
            "is_locked": self.is_locked,
            "metadata": self.metadata,
        }


@dataclass
class ThreadStats:
    """Thread statistics for analytics and monitoring."""

    thread_id: str
    message_count: int
    participant_count: int
    last_activity: datetime

    # Response metrics
    avg_response_time_seconds: Optional[float] = None
    first_response_time_seconds: Optional[float] = None

    # Participation breakdown
    participants: list[str] = field(default_factory=list)  # User IDs

    # Content metrics
    total_reactions: int = 0
    total_attachments: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize thread stats to dictionary."""
        return {
            "thread_id": self.thread_id,
            "message_count": self.message_count,
            "participant_count": self.participant_count,
            "last_activity": self.last_activity.isoformat(),
            "avg_response_time_seconds": self.avg_response_time_seconds,
            "first_response_time_seconds": self.first_response_time_seconds,
            "participants": self.participants,
            "total_reactions": self.total_reactions,
            "total_attachments": self.total_attachments,
        }


@dataclass
class ThreadParticipant:
    """Information about a thread participant."""

    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    message_count: int = 0
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    is_bot: bool = False


class ThreadManager(ABC):
    """
    Abstract base class for platform-specific thread management.

    Provides a unified interface for thread operations across
    different chat platforms.

    Subclasses must implement the abstract methods to handle
    platform-specific thread APIs.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name (e.g., 'slack', 'teams')."""
        ...

    @abstractmethod
    async def get_thread(self, thread_id: str, channel_id: str) -> ThreadInfo:
        """
        Get thread information.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID

        Returns:
            ThreadInfo object with thread metadata

        Raises:
            ThreadNotFoundError: If thread doesn't exist
        """
        ...

    @abstractmethod
    async def get_thread_messages(
        self,
        thread_id: str,
        channel_id: str,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> tuple[list["ChatMessage"], Optional[str]]:
        """
        Get messages in a thread with pagination.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID
            limit: Maximum messages to return
            cursor: Pagination cursor for next page

        Returns:
            Tuple of (messages list, next page cursor or None)
        """
        ...

    @abstractmethod
    async def list_threads(
        self,
        channel_id: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> tuple[list[ThreadInfo], Optional[str]]:
        """
        List threads in a channel.

        Args:
            channel_id: Channel to list threads from
            limit: Maximum threads to return
            cursor: Pagination cursor for next page

        Returns:
            Tuple of (threads list, next page cursor or None)
        """
        ...

    @abstractmethod
    async def reply_to_thread(
        self,
        thread_id: str,
        channel_id: str,
        message: str,
        **kwargs: Any,
    ) -> "ChatMessage":
        """
        Reply to an existing thread.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID
            message: Reply message text
            **kwargs: Additional platform-specific options

        Returns:
            The sent message
        """
        ...

    @abstractmethod
    async def get_thread_stats(
        self,
        thread_id: str,
        channel_id: str,
    ) -> ThreadStats:
        """
        Get statistics for a thread.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID

        Returns:
            ThreadStats object with thread statistics
        """
        ...

    @abstractmethod
    async def get_thread_participants(
        self,
        thread_id: str,
        channel_id: str,
    ) -> list[ThreadParticipant]:
        """
        Get participants in a thread.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID

        Returns:
            List of ThreadParticipant objects
        """
        ...

    # ==========================================================================
    # Optional methods with default implementations
    # ==========================================================================

    async def archive_thread(
        self,
        thread_id: str,
        channel_id: str,
    ) -> bool:
        """
        Archive a thread (if supported by platform).

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID

        Returns:
            True if archived, False if not supported
        """
        return False

    async def lock_thread(
        self,
        thread_id: str,
        channel_id: str,
    ) -> bool:
        """
        Lock a thread to prevent new replies (if supported).

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID

        Returns:
            True if locked, False if not supported
        """
        return False

    async def get_thread_context(
        self,
        thread_id: str,
        channel_id: str,
        max_messages: int = 20,
    ) -> dict[str, Any]:
        """
        Get thread context for AI prompts.

        Returns a summary suitable for injecting into debate prompts.

        Args:
            thread_id: Platform-specific thread identifier
            channel_id: Parent channel/conversation ID
            max_messages: Maximum recent messages to include

        Returns:
            Dictionary with thread context for prompt building
        """
        thread = await self.get_thread(thread_id, channel_id)
        messages, _ = await self.get_thread_messages(thread_id, channel_id, limit=max_messages)
        participants = await self.get_thread_participants(thread_id, channel_id)

        return {
            "thread_id": thread.id,
            "channel_id": thread.channel_id,
            "platform": thread.platform,
            "title": thread.title,
            "message_count": thread.message_count,
            "participant_count": thread.participant_count,
            "created_at": thread.created_at.isoformat(),
            "participants": [
                {"user_id": p.user_id, "display_name": p.display_name}
                for p in participants[:10]  # Limit for context size
            ],
            "recent_messages": [
                {
                    "author": m.author.display_name or m.author.username,
                    "content": m.content[:500],  # Truncate long messages
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in messages[-max_messages:]
            ],
        }


class ThreadNotFoundError(Exception):
    """Raised when a thread cannot be found."""

    def __init__(self, thread_id: str, channel_id: str, platform: str):
        self.thread_id = thread_id
        self.channel_id = channel_id
        self.platform = platform
        super().__init__(f"Thread {thread_id} not found in channel {channel_id} on {platform}")
