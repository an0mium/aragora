"""
Chat platform data models.

Unified data structures for cross-platform chat integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    """Type of chat message."""

    TEXT = "text"
    RICH = "rich"  # Formatted/structured message
    FILE = "file"
    VOICE = "voice"
    COMMAND = "command"
    INTERACTION = "interaction"


class InteractionType(str, Enum):
    """Type of user interaction."""

    BUTTON_CLICK = "button_click"
    SELECT_MENU = "select_menu"
    MODAL_SUBMIT = "modal_submit"
    SHORTCUT = "shortcut"


class UserRole(str, Enum):
    """User role in a channel or workspace."""

    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    GUEST = "guest"
    UNKNOWN = "unknown"


@dataclass
class ChatUser:
    """Represents a user across chat platforms."""

    id: str
    platform: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    # Enrichment fields for context injection
    timezone: Optional[str] = None  # IANA timezone (e.g., "America/New_York")
    language: Optional[str] = None  # ISO 639-1 code (e.g., "en", "es")
    locale: Optional[str] = None  # Full locale (e.g., "en-US")
    role: UserRole = UserRole.UNKNOWN
    # Status and activity
    status: Optional[str] = None  # "online", "away", "dnd", "offline"
    last_active: Optional[datetime] = None
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cache timestamp for TTL
    _enriched_at: Optional[datetime] = field(default=None, repr=False)

    @property
    def is_enriched(self) -> bool:
        """Check if user has enrichment data."""
        return self.timezone is not None or self.language is not None

    def to_context_dict(self) -> dict[str, Any]:
        """Export enrichment data for debate prompt context."""
        return {
            "user_id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "timezone": self.timezone,
            "language": self.language,
            "locale": self.locale,
            "role": self.role.value if self.role else None,
            "status": self.status,
            "is_bot": self.is_bot,
        }


class ChannelType(str, Enum):
    """Channel/conversation type."""

    PUBLIC = "public"  # Public channel anyone can join
    PRIVATE = "private"  # Private channel with restricted access
    DM = "dm"  # Direct message between two users
    GROUP_DM = "group_dm"  # Group direct message
    THREAD = "thread"  # Thread within a channel
    UNKNOWN = "unknown"


@dataclass
class ChatChannel:
    """Represents a channel/conversation across platforms."""

    id: str
    platform: str
    name: Optional[str] = None
    is_private: bool = False
    is_dm: bool = False
    team_id: Optional[str] = None  # Workspace/Guild/Organization
    # Enrichment fields for context injection
    channel_type: ChannelType = ChannelType.UNKNOWN
    topic: Optional[str] = None  # Channel topic/purpose
    description: Optional[str] = None  # Longer channel description
    member_count: Optional[int] = None
    # Timestamps
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cache timestamp for TTL
    _enriched_at: Optional[datetime] = field(default=None, repr=False)

    @property
    def is_enriched(self) -> bool:
        """Check if channel has enrichment data."""
        return self.topic is not None or self.member_count is not None

    def to_context_dict(self) -> dict[str, Any]:
        """Export enrichment data for debate prompt context."""
        return {
            "channel_id": self.id,
            "channel_name": self.name,
            "channel_type": self.channel_type.value if self.channel_type else None,
            "topic": self.topic,
            "description": self.description,
            "member_count": self.member_count,
            "team_id": self.team_id,
            "is_private": self.is_private,
            "is_dm": self.is_dm,
        }


@dataclass
class ChatMessage:
    """Unified message structure for all chat platforms."""

    id: str
    platform: str
    channel: ChatChannel
    author: ChatUser
    content: str
    message_type: MessageType = MessageType.TEXT

    # Threading
    thread_id: Optional[str] = None
    reply_to_id: Optional[str] = None

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    edited_at: Optional[datetime] = None

    # Rich content
    blocks: Optional[list[dict[str, Any]]] = None  # Platform-specific rich content
    attachments: list[dict[str, Any]] = field(default_factory=list)

    # Platform-specific
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "platform": self.platform,
            "channel": {
                "id": self.channel.id,
                "name": self.channel.name,
                "is_private": self.channel.is_private,
            },
            "author": {
                "id": self.author.id,
                "username": self.author.username,
                "display_name": self.author.display_name,
                "is_bot": self.author.is_bot,
            },
            "content": self.content,
            "message_type": self.message_type.value,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "metadata": self.metadata,
        }


@dataclass
class BotCommand:
    """Represents a slash command or bot command."""

    name: str
    text: str  # Full command text
    args: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    user: Optional[ChatUser] = None
    channel: Optional[ChatChannel] = None
    platform: str = ""
    response_url: Optional[str] = None  # For async responses
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserInteraction:
    """Represents a user interaction (button click, menu select, etc.)."""

    id: str
    interaction_type: InteractionType
    action_id: str
    value: Optional[str] = None
    values: list[str] = field(default_factory=list)
    user: Optional[ChatUser] = None
    channel: Optional[ChatChannel] = None
    message_id: Optional[str] = None
    platform: str = ""
    response_url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageBlock:
    """Generic rich message block."""

    type: str
    text: Optional[str] = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    elements: list[dict[str, Any]] = field(default_factory=list)
    accessory: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageButton:
    """Interactive button element."""

    text: str
    action_id: str
    value: Optional[str] = None
    style: str = "default"  # default, primary, danger
    url: Optional[str] = None  # For link buttons
    confirm: Optional[dict[str, Any]] = None  # Confirmation dialog


@dataclass
class FileAttachment:
    """File attachment for messages."""

    id: str
    filename: str
    content_type: str
    size: int
    url: Optional[str] = None
    content: Optional[bytes] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceMessage:
    """Voice/audio message for transcription."""

    id: str
    channel: ChatChannel
    author: ChatUser
    duration_seconds: float
    file: FileAttachment
    transcription: Optional[str] = None
    platform: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendMessageRequest:
    """Request to send a message."""

    channel_id: str
    text: str
    blocks: Optional[list[dict[str, Any]]] = None
    thread_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    ephemeral: bool = False  # Only visible to specific user
    ephemeral_user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendMessageResponse:
    """Response from sending a message."""

    success: bool
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Alias for backwards compatibility
MessageSendResult = SendMessageResponse


@dataclass
class WebhookEvent:
    """Generic webhook event from any platform."""

    platform: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_payload: dict[str, Any] = field(default_factory=dict)
    message: Optional[ChatMessage] = None
    command: Optional[BotCommand] = None
    interaction: Optional[UserInteraction] = None
    voice_message: Optional[VoiceMessage] = None
    challenge: Optional[str] = None  # For URL verification
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verification(self) -> bool:
        """Check if this is a URL verification challenge."""
        return self.challenge is not None


@dataclass
class ChatEvidence:
    """Evidence collected from chat messages for debate grounding.

    Represents a chat message or thread that can be used as evidence
    in debates, with provenance tracking and relevance scoring.
    """

    id: str
    source_type: str = "chat"  # Always "chat" for this type
    source_id: str = ""  # Message ID or thread ID
    platform: str = ""  # slack, discord, teams, etc.
    channel_id: str = ""
    channel_name: Optional[str] = None

    # Content
    content: str = ""  # Message text
    title: str = ""  # Thread title or first message summary

    # Author info
    author_id: str = ""
    author_name: Optional[str] = None
    author_is_bot: bool = False

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    collected_at: datetime = field(default_factory=datetime.utcnow)

    # Threading
    thread_id: Optional[str] = None
    is_thread_root: bool = False
    reply_count: int = 0

    # Evidence quality indicators
    relevance_score: float = 1.0  # How relevant to the query (0-1)
    confidence: float = 0.5  # Base confidence in source
    freshness: float = 1.0  # Temporal freshness (1.0 = current)

    # Original message reference
    source_message: Optional[ChatMessage] = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reliability_score(self) -> float:
        """Combined reliability score for evidence weighting."""
        # Weight factors based on source characteristics
        # - Relevance is most important for debate grounding
        # - Freshness matters for time-sensitive topics
        # - Confidence based on source authority
        return 0.5 * self.relevance_score + 0.3 * self.freshness + 0.2 * self.confidence

    @property
    def source_url(self) -> Optional[str]:
        """Get a URL to the original message if available."""
        url: Optional[str] = self.metadata.get("permalink")
        return url

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "platform": self.platform,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "content": self.content,
            "title": self.title,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "timestamp": self.timestamp.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "thread_id": self.thread_id,
            "is_thread_root": self.is_thread_root,
            "reply_count": self.reply_count,
            "relevance_score": self.relevance_score,
            "reliability_score": self.reliability_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_message(
        cls,
        message: ChatMessage,
        query: str,
        relevance_score: float = 1.0,
    ) -> ChatEvidence:
        """Create ChatEvidence from a ChatMessage.

        Args:
            message: The source chat message
            query: The search query that found this message
            relevance_score: Relevance score for this evidence (0-1)

        Returns:
            ChatEvidence instance with data from the message
        """
        import uuid

        return cls(
            id=f"evidence_{uuid.uuid4().hex[:12]}",
            source_id=message.id,
            platform=message.platform,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            content=message.content,
            title=message.content[:100] if message.content else "",
            author_id=message.author.id,
            author_name=message.author.display_name or message.author.username,
            author_is_bot=message.author.is_bot,
            timestamp=message.timestamp,
            thread_id=message.thread_id,
            is_thread_root=message.thread_id == message.id,
            relevance_score=relevance_score,
            source_message=message,
            metadata={"query": query, **message.metadata},
        )


@dataclass
class ChannelContext:
    """
    Context fetched from a chat channel for deliberation.

    Used by the orchestration handler to auto-fetch context from
    channels before starting a deliberation.
    """

    channel: ChatChannel
    messages: list[ChatMessage] = field(default_factory=list)
    participants: list[ChatUser] = field(default_factory=list)

    # Time range of fetched messages
    oldest_timestamp: Optional[datetime] = None
    newest_timestamp: Optional[datetime] = None

    # Summary statistics
    message_count: int = 0
    participant_count: int = 0

    # Any errors or warnings during fetch
    warnings: list[str] = field(default_factory=list)

    # Metadata
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self, max_messages: int = 50) -> str:
        """
        Convert to a string suitable for deliberation context.

        Args:
            max_messages: Maximum messages to include in context
        """
        lines = [
            f"# Channel Context: {self.channel.name or self.channel.id}",
            f"Platform: {self.channel.platform}",
            f"Messages: {len(self.messages)} (showing last {min(len(self.messages), max_messages)})",
            f"Participants: {len(self.participants)}",
            "",
            "## Recent Messages",
            "",
        ]

        for msg in self.messages[-max_messages:]:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            author = msg.author.display_name or msg.author.username or msg.author.id
            lines.append(f"[{timestamp}] **{author}**: {msg.content}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "channel": {
                "id": self.channel.id,
                "platform": self.channel.platform,
                "name": self.channel.name,
            },
            "messages": [m.to_dict() for m in self.messages],
            "participants": [
                {
                    "id": p.id,
                    "username": p.username,
                    "display_name": p.display_name,
                }
                for p in self.participants
            ],
            "message_count": len(self.messages),
            "participant_count": len(self.participants),
            "oldest_timestamp": self.oldest_timestamp.isoformat()
            if self.oldest_timestamp
            else None,
            "newest_timestamp": self.newest_timestamp.isoformat()
            if self.newest_timestamp
            else None,
            "fetched_at": self.fetched_at.isoformat(),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_message(
        message: ChatMessage,
        query: Optional[str] = None,
        relevance_score: float = 1.0,
    ) -> "ChatEvidence":
        """Create ChatEvidence from a ChatMessage."""
        import hashlib

        evidence_id = hashlib.sha256(
            f"{message.platform}:{message.channel.id}:{message.id}".encode()
        ).hexdigest()[:16]

        return ChatEvidence(
            id=evidence_id,
            source_id=message.id,
            platform=message.platform,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            content=message.content,
            title=message.content[:100] if message.content else "",
            author_id=message.author.id,
            author_name=message.author.display_name or message.author.username,
            author_is_bot=message.author.is_bot,
            timestamp=message.timestamp,
            thread_id=message.thread_id,
            is_thread_root=message.thread_id == message.id,
            relevance_score=relevance_score,
            source_message=message,
            metadata=message.metadata,
        )


# =============================================================================
# Metadata Cache for TTL-based enrichment caching
# =============================================================================


@dataclass
class MetadataCacheEntry:
    """Cache entry for user or channel metadata."""

    data: dict[str, Any]
    enriched_at: datetime
    ttl_seconds: int = 3600  # Default 1 hour TTL

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        from datetime import timedelta

        return datetime.utcnow() > self.enriched_at + timedelta(seconds=self.ttl_seconds)


class MetadataCache:
    """
    Simple in-memory cache for user and channel metadata with TTL.

    Used to avoid repeated API calls for enrichment data that doesn't
    change frequently (timezone, language, role, channel topic, etc.).
    """

    def __init__(self, default_ttl: int = 3600):
        """
        Initialize metadata cache.

        Args:
            default_ttl: Default TTL in seconds (default: 1 hour)
        """
        self.default_ttl = default_ttl
        self._user_cache: dict[str, MetadataCacheEntry] = {}
        self._channel_cache: dict[str, MetadataCacheEntry] = {}

    def get_user(self, user_id: str, platform: str) -> Optional[dict[str, Any]]:
        """
        Get cached user metadata.

        Args:
            user_id: User ID
            platform: Platform name

        Returns:
            Cached metadata dict or None if not cached/expired
        """
        key = f"{platform}:{user_id}"
        entry = self._user_cache.get(key)
        if entry and not entry.is_expired:
            return entry.data
        elif entry:
            # Expired, clean up
            del self._user_cache[key]
        return None

    def set_user(
        self,
        user_id: str,
        platform: str,
        metadata: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache user metadata.

        Args:
            user_id: User ID
            platform: Platform name
            metadata: Metadata to cache
            ttl: TTL in seconds (default: use cache default)
        """
        key = f"{platform}:{user_id}"
        self._user_cache[key] = MetadataCacheEntry(
            data=metadata,
            enriched_at=datetime.utcnow(),
            ttl_seconds=ttl or self.default_ttl,
        )

    def get_channel(self, channel_id: str, platform: str) -> Optional[dict[str, Any]]:
        """
        Get cached channel metadata.

        Args:
            channel_id: Channel ID
            platform: Platform name

        Returns:
            Cached metadata dict or None if not cached/expired
        """
        key = f"{platform}:{channel_id}"
        entry = self._channel_cache.get(key)
        if entry and not entry.is_expired:
            return entry.data
        elif entry:
            # Expired, clean up
            del self._channel_cache[key]
        return None

    def set_channel(
        self,
        channel_id: str,
        platform: str,
        metadata: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache channel metadata.

        Args:
            channel_id: Channel ID
            platform: Platform name
            metadata: Metadata to cache
            ttl: TTL in seconds (default: use cache default)
        """
        key = f"{platform}:{channel_id}"
        self._channel_cache[key] = MetadataCacheEntry(
            data=metadata,
            enriched_at=datetime.utcnow(),
            ttl_seconds=ttl or self.default_ttl,
        )

    def invalidate_user(self, user_id: str, platform: str) -> None:
        """Invalidate cached user metadata."""
        key = f"{platform}:{user_id}"
        self._user_cache.pop(key, None)

    def invalidate_channel(self, channel_id: str, platform: str) -> None:
        """Invalidate cached channel metadata."""
        key = f"{platform}:{channel_id}"
        self._channel_cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached metadata."""
        self._user_cache.clear()
        self._channel_cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "user_entries": len(self._user_cache),
            "channel_entries": len(self._channel_cache),
            "default_ttl_seconds": self.default_ttl,
        }


# Global metadata cache instance
_metadata_cache: Optional[MetadataCache] = None


def get_metadata_cache() -> MetadataCache:
    """Get or create global metadata cache."""
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = MetadataCache()
    return _metadata_cache


def build_chat_context(
    user: Optional[ChatUser] = None,
    channel: Optional[ChatChannel] = None,
    include_user: bool = True,
    include_channel: bool = True,
) -> dict[str, Any]:
    """
    Build context dict from user and channel for debate prompt injection.

    Args:
        user: ChatUser with enrichment data
        channel: ChatChannel with enrichment data
        include_user: Whether to include user context
        include_channel: Whether to include channel context

    Returns:
        Context dict suitable for debate prompt injection
    """
    context: dict[str, Any] = {}

    if include_user and user:
        user_ctx = user.to_context_dict()
        # Filter out None values
        context["user"] = {k: v for k, v in user_ctx.items() if v is not None}

    if include_channel and channel:
        channel_ctx = channel.to_context_dict()
        # Filter out None values
        context["channel"] = {k: v for k, v in channel_ctx.items() if v is not None}

    return context
