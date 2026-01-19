"""
Base classes for the unified bot framework.

Provides platform-agnostic abstractions for bot commands, messages,
and user interactions across Slack, Discord, Teams, and Zoom.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported chat platforms."""

    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    ZOOM = "zoom"


@dataclass
class BotUser:
    """Represents a user on any platform."""

    id: str
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    is_bot: bool = False
    platform: Platform = Platform.SLACK
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def mention(self) -> str:
        """Get platform-specific mention string."""
        if self.platform == Platform.SLACK:
            return f"<@{self.id}>"
        elif self.platform == Platform.DISCORD:
            return f"<@{self.id}>"
        elif self.platform == Platform.TEAMS:
            return f"<at>{self.display_name or self.username}</at>"
        elif self.platform == Platform.ZOOM:
            return f"@{self.username}"
        return f"@{self.username}"


@dataclass
class BotChannel:
    """Represents a channel/conversation on any platform."""

    id: str
    name: Optional[str] = None
    is_private: bool = False
    is_dm: bool = False
    platform: Platform = Platform.SLACK
    thread_id: Optional[str] = None  # For threaded conversations
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BotMessage:
    """Represents a message on any platform."""

    id: str
    text: str
    user: BotUser
    channel: BotChannel
    timestamp: datetime
    platform: Platform
    thread_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_threaded(self) -> bool:
        return self.thread_id is not None


@dataclass
class CommandContext:
    """Context passed to command handlers."""

    message: BotMessage
    user: BotUser
    channel: BotChannel
    platform: Platform
    args: List[str] = field(default_factory=list)
    raw_args: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str:
        return self.user.id

    @property
    def channel_id(self) -> str:
        return self.channel.id

    @property
    def thread_id(self) -> Optional[str]:
        return self.message.thread_id or self.channel.thread_id


@dataclass
class CommandResult:
    """Result from a command execution."""

    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    ephemeral: bool = False  # Only visible to the user who triggered

    # Platform-specific formatting
    slack_blocks: Optional[List[Dict[str, Any]]] = None
    discord_embed: Optional[Dict[str, Any]] = None
    teams_card: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, message: str, **kwargs: Any) -> "CommandResult":
        """Create a successful result."""
        return cls(success=True, message=message, **kwargs)

    @classmethod
    def fail(cls, error: str, **kwargs: Any) -> "CommandResult":
        """Create a failed result."""
        return cls(success=False, error=error, **kwargs)


@dataclass
class BotConfig:
    """Configuration for a bot instance."""

    platform: Platform
    token: str
    app_id: Optional[str] = None
    client_secret: Optional[str] = None
    signing_secret: Optional[str] = None  # For webhook verification
    api_base: str = "http://localhost:8080"
    ws_url: str = "ws://localhost:8080/ws"

    # Rate limiting
    rate_limit_per_user: int = 10  # Commands per minute per user
    rate_limit_global: int = 100  # Total commands per minute

    # Feature flags
    enable_threading: bool = True
    enable_reactions: bool = True
    enable_file_uploads: bool = True

    # Metadata
    bot_name: str = "Aragora"
    bot_username: str = "aragora"


class BaseBotClient(ABC):
    """Abstract base class for platform-specific bot clients."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.platform = config.platform
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""
        pass

    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        text: str,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Send a message to a channel. Returns message ID if successful."""
        pass

    @abstractmethod
    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
    ) -> bool:
        """Send an ephemeral message visible only to one user."""
        pass

    @abstractmethod
    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """Add a reaction to a message."""
        pass

    @abstractmethod
    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Update an existing message."""
        pass

    async def send_result(
        self,
        ctx: CommandContext,
        result: CommandResult,
    ) -> Optional[str]:
        """Send a command result to the appropriate channel."""
        if result.ephemeral:
            await self.send_ephemeral(
                ctx.channel_id,
                ctx.user_id,
                result.message or result.error or "Done",
            )
            return None

        text = result.message or result.error or "Command completed"
        attachments = result.attachments

        # Use platform-specific formatting if available
        if self.platform == Platform.SLACK and result.slack_blocks:
            attachments = [{"blocks": result.slack_blocks}]
        elif self.platform == Platform.DISCORD and result.discord_embed:
            attachments = [result.discord_embed]
        elif self.platform == Platform.TEAMS and result.teams_card:
            attachments = [result.teams_card]

        return await self.send_message(
            ctx.channel_id,
            text,
            thread_id=ctx.thread_id,
            attachments=attachments,
        )


class BotEventHandler(ABC):
    """Abstract base class for handling platform events."""

    def __init__(self, client: BaseBotClient):
        self.client = client

    @abstractmethod
    async def on_message(self, message: BotMessage) -> None:
        """Handle incoming message."""
        pass

    @abstractmethod
    async def on_reaction_added(
        self,
        channel_id: str,
        message_id: str,
        user_id: str,
        emoji: str,
    ) -> None:
        """Handle reaction added to a message."""
        pass

    @abstractmethod
    async def on_command(self, ctx: CommandContext) -> None:
        """Handle a parsed command."""
        pass

    async def on_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle errors during event processing."""
        logger.error(f"Bot error: {error}", exc_info=True, extra={"context": context})
