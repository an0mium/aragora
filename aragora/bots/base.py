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
from typing import Any, Dict, List, Optional

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
    """Configuration for a bot instance.

    Note: api_base and ws_url are REQUIRED in production.
    Set ARAGORA_API_BASE and ARAGORA_WS_URL environment variables.
    """

    platform: Platform
    token: str
    app_id: Optional[str] = None
    client_secret: Optional[str] = None
    signing_secret: Optional[str] = None  # For webhook verification
    api_base: str = ""  # Required in production, defaults to localhost only in dev
    ws_url: str = ""  # Required in production, defaults to localhost only in dev

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

        # Validate and apply URL defaults based on environment
        import os

        env = os.environ.get("ARAGORA_ENV", "development").lower()
        is_production = env in ("production", "prod", "live")

        # Apply localhost defaults only in development
        if not config.api_base:
            if is_production:
                raise ValueError(
                    "api_base is required in production. "
                    "Set ARAGORA_API_BASE environment variable."
                )
            config.api_base = "http://localhost:8080"
            logger.debug("Using localhost api_base (development mode)")

        if not config.ws_url:
            if is_production:
                raise ValueError(
                    "ws_url is required in production. " "Set ARAGORA_WS_URL environment variable."
                )
            config.ws_url = "ws://localhost:8080/ws"
            logger.debug("Using localhost ws_url (development mode)")

        # Error if localhost is explicitly set in production
        if is_production:
            if "localhost" in config.api_base or "127.0.0.1" in config.api_base:
                raise ValueError(
                    f"Bot api_base cannot use localhost in production ({config.api_base}). "
                    "Set ARAGORA_API_BASE to production URL."
                )
            if "localhost" in config.ws_url or "127.0.0.1" in config.ws_url:
                raise ValueError(
                    f"Bot ws_url cannot use localhost in production ({config.ws_url}). "
                    "Set ARAGORA_WS_URL to production URL."
                )

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


class DefaultBotEventHandler(BotEventHandler):
    """Default implementation of BotEventHandler with routing logic.

    Provides sensible defaults for message routing, command parsing,
    and reaction handling that can be customized by subclasses.
    """

    def __init__(
        self,
        client: BaseBotClient,
        command_prefix: str = "/",
        debate_keywords: Optional[List[str]] = None,
    ):
        super().__init__(client)
        self.command_prefix = command_prefix
        self.debate_keywords = debate_keywords or ["debate", "discuss", "argue"]
        self._active_debates: Dict[str, str] = {}  # channel_id -> debate_id
        self._registry: Optional[Any] = None

    def set_command_registry(self, registry: Any) -> None:
        """Set the command registry for command execution."""
        self._registry = registry

    async def on_message(self, message: BotMessage) -> None:
        """Route message to appropriate handler.

        Routing logic:
        1. Commands (starting with prefix) -> on_command()
        2. Messages in active debate channels -> _route_to_debate()
        3. All other messages -> _handle_general_message()
        """
        # Skip bot messages
        if message.user.is_bot:
            return

        text = message.text.strip()

        # Route commands
        if text.startswith(self.command_prefix):
            ctx = self._parse_command(message)
            if ctx:
                await self.on_command(ctx)
                return

        # Route debate messages
        if self._is_debate_context(message):
            await self._route_to_debate(message)
            return

        # Handle general messages
        await self._handle_general_message(message)

    async def on_reaction_added(
        self,
        channel_id: str,
        message_id: str,
        user_id: str,
        emoji: str,
    ) -> None:
        """Handle reaction added to a message.

        Default implementation handles debate-related reactions:
        - thumbsup/thumbsdown for voting
        - question mark for clarification requests
        """
        # Check if this is in an active debate
        if channel_id in self._active_debates:
            await self._handle_debate_reaction(channel_id, message_id, user_id, emoji)
            return

        # Log other reactions
        logger.debug(
            f"Reaction added: emoji={emoji} channel={channel_id} "
            f"message={message_id} user={user_id}"
        )

    async def on_command(self, ctx: CommandContext) -> None:
        """Execute a command through the registry.

        Default implementation uses the command registry if available,
        otherwise logs a warning.
        """
        if not self._registry:
            logger.warning(f"No command registry set, cannot execute: {ctx.args}")
            return

        try:
            result = await self._registry.execute(ctx)
            await self.client.send_result(ctx, result)
        except Exception as e:
            await self.on_error(e, {"command": ctx.args, "user": ctx.user_id})
            await self.client.send_message(
                ctx.channel_id,
                f"Error executing command: {str(e)}",
                thread_id=ctx.thread_id,
            )

    def _parse_command(self, message: BotMessage) -> Optional[CommandContext]:
        """Parse a message into a CommandContext if it's a command."""
        text = message.text.strip()

        if not text.startswith(self.command_prefix):
            return None

        # Remove prefix and split
        text = text[len(self.command_prefix) :]
        parts = text.split()

        if not parts:
            return None

        return CommandContext(
            message=message,
            user=message.user,
            channel=message.channel,
            platform=message.platform,
            args=parts,
            raw_args=" ".join(parts[1:]) if len(parts) > 1 else "",
        )

    def _is_debate_context(self, message: BotMessage) -> bool:
        """Check if a message is in a debate context.

        Returns True if:
        - The channel has an active debate
        - The message mentions debate keywords
        """
        # Check for active debate in channel
        if message.channel.id in self._active_debates:
            return True

        # Check for debate keywords
        text_lower = message.text.lower()
        return any(keyword in text_lower for keyword in self.debate_keywords)

    async def _route_to_debate(self, message: BotMessage) -> None:
        """Route a message to the debate system.

        Override this method to integrate with your debate backend.
        """
        debate_id = self._active_debates.get(message.channel.id)

        if debate_id:
            logger.info(f"Routing message to debate {debate_id}: {message.text[:50]}...")
            # Subclasses should override to send to debate API
        else:
            logger.debug(f"Debate keyword detected but no active debate: {message.text[:50]}...")

    async def _handle_general_message(self, message: BotMessage) -> None:
        """Handle messages that aren't commands or debate-related.

        Override this method for custom general message handling.
        """
        logger.debug(f"General message from {message.user.username}: {message.text[:50]}...")

    async def _handle_debate_reaction(
        self,
        channel_id: str,
        message_id: str,
        user_id: str,
        emoji: str,
    ) -> None:
        """Handle reactions in debate contexts.

        Translates emoji reactions to debate actions:
        - +1/thumbsup -> upvote
        - -1/thumbsdown -> downvote
        - ? -> clarification request
        """
        debate_id = self._active_debates.get(channel_id)
        if not debate_id:
            return

        # Map emoji to action
        action_map = {
            "+1": "upvote",
            "thumbsup": "upvote",
            "-1": "downvote",
            "thumbsdown": "downvote",
            "question": "clarify",
            "thinking_face": "clarify",
        }

        action = action_map.get(emoji.lower().strip(":"))
        if action:
            logger.info(
                f"Debate reaction: debate={debate_id} action={action} "
                f"user={user_id} message={message_id}"
            )
            # Subclasses should override to send to debate API

    def start_debate(self, channel_id: str, debate_id: str) -> None:
        """Mark a channel as having an active debate."""
        self._active_debates[channel_id] = debate_id
        logger.info(f"Started debate {debate_id} in channel {channel_id}")

    def end_debate(self, channel_id: str) -> Optional[str]:
        """End a debate in a channel. Returns the debate ID if found."""
        debate_id = self._active_debates.pop(channel_id, None)
        if debate_id:
            logger.info(f"Ended debate {debate_id} in channel {channel_id}")
        return debate_id

    def get_active_debate(self, channel_id: str) -> Optional[str]:
        """Get the active debate ID for a channel, if any."""
        return self._active_debates.get(channel_id)
