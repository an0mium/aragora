"""
Command framework data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class CommandPermission(Enum):
    """Permission levels for commands."""

    PUBLIC = "public"  # Anyone can use
    AUTHENTICATED = "authenticated"  # Must be logged in
    MEMBER = "member"  # Must be workspace member
    ADMIN = "admin"  # Must be workspace admin
    OWNER = "owner"  # Must be org owner


@dataclass
class CommandDefinition:
    """Definition of a slash command."""

    name: str
    description: str
    usage: str
    examples: list[str] = field(default_factory=list)
    permission: CommandPermission = CommandPermission.AUTHENTICATED
    aliases: list[str] = field(default_factory=list)
    args_schema: Optional[dict[str, Any]] = None
    hidden: bool = False  # Hidden from help
    rate_limit: int = 10  # Max calls per minute per user
    platforms: list[str] = field(default_factory=lambda: ["slack", "teams", "discord", "telegram"])


@dataclass
class CommandContext:
    """Context for command execution."""

    command_name: str
    args: list[str]
    raw_text: str
    user_id: str
    user_name: str
    channel_id: str
    channel_name: Optional[str]
    platform: str  # slack, teams, discord, telegram
    workspace_id: Optional[str]
    tenant_id: Optional[str]
    thread_id: Optional[str] = None
    response_url: Optional[str] = None  # For Slack delayed responses
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def arg_string(self) -> str:
        """Get args as a single string."""
        return " ".join(self.args)

    def get_arg(self, index: int, default: Optional[str] = None) -> Optional[str]:
        """Get argument by index safely."""
        if 0 <= index < len(self.args):
            return self.args[index]
        return default


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    message: str
    blocks: Optional[list[dict[str, Any]]] = None  # Platform-specific blocks
    ephemeral: bool = False  # Only visible to user
    thread_reply: bool = False  # Reply in thread
    attachments: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        message: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        ephemeral: bool = False,
        **kwargs: Any,
    ) -> CommandResult:
        """Create a successful result."""
        return cls(success=True, message=message, blocks=blocks, ephemeral=ephemeral, **kwargs)

    @classmethod
    def error(cls, message: str, ephemeral: bool = True) -> CommandResult:
        """Create an error result."""
        return cls(success=False, message=message, ephemeral=ephemeral)


@dataclass
class CommandError(Exception):
    """Exception raised during command execution."""

    message: str
    code: str = "COMMAND_ERROR"
    details: Optional[dict[str, Any]] = None

    def to_result(self) -> CommandResult:
        """Convert to a CommandResult."""
        return CommandResult.error(self.message)


# Type alias for command handler functions
CommandHandler = Callable[[CommandContext], Coroutine[Any, Any, CommandResult]]
