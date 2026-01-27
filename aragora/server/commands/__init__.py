"""
Slash Command Framework for Aragora.

Provides a unified command handling system for Slack, Teams, Discord, and other chat platforms.
"""

from __future__ import annotations

from .registry import CommandRegistry, get_command_registry
from .router import CommandRouter
from .models import (
    CommandDefinition,
    CommandContext,
    CommandResult,
    CommandError,
    CommandPermission,
)
from .handlers import (
    BaseCommandHandler,
    DebateCommandHandler,
    StatusCommandHandler,
    HelpCommandHandler,
    HistoryCommandHandler,
)

__all__ = [
    # Registry
    "CommandRegistry",
    "get_command_registry",
    # Router
    "CommandRouter",
    # Models
    "CommandDefinition",
    "CommandContext",
    "CommandResult",
    "CommandError",
    "CommandPermission",
    # Handlers
    "BaseCommandHandler",
    "DebateCommandHandler",
    "StatusCommandHandler",
    "HelpCommandHandler",
    "HistoryCommandHandler",
]
