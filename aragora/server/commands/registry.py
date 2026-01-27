"""
Command registry for managing slash commands.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Optional

from .models import CommandDefinition, CommandContext, CommandResult, CommandPermission

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Registry for slash commands.

    Manages command definitions, handlers, and discovery.
    """

    def __init__(self) -> None:
        self._commands: dict[str, CommandDefinition] = {}
        self._handlers: dict[
            str, Callable[[CommandContext], Coroutine[Any, Any, CommandResult]]
        ] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical name

    def register(
        self,
        definition: CommandDefinition,
        handler: Callable[[CommandContext], Coroutine[Any, Any, CommandResult]],
    ) -> None:
        """
        Register a command with its handler.

        Args:
            definition: Command definition including name, description, etc.
            handler: Async function to handle the command.
        """
        name = definition.name.lower()
        if name in self._commands:
            logger.warning(f"Overwriting existing command: {name}")

        self._commands[name] = definition
        self._handlers[name] = handler

        # Register aliases
        for alias in definition.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._aliases:
                logger.warning(f"Alias {alias_lower} already registered")
            self._aliases[alias_lower] = name

        logger.info(f"Registered command: {name} with {len(definition.aliases)} aliases")

    def unregister(self, name: str) -> bool:
        """
        Unregister a command.

        Args:
            name: Command name to unregister.

        Returns:
            True if command was unregistered, False if not found.
        """
        name = name.lower()
        if name not in self._commands:
            return False

        definition = self._commands[name]

        # Remove aliases
        for alias in definition.aliases:
            self._aliases.pop(alias.lower(), None)

        del self._commands[name]
        del self._handlers[name]

        logger.info(f"Unregistered command: {name}")
        return True

    def get_command(self, name: str) -> Optional[CommandDefinition]:
        """
        Get command definition by name or alias.

        Args:
            name: Command name or alias.

        Returns:
            CommandDefinition if found, None otherwise.
        """
        name = name.lower()

        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        return self._commands.get(name)

    def get_handler(
        self, name: str
    ) -> Optional[Callable[[CommandContext], Coroutine[Any, Any, CommandResult]]]:
        """
        Get command handler by name or alias.

        Args:
            name: Command name or alias.

        Returns:
            Handler function if found, None otherwise.
        """
        name = name.lower()

        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        return self._handlers.get(name)

    def list_commands(
        self,
        platform: Optional[str] = None,
        permission: Optional[CommandPermission] = None,
        include_hidden: bool = False,
    ) -> list[CommandDefinition]:
        """
        List all registered commands.

        Args:
            platform: Filter by platform (slack, teams, etc.)
            permission: Filter by max permission level
            include_hidden: Include hidden commands

        Returns:
            List of command definitions.
        """
        commands = []
        for cmd in self._commands.values():
            # Filter by platform
            if platform and platform not in cmd.platforms:
                continue

            # Filter hidden
            if cmd.hidden and not include_hidden:
                continue

            # Filter by permission (include commands at or below the level)
            if permission:
                permission_order = list(CommandPermission)
                if permission_order.index(cmd.permission) > permission_order.index(permission):
                    continue

            commands.append(cmd)

        return sorted(commands, key=lambda c: c.name)

    def get_help_text(self, platform: Optional[str] = None) -> str:
        """
        Generate help text for all visible commands.

        Args:
            platform: Filter by platform.

        Returns:
            Formatted help text.
        """
        commands = self.list_commands(platform=platform)
        if not commands:
            return "No commands available."

        lines = ["*Available Commands:*\n"]
        for cmd in commands:
            lines.append(f"  `/{cmd.name}` - {cmd.description}")
            if cmd.aliases:
                lines.append(f"    Aliases: {', '.join(cmd.aliases)}")

        lines.append("\nUse `/help <command>` for detailed usage.")
        return "\n".join(lines)

    def command_exists(self, name: str) -> bool:
        """Check if a command exists by name or alias."""
        name = name.lower()
        return name in self._commands or name in self._aliases

    @property
    def command_count(self) -> int:
        """Number of registered commands."""
        return len(self._commands)


# Global registry instance
_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get or create the global command registry."""
    global _registry
    if _registry is None:
        _registry = CommandRegistry()
    return _registry
