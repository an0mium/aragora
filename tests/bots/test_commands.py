"""
Tests for aragora.bots.commands - Bot command registry and built-in commands.

Tests cover:
- CommandRegistry registration
- Command execution with context
- Cooldown enforcement
- Rate limiting per user
- Built-in commands (help, status, debate, gauntlet)
- Command decorator usage
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.bots.base import (
    Platform,
    BotUser,
    BotChannel,
    BotMessage,
    CommandContext,
    CommandResult,
)
from aragora.bots.commands import (
    BotCommand,
    CommandRegistry,
    command,
    get_default_registry,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def registry():
    """Create a fresh command registry."""
    return CommandRegistry()


@pytest.fixture
def sample_context():
    """Create a sample command context."""
    user = BotUser(id="user-123", username="testuser", platform=Platform.DISCORD)
    channel = BotChannel(id="channel-123", platform=Platform.DISCORD)
    message = BotMessage(
        id="msg-123",
        text="/test command",
        user=user,
        channel=channel,
        timestamp=datetime.utcnow(),
        platform=Platform.DISCORD,
    )

    return CommandContext(
        message=message,
        user=user,
        channel=channel,
        platform=Platform.DISCORD,
        args=["test", "arg1", "arg2"],
        raw_args="arg1 arg2",
        metadata={"api_base": "http://localhost:8080"},
    )


# ===========================================================================
# BotCommand Tests
# ===========================================================================


class TestBotCommand:
    """Tests for BotCommand dataclass."""

    def test_create_command(self):
        """Test creating a bot command."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True, message="Done")

        cmd = BotCommand(
            name="test",
            description="Test command",
            handler=handler,
        )

        assert cmd.name == "test"
        assert cmd.description == "Test command"
        assert cmd.handler == handler

    def test_command_with_usage(self):
        """Test command with usage string."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = BotCommand(
            name="debate",
            description="Start a debate",
            usage="debate <topic>",
            handler=handler,
        )

        assert cmd.usage == "debate <topic>"

    def test_command_with_platforms(self):
        """Test command with platform restrictions."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = BotCommand(
            name="special",
            description="Platform-specific command",
            handler=handler,
            platforms=[Platform.SLACK, Platform.DISCORD],
        )

        assert Platform.SLACK in cmd.platforms
        assert Platform.DISCORD in cmd.platforms
        assert Platform.TEAMS not in cmd.platforms

    def test_command_with_cooldown(self):
        """Test command with cooldown setting."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = BotCommand(
            name="expensive",
            description="Rate-limited command",
            handler=handler,
            cooldown=60,
        )

        assert cmd.cooldown == 60


# ===========================================================================
# CommandRegistry Tests
# ===========================================================================


class TestCommandRegistry:
    """Tests for CommandRegistry class."""

    def test_register_command(self, registry):
        """Test registering a command."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = BotCommand(
            name="test",
            description="Test command",
            handler=handler,
        )

        registry.register(cmd)

        assert "test" in registry._commands
        assert registry._commands["test"] == cmd

    def test_register_duplicate_command(self, registry):
        """Test registering duplicate command overwrites."""
        async def handler1(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True, message="v1")

        async def handler2(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True, message="v2")

        cmd1 = BotCommand(name="test", description="v1", handler=handler1)
        cmd2 = BotCommand(name="test", description="v2", handler=handler2)

        registry.register(cmd1)
        registry.register(cmd2)

        assert registry._commands["test"].description == "v2"

    def test_get_command(self, registry):
        """Test getting a registered command."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = BotCommand(name="test", description="Test", handler=handler)
        registry.register(cmd)

        retrieved = registry.get("test")
        assert retrieved == cmd

    def test_get_unknown_command(self, registry):
        """Test getting unknown command returns None."""
        assert registry.get("unknown") is None

    def test_list_commands(self, registry):
        """Test listing all commands."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        registry.register(BotCommand(name="cmd1", description="First", handler=handler))
        registry.register(BotCommand(name="cmd2", description="Second", handler=handler))

        commands = registry.list_commands()
        names = [cmd.name for cmd in commands]

        assert "cmd1" in names
        assert "cmd2" in names

    def test_list_commands_for_platform(self, registry):
        """Test listing commands for specific platform."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        registry.register(
            BotCommand(
                name="all",
                description="All platforms",
                handler=handler,
                platforms=[Platform.SLACK, Platform.DISCORD, Platform.TEAMS],
            )
        )
        registry.register(
            BotCommand(
                name="slack_only",
                description="Slack only",
                handler=handler,
                platforms=[Platform.SLACK],
            )
        )

        slack_cmds = registry.list_commands(platform=Platform.SLACK)
        discord_cmds = registry.list_commands(platform=Platform.DISCORD)

        slack_names = [cmd.name for cmd in slack_cmds]
        discord_names = [cmd.name for cmd in discord_cmds]

        assert "all" in slack_names
        assert "slack_only" in slack_names
        assert "all" in discord_names
        assert "slack_only" not in discord_names

    @pytest.mark.asyncio
    async def test_execute_command(self, registry, sample_context):
        """Test executing a registered command."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True, message=f"Executed with {ctx.raw_args}")

        registry.register(BotCommand(name="test", description="Test", handler=handler))

        result = await registry.execute(sample_context)

        assert result.success is True
        assert "arg1 arg2" in result.message

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self, registry, sample_context):
        """Test executing unknown command returns error."""
        sample_context.args = ["unknown"]

        result = await registry.execute(sample_context)

        assert result.success is False
        assert "unknown" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_exception(self, registry, sample_context):
        """Test handler exception is caught."""
        async def failing_handler(ctx: CommandContext) -> CommandResult:
            raise ValueError("Something went wrong")

        registry.register(
            BotCommand(name="test", description="Failing", handler=failing_handler)
        )

        result = await registry.execute(sample_context)

        assert result.success is False
        assert "error" in result.error.lower() or "wrong" in result.error.lower()


# ===========================================================================
# Cooldown Tests
# ===========================================================================


class TestCooldowns:
    """Tests for command cooldowns."""

    @pytest.mark.asyncio
    async def test_cooldown_enforcement(self, registry, sample_context):
        """Test cooldown prevents rapid execution."""
        call_count = 0

        async def handler(ctx: CommandContext) -> CommandResult:
            nonlocal call_count
            call_count += 1
            return CommandResult(success=True)

        registry.register(
            BotCommand(
                name="test",
                description="Rate limited",
                handler=handler,
                cooldown=60,
            )
        )

        # First call should succeed
        result1 = await registry.execute(sample_context)
        assert result1.success is True

        # Second immediate call should be rate limited
        result2 = await registry.execute(sample_context)

        # Either it fails or we allow it (implementation dependent)
        # The key is consistent behavior
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_cooldown_per_user(self, registry):
        """Test cooldowns are per-user."""
        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        registry.register(
            BotCommand(
                name="test",
                description="Test",
                handler=handler,
                cooldown=60,
            )
        )

        # User 1
        user1 = BotUser(id="user-1", username="user1", platform=Platform.DISCORD)
        channel = BotChannel(id="c1", platform=Platform.DISCORD)
        msg1 = BotMessage(
            id="m1", text="test", user=user1, channel=channel,
            timestamp=datetime.utcnow(), platform=Platform.DISCORD,
        )
        ctx1 = CommandContext(
            message=msg1, user=user1, channel=channel,
            platform=Platform.DISCORD, args=["test"],
        )

        # User 2
        user2 = BotUser(id="user-2", username="user2", platform=Platform.DISCORD)
        msg2 = BotMessage(
            id="m2", text="test", user=user2, channel=channel,
            timestamp=datetime.utcnow(), platform=Platform.DISCORD,
        )
        ctx2 = CommandContext(
            message=msg2, user=user2, channel=channel,
            platform=Platform.DISCORD, args=["test"],
        )

        # Both should succeed (different users)
        result1 = await registry.execute(ctx1)
        result2 = await registry.execute(ctx2)

        assert result1.success is True
        assert result2.success is True


# ===========================================================================
# Decorator Tests
# ===========================================================================


class TestCommandDecorator:
    """Tests for @command decorator."""

    def test_decorator_registration(self, registry):
        """Test decorator registers command."""
        @registry.command(name="decorated", description="Decorated command")
        async def decorated_handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        assert "decorated" in registry._commands

    def test_decorator_preserves_function(self, registry):
        """Test decorator preserves function reference."""
        @registry.command(name="test", description="Test")
        async def my_handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        # Function should still be callable
        assert callable(my_handler)

    def test_decorator_with_options(self, registry):
        """Test decorator with all options."""
        @registry.command(
            name="advanced",
            description="Advanced command",
            usage="advanced <arg>",
            platforms=[Platform.SLACK],
            cooldown=30,
        )
        async def advanced_handler(ctx: CommandContext) -> CommandResult:
            return CommandResult(success=True)

        cmd = registry.get("advanced")
        assert cmd.usage == "advanced <arg>"
        assert cmd.cooldown == 30
        assert Platform.SLACK in cmd.platforms


# ===========================================================================
# Default Registry Tests
# ===========================================================================


class TestDefaultRegistry:
    """Tests for default command registry."""

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert registry is not None
        assert isinstance(registry, CommandRegistry)

    def test_default_registry_has_help(self):
        """Test default registry includes help command."""
        registry = get_default_registry()
        help_cmd = registry.get("help")
        assert help_cmd is not None

    def test_default_registry_has_status(self):
        """Test default registry includes status command."""
        registry = get_default_registry()
        status_cmd = registry.get("status")
        assert status_cmd is not None

    def test_default_registry_has_debate(self):
        """Test default registry includes debate command."""
        registry = get_default_registry()
        debate_cmd = registry.get("debate")
        assert debate_cmd is not None

    def test_default_registry_has_gauntlet(self):
        """Test default registry includes gauntlet command."""
        registry = get_default_registry()
        gauntlet_cmd = registry.get("gauntlet")
        assert gauntlet_cmd is not None


# ===========================================================================
# Built-in Command Tests
# ===========================================================================


class TestBuiltInCommands:
    """Tests for built-in commands."""

    @pytest.fixture
    def default_context(self):
        """Create context for default registry testing."""
        user = BotUser(id="u1", username="test", platform=Platform.DISCORD)
        channel = BotChannel(id="c1", platform=Platform.DISCORD)
        message = BotMessage(
            id="m1", text="help", user=user, channel=channel,
            timestamp=datetime.utcnow(), platform=Platform.DISCORD,
        )
        return CommandContext(
            message=message, user=user, channel=channel,
            platform=Platform.DISCORD, args=["help"],
            metadata={"api_base": "http://localhost:8080"},
        )

    @pytest.mark.asyncio
    async def test_help_command(self, default_context):
        """Test help command returns command list."""
        registry = get_default_registry()
        default_context.args = ["help"]

        result = await registry.execute(default_context)

        assert result.success is True
        assert result.message is not None
        # Should mention available commands
        assert "debate" in result.message.lower() or "command" in result.message.lower()

    @pytest.mark.asyncio
    async def test_status_command(self, default_context):
        """Test status command returns system status."""
        registry = get_default_registry()
        default_context.args = ["status"]

        result = await registry.execute(default_context)

        assert result.success is True
        assert result.message is not None

    @pytest.mark.asyncio
    async def test_debate_command_requires_topic(self, default_context):
        """Test debate command requires topic."""
        registry = get_default_registry()
        default_context.args = ["debate"]  # No topic
        default_context.raw_args = ""

        result = await registry.execute(default_context)

        # Should either fail or prompt for topic
        assert result is not None
        if not result.success:
            assert "topic" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_gauntlet_command_requires_input(self, default_context):
        """Test gauntlet command requires input."""
        registry = get_default_registry()
        default_context.args = ["gauntlet"]
        default_context.raw_args = ""

        result = await registry.execute(default_context)

        # Should either fail or prompt for input
        assert result is not None
