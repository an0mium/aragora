"""
Tests for the slash command framework.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.commands.models import (
    CommandDefinition,
    CommandContext,
    CommandResult,
    CommandError,
    CommandPermission,
)
from aragora.server.commands.registry import CommandRegistry
from aragora.server.commands.router import CommandRouter, parse_command_text
from aragora.server.commands.handlers import (
    HelpCommandHandler,
    DebateCommandHandler,
    StatusCommandHandler,
    HistoryCommandHandler,
    register_default_commands,
)


class TestCommandDefinition:
    """Tests for CommandDefinition."""

    def test_default_values(self):
        """Test default values for command definition."""
        defn = CommandDefinition(
            name="test",
            description="A test command",
            usage="/test",
        )
        assert defn.permission == CommandPermission.AUTHENTICATED
        assert defn.aliases == []
        assert defn.hidden is False
        assert defn.rate_limit == 10
        assert "slack" in defn.platforms
        assert "teams" in defn.platforms

    def test_custom_values(self):
        """Test custom values for command definition."""
        defn = CommandDefinition(
            name="admin",
            description="Admin command",
            usage="/admin",
            permission=CommandPermission.ADMIN,
            aliases=["a", "adm"],
            hidden=True,
            rate_limit=5,
            platforms=["slack"],
        )
        assert defn.permission == CommandPermission.ADMIN
        assert defn.aliases == ["a", "adm"]
        assert defn.hidden is True
        assert defn.rate_limit == 5
        assert defn.platforms == ["slack"]


class TestCommandContext:
    """Tests for CommandContext."""

    def test_arg_string(self):
        """Test arg_string property."""
        ctx = CommandContext(
            command_name="test",
            args=["arg1", "arg2", "arg3"],
            raw_text="/test arg1 arg2 arg3",
            user_id="U123",
            user_name="testuser",
            channel_id="C123",
            channel_name="general",
            platform="slack",
            workspace_id="W123",
            tenant_id="T123",
        )
        assert ctx.arg_string == "arg1 arg2 arg3"

    def test_get_arg(self):
        """Test get_arg method."""
        ctx = CommandContext(
            command_name="test",
            args=["first", "second"],
            raw_text="/test first second",
            user_id="U123",
            user_name="testuser",
            channel_id="C123",
            channel_name="general",
            platform="slack",
            workspace_id="W123",
            tenant_id="T123",
        )
        assert ctx.get_arg(0) == "first"
        assert ctx.get_arg(1) == "second"
        assert ctx.get_arg(2) is None
        assert ctx.get_arg(2, "default") == "default"
        assert ctx.get_arg(-1) is None


class TestCommandResult:
    """Tests for CommandResult."""

    def test_ok_result(self):
        """Test creating success result."""
        result = CommandResult.ok("Success!", ephemeral=True)
        assert result.success is True
        assert result.message == "Success!"
        assert result.ephemeral is True

    def test_error_result(self):
        """Test creating error result."""
        result = CommandResult.error("Something went wrong")
        assert result.success is False
        assert result.message == "Something went wrong"
        assert result.ephemeral is True

    def test_result_with_blocks(self):
        """Test result with blocks."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]
        result = CommandResult.ok("Hello", blocks=blocks)
        assert result.blocks == blocks


class TestCommandError:
    """Tests for CommandError."""

    def test_to_result(self):
        """Test converting error to result."""
        error = CommandError(message="Test error", code="TEST_ERR")
        result = error.to_result()
        assert result.success is False
        assert result.message == "Test error"
        assert result.ephemeral is True


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def test_register_command(self):
        """Test registering a command."""
        registry = CommandRegistry()

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("Done")

        defn = CommandDefinition(name="test", description="Test", usage="/test", aliases=["t"])
        registry.register(defn, handler)

        assert registry.command_exists("test")
        assert registry.command_exists("t")  # alias
        assert registry.command_count == 1

    def test_get_command(self):
        """Test getting command by name and alias."""
        registry = CommandRegistry()

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("Done")

        defn = CommandDefinition(
            name="mycommand", description="My command", usage="/mycommand", aliases=["mc"]
        )
        registry.register(defn, handler)

        assert registry.get_command("mycommand") == defn
        assert registry.get_command("mc") == defn
        assert registry.get_command("unknown") is None

    def test_unregister_command(self):
        """Test unregistering a command."""
        registry = CommandRegistry()

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("Done")

        defn = CommandDefinition(name="temp", description="Temp", usage="/temp", aliases=["tmp"])
        registry.register(defn, handler)

        assert registry.command_exists("temp")
        assert registry.unregister("temp") is True
        assert not registry.command_exists("temp")
        assert not registry.command_exists("tmp")
        assert registry.unregister("temp") is False  # Already unregistered

    def test_list_commands(self):
        """Test listing commands with filters."""
        registry = CommandRegistry()

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("Done")

        # Public command on all platforms
        registry.register(
            CommandDefinition(
                name="public",
                description="Public",
                usage="/public",
                permission=CommandPermission.PUBLIC,
            ),
            handler,
        )

        # Admin command only on Slack
        registry.register(
            CommandDefinition(
                name="admin",
                description="Admin",
                usage="/admin",
                permission=CommandPermission.ADMIN,
                platforms=["slack"],
            ),
            handler,
        )

        # Hidden command
        registry.register(
            CommandDefinition(
                name="hidden",
                description="Hidden",
                usage="/hidden",
                hidden=True,
            ),
            handler,
        )

        # List all visible
        commands = registry.list_commands()
        assert len(commands) == 2  # public + admin (not hidden)

        # Filter by platform
        slack_commands = registry.list_commands(platform="slack")
        assert len(slack_commands) == 2

        teams_commands = registry.list_commands(platform="teams")
        assert len(teams_commands) == 1  # Only public

        # Include hidden
        all_commands = registry.list_commands(include_hidden=True)
        assert len(all_commands) == 3

    def test_get_help_text(self):
        """Test generating help text."""
        registry = CommandRegistry()

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("Done")

        registry.register(
            CommandDefinition(name="foo", description="Foo command", usage="/foo", aliases=["f"]),
            handler,
        )
        registry.register(
            CommandDefinition(name="bar", description="Bar command", usage="/bar"),
            handler,
        )

        help_text = registry.get_help_text()
        assert "Available Commands" in help_text
        assert "/foo" in help_text
        assert "/bar" in help_text
        assert "Foo command" in help_text


class TestCommandRouter:
    """Tests for CommandRouter."""

    @pytest.fixture
    def registry(self):
        """Create a registry with test commands."""
        registry = CommandRegistry()

        async def echo_handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok(f"Echo: {ctx.arg_string}")

        async def error_handler(ctx: CommandContext) -> CommandResult:
            raise CommandError("Test error", code="TEST")

        async def exception_handler(ctx: CommandContext) -> CommandResult:
            raise ValueError("Unexpected error")

        registry.register(
            CommandDefinition(name="echo", description="Echo", usage="/echo"),
            echo_handler,
        )
        registry.register(
            CommandDefinition(name="error", description="Error", usage="/error"),
            error_handler,
        )
        registry.register(
            CommandDefinition(name="exception", description="Exception", usage="/exception"),
            exception_handler,
        )
        registry.register(
            CommandDefinition(
                name="slackonly",
                description="Slack only",
                usage="/slackonly",
                platforms=["slack"],
            ),
            echo_handler,
        )

        return registry

    @pytest.fixture
    def router(self, registry):
        """Create a router with test registry."""
        return CommandRouter(registry)

    @pytest.fixture
    def ctx(self):
        """Create a test context."""
        return CommandContext(
            command_name="echo",
            args=["hello", "world"],
            raw_text="/echo hello world",
            user_id="U123",
            user_name="testuser",
            channel_id="C123",
            channel_name="general",
            platform="slack",
            workspace_id="W123",
            tenant_id="T123",
        )

    @pytest.mark.asyncio
    async def test_route_success(self, router, ctx):
        """Test successful command routing."""
        result = await router.route(ctx)
        assert result.success is True
        assert "Echo: hello world" in result.message

    @pytest.mark.asyncio
    async def test_route_unknown_command(self, router, ctx):
        """Test routing unknown command."""
        ctx.command_name = "unknown"
        result = await router.route(ctx)
        assert result.success is False
        assert "Unknown command" in result.message

    @pytest.mark.asyncio
    async def test_route_command_error(self, router, ctx):
        """Test routing command that raises CommandError."""
        ctx.command_name = "error"
        result = await router.route(ctx)
        assert result.success is False
        assert "Test error" in result.message

    @pytest.mark.asyncio
    async def test_route_exception(self, router, ctx):
        """Test routing command that raises unexpected exception."""
        ctx.command_name = "exception"
        result = await router.route(ctx)
        assert result.success is False
        assert "error occurred" in result.message.lower()

    @pytest.mark.asyncio
    async def test_route_platform_restriction(self, router, ctx):
        """Test platform restriction."""
        ctx.command_name = "slackonly"
        ctx.platform = "teams"
        result = await router.route(ctx)
        assert result.success is False
        assert "not available on teams" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, router, ctx):
        """Test rate limiting."""

        # Register a command with low rate limit
        async def limited_handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("OK")

        router.registry.register(
            CommandDefinition(
                name="limited", description="Limited", usage="/limited", rate_limit=2
            ),
            limited_handler,
        )

        ctx.command_name = "limited"

        # First two calls should succeed
        result1 = await router.route(ctx)
        result2 = await router.route(ctx)
        assert result1.success is True
        assert result2.success is True

        # Third call should be rate limited
        result3 = await router.route(ctx)
        assert result3.success is False
        assert "rate limit" in result3.message.lower()


class TestParseCommandText:
    """Tests for parse_command_text function."""

    def test_simple_command(self):
        """Test parsing simple command."""
        name, args = parse_command_text("/test")
        assert name == "test"
        assert args == []

    def test_command_with_args(self):
        """Test parsing command with arguments."""
        name, args = parse_command_text("/echo hello world")
        assert name == "echo"
        assert args == ["hello", "world"]

    def test_command_with_quoted_args(self):
        """Test parsing command with quoted arguments."""
        name, args = parse_command_text('/debate "should we use TypeScript" for frontend')
        assert name == "debate"
        assert "should we use TypeScript" in args
        assert "for" in args
        assert "frontend" in args

    def test_command_without_slash(self):
        """Test parsing command without leading slash."""
        name, args = parse_command_text("help debate")
        assert name == "help"
        assert args == ["debate"]

    def test_empty_command(self):
        """Test parsing empty command."""
        name, args = parse_command_text("")
        assert name == ""
        assert args == []


class TestHelpCommandHandler:
    """Tests for HelpCommandHandler."""

    @pytest.fixture
    def handler(self):
        return HelpCommandHandler()

    @pytest.fixture
    def ctx(self):
        return CommandContext(
            command_name="help",
            args=[],
            raw_text="/help",
            user_id="U123",
            user_name="testuser",
            channel_id="C123",
            channel_name="general",
            platform="slack",
            workspace_id="W123",
            tenant_id="T123",
        )

    def test_definition(self, handler):
        """Test handler definition."""
        defn = handler.definition
        assert defn.name == "help"
        assert "?" in defn.aliases
        assert defn.permission == CommandPermission.PUBLIC

    @pytest.mark.asyncio
    async def test_general_help(self, handler, ctx):
        """Test general help output."""
        # Register the help command itself
        from aragora.server.commands.registry import get_command_registry

        registry = get_command_registry()
        registry.register(handler.definition, handler)

        result = await handler.execute(ctx)
        assert result.success is True
        assert "help" in result.message.lower()

    @pytest.mark.asyncio
    async def test_specific_command_help(self, handler, ctx):
        """Test help for specific command."""
        from aragora.server.commands.registry import get_command_registry

        registry = get_command_registry()
        registry.register(handler.definition, handler)

        ctx.args = ["help"]
        result = await handler.execute(ctx)
        assert result.success is True
        assert "Command: /help" in result.message

    @pytest.mark.asyncio
    async def test_unknown_command_help(self, handler, ctx):
        """Test help for unknown command."""
        ctx.args = ["unknowncommand"]
        result = await handler.execute(ctx)
        assert result.success is False
        assert "Unknown command" in result.message


class TestRegisterDefaultCommands:
    """Tests for register_default_commands."""

    def test_registers_commands(self):
        """Test that default commands are registered."""
        # Create fresh registry
        from aragora.server.commands import registry

        registry._registry = None
        reg = registry.get_command_registry()

        register_default_commands()

        assert reg.command_exists("debate")
        assert reg.command_exists("status")
        assert reg.command_exists("help")
        assert reg.command_exists("history")
        assert reg.command_exists("results")

        # Check aliases
        assert reg.command_exists("d")  # debate alias
        assert reg.command_exists("s")  # status alias
        assert reg.command_exists("h")  # help alias
