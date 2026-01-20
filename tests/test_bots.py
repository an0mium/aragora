"""
Tests for the bot framework module.

Tests cover:
- Base classes (Platform, BotUser, BotChannel, BotMessage, etc.)
- Command registry and built-in commands
- Discord bot implementation
- Teams bot implementation
- Zoom bot implementation
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.bots.base import (
    Platform,
    BotUser,
    BotChannel,
    BotMessage,
    CommandContext,
    CommandResult,
    BotConfig,
    BaseBotClient,
    BotEventHandler,
)


class TestPlatform:
    """Tests for Platform enum."""

    def test_platform_values(self):
        """Test all platform values exist."""
        assert Platform.SLACK.value == "slack"
        assert Platform.DISCORD.value == "discord"
        assert Platform.TEAMS.value == "teams"
        assert Platform.ZOOM.value == "zoom"

    def test_platform_iteration(self):
        """Test iterating over platforms."""
        platforms = list(Platform)
        assert len(platforms) == 4


class TestBotUser:
    """Tests for BotUser dataclass."""

    def test_basic_user(self):
        """Test basic user creation."""
        user = BotUser(
            id="U123",
            username="testuser",
        )

        assert user.id == "U123"
        assert user.username == "testuser"
        assert user.display_name is None
        assert user.email is None
        assert user.is_bot is False
        assert user.platform == Platform.SLACK  # Default
        assert user.raw_data == {}

    def test_user_with_all_fields(self):
        """Test user with all fields."""
        user = BotUser(
            id="U456",
            username="fulluser",
            display_name="Full User",
            email="full@example.com",
            is_bot=True,
            platform=Platform.DISCORD,
            raw_data={"extra": "data"},
        )

        assert user.display_name == "Full User"
        assert user.email == "full@example.com"
        assert user.is_bot is True
        assert user.platform == Platform.DISCORD
        assert user.raw_data == {"extra": "data"}

    def test_slack_mention(self):
        """Test Slack mention format."""
        user = BotUser(id="U123", username="test", platform=Platform.SLACK)
        assert user.mention == "<@U123>"

    def test_discord_mention(self):
        """Test Discord mention format."""
        user = BotUser(id="123456", username="test", platform=Platform.DISCORD)
        assert user.mention == "<@123456>"

    def test_teams_mention(self):
        """Test Teams mention format."""
        user = BotUser(
            id="user@example.com",
            username="test",
            display_name="Test User",
            platform=Platform.TEAMS,
        )
        assert user.mention == "<at>Test User</at>"

    def test_zoom_mention(self):
        """Test Zoom mention format."""
        user = BotUser(id="zoom123", username="test", platform=Platform.ZOOM)
        assert user.mention == "@test"


class TestBotChannel:
    """Tests for BotChannel dataclass."""

    def test_basic_channel(self):
        """Test basic channel creation."""
        channel = BotChannel(id="C123")

        assert channel.id == "C123"
        assert channel.name is None
        assert channel.is_private is False
        assert channel.is_dm is False
        assert channel.platform == Platform.SLACK
        assert channel.thread_id is None

    def test_channel_with_thread(self):
        """Test channel with thread."""
        channel = BotChannel(
            id="C123",
            name="general",
            thread_id="1234567890.123456",
        )

        assert channel.thread_id == "1234567890.123456"


class TestBotMessage:
    """Tests for BotMessage dataclass."""

    def test_basic_message(self):
        """Test basic message creation."""
        user = BotUser(id="U123", username="test")
        channel = BotChannel(id="C123")

        message = BotMessage(
            id="M123",
            text="Hello world",
            user=user,
            channel=channel,
            timestamp=datetime.now(),
            platform=Platform.SLACK,
        )

        assert message.id == "M123"
        assert message.text == "Hello world"
        assert message.user == user
        assert message.channel == channel
        assert message.is_threaded is False

    def test_threaded_message(self):
        """Test threaded message."""
        user = BotUser(id="U123", username="test")
        channel = BotChannel(id="C123")

        message = BotMessage(
            id="M123",
            text="Thread reply",
            user=user,
            channel=channel,
            timestamp=datetime.now(),
            platform=Platform.SLACK,
            thread_id="1234567890.123456",
        )

        assert message.is_threaded is True


class TestCommandContext:
    """Tests for CommandContext dataclass."""

    def test_basic_context(self):
        """Test basic context creation."""
        user = BotUser(id="U123", username="test")
        channel = BotChannel(id="C123")
        message = BotMessage(
            id="M123",
            text="/debate topic",
            user=user,
            channel=channel,
            timestamp=datetime.now(),
            platform=Platform.SLACK,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.SLACK,
            args=["debate", "topic"],
            raw_args="topic",
        )

        assert ctx.user_id == "U123"
        assert ctx.channel_id == "C123"
        assert ctx.thread_id is None
        assert ctx.args == ["debate", "topic"]
        assert ctx.raw_args == "topic"


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = CommandResult(success=True, message="Done!")

        assert result.success is True
        assert result.message == "Done!"
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = CommandResult(success=False, error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_ok_factory(self):
        """Test CommandResult.ok factory method."""
        result = CommandResult.ok("Success!")

        assert result.success is True
        assert result.message == "Success!"

    def test_fail_factory(self):
        """Test CommandResult.fail factory method."""
        result = CommandResult.fail("Error!")

        assert result.success is False
        assert result.error == "Error!"

    def test_result_with_data(self):
        """Test result with data."""
        result = CommandResult.ok(
            "Debate started",
            data={"debate_id": "123"},
        )

        assert result.data == {"debate_id": "123"}

    def test_ephemeral_result(self):
        """Test ephemeral result."""
        result = CommandResult.ok("Secret message", ephemeral=True)

        assert result.ephemeral is True


class TestBotConfig:
    """Tests for BotConfig dataclass."""

    def test_basic_config(self):
        """Test basic config."""
        config = BotConfig(
            platform=Platform.SLACK,
            token="xoxb-123",
        )

        assert config.platform == Platform.SLACK
        assert config.token == "xoxb-123"
        assert config.api_base == "http://localhost:8080"
        assert config.bot_name == "Aragora"

    def test_config_rate_limits(self):
        """Test config rate limits."""
        config = BotConfig(
            platform=Platform.DISCORD,
            token="test",
            rate_limit_per_user=5,
            rate_limit_global=50,
        )

        assert config.rate_limit_per_user == 5
        assert config.rate_limit_global == 50


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def test_register_command(self):
        """Test registering a command."""
        from aragora.bots.commands import CommandRegistry, BotCommand

        registry = CommandRegistry()

        async def test_handler(ctx):
            return CommandResult.ok("Test")

        cmd = BotCommand(
            name="test",
            handler=test_handler,
            description="Test command",
        )
        registry.register(cmd)

        assert registry.get("test") is not None
        assert registry.get("test").name == "test"

    def test_get_command_by_alias(self):
        """Test getting command by alias."""
        from aragora.bots.commands import CommandRegistry, BotCommand

        registry = CommandRegistry()

        async def test_handler(ctx):
            return CommandResult.ok("Test")

        cmd = BotCommand(
            name="test",
            handler=test_handler,
            aliases=["t", "tst"],
        )
        registry.register(cmd)

        assert registry.get("t") is not None
        assert registry.get("t").name == "test"
        assert registry.get("tst").name == "test"

    def test_unknown_command(self):
        """Test getting unknown command."""
        from aragora.bots.commands import CommandRegistry

        registry = CommandRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister_command(self):
        """Test unregistering a command."""
        from aragora.bots.commands import CommandRegistry, BotCommand

        registry = CommandRegistry()

        async def test_handler(ctx):
            return CommandResult.ok("Test")

        cmd = BotCommand(name="test", handler=test_handler)
        registry.register(cmd)

        assert registry.unregister("test") is True
        assert registry.get("test") is None

    def test_list_for_platform(self):
        """Test listing commands for a platform."""
        from aragora.bots.commands import CommandRegistry, BotCommand

        registry = CommandRegistry()

        async def handler(ctx):
            return CommandResult.ok("OK")

        # Command for all platforms
        cmd1 = BotCommand(name="all", handler=handler)
        registry.register(cmd1)

        # Command only for Slack
        cmd2 = BotCommand(
            name="slack_only",
            handler=handler,
            platforms={Platform.SLACK},
        )
        registry.register(cmd2)

        slack_cmds = registry.list_for_platform(Platform.SLACK)
        discord_cmds = registry.list_for_platform(Platform.DISCORD)

        assert len(slack_cmds) == 2
        assert len(discord_cmds) == 1
        assert any(c.name == "slack_only" for c in slack_cmds)
        assert not any(c.name == "slack_only" for c in discord_cmds)

    def test_command_decorator(self):
        """Test command decorator."""
        from aragora.bots.commands import CommandRegistry

        registry = CommandRegistry()

        @registry.command("decorated", description="A decorated command")
        async def decorated_handler(ctx):
            return CommandResult.ok("Decorated!")

        cmd = registry.get("decorated")
        assert cmd is not None
        assert cmd.description == "A decorated command"

    def test_validate_args(self):
        """Test argument validation."""
        from aragora.bots.commands import BotCommand

        async def handler(ctx):
            return CommandResult.ok("OK")

        cmd = BotCommand(
            name="test",
            handler=handler,
            requires_args=True,
            min_args=2,
            max_args=5,
        )

        assert cmd.validate_args([]) is not None  # Too few
        assert cmd.validate_args(["a"]) is not None  # Too few
        assert cmd.validate_args(["a", "b"]) is None  # OK
        assert cmd.validate_args(["a", "b", "c", "d", "e"]) is None  # OK
        assert cmd.validate_args(["a", "b", "c", "d", "e", "f"]) is not None  # Too many

    @pytest.mark.asyncio
    async def test_execute_command(self):
        """Test executing a command."""
        from aragora.bots.commands import CommandRegistry, BotCommand

        registry = CommandRegistry()

        async def echo_handler(ctx):
            return CommandResult.ok(f"Echo: {ctx.raw_args}")

        cmd = BotCommand(name="echo", handler=echo_handler)
        registry.register(cmd)

        user = BotUser(id="U123", username="test")
        channel = BotChannel(id="C123")
        message = BotMessage(
            id="M123",
            text="/echo hello",
            user=user,
            channel=channel,
            timestamp=datetime.now(),
            platform=Platform.SLACK,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.SLACK,
            args=["echo", "hello"],
            raw_args="hello",
        )

        result = await registry.execute(ctx)
        assert result.success is True
        assert "Echo:" in result.message


class TestDefaultRegistry:
    """Tests for default command registry."""

    def test_get_default_registry(self):
        """Test getting default registry."""
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()
        assert registry is not None

    def test_builtin_help_command(self):
        """Test help command is registered."""
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()
        help_cmd = registry.get("help")

        assert help_cmd is not None
        assert "?" in help_cmd.aliases
        assert "commands" in help_cmd.aliases

    def test_builtin_status_command(self):
        """Test status command is registered."""
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()
        status_cmd = registry.get("status")

        assert status_cmd is not None
        assert "ping" in status_cmd.aliases
        assert "health" in status_cmd.aliases

    def test_builtin_debate_command(self):
        """Test debate command is registered."""
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()
        debate_cmd = registry.get("debate")

        assert debate_cmd is not None
        assert debate_cmd.requires_args is True
        assert debate_cmd.cooldown == 30

    def test_builtin_gauntlet_command(self):
        """Test gauntlet command is registered."""
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()
        gauntlet_cmd = registry.get("gauntlet")

        assert gauntlet_cmd is not None
        assert gauntlet_cmd.requires_args is True
        assert gauntlet_cmd.cooldown == 60


class TestDiscordBot:
    """Tests for Discord bot implementation."""

    def test_discord_availability_check(self):
        """Test Discord availability check."""
        from aragora.bots.discord_bot import _check_discord_available

        available, error = _check_discord_available()
        # Result depends on whether discord.py is installed
        assert isinstance(available, bool)
        if not available:
            assert "discord.py" in error

    def test_create_discord_bot_requires_token(self):
        """Test creating Discord bot requires token."""
        from aragora.bots.discord_bot import create_discord_bot

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("DISCORD_BOT_TOKEN", None)

            with pytest.raises(ValueError) as exc_info:
                create_discord_bot()

            assert "token" in str(exc_info.value).lower()

    def test_create_discord_bot_with_token(self):
        """Test creating Discord bot with token."""
        from aragora.bots.discord_bot import create_discord_bot

        bot = create_discord_bot(token="test-token", application_id="app-123")

        assert bot.token == "test-token"
        assert bot.application_id == "app-123"
        assert bot.config.platform == Platform.DISCORD


class TestTeamsBot:
    """Tests for Teams bot implementation."""

    def test_teams_availability_check(self):
        """Test Teams availability check."""
        from aragora.bots.teams_bot import _check_botframework_available

        available, error = _check_botframework_available()
        assert isinstance(available, bool)
        if not available:
            assert "botbuilder" in error

    def test_create_teams_bot_requires_credentials(self):
        """Test creating Teams bot requires credentials."""
        from aragora.bots.teams_bot import create_teams_bot

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("TEAMS_APP_ID", None)
            os.environ.pop("TEAMS_APP_PASSWORD", None)

            with pytest.raises(ValueError) as exc_info:
                create_teams_bot()

            assert "credentials" in str(exc_info.value).lower()

    def test_create_teams_bot_with_credentials(self):
        """Test creating Teams bot with credentials."""
        from aragora.bots.teams_bot import create_teams_bot

        bot = create_teams_bot(
            app_id="test-app-id",
            app_password="test-password",
        )

        assert bot.app_id == "test-app-id"
        assert bot.app_password == "test-password"
        assert bot.config.platform == Platform.TEAMS


class TestZoomBot:
    """Tests for Zoom bot implementation."""

    def test_create_zoom_bot_requires_credentials(self):
        """Test creating Zoom bot requires credentials."""
        from aragora.bots.zoom_bot import create_zoom_bot

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("ZOOM_CLIENT_ID", None)
            os.environ.pop("ZOOM_CLIENT_SECRET", None)

            with pytest.raises(ValueError) as exc_info:
                create_zoom_bot()

            assert "credentials" in str(exc_info.value).lower()

    def test_create_zoom_bot_with_credentials(self):
        """Test creating Zoom bot with credentials."""
        from aragora.bots.zoom_bot import create_zoom_bot

        bot = create_zoom_bot(
            client_id="test-client-id",
            client_secret="test-secret",
            bot_jid="bot@zoom.us",
        )

        assert bot.client_id == "test-client-id"
        assert bot.client_secret == "test-secret"
        assert bot.bot_jid == "bot@zoom.us"
        assert bot.config.platform == Platform.ZOOM

    def test_zoom_webhook_verification(self):
        """Test Zoom webhook signature verification."""
        from aragora.bots.zoom_bot import AragoraZoomBot
        import hashlib
        import hmac

        bot = AragoraZoomBot(
            client_id="test",
            client_secret="test",
            secret_token="secret123",
        )

        payload = b'{"event": "test"}'
        timestamp = "1234567890"

        # Generate valid signature
        message = f"v0:{timestamp}:{payload.decode('utf-8')}"
        expected_sig = (
            "v0="
            + hmac.new(
                b"secret123",
                message.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        # Test valid signature
        assert bot.verify_webhook(payload, timestamp, expected_sig) is True

        # Test invalid signature
        assert bot.verify_webhook(payload, timestamp, "v0=invalid") is False

    def test_zoom_oauth_manager_init(self):
        """Test Zoom OAuth manager initialization."""
        from aragora.bots.zoom_bot import ZoomOAuthManager

        oauth = ZoomOAuthManager(
            client_id="test-id",
            client_secret="test-secret",
        )

        assert oauth.client_id == "test-id"
        assert oauth.client_secret == "test-secret"
        assert oauth._access_token is None

    @pytest.mark.asyncio
    async def test_zoom_handle_url_validation(self):
        """Test Zoom URL validation event handling."""
        from aragora.bots.zoom_bot import AragoraZoomBot
        import hashlib
        import hmac

        bot = AragoraZoomBot(
            client_id="test",
            client_secret="test",
            secret_token="secret123",
        )

        event = {
            "event": "endpoint.url_validation",
            "payload": {
                "plainToken": "abc123",
            },
        }

        response = await bot.handle_event(event)

        assert "plainToken" in response
        assert response["plainToken"] == "abc123"
        assert "encryptedToken" in response

        # Verify encryption
        expected = hmac.new(
            b"secret123",
            b"abc123",
            hashlib.sha256,
        ).hexdigest()
        assert response["encryptedToken"] == expected


class TestBotModuleExports:
    """Tests for bot module exports."""

    def test_base_exports(self):
        """Test base module exports."""
        from aragora.bots.base import (
            Platform,
            BotUser,
            BotChannel,
            BotMessage,
            CommandContext,
            CommandResult,
            BotConfig,
        )

        assert Platform is not None
        assert BotUser is not None
        assert BotChannel is not None
        assert BotMessage is not None
        assert CommandContext is not None
        assert CommandResult is not None
        assert BotConfig is not None
        assert BaseBotClient is not None
        assert BotEventHandler is not None

    def test_commands_exports(self):
        """Test commands module exports."""
        from aragora.bots.commands import (
            BotCommand,
            CommandRegistry,
            get_default_registry,
            command,
        )

        assert BotCommand is not None
        assert CommandRegistry is not None
        assert get_default_registry is not None
        assert command is not None

    def test_discord_exports(self):
        """Test discord module exports."""
        from aragora.bots.discord_bot import (
            AragoraDiscordBot,
            run_discord_bot,
            create_discord_bot,
        )

        assert AragoraDiscordBot is not None
        assert run_discord_bot is not None
        assert create_discord_bot is not None

    def test_teams_exports(self):
        """Test teams module exports."""
        from aragora.bots.teams_bot import (
            AragoraTeamsBot,
            create_teams_bot,
        )

        assert AragoraTeamsBot is not None
        assert create_teams_bot is not None

    def test_zoom_exports(self):
        """Test zoom module exports."""
        from aragora.bots.zoom_bot import (
            AragoraZoomBot,
            ZoomOAuthManager,
            create_zoom_bot,
        )

        assert AragoraZoomBot is not None
        assert ZoomOAuthManager is not None
        assert create_zoom_bot is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
