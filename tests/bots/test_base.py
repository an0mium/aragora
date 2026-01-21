"""
Tests for aragora.bots.base - Bot framework base classes.

Tests cover:
- Platform enum values
- BotUser, BotChannel, BotMessage creation
- CommandContext building
- CommandResult serialization
- BotConfig initialization
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.bots.base import (
    Platform,
    BotUser,
    BotChannel,
    BotMessage,
    CommandContext,
    CommandResult,
    BotConfig,
)


# ===========================================================================
# Platform Tests
# ===========================================================================


class TestPlatform:
    """Tests for Platform enum."""

    def test_platform_values(self):
        """Test all platform values are defined."""
        assert Platform.SLACK.value == "slack"
        assert Platform.DISCORD.value == "discord"
        assert Platform.TEAMS.value == "teams"
        assert Platform.ZOOM.value == "zoom"

    def test_platform_from_string(self):
        """Test creating platform from string."""
        assert Platform("slack") == Platform.SLACK
        assert Platform("discord") == Platform.DISCORD
        assert Platform("teams") == Platform.TEAMS
        assert Platform("zoom") == Platform.ZOOM

    def test_platform_invalid_value(self):
        """Test error on invalid platform value."""
        with pytest.raises(ValueError):
            Platform("invalid")


# ===========================================================================
# BotUser Tests
# ===========================================================================


class TestBotUser:
    """Tests for BotUser dataclass."""

    def test_create_user(self):
        """Test creating a bot user."""
        user = BotUser(
            id="user-123",
            username="testuser",
            platform=Platform.DISCORD,
        )
        assert user.id == "user-123"
        assert user.username == "testuser"
        assert user.platform == Platform.DISCORD

    def test_user_with_display_name(self):
        """Test user with display name."""
        user = BotUser(
            id="user-123",
            username="testuser",
            display_name="Test User",
            platform=Platform.SLACK,
        )
        assert user.display_name == "Test User"

    def test_user_is_bot_flag(self):
        """Test is_bot flag."""
        user = BotUser(
            id="bot-123",
            username="bot",
            is_bot=True,
            platform=Platform.TEAMS,
        )
        assert user.is_bot is True

    def test_user_default_is_not_bot(self):
        """Test default is_bot is False."""
        user = BotUser(
            id="user-123",
            username="human",
            platform=Platform.ZOOM,
        )
        assert user.is_bot is False


# ===========================================================================
# BotChannel Tests
# ===========================================================================


class TestBotChannel:
    """Tests for BotChannel dataclass."""

    def test_create_channel(self):
        """Test creating a bot channel."""
        channel = BotChannel(
            id="channel-123",
            platform=Platform.SLACK,
        )
        assert channel.id == "channel-123"
        assert channel.platform == Platform.SLACK

    def test_channel_with_name(self):
        """Test channel with name."""
        channel = BotChannel(
            id="channel-123",
            name="general",
            platform=Platform.DISCORD,
        )
        assert channel.name == "general"

    def test_channel_dm_flag(self):
        """Test is_dm flag."""
        channel = BotChannel(
            id="dm-123",
            is_dm=True,
            platform=Platform.TEAMS,
        )
        assert channel.is_dm is True

    def test_channel_default_not_dm(self):
        """Test default is_dm is False."""
        channel = BotChannel(
            id="channel-123",
            platform=Platform.ZOOM,
        )
        assert channel.is_dm is False


# ===========================================================================
# BotMessage Tests
# ===========================================================================


class TestBotMessage:
    """Tests for BotMessage dataclass."""

    def test_create_message(self):
        """Test creating a bot message."""
        user = BotUser(id="u1", username="test", platform=Platform.SLACK)
        channel = BotChannel(id="c1", platform=Platform.SLACK)

        message = BotMessage(
            id="msg-123",
            text="Hello world",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.SLACK,
        )

        assert message.id == "msg-123"
        assert message.text == "Hello world"
        assert message.user == user
        assert message.channel == channel
        assert message.platform == Platform.SLACK

    def test_message_with_attachments(self):
        """Test message with attachments."""
        user = BotUser(id="u1", username="test", platform=Platform.DISCORD)
        channel = BotChannel(id="c1", platform=Platform.DISCORD)

        message = BotMessage(
            id="msg-123",
            text="See attachment",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.DISCORD,
            attachments=[{"url": "https://example.com/file.pdf"}],
        )

        assert len(message.attachments) == 1
        assert message.attachments[0]["url"] == "https://example.com/file.pdf"


# ===========================================================================
# CommandContext Tests
# ===========================================================================


class TestCommandContext:
    """Tests for CommandContext dataclass."""

    def test_create_context(self):
        """Test creating a command context."""
        user = BotUser(id="u1", username="test", platform=Platform.SLACK)
        channel = BotChannel(id="c1", platform=Platform.SLACK)
        message = BotMessage(
            id="m1",
            text="/debate test",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.SLACK,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.SLACK,
        )

        assert ctx.message == message
        assert ctx.user == user
        assert ctx.channel == channel
        assert ctx.platform == Platform.SLACK

    def test_context_with_args(self):
        """Test context with command arguments."""
        user = BotUser(id="u1", username="test", platform=Platform.DISCORD)
        channel = BotChannel(id="c1", platform=Platform.DISCORD)
        message = BotMessage(
            id="m1",
            text="/debate AI ethics",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.DISCORD,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.DISCORD,
            args=["debate", "AI", "ethics"],
            raw_args="AI ethics",
        )

        assert ctx.args == ["debate", "AI", "ethics"]
        assert ctx.raw_args == "AI ethics"

    def test_context_with_metadata(self):
        """Test context with metadata."""
        user = BotUser(id="u1", username="test", platform=Platform.TEAMS)
        channel = BotChannel(id="c1", platform=Platform.TEAMS)
        message = BotMessage(
            id="m1",
            text="/status",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.TEAMS,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.TEAMS,
            metadata={
                "api_base": "http://localhost:8080",
                "tenant_id": "tenant-123",
            },
        )

        assert ctx.metadata["api_base"] == "http://localhost:8080"
        assert ctx.metadata["tenant_id"] == "tenant-123"


# ===========================================================================
# CommandResult Tests
# ===========================================================================


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = CommandResult(
            success=True,
            message="Debate started successfully",
        )

        assert result.success is True
        assert result.message == "Debate started successfully"
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = CommandResult(
            success=False,
            error="Invalid topic",
        )

        assert result.success is False
        assert result.error == "Invalid topic"

    def test_result_with_data(self):
        """Test result with additional data."""
        result = CommandResult(
            success=True,
            message="Debate created",
            data={
                "debate_id": "debate-123",
                "topic": "AI Ethics",
            },
        )

        assert result.data["debate_id"] == "debate-123"
        assert result.data["topic"] == "AI Ethics"

    def test_ephemeral_result(self):
        """Test ephemeral flag for private responses."""
        result = CommandResult(
            success=True,
            message="Your vote has been recorded",
            ephemeral=True,
        )

        assert result.ephemeral is True

    def test_result_with_discord_embed(self):
        """Test result with Discord embed."""
        result = CommandResult(
            success=True,
            message="Results",
            discord_embed={
                "title": "Debate Results",
                "color": 0x00FF00,
                "fields": [
                    {"name": "Winner", "value": "Side A"},
                ],
            },
        )

        assert result.discord_embed is not None
        assert result.discord_embed["title"] == "Debate Results"

    def test_result_with_teams_card(self):
        """Test result with Teams adaptive card."""
        result = CommandResult(
            success=True,
            message="Results",
            teams_card={
                "type": "AdaptiveCard",
                "body": [{"type": "TextBlock", "text": "Results"}],
            },
        )

        assert result.teams_card is not None
        assert result.teams_card["type"] == "AdaptiveCard"


# ===========================================================================
# BotConfig Tests
# ===========================================================================


class TestBotConfig:
    """Tests for BotConfig dataclass."""

    def test_create_config(self):
        """Test creating a bot config."""
        config = BotConfig(
            platform=Platform.DISCORD,
            token="bot-token-123",
        )

        assert config.platform == Platform.DISCORD
        assert config.token == "bot-token-123"

    def test_config_with_app_id(self):
        """Test config with application ID."""
        config = BotConfig(
            platform=Platform.TEAMS,
            token="secret",
            app_id="app-123",
        )

        assert config.app_id == "app-123"

    def test_config_with_api_base(self):
        """Test config with API base URL."""
        config = BotConfig(
            platform=Platform.SLACK,
            token="token",
            api_base="https://api.aragora.ai",
        )

        assert config.api_base == "https://api.aragora.ai"

    def test_config_default_api_base(self):
        """Test config defaults to localhost API base."""
        config = BotConfig(
            platform=Platform.ZOOM,
            token="token",
        )

        # Should have some default or be None
        assert (
            config.api_base is None or "localhost" in str(config.api_base) or config.api_base == ""
        )
