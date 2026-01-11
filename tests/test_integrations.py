"""
Tests for Slack and Discord integrations.

Tests cover:
- SlackConfig and DiscordConfig creation
- Message formatting
- Rate limiting
- Integration manager patterns
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.slack import (
    SlackConfig,
    SlackMessage,
    SlackIntegration,
)
from aragora.integrations.discord import (
    DiscordConfig,
    DiscordEmbed,
    DiscordIntegration,
    DiscordWebhookManager,
    create_discord_integration,
)


class TestSlackConfig:
    """Test SlackConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        assert config.webhook_url == "https://hooks.slack.com/test"
        assert config.channel == "#debates"
        assert config.bot_name == "Aragora"

    def test_full_config(self):
        """Test creating config with all fields."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            channel="#my-debates",
            bot_name="DebateBot",
            icon_emoji=":robot_face:",
            notify_on_consensus=True,
            notify_on_debate_end=True,
            notify_on_error=True,
            min_consensus_confidence=0.8,
            max_messages_per_minute=20,
        )
        assert config.channel == "#my-debates"
        assert config.bot_name == "DebateBot"
        assert config.icon_emoji == ":robot_face:"
        assert config.min_consensus_confidence == 0.8

    def test_config_requires_webhook_url(self):
        """Test config raises on empty webhook URL."""
        with pytest.raises(ValueError, match="webhook URL"):
            SlackConfig(webhook_url="")


class TestSlackMessage:
    """Test SlackMessage dataclass."""

    def test_simple_message(self):
        """Test creating simple message."""
        msg = SlackMessage(text="Hello world")
        assert msg.text == "Hello world"
        assert msg.blocks == []
        assert msg.attachments == []

    def test_message_to_payload(self):
        """Test converting message to payload."""
        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        msg = SlackMessage(text="Test message")
        payload = msg.to_payload(config)
        assert payload["text"] == "Test message"
        assert payload["username"] == "Aragora"
        assert payload["icon_emoji"] == ":speech_balloon:"

    def test_message_with_blocks(self):
        """Test message with Block Kit blocks."""
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}
        ]
        msg = SlackMessage(text="Fallback", blocks=blocks)
        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        payload = msg.to_payload(config)
        assert payload["blocks"] == blocks

    def test_message_with_attachments(self):
        """Test message with attachments."""
        attachments = [{"color": "good", "text": "Success!"}]
        msg = SlackMessage(text="Test", attachments=attachments)
        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        payload = msg.to_payload(config)
        assert payload["attachments"] == attachments


class TestSlackIntegration:
    """Test SlackIntegration class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            channel="#test",
        )

    @pytest.fixture
    def integration(self, config):
        """Create test integration."""
        return SlackIntegration(config)

    def test_create_integration(self, config):
        """Test creating integration."""
        integration = SlackIntegration(config)
        assert integration.config == config

    def test_rate_limit_check(self, integration):
        """Test rate limit checking."""
        # First call should succeed
        assert integration._check_rate_limit() is True
        # Subsequent calls within limit should succeed
        for _ in range(integration.config.max_messages_per_minute - 1):
            integration._check_rate_limit()
        # Over limit should fail
        assert integration._check_rate_limit() is False


class TestSlackIntegrationAsync:
    """Async tests for SlackIntegration."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SlackConfig(
            webhook_url="https://hooks.slack.com/test",
        )

    @pytest.mark.asyncio
    async def test_send_disabled_notifications(self, config):
        """Test notifications respect config flags."""
        config.notify_on_consensus = False
        integration = SlackIntegration(config)
        # Should return True (skipped, not failed)
        result = await integration.send_consensus_alert(
            debate_id="test",
            confidence=0.9,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(self, config):
        """Test successful consensus alert send."""
        integration = SlackIntegration(config)

        with patch.object(
            integration, "_send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_consensus_alert(
                debate_id="test-123",
                confidence=0.9,
                winner="claude",
                task="Test task",
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_low_confidence_skipped(self, config):
        """Test low confidence alerts are skipped."""
        config.min_consensus_confidence = 0.8
        integration = SlackIntegration(config)
        # Should return True (skipped) for low confidence
        result = await integration.send_consensus_alert(
            debate_id="test",
            confidence=0.5,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_close_session(self, config):
        """Test closing aiohttp session."""
        integration = SlackIntegration(config)
        session = await integration._get_session()
        assert not session.closed
        await integration.close()
        assert session.closed


class TestDiscordConfig:
    """Test DiscordConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = DiscordConfig(webhook_url="https://discord.com/api/webhooks/test")
        assert config.webhook_url == "https://discord.com/api/webhooks/test"
        assert config.username == "Aragora Debates"
        assert config.enabled is True

    def test_full_config(self):
        """Test creating config with all fields."""
        config = DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test",
            username="DebateBot",
            avatar_url="https://example.com/avatar.png",
            enabled=True,
            include_agent_details=True,
            include_vote_breakdown=True,
            rate_limit_per_minute=60,
        )
        assert config.username == "DebateBot"
        assert config.avatar_url == "https://example.com/avatar.png"
        assert config.rate_limit_per_minute == 60


class TestDiscordEmbed:
    """Test DiscordEmbed dataclass."""

    def test_minimal_embed(self):
        """Test creating minimal embed."""
        embed = DiscordEmbed(title="Test")
        data = embed.to_dict()
        assert data["title"] == "Test"

    def test_full_embed(self):
        """Test creating full embed."""
        embed = DiscordEmbed(
            title="Test Title",
            description="Test description",
            color=0xFF0000,
            url="https://example.com",
            timestamp="2024-01-15T12:00:00Z",
            footer={"text": "Footer text"},
            fields=[
                {"name": "Field 1", "value": "Value 1", "inline": True}
            ],
        )
        data = embed.to_dict()
        assert data["title"] == "Test Title"
        assert data["description"] == "Test description"
        assert data["color"] == 0xFF0000
        assert data["footer"]["text"] == "Footer text"
        assert len(data["fields"]) == 1

    def test_embed_truncates_description(self):
        """Test description is truncated at Discord limit."""
        long_desc = "x" * 3000
        embed = DiscordEmbed(description=long_desc)
        data = embed.to_dict()
        assert len(data["description"]) <= 2048

    def test_embed_truncates_fields(self):
        """Test fields are limited to 25."""
        embed = DiscordEmbed(
            fields=[{"name": f"Field {i}", "value": f"Value {i}"} for i in range(30)]
        )
        data = embed.to_dict()
        assert len(data["fields"]) <= 25


class TestDiscordIntegration:
    """Test DiscordIntegration class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=True,
        )

    @pytest.fixture
    def integration(self, config):
        """Create test integration."""
        return DiscordIntegration(config)

    def test_create_integration(self, config):
        """Test creating integration."""
        integration = DiscordIntegration(config)
        assert integration.config == config

    def test_colors_defined(self, integration):
        """Test color constants are defined."""
        assert "debate_start" in integration.COLORS
        assert "consensus" in integration.COLORS
        assert "error" in integration.COLORS

    def test_truncate(self, integration):
        """Test text truncation."""
        long_text = "x" * 2000
        truncated = integration._truncate(long_text, 1024)
        assert len(truncated) <= 1024
        assert truncated.endswith("...")


class TestDiscordIntegrationAsync:
    """Async tests for DiscordIntegration."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_send_disabled(self, config):
        """Test send returns False when disabled."""
        config.enabled = False
        integration = DiscordIntegration(config)
        result = await integration.send_debate_start(
            debate_id="test",
            topic="Test",
            agents=[],
            config={},
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_send_debate_start_success(self, config):
        """Test successful debate start send."""
        integration = DiscordIntegration(config)

        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_debate_start(
                debate_id="test-123",
                topic="Test topic",
                agents=["claude", "gpt-4"],
                config={"rounds": 3, "consensus_mode": "majority"},
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_reached(self, config):
        """Test sending consensus notification."""
        integration = DiscordIntegration(config)

        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_consensus_reached(
                debate_id="test-123",
                topic="Test",
                consensus_type="unanimous",
                result={"winner": "A", "confidence": 0.95, "votes": {"A": 3, "B": 0}},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_no_consensus(self, config):
        """Test sending no-consensus notification."""
        integration = DiscordIntegration(config)

        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_no_consensus(
                debate_id="test-123",
                topic="Test topic",
                final_state={"rounds_completed": 5, "final_votes": {"A": 2, "B": 2}},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_error(self, config):
        """Test sending error notification."""
        integration = DiscordIntegration(config)

        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_error(
                error_type="AgentTimeout",
                message="Agent failed to respond",
                debate_id="test-123",
                details={"agent": "claude", "timeout_s": 30},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_round_summary(self, config):
        """Test sending round summary."""
        integration = DiscordIntegration(config)

        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            result = await integration.send_round_summary(
                debate_id="test-123",
                round_number=2,
                total_rounds=5,
                summary="Agents discussed the pros and cons.",
                agent_positions={"claude": "For", "gpt-4": "Against"},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_close_session(self, config):
        """Test closing aiohttp session."""
        integration = DiscordIntegration(config)
        # Get session to create it
        session = await integration._get_session()
        assert not session.closed
        # Close
        await integration.close()
        assert session.closed


class TestDiscordWebhookManager:
    """Test DiscordWebhookManager class."""

    def test_register_integration(self):
        """Test registering an integration."""
        manager = DiscordWebhookManager()
        config = DiscordConfig(webhook_url="https://discord.com/test")
        manager.register("test", config)
        assert "test" in manager._integrations

    def test_unregister_integration(self):
        """Test unregistering an integration."""
        manager = DiscordWebhookManager()
        config = DiscordConfig(webhook_url="https://discord.com/test")
        manager.register("test", config)
        manager.unregister("test")
        assert "test" not in manager._integrations

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent integration doesn't raise."""
        manager = DiscordWebhookManager()
        manager.unregister("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to multiple integrations."""
        manager = DiscordWebhookManager()
        config1 = DiscordConfig(webhook_url="https://discord.com/test1")
        config2 = DiscordConfig(webhook_url="https://discord.com/test2")
        manager.register("hook1", config1)
        manager.register("hook2", config2)

        # Mock the send methods
        with patch.object(
            manager._integrations["hook1"],
            "send_debate_start",
            new_callable=AsyncMock,
        ) as mock1, patch.object(
            manager._integrations["hook2"],
            "send_debate_start",
            new_callable=AsyncMock,
        ) as mock2:
            mock1.return_value = True
            mock2.return_value = True

            results = await manager.broadcast(
                "send_debate_start",
                debate_id="test",
                topic="Test",
                agents=[],
                config={},
            )

            assert results["hook1"] is True
            assert results["hook2"] is True

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all integrations."""
        manager = DiscordWebhookManager()
        config = DiscordConfig(webhook_url="https://discord.com/test")
        manager.register("test", config)

        with patch.object(
            manager._integrations["test"],
            "close",
            new_callable=AsyncMock,
        ) as mock_close:
            await manager.close_all()
            mock_close.assert_called_once()


class TestCreateDiscordIntegration:
    """Test factory function."""

    def test_create_with_minimal_args(self):
        """Test creating integration with minimal arguments."""
        integration = create_discord_integration(
            webhook_url="https://discord.com/test"
        )
        assert integration.config.webhook_url == "https://discord.com/test"

    def test_create_with_kwargs(self):
        """Test creating integration with keyword arguments."""
        integration = create_discord_integration(
            webhook_url="https://discord.com/test",
            username="CustomBot",
            enabled=False,
        )
        assert integration.config.username == "CustomBot"
        assert integration.config.enabled is False
