"""Tests for Discord integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.discord import (
    DiscordConfig,
    DiscordEmbed,
    DiscordIntegration,
    DiscordWebhookManager,
    create_discord_integration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return DiscordConfig(webhook_url="https://discord.com/api/webhooks/test/token")


@pytest.fixture
def disabled_config():
    return DiscordConfig(
        webhook_url="https://discord.com/api/webhooks/test/token",
        enabled=False,
    )


@pytest.fixture
def integration(config):
    return DiscordIntegration(config)


@pytest.fixture
def disabled_integration(disabled_config):
    return DiscordIntegration(disabled_config)


def _mock_response(status=204, headers=None, text=""):
    resp = AsyncMock()
    resp.status = status
    resp.headers = headers or {}
    resp.text = AsyncMock(return_value=text)
    return resp


# =============================================================================
# DiscordConfig Tests
# =============================================================================


class TestDiscordConfig:
    def test_default_values(self):
        cfg = DiscordConfig(webhook_url="https://example.com")
        assert cfg.username == "Aragora Debates"
        assert cfg.avatar_url == ""
        assert cfg.enabled is True
        assert cfg.include_agent_details is True
        assert cfg.include_vote_breakdown is True
        assert cfg.max_summary_length == 1900
        assert cfg.rate_limit_per_minute == 30
        assert cfg.retry_count == 3
        assert cfg.retry_delay == 1.0

    def test_custom_values(self):
        cfg = DiscordConfig(
            webhook_url="https://example.com",
            username="Custom Bot",
            rate_limit_per_minute=10,
        )
        assert cfg.username == "Custom Bot"
        assert cfg.rate_limit_per_minute == 10


# =============================================================================
# DiscordEmbed Tests
# =============================================================================


class TestDiscordEmbed:
    def test_to_dict_minimal(self):
        embed = DiscordEmbed()
        result = embed.to_dict()
        # Only color is set by default (non-zero)
        assert result.get("color") == 0x5865F2

    def test_to_dict_full(self):
        embed = DiscordEmbed(
            title="Test Title",
            description="Test Desc",
            color=0xFF0000,
            url="https://example.com",
            timestamp="2024-01-01T00:00:00Z",
            footer={"text": "footer"},
            author={"name": "author"},
            fields=[{"name": "f1", "value": "v1"}],
            thumbnail={"url": "https://example.com/img.png"},
        )
        result = embed.to_dict()
        assert result["title"] == "Test Title"
        assert result["description"] == "Test Desc"
        assert result["color"] == 0xFF0000
        assert result["url"] == "https://example.com"
        assert result["timestamp"] == "2024-01-01T00:00:00Z"
        assert result["footer"] == {"text": "footer"}
        assert result["author"] == {"name": "author"}
        assert len(result["fields"]) == 1
        assert result["thumbnail"] == {"url": "https://example.com/img.png"}

    def test_description_truncated_at_2048(self):
        embed = DiscordEmbed(description="x" * 3000)
        result = embed.to_dict()
        assert len(result["description"]) == 2048

    def test_fields_capped_at_25(self):
        fields = [{"name": f"f{i}", "value": f"v{i}"} for i in range(30)]
        embed = DiscordEmbed(fields=fields)
        result = embed.to_dict()
        assert len(result["fields"]) == 25


# =============================================================================
# DiscordIntegration Tests
# =============================================================================


class TestDiscordIntegration:
    def test_initialization(self, integration):
        assert integration.config.webhook_url == "https://discord.com/api/webhooks/test/token"
        assert integration._request_times == []
        assert integration._session is None

    def test_truncate_within_limit(self, integration):
        result = integration._truncate("short", 1024)
        assert result == "short"

    def test_truncate_over_limit(self, integration):
        result = integration._truncate("x" * 2000, 1024)
        assert len(result) == 1024
        assert result.endswith("...")

    def test_colors_defined(self, integration):
        assert "debate_start" in DiscordIntegration.COLORS
        assert "consensus" in DiscordIntegration.COLORS
        assert "no_consensus" in DiscordIntegration.COLORS
        assert "error" in DiscordIntegration.COLORS

    @pytest.mark.asyncio
    async def test_send_webhook_disabled(self, disabled_integration):
        result = await disabled_integration._send_webhook([], content="test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, integration):
        mock_resp = _mock_response(status=204)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with patch.object(integration, "_get_session", return_value=mock_session):
            with patch.object(integration, "_check_rate_limit", return_value=None):
                result = await integration._send_webhook([DiscordEmbed(title="Test")])
                assert result is True

    @pytest.mark.asyncio
    async def test_send_webhook_client_error(self, integration):
        mock_resp = _mock_response(status=400, text="Bad Request")
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with patch.object(integration, "_get_session", return_value=mock_session):
            with patch.object(integration, "_check_rate_limit", return_value=None):
                result = await integration._send_webhook([])
                assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook_empty_url(self):
        cfg = DiscordConfig(webhook_url="")
        integ = DiscordIntegration(cfg)
        result = await integ.verify_webhook()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_debate_start(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_debate_start(
                debate_id="debate-123456789",
                topic="Test topic for debate",
                agents=["claude", "gpt4", "gemini"],
                config={"rounds": 3, "consensus_mode": "majority"},
            )
            assert result is True
            mock_send.assert_called_once()
            embeds = mock_send.call_args[0][0]
            assert len(embeds) == 1
            assert embeds[0].title == "Debate Started"

    @pytest.mark.asyncio
    async def test_send_debate_start_no_agent_details(self):
        cfg = DiscordConfig(
            webhook_url="https://example.com",
            include_agent_details=False,
        )
        integ = DiscordIntegration(cfg)
        with patch.object(
            integ, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integ.send_debate_start(
                debate_id="d1",
                topic="T",
                agents=["a1"],
                config={},
            )
            assert result is True
            embeds = mock_send.call_args[0][0]
            # Should not include agent field when include_agent_details=False
            field_names = [f["name"] for f in embeds[0].fields]
            assert "Participating Agents" not in field_names

    @pytest.mark.asyncio
    async def test_send_consensus_reached(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_consensus_reached(
                debate_id="d-123",
                topic="Test",
                consensus_type="unanimous",
                result={"winner": "claude", "confidence": 0.95, "votes": {"A": 3, "B": 1}},
            )
            assert result is True
            embeds = mock_send.call_args[0][0]
            assert embeds[0].title == "Consensus Reached"

    @pytest.mark.asyncio
    async def test_send_no_consensus(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_no_consensus(
                debate_id="d-456",
                topic="Unresolved topic",
                final_state={"rounds_completed": 5, "final_votes": {"A": 2, "B": 2}},
            )
            assert result is True
            embeds = mock_send.call_args[0][0]
            assert "No Consensus" in embeds[0].title

    @pytest.mark.asyncio
    async def test_send_error(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_error(
                error_type="AgentTimeout",
                message="Agent did not respond",
                debate_id="d-789",
                details={"agent": "gpt4", "timeout": "30s"},
            )
            assert result is True
            embeds = mock_send.call_args[0][0]
            assert "AgentTimeout" in embeds[0].title

    @pytest.mark.asyncio
    async def test_send_error_no_debate_id(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_error(
                error_type="SystemError",
                message="Something failed",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_round_summary(self, integration):
        with patch.object(
            integration, "_send_webhook", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_round_summary(
                debate_id="d-101",
                round_number=2,
                total_rounds=5,
                summary="Agents debated rate limiting.",
                agent_positions={"claude": "Token bucket", "gpt4": "Sliding window"},
            )
            assert result is True
            embeds = mock_send.call_args[0][0]
            assert "Round 2/5" in embeds[0].title

    @pytest.mark.asyncio
    async def test_close_session(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, integration):
        # Should not raise when no session
        await integration.close()


# =============================================================================
# DiscordWebhookManager Tests
# =============================================================================


class TestDiscordWebhookManager:
    def test_register_and_unregister(self):
        manager = DiscordWebhookManager()
        cfg = DiscordConfig(webhook_url="https://example.com")
        manager.register("test", cfg)
        assert "test" in manager._integrations
        manager.unregister("test")
        assert "test" not in manager._integrations

    def test_unregister_nonexistent(self):
        manager = DiscordWebhookManager()
        # Should not raise
        manager.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_broadcast(self):
        manager = DiscordWebhookManager()
        cfg = DiscordConfig(webhook_url="https://example.com")
        manager.register("ch1", cfg)
        manager.register("ch2", cfg)

        with patch.object(
            DiscordIntegration, "send_error", new_callable=AsyncMock, return_value=True
        ):
            results = await manager.broadcast(
                "send_error",
                error_type="Test",
                message="msg",
            )
            assert len(results) == 2
            assert all(v is True for v in results.values())

    @pytest.mark.asyncio
    async def test_broadcast_invalid_method(self):
        manager = DiscordWebhookManager()
        cfg = DiscordConfig(webhook_url="https://example.com")
        manager.register("ch1", cfg)
        results = await manager.broadcast("nonexistent_method")
        assert results == {}

    @pytest.mark.asyncio
    async def test_broadcast_handles_exception(self):
        manager = DiscordWebhookManager()
        cfg = DiscordConfig(webhook_url="https://example.com")
        manager.register("ch1", cfg)

        with patch.object(
            DiscordIntegration, "send_error", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            results = await manager.broadcast("send_error", error_type="Test", message="msg")
            assert results["ch1"] is False

    @pytest.mark.asyncio
    async def test_close_all(self):
        manager = DiscordWebhookManager()
        cfg = DiscordConfig(webhook_url="https://example.com")
        manager.register("ch1", cfg)
        manager.register("ch2", cfg)

        with patch.object(DiscordIntegration, "close", new_callable=AsyncMock):
            await manager.close_all()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateDiscordIntegration:
    def test_factory(self):
        integ = create_discord_integration("https://example.com/webhook", username="MyBot")
        assert isinstance(integ, DiscordIntegration)
        assert integ.config.username == "MyBot"
        assert integ.config.webhook_url == "https://example.com/webhook"
