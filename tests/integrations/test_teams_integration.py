"""Tests for Microsoft Teams integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.teams import (
    AdaptiveCard,
    TeamsConfig,
    TeamsIntegration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return TeamsConfig(webhook_url="https://xxx.webhook.office.com/webhookb2/test")


@pytest.fixture
def integration(config):
    return TeamsIntegration(config)


@pytest.fixture
def unconfigured_integration():
    return TeamsIntegration(TeamsConfig(webhook_url=""))


def _make_debate_result(**kwargs):
    result = MagicMock()
    result.task = kwargs.get("task", "Design a rate limiter")
    result.final_answer = kwargs.get("final_answer", "Use token bucket")
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.rounds_completed = kwargs.get("rounds_completed", 3)
    result.winner = kwargs.get("winner", "claude")
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# TeamsConfig Tests
# =============================================================================


class TestTeamsConfig:
    def test_default_values(self):
        cfg = TeamsConfig(webhook_url="https://test.com")
        assert cfg.bot_name == "Aragora"
        assert cfg.notify_on_consensus is True
        assert cfg.notify_on_debate_end is True
        assert cfg.notify_on_error is True
        assert cfg.notify_on_leaderboard is False
        assert cfg.min_consensus_confidence == 0.7
        assert cfg.max_messages_per_minute == 10

    def test_env_fallback(self):
        with patch.dict("os.environ", {"TEAMS_WEBHOOK_URL": "https://env.webhook.com"}):
            cfg = TeamsConfig()
            assert cfg.webhook_url == "https://env.webhook.com"


# =============================================================================
# AdaptiveCard Tests
# =============================================================================


class TestAdaptiveCard:
    def test_to_payload_basic(self):
        card = AdaptiveCard(title="Test Title")
        payload = card.to_payload()
        assert payload["type"] == "message"
        assert len(payload["attachments"]) == 1
        content = payload["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert content["version"] == "1.4"
        # Title should be the first body element
        assert content["body"][0]["text"] == "Test Title"

    def test_to_payload_with_body(self):
        body = [{"type": "TextBlock", "text": "Hello"}]
        card = AdaptiveCard(title="Title", body=body)
        payload = card.to_payload()
        content = payload["attachments"][0]["content"]
        assert len(content["body"]) == 2  # title + body element

    def test_to_payload_with_actions(self):
        actions = [{"type": "Action.OpenUrl", "title": "View", "url": "https://example.com"}]
        card = AdaptiveCard(title="Title", actions=actions)
        payload = card.to_payload()
        content = payload["attachments"][0]["content"]
        assert "actions" in content
        assert content["actions"][0]["title"] == "View"

    def test_to_payload_without_actions(self):
        card = AdaptiveCard(title="Title")
        payload = card.to_payload()
        content = payload["attachments"][0]["content"]
        assert "actions" not in content


# =============================================================================
# TeamsIntegration Tests
# =============================================================================


class TestTeamsIntegration:
    def test_initialization(self, integration):
        assert integration._session is None
        assert integration._message_count == 0

    def test_is_configured(self, integration):
        assert integration.is_configured is True

    def test_is_not_configured(self, unconfigured_integration):
        assert unconfigured_integration.is_configured is False

    def test_check_rate_limit_allows(self, integration):
        assert integration._check_rate_limit() is True
        assert integration._message_count == 1

    def test_check_rate_limit_blocks(self, integration):
        integration._message_count = 10
        integration._last_reset = datetime.now()
        assert integration._check_rate_limit() is False

    def test_check_rate_limit_resets(self, integration):
        integration._message_count = 10
        integration._last_reset = datetime.now() - timedelta(seconds=61)
        assert integration._check_rate_limit() is True

    @pytest.mark.asyncio
    async def test_send_card_not_configured(self, unconfigured_integration):
        card = AdaptiveCard(title="Test")
        result = await unconfigured_integration._send_card(card)
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook_not_configured(self, unconfigured_integration):
        result = await unconfigured_integration.verify_webhook()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook(self, integration):
        with patch.object(integration, "_send_card", new_callable=AsyncMock, return_value=True):
            result = await integration.verify_webhook()
            assert result is True

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(self, integration):
        integration.config.notify_on_debate_end = False
        result = _make_debate_result()
        success = await integration.post_debate_summary(result)
        assert success is False

    @pytest.mark.asyncio
    async def test_post_debate_summary(self, integration):
        with patch.object(
            integration, "_send_card", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = _make_debate_result()
            success = await integration.post_debate_summary(result)
            assert success is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, integration):
        integration.config.notify_on_consensus = False
        result = await integration.send_consensus_alert("d-1", "Answer", 0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, integration):
        result = await integration.send_consensus_alert("d-1", "Answer", 0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_consensus_alert(self, integration):
        with patch.object(integration, "_send_card", new_callable=AsyncMock, return_value=True):
            result = await integration.send_consensus_alert(
                debate_id="d-1",
                answer="Use token bucket",
                confidence=0.9,
                agents=["claude", "gpt4"],
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, integration):
        integration.config.notify_on_error = False
        result = await integration.send_error_alert("d-1", "Error msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_error_alert(self, integration):
        with patch.object(integration, "_send_card", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(
                debate_id="d-1",
                error="Agent timeout",
                phase="proposal",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_no_phase(self, integration):
        with patch.object(integration, "_send_card", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(
                debate_id="d-1",
                error="System error",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_disabled(self, integration):
        result = await integration.send_leaderboard_update([])
        assert result is False  # notify_on_leaderboard is False by default

    @pytest.mark.asyncio
    async def test_send_leaderboard_update(self, integration):
        integration.config.notify_on_leaderboard = True
        with patch.object(integration, "_send_card", new_callable=AsyncMock, return_value=True):
            rankings = [
                {"name": "claude", "elo": 1800, "wins": 10, "losses": 2},
                {"name": "gpt4", "elo": 1750, "wins": 8, "losses": 4},
            ]
            result = await integration.send_leaderboard_update(rankings, domain="coding")
            assert result is True

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        async with TeamsIntegration(config) as teams:
            assert isinstance(teams, TeamsIntegration)

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
