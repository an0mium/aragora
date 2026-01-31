"""Tests for Slack integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.slack import (
    SlackConfig,
    SlackIntegration,
    SlackMessage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/X")


@pytest.fixture
def integration(config):
    return SlackIntegration(config)


def _make_debate_result(**kwargs):
    result = MagicMock()
    result.task = kwargs.get("task", "Design a rate limiter")
    result.final_answer = kwargs.get("final_answer", "Use token bucket algorithm")
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# SlackConfig Tests
# =============================================================================


class TestSlackConfig:
    def test_default_values(self):
        cfg = SlackConfig(webhook_url="https://hooks.slack.com/services/test")
        assert cfg.channel == "#debates"
        assert cfg.bot_name == "Aragora"
        assert cfg.icon_emoji == ":speech_balloon:"
        assert cfg.notify_on_consensus is True
        assert cfg.notify_on_debate_end is True
        assert cfg.notify_on_error is True
        assert cfg.min_consensus_confidence == 0.7
        assert cfg.max_messages_per_minute == 10

    def test_requires_webhook_url(self):
        with pytest.raises(ValueError, match="Slack webhook URL is required"):
            SlackConfig(webhook_url="")


# =============================================================================
# SlackMessage Tests
# =============================================================================


class TestSlackMessage:
    def test_basic_payload(self):
        msg = SlackMessage(text="Hello Slack")
        cfg = SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/X")
        payload = msg.to_payload(cfg)
        assert payload["text"] == "Hello Slack"
        assert payload["username"] == "Aragora"
        assert payload["icon_emoji"] == ":speech_balloon:"
        assert "blocks" not in payload

    def test_payload_with_blocks(self):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]
        msg = SlackMessage(text="Hello", blocks=blocks)
        cfg = SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/X")
        payload = msg.to_payload(cfg)
        assert payload["blocks"] == blocks

    def test_payload_with_attachments(self):
        attachments = [{"color": "#FF0000", "text": "Error"}]
        msg = SlackMessage(text="Alert", attachments=attachments)
        cfg = SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/X")
        payload = msg.to_payload(cfg)
        assert payload["attachments"] == attachments


# =============================================================================
# SlackIntegration Tests
# =============================================================================


class TestSlackIntegration:
    def test_initialization(self, integration):
        assert integration._session is None
        assert integration._message_count == 0

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
        assert integration._message_count == 1

    @pytest.mark.asyncio
    async def test_verify_webhook_empty_url(self):
        # SlackConfig requires non-empty URL, so use patch
        cfg = SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/X")
        integ = SlackIntegration(cfg)
        cfg.webhook_url = ""
        result = await integ.verify_webhook()
        assert result is False

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(self, integration):
        integration.config.notify_on_debate_end = False
        result = _make_debate_result()
        success = await integration.post_debate_summary(result)
        assert success is True  # Returns True when disabled

    @pytest.mark.asyncio
    async def test_post_debate_summary(self, integration):
        with patch.object(
            integration, "_send_message", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = _make_debate_result()
            success = await integration.post_debate_summary(result)
            assert success is True
            mock_send.assert_called_once()

    def test_build_debate_summary_blocks(self, integration):
        result = _make_debate_result()
        blocks = integration._build_debate_summary_blocks(result)
        assert len(blocks) > 0
        # Header block
        assert blocks[0]["type"] == "header"
        # Task section
        assert "Task:" in str(blocks[1])

    def test_build_debate_summary_blocks_no_consensus(self, integration):
        result = _make_debate_result(consensus_reached=False, final_answer=None)
        blocks = integration._build_debate_summary_blocks(result)
        # Should not include final proposal block
        block_texts = str(blocks)
        assert "Final Proposal" not in block_texts

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, integration):
        integration.config.notify_on_consensus = False
        result = await integration.send_consensus_alert("d-1", 0.9)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, integration):
        result = await integration.send_consensus_alert("d-1", 0.5)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert(self, integration):
        with patch.object(
            integration, "_send_message", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await integration.send_consensus_alert(
                debate_id="d-1", confidence=0.9, winner="claude", task="Test task"
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, integration):
        integration.config.notify_on_error = False
        result = await integration.send_error_alert("TestError", "msg")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(
                error_type="AgentTimeout",
                error_message="Agent failed",
                debate_id="d-123",
                severity="error",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            rankings = [
                {"name": "claude", "elo": 1800, "wins": 10},
                {"name": "gpt4", "elo": 1750, "wins": 8},
            ]
            result = await integration.send_leaderboard_update(rankings)
            assert result is True

    @pytest.mark.asyncio
    async def test_post_debate_with_voting(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.post_debate_with_voting(
                debate_id="d-123",
                task="Test task",
                agents=["claude", "gpt4", "gemini"],
                current_round=1,
                total_rounds=3,
            )
            assert result == "d-123"

    @pytest.mark.asyncio
    async def test_post_debate_with_voting_failure(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=False):
            result = await integration.post_debate_with_voting(
                debate_id="d-123",
                task="Test",
                agents=["claude"],
                current_round=1,
                total_rounds=3,
            )
            assert result is None

    def test_build_debate_with_voting_blocks(self, integration):
        blocks = integration._build_debate_with_voting_blocks(
            debate_id="d-123",
            task="Test task",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        # Check for header, section, divider, voting prompt, action blocks, context
        assert any(b["type"] == "header" for b in blocks)
        assert any(b["type"] == "actions" for b in blocks)

    @pytest.mark.asyncio
    async def test_update_debate_progress(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.update_debate_progress(
                debate_id="d-123",
                current_round=2,
                total_rounds=5,
                latest_argument="Token bucket is best because...",
                agent_name="claude",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_post_consensus_with_votes(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            debate_result = _make_debate_result()
            user_votes = {"claude": 5, "gpt4": 3}
            result = await integration.post_consensus_with_votes(
                debate_id="d-123",
                result=debate_result,
                user_votes=user_votes,
            )
            assert result is True

    def test_build_consensus_with_votes_blocks(self, integration):
        debate_result = _make_debate_result()
        user_votes = {"claude": 5, "gpt4": 3}
        blocks = integration._build_consensus_with_votes_blocks(
            debate_id="d-123",
            result=debate_result,
            user_votes=user_votes,
        )
        block_text = str(blocks)
        assert "User Votes" in block_text
        assert "5 votes" in block_text

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
