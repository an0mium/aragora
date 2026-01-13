"""
Unit tests for Slack integration.

Tests SlackIntegration message sending, formatting, and rate limiting.
"""

import pytest
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from aragora.integrations.slack import (
    SlackConfig,
    SlackMessage,
    SlackIntegration,
)
from aragora.core import DebateResult


@pytest.fixture
def slack_config():
    """Create a test SlackConfig."""
    return SlackConfig(
        webhook_url="https://hooks.slack.com/test/webhook",
        channel="#test-debates",
        bot_name="TestBot",
        icon_emoji=":test:",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
        min_consensus_confidence=0.7,
        max_messages_per_minute=10,
    )


@pytest.fixture
def slack_integration(slack_config):
    """Create a SlackIntegration instance."""
    return SlackIntegration(slack_config)


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock(spec=DebateResult)
    result.task = "Should AI be regulated in healthcare?"
    result.consensus_reached = True
    result.winner = "Agent A"
    result.confidence = 0.85
    result.rounds_completed = 3
    result.final_proposal = "AI should be regulated with careful oversight..."
    result.final_answer = "AI should be regulated with careful oversight..."
    return result


# SlackConfig Tests


class TestSlackConfig:
    """Tests for SlackConfig validation."""

    def test_config_requires_webhook_url(self):
        """Test that webhook URL is required."""
        with pytest.raises(ValueError, match="webhook URL is required"):
            SlackConfig(webhook_url="")

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SlackConfig(webhook_url="https://hooks.slack.com/test")

        assert config.channel == "#debates"
        assert config.bot_name == "Aragora"
        assert config.notify_on_consensus is True
        assert config.max_messages_per_minute == 10


# SlackMessage Tests


class TestSlackMessage:
    """Tests for SlackMessage formatting."""

    def test_message_to_payload_basic(self, slack_config):
        """Test basic message payload."""
        message = SlackMessage(text="Hello World")
        payload = message.to_payload(slack_config)

        assert payload["text"] == "Hello World"
        assert payload["username"] == "TestBot"
        assert payload["icon_emoji"] == ":test:"

    def test_message_to_payload_with_blocks(self, slack_config):
        """Test message payload with blocks."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Test"}}]
        message = SlackMessage(text="Fallback", blocks=blocks)
        payload = message.to_payload(slack_config)

        assert "blocks" in payload
        assert len(payload["blocks"]) == 1

    def test_message_to_payload_with_attachments(self, slack_config):
        """Test message payload with attachments."""
        attachments = [{"color": "good", "text": "Attachment"}]
        message = SlackMessage(text="Fallback", attachments=attachments)
        payload = message.to_payload(slack_config)

        assert "attachments" in payload
        assert len(payload["attachments"]) == 1

    def test_message_omits_empty_blocks(self, slack_config):
        """Test that empty blocks are not included."""
        message = SlackMessage(text="Just text")
        payload = message.to_payload(slack_config)

        assert "blocks" not in payload
        assert "attachments" not in payload


# Rate Limiting Tests


class TestSlackRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_allows_within_limit(self, slack_integration):
        """Test that messages within limit are allowed."""
        assert slack_integration._check_rate_limit() is True

    def test_rate_limit_blocks_at_limit(self, slack_integration):
        """Test that messages at limit are blocked."""
        # Exhaust rate limit
        for _ in range(10):
            slack_integration._check_rate_limit()

        # 11th call should be blocked
        assert slack_integration._check_rate_limit() is False

    def test_rate_limit_resets_after_minute(self, slack_integration):
        """Test that rate limit resets after a minute."""
        # Exhaust rate limit
        for _ in range(10):
            slack_integration._check_rate_limit()

        # Simulate time passing
        slack_integration._last_reset = datetime.now()
        from datetime import timedelta

        slack_integration._last_reset -= timedelta(seconds=61)

        # Should now be allowed
        assert slack_integration._check_rate_limit() is True
        assert slack_integration._message_count == 1  # Reset to 1


# Message Sending Tests


class TestSlackMessageSending:
    """Tests for message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, slack_integration):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False  # Prevent _get_session from creating new session
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        message = SlackMessage(text="Test message")
        result = await slack_integration._send_message(message)

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, slack_integration):
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="invalid_payload")

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        message = SlackMessage(text="Test message")
        result = await slack_integration._send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_connection_error(self, slack_integration):
        """Test handling of connection error."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        slack_integration._session = mock_session

        message = SlackMessage(text="Test message")
        result = await slack_integration._send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_rate_limited(self, slack_integration):
        """Test that rate limited messages are skipped."""
        # Exhaust rate limit
        slack_integration._message_count = 10

        message = SlackMessage(text="Test message")
        result = await slack_integration._send_message(message)

        assert result is False


# Debate Summary Tests


class TestDebateSummary:
    """Tests for debate summary posting."""

    @pytest.mark.asyncio
    async def test_post_debate_summary_creates_blocks(self, slack_integration, mock_debate_result):
        """Test that debate summary creates proper blocks."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        result = await slack_integration.post_debate_summary(mock_debate_result)

        assert result is True

        # Check that blocks were created in the payload
        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]
        assert "blocks" in payload
        assert len(payload["blocks"]) > 0

    @pytest.mark.asyncio
    async def test_post_debate_summary_skipped_when_disabled(
        self, slack_config, mock_debate_result
    ):
        """Test that summary is skipped when notifications disabled."""
        slack_config.notify_on_debate_end = False
        integration = SlackIntegration(slack_config)

        result = await integration.post_debate_summary(mock_debate_result)

        assert result is True  # Returns True but doesn't send

    def test_build_debate_summary_blocks_consensus_reached(
        self, slack_integration, mock_debate_result
    ):
        """Test block building for consensus reached."""
        blocks = slack_integration._build_debate_summary_blocks(mock_debate_result)

        # Should have header with check mark
        header = blocks[0]
        assert header["type"] == "header"
        assert ":white_check_mark:" in header["text"]["text"]

        # Should have task section
        task_section = blocks[1]
        assert mock_debate_result.task in task_section["text"]["text"]

    def test_build_debate_summary_blocks_no_consensus(self, slack_integration):
        """Test block building when no consensus."""
        result = MagicMock(spec=DebateResult)
        result.task = "Test task"
        result.consensus_reached = False
        result.winner = None
        result.confidence = 0.3
        result.rounds_completed = 5
        result.final_proposal = None

        blocks = slack_integration._build_debate_summary_blocks(result)

        # Should have header with X
        header = blocks[0]
        assert ":x:" in header["text"]["text"]

    def test_build_debate_summary_blocks_truncates_proposal(
        self, slack_integration, mock_debate_result
    ):
        """Test that long proposals are truncated."""
        long_text = "A" * 1000  # Long text
        mock_debate_result.final_answer = long_text

        blocks = slack_integration._build_debate_summary_blocks(mock_debate_result)

        # Find the proposal block
        proposal_block = None
        for block in blocks:
            if block.get("type") == "section" and "Final" in str(block):
                proposal_block = block
                break

        assert proposal_block is not None
        # Should be truncated - the code adds "..." when text > 500 chars
        assert "..." in proposal_block["text"]["text"]


# Consensus Alert Tests


class TestConsensusAlert:
    """Tests for consensus alert sending."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(self, slack_integration):
        """Test successful consensus alert."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        result = await slack_integration.send_consensus_alert(
            debate_id="debate-123",
            confidence=0.85,
            winner="Agent A",
            task="Test task",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_skipped_below_threshold(self, slack_integration):
        """Test that low confidence alerts are skipped."""
        result = await slack_integration.send_consensus_alert(
            debate_id="debate-123",
            confidence=0.5,  # Below threshold of 0.7
        )

        assert result is True  # Returns True but doesn't send

    @pytest.mark.asyncio
    async def test_send_consensus_alert_skipped_when_disabled(self, slack_config):
        """Test that alerts are skipped when disabled."""
        slack_config.notify_on_consensus = False
        integration = SlackIntegration(slack_config)

        result = await integration.send_consensus_alert(
            debate_id="debate-123",
            confidence=0.9,
        )

        assert result is True


# Error Alert Tests


class TestErrorAlert:
    """Tests for error alert sending."""

    @pytest.mark.asyncio
    async def test_send_error_alert_success(self, slack_integration):
        """Test successful error alert."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        result = await slack_integration.send_error_alert(
            error_type="Agent Timeout",
            error_message="Agent failed to respond within 30 seconds",
            debate_id="debate-123",
            severity="error",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_skipped_when_disabled(self, slack_config):
        """Test that error alerts are skipped when disabled."""
        slack_config.notify_on_error = False
        integration = SlackIntegration(slack_config)

        result = await integration.send_error_alert(
            error_type="Test Error",
            error_message="Test message",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_severity_emoji(self, slack_integration):
        """Test that correct emoji is used for severity."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        await slack_integration.send_error_alert(
            error_type="Critical Error",
            error_message="Test",
            severity="critical",
        )

        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]

        # Check for rotating light emoji in header
        header = payload["blocks"][0]
        assert ":rotating_light:" in header["text"]["text"]


# Leaderboard Tests


class TestLeaderboardUpdate:
    """Tests for leaderboard update sending."""

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_success(self, slack_integration):
        """Test successful leaderboard update."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        rankings = [
            {"name": "Agent A", "elo": 1650, "wins": 10},
            {"name": "Agent B", "elo": 1600, "wins": 8},
            {"name": "Agent C", "elo": 1550, "wins": 6},
        ]

        result = await slack_integration.send_leaderboard_update(rankings, top_n=3)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_with_medals(self, slack_integration):
        """Test that top 3 get medal emojis."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        rankings = [
            {"name": "Gold", "elo": 1700, "wins": 12},
            {"name": "Silver", "elo": 1650, "wins": 10},
            {"name": "Bronze", "elo": 1600, "wins": 8},
            {"name": "Fourth", "elo": 1550, "wins": 6},
        ]

        await slack_integration.send_leaderboard_update(rankings, top_n=4)

        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]
        blocks_text = str(payload["blocks"])

        # Check for medals
        assert "🥇" in blocks_text
        assert "🥈" in blocks_text
        assert "🥉" in blocks_text
        assert "#4" in blocks_text


# Session Management Tests


class TestSessionManagement:
    """Tests for aiohttp session management."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, slack_integration):
        """Test that get_session creates a new session."""
        assert slack_integration._session is None

        session = await slack_integration._get_session()

        assert session is not None
        assert slack_integration._session is session

        await slack_integration.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, slack_integration):
        """Test that get_session reuses existing session."""
        session1 = await slack_integration._get_session()
        session2 = await slack_integration._get_session()

        assert session1 is session2

        await slack_integration.close()

    @pytest.mark.asyncio
    async def test_close_closes_session(self, slack_integration):
        """Test that close() closes the session."""
        session = await slack_integration._get_session()
        await slack_integration.close()

        assert slack_integration._session.closed


# Integration Tests


class TestSlackIntegrationEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_multiple_messages_respect_rate_limit(self, slack_integration):
        """Test that multiple messages respect rate limit."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_context

        slack_integration._session = mock_session

        # Send 15 messages (limit is 10)
        results = []
        for i in range(15):
            result = await slack_integration._send_message(SlackMessage(text=f"Message {i}"))
            results.append(result)

        # First 10 should succeed, last 5 should fail
        assert sum(results) == 10
        assert results[:10] == [True] * 10
        assert results[10:] == [False] * 5
