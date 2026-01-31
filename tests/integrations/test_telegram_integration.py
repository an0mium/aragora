"""Tests for Telegram integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.telegram import (
    InlineButton,
    TelegramConfig,
    TelegramIntegration,
    TelegramMessage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return TelegramConfig(bot_token="123456:ABC-DEF", chat_id="-1001234567890")


@pytest.fixture
def integration(config):
    return TelegramIntegration(config)


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
# TelegramConfig Tests
# =============================================================================


class TestTelegramConfig:
    def test_default_values(self):
        cfg = TelegramConfig(bot_token="tok", chat_id="cid")
        assert cfg.notify_on_consensus is True
        assert cfg.notify_on_debate_end is True
        assert cfg.notify_on_error is True
        assert cfg.min_consensus_confidence == 0.7
        assert cfg.max_messages_per_minute == 20
        assert cfg.max_retries == 3
        assert cfg.parse_mode == "HTML"

    def test_requires_bot_token(self):
        with pytest.raises(ValueError, match="bot token is required"):
            TelegramConfig(bot_token="", chat_id="cid")

    def test_requires_chat_id(self):
        with pytest.raises(ValueError, match="chat ID is required"):
            TelegramConfig(bot_token="tok", chat_id="")

    def test_api_base(self):
        cfg = TelegramConfig(bot_token="123456:ABC", chat_id="cid")
        assert cfg.api_base == "https://api.telegram.org/bot123456:ABC"


# =============================================================================
# InlineButton Tests
# =============================================================================


class TestInlineButton:
    def test_url_button(self):
        btn = InlineButton(text="Click", url="https://example.com")
        d = btn.to_dict()
        assert d["text"] == "Click"
        assert d["url"] == "https://example.com"
        assert "callback_data" not in d

    def test_callback_button(self):
        btn = InlineButton(text="Vote", callback_data="vote_claude")
        d = btn.to_dict()
        assert d["text"] == "Vote"
        assert d["callback_data"] == "vote_claude"
        assert "url" not in d

    def test_text_only(self):
        btn = InlineButton(text="Just text")
        d = btn.to_dict()
        assert d == {"text": "Just text"}


# =============================================================================
# TelegramMessage Tests
# =============================================================================


class TestTelegramMessage:
    def test_basic_payload(self):
        msg = TelegramMessage(text="Hello")
        cfg = TelegramConfig(bot_token="tok", chat_id="123")
        payload = msg.to_payload(cfg)
        assert payload["chat_id"] == "123"
        assert payload["text"] == "Hello"
        assert payload["parse_mode"] == "HTML"
        assert payload["disable_web_page_preview"] is True
        assert "reply_markup" not in payload

    def test_payload_with_keyboard(self):
        buttons = [[InlineButton(text="Click", url="https://example.com")]]
        msg = TelegramMessage(text="Hello", reply_markup=buttons)
        cfg = TelegramConfig(bot_token="tok", chat_id="123")
        payload = msg.to_payload(cfg)
        assert "reply_markup" in payload
        assert "inline_keyboard" in payload["reply_markup"]

    def test_payload_notification_disabled(self):
        msg = TelegramMessage(text="Hello", disable_notification=True)
        cfg = TelegramConfig(bot_token="tok", chat_id="123")
        payload = msg.to_payload(cfg)
        assert payload["disable_notification"] is True


# =============================================================================
# TelegramIntegration Tests
# =============================================================================


class TestTelegramIntegration:
    def test_initialization(self, integration):
        assert integration._session is None
        assert integration._message_count == 0

    def test_escape_html(self, integration):
        assert integration._escape_html("<b>test&</b>") == "&lt;b&gt;test&amp;&lt;/b&gt;"

    def test_check_rate_limit_allows(self, integration):
        assert integration._check_rate_limit() is True
        assert integration._message_count == 1

    def test_check_rate_limit_blocks(self, integration):
        integration._message_count = 20
        integration._last_reset = datetime.now()
        assert integration._check_rate_limit() is False

    def test_check_rate_limit_resets(self, integration):
        integration._message_count = 20
        integration._last_reset = datetime.now() - timedelta(seconds=61)
        assert integration._check_rate_limit() is True

    @pytest.mark.asyncio
    async def test_verify_connection_no_token(self):
        cfg = TelegramConfig(bot_token="tok", chat_id="cid")
        integ = TelegramIntegration(cfg)
        integ.config.bot_token = ""
        result = await integ.verify_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(self, integration):
        integration.config.notify_on_debate_end = False
        result = _make_debate_result()
        success = await integration.post_debate_summary(result)
        assert success is True

    @pytest.mark.asyncio
    async def test_post_debate_summary(self, integration):
        with patch.object(
            integration, "_send_message", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = _make_debate_result()
            success = await integration.post_debate_summary(result)
            assert success is True
            mock_send.assert_called_once()

    def test_build_debate_summary_html(self, integration):
        result = _make_debate_result()
        html = integration._build_debate_summary_html(result)
        assert "<b>" in html
        assert "Debate Completed" in html
        assert "Design a rate limiter" in html

    def test_build_debate_summary_html_no_consensus(self, integration):
        result = _make_debate_result(consensus_reached=False, final_answer=None)
        html = integration._build_debate_summary_html(result)
        assert "Final Proposal" not in html

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
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_consensus_alert(
                debate_id="d-123",
                confidence=0.9,
                winner="claude",
                task="Test task",
            )
            assert result is True

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
    async def test_send_error_alert_with_debate_id(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(
                error_type="Err",
                error_message="msg",
                debate_id="d-456",
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
    async def test_send_debate_started(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_debate_started(
                debate_id="d-123",
                task="Test task",
                agents=["claude", "gpt4", "gemini"],
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_debate_started_many_agents(self, integration):
        with patch.object(integration, "_send_message", new_callable=AsyncMock, return_value=True):
            agents = [f"agent-{i}" for i in range(10)]
            result = await integration.send_debate_started(
                debate_id="d-123", task="Test", agents=agents
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
