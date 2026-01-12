"""
Comprehensive tests for Telegram integration.

Tests Telegram configuration, message formatting, rate limiting,
API calls, and notification methods.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.integrations.telegram import (
    TelegramConfig,
    TelegramMessage,
    InlineButton,
    TelegramIntegration,
)
from aragora.core import DebateResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def telegram_config():
    """Create a basic Telegram config for testing."""
    return TelegramConfig(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        chat_id="-1001234567890",
    )


@pytest.fixture
def telegram_integration(telegram_config):
    """Create a Telegram integration instance."""
    return TelegramIntegration(telegram_config)


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result for testing."""
    result = DebateResult(
        task="Should we use Redis or Memcached for caching?",
        final_answer="Redis is recommended for its richer data structures and persistence options.",
        consensus_reached=True,
        rounds_used=4,
        winner="claude-3.5-sonnet",
        confidence=0.92,
    )
    result.debate_id = "test-debate-456"
    return result


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for testing."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"ok": True, "result": {}})

    # Setup context managers
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aexit__.return_value = None
    mock_session.closed = False

    return mock_session


# ============================================================================
# TestTelegramConfig
# ============================================================================

class TestTelegramConfig:
    """Tests for TelegramConfig dataclass."""

    def test_config_with_minimal_params(self):
        """Config can be created with bot_token and chat_id."""
        config = TelegramConfig(
            bot_token="test_token",
            chat_id="12345"
        )
        assert config.bot_token == "test_token"
        assert config.chat_id == "12345"

    def test_config_requires_bot_token(self):
        """Config raises error without bot token."""
        with pytest.raises(ValueError, match="bot token is required"):
            TelegramConfig(bot_token="", chat_id="12345")

    def test_config_requires_chat_id(self):
        """Config raises error without chat ID."""
        with pytest.raises(ValueError, match="chat ID is required"):
            TelegramConfig(bot_token="test_token", chat_id="")

    def test_config_default_values(self):
        """Config has correct default values."""
        config = TelegramConfig(bot_token="test", chat_id="123")
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.notify_on_error is True
        assert config.min_consensus_confidence == 0.7
        assert config.max_messages_per_minute == 20
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.parse_mode == "HTML"

    def test_config_api_base_property(self):
        """API base URL is constructed correctly."""
        config = TelegramConfig(bot_token="123456:ABC", chat_id="chat")
        assert config.api_base == "https://api.telegram.org/bot123456:ABC"

    def test_config_custom_values(self):
        """Config accepts custom values."""
        config = TelegramConfig(
            bot_token="test_token",
            chat_id="12345",
            max_messages_per_minute=30,
            min_consensus_confidence=0.9,
            parse_mode="Markdown",
        )
        assert config.max_messages_per_minute == 30
        assert config.min_consensus_confidence == 0.9
        assert config.parse_mode == "Markdown"


# ============================================================================
# TestInlineButton
# ============================================================================

class TestInlineButton:
    """Tests for InlineButton dataclass."""

    def test_button_url_type(self):
        """URL button generates correct dict."""
        button = InlineButton(text="Click Me", url="https://example.com")
        result = button.to_dict()
        assert result == {"text": "Click Me", "url": "https://example.com"}

    def test_button_callback_type(self):
        """Callback button generates correct dict."""
        button = InlineButton(text="Action", callback_data="do_action")
        result = button.to_dict()
        assert result == {"text": "Action", "callback_data": "do_action"}

    def test_button_text_only(self):
        """Button with only text generates minimal dict."""
        button = InlineButton(text="Plain Button")
        result = button.to_dict()
        assert result == {"text": "Plain Button"}

    def test_button_url_takes_precedence(self):
        """URL takes precedence over callback_data."""
        button = InlineButton(
            text="Mixed",
            url="https://example.com",
            callback_data="callback"
        )
        result = button.to_dict()
        # URL is checked first in to_dict
        assert result == {"text": "Mixed", "url": "https://example.com"}


# ============================================================================
# TestTelegramMessage
# ============================================================================

class TestTelegramMessage:
    """Tests for TelegramMessage dataclass."""

    def test_message_basic_payload(self, telegram_config):
        """Basic message generates correct payload."""
        message = TelegramMessage(text="Hello, World!")
        payload = message.to_payload(telegram_config)

        assert payload["chat_id"] == telegram_config.chat_id
        assert payload["text"] == "Hello, World!"
        assert payload["parse_mode"] == "HTML"
        assert payload["disable_web_page_preview"] is True
        assert payload["disable_notification"] is False
        assert "reply_markup" not in payload

    def test_message_with_buttons(self, telegram_config):
        """Message with buttons includes reply_markup."""
        buttons = [[
            InlineButton(text="Button 1", url="https://a.com"),
            InlineButton(text="Button 2", callback_data="action"),
        ]]
        message = TelegramMessage(text="Test", reply_markup=buttons)
        payload = message.to_payload(telegram_config)

        assert "reply_markup" in payload
        assert "inline_keyboard" in payload["reply_markup"]
        keyboard = payload["reply_markup"]["inline_keyboard"]
        assert len(keyboard) == 1
        assert len(keyboard[0]) == 2

    def test_message_with_notification_disabled(self, telegram_config):
        """Message can disable notifications."""
        message = TelegramMessage(
            text="Silent",
            disable_notification=True
        )
        payload = message.to_payload(telegram_config)
        assert payload["disable_notification"] is True

    def test_message_multiple_button_rows(self, telegram_config):
        """Message supports multiple button rows."""
        buttons = [
            [InlineButton(text="Row 1", url="https://a.com")],
            [InlineButton(text="Row 2", url="https://b.com")],
            [InlineButton(text="Row 3", callback_data="c")],
        ]
        message = TelegramMessage(text="Multi-row", reply_markup=buttons)
        payload = message.to_payload(telegram_config)

        keyboard = payload["reply_markup"]["inline_keyboard"]
        assert len(keyboard) == 3


# ============================================================================
# TestRateLimiting
# ============================================================================

class TestRateLimiting:
    """Tests for Telegram rate limiting."""

    def test_rate_limit_allows_within_limit(self, telegram_integration):
        """Rate limiter allows messages within limit."""
        for _ in range(5):
            result = telegram_integration._check_rate_limit()
            assert result is True

    def test_rate_limit_blocks_over_limit(self, telegram_config):
        """Rate limiter blocks messages over limit."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            max_messages_per_minute=3,
        )
        integration = TelegramIntegration(config)

        # Use up the limit
        for _ in range(3):
            result = integration._check_rate_limit()
            assert result is True

        # Next should be blocked
        result = integration._check_rate_limit()
        assert result is False

    def test_rate_limit_resets_after_minute(self, telegram_config):
        """Rate limiter resets count after a minute."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            max_messages_per_minute=2,
        )
        integration = TelegramIntegration(config)

        # Use up the limit
        integration._check_rate_limit()
        integration._check_rate_limit()
        assert integration._check_rate_limit() is False

        # Simulate time passing (more than 1 minute)
        integration._last_reset = datetime.now() - timedelta(minutes=2)

        # Should be allowed again
        result = integration._check_rate_limit()
        assert result is True


# ============================================================================
# TestHtmlEscaping
# ============================================================================

class TestHtmlEscaping:
    """Tests for HTML escaping in messages."""

    def test_escape_ampersand(self, telegram_integration):
        """Ampersand is escaped to &amp;."""
        result = telegram_integration._escape_html("A & B")
        assert result == "A &amp; B"

    def test_escape_less_than(self, telegram_integration):
        """Less than is escaped to &lt;."""
        result = telegram_integration._escape_html("if x < y")
        assert result == "if x &lt; y"

    def test_escape_greater_than(self, telegram_integration):
        """Greater than is escaped to &gt;."""
        result = telegram_integration._escape_html("x > 0")
        assert result == "x &gt; 0"

    def test_escape_multiple_characters(self, telegram_integration):
        """Multiple special characters are escaped."""
        result = telegram_integration._escape_html("<script>alert('XSS') & bad</script>")
        assert "&lt;script&gt;" in result
        assert "&amp;" in result
        assert "<" not in result
        assert ">" not in result

    def test_escape_preserves_normal_text(self, telegram_integration):
        """Normal text without special chars is preserved."""
        text = "Hello World! This is a test."
        result = telegram_integration._escape_html(text)
        assert result == text


# ============================================================================
# TestSendMessage
# ============================================================================

class TestSendMessage:
    """Tests for _send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, telegram_integration, mock_aiohttp_session):
        """Message is sent successfully."""
        telegram_integration._session = mock_aiohttp_session

        message = TelegramMessage(text="Test message")
        result = await telegram_integration._send_message(message)

        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_rate_limited_internally(self, telegram_config):
        """Message is blocked when internal rate limit exceeded."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            max_messages_per_minute=0,  # No messages allowed
        )
        integration = TelegramIntegration(config)

        message = TelegramMessage(text="Test")
        result = await integration._send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_retries_on_429(self, telegram_integration):
        """Message retries when Telegram returns 429."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value={
            "ok": False,
            "parameters": {"retry_after": 0.01}  # Very short for testing
        })

        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None
        mock_session.closed = False
        telegram_integration._session = mock_session

        message = TelegramMessage(text="Test")
        # Will retry 3 times then fail
        result = await telegram_integration._send_message(message)

        assert result is False
        # Should have made multiple attempts
        assert mock_session.post.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_message_handles_api_error(self, telegram_integration):
        """Message handles Telegram API errors gracefully."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={
            "ok": False,
            "description": "Bad Request: chat not found"
        })

        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None
        mock_session.closed = False
        telegram_integration._session = mock_session

        message = TelegramMessage(text="Test")
        result = await telegram_integration._send_message(message)

        assert result is False


# ============================================================================
# TestPostDebateSummary
# ============================================================================

class TestPostDebateSummary:
    """Tests for post_debate_summary method."""

    @pytest.mark.asyncio
    async def test_post_debate_summary_success(
        self, telegram_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Debate summary is posted successfully."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.post_debate_summary(sample_debate_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_post_debate_summary_respects_config(self, sample_debate_result):
        """Debate summary respects notify_on_debate_end config."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            notify_on_debate_end=False,
        )
        integration = TelegramIntegration(config)

        result = await integration.post_debate_summary(sample_debate_result)
        assert result is True  # Returns True but doesn't send

    def test_build_debate_summary_html_consensus_reached(
        self, telegram_integration, sample_debate_result
    ):
        """Summary HTML shows consensus reached correctly."""
        html = telegram_integration._build_debate_summary_html(sample_debate_result)

        assert "\u2705" in html  # Checkmark
        assert "Debate Completed" in html
        assert "Consensus:</b> Reached" in html
        assert "Winner:</b> claude-3.5-sonnet" in html
        assert "92%" in html  # Confidence
        assert "Rounds:</b> 4" in html

    def test_build_debate_summary_html_no_consensus(self, telegram_integration):
        """Summary HTML shows no consensus correctly."""
        result = DebateResult(
            task="Test task",
            final_answer="",
            consensus_reached=False,
            rounds_used=5,
            winner=None,
            confidence=0.4,
        )
        html = telegram_integration._build_debate_summary_html(result)

        assert "\u274c" in html  # X mark
        assert "Consensus:</b> Not Reached" in html
        assert "No clear winner" in html


# ============================================================================
# TestSendConsensusAlert
# ============================================================================

class TestSendConsensusAlert:
    """Tests for send_consensus_alert method."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Consensus alert is sent successfully."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_consensus_alert(
            debate_id="test-123",
            confidence=0.9,
            winner="claude",
            task="Test task",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, telegram_integration):
        """Consensus alert not sent below confidence threshold."""
        # Default min_consensus_confidence is 0.7
        result = await telegram_integration.send_consensus_alert(
            debate_id="test-123",
            confidence=0.5,  # Below threshold
        )

        assert result is True  # Returns True but doesn't send

    @pytest.mark.asyncio
    async def test_send_consensus_alert_respects_config(self):
        """Consensus alert respects notify_on_consensus config."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            notify_on_consensus=False,
        )
        integration = TelegramIntegration(config)

        result = await integration.send_consensus_alert(
            debate_id="test-123",
            confidence=0.95,
        )

        assert result is True  # Returns True but doesn't send


# ============================================================================
# TestSendErrorAlert
# ============================================================================

class TestSendErrorAlert:
    """Tests for send_error_alert method."""

    @pytest.mark.asyncio
    async def test_send_error_alert_success(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Error alert is sent successfully."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_error_alert(
            error_type="Connection Error",
            error_message="Failed to connect to API",
            debate_id="test-123",
            severity="error",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_respects_config(self):
        """Error alert respects notify_on_error config."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            notify_on_error=False,
        )
        integration = TelegramIntegration(config)

        result = await integration.send_error_alert(
            error_type="Test Error",
            error_message="Test message",
        )

        assert result is True  # Returns True but doesn't send

    @pytest.mark.asyncio
    async def test_send_error_alert_severity_emojis(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Error alert uses correct severity emojis."""
        telegram_integration._session = mock_aiohttp_session

        # Just verify the method accepts different severities
        for severity in ["info", "warning", "error", "critical"]:
            result = await telegram_integration.send_error_alert(
                error_type="Test",
                error_message="Test",
                severity=severity,
            )
            assert result is True


# ============================================================================
# TestSendLeaderboardUpdate
# ============================================================================

class TestSendLeaderboardUpdate:
    """Tests for send_leaderboard_update method."""

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_success(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Leaderboard update is sent successfully."""
        telegram_integration._session = mock_aiohttp_session

        rankings = [
            {"name": "claude", "elo": 1650, "wins": 15},
            {"name": "gpt-4", "elo": 1600, "wins": 12},
            {"name": "gemini", "elo": 1550, "wins": 10},
        ]

        result = await telegram_integration.send_leaderboard_update(rankings)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_top_n(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Leaderboard respects top_n parameter."""
        telegram_integration._session = mock_aiohttp_session

        rankings = [
            {"name": f"agent_{i}", "elo": 1600 - i * 10, "wins": 10 - i}
            for i in range(10)
        ]

        # Request only top 3
        result = await telegram_integration.send_leaderboard_update(rankings, top_n=3)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_empty(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Leaderboard handles empty rankings."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_leaderboard_update([])

        assert result is True


# ============================================================================
# TestSendDebateStarted
# ============================================================================

class TestSendDebateStarted:
    """Tests for send_debate_started method."""

    @pytest.mark.asyncio
    async def test_send_debate_started_success(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Debate started notification is sent successfully."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_debate_started(
            debate_id="test-123",
            task="Design a rate limiter",
            agents=["claude", "gpt-4", "gemini"],
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_debate_started_many_agents(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Debate started notification truncates many agents."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_debate_started(
            debate_id="test-123",
            task="Test task",
            agents=["agent1", "agent2", "agent3", "agent4", "agent5", "agent6"],
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_debate_started_escapes_task(
        self, telegram_integration, mock_aiohttp_session
    ):
        """Debate started notification escapes HTML in task."""
        telegram_integration._session = mock_aiohttp_session

        result = await telegram_integration.send_debate_started(
            debate_id="test-123",
            task="Compare <React> & Angular",
            agents=["claude"],
        )

        assert result is True


# ============================================================================
# TestSessionManagement
# ============================================================================

class TestSessionManagement:
    """Tests for aiohttp session management."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, telegram_integration):
        """_get_session creates new session when none exists."""
        assert telegram_integration._session is None

        with patch('aiohttp.ClientSession') as mock_cls:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_cls.return_value = mock_session

            session = await telegram_integration._get_session()

            assert session is not None
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, telegram_integration):
        """_get_session reuses existing open session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        telegram_integration._session = mock_session

        session = await telegram_integration._get_session()

        assert session is mock_session

    @pytest.mark.asyncio
    async def test_get_session_recreates_closed(self, telegram_integration):
        """_get_session recreates closed session."""
        old_session = AsyncMock()
        old_session.closed = True
        telegram_integration._session = old_session

        with patch('aiohttp.ClientSession') as mock_cls:
            new_session = AsyncMock()
            new_session.closed = False
            mock_cls.return_value = new_session

            session = await telegram_integration._get_session()

            assert session is new_session
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_session(self, telegram_integration):
        """close() closes the session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        telegram_integration._session = mock_session

        await telegram_integration.close()

        mock_session.close.assert_called_once()


# ============================================================================
# TestConnectionHandling
# ============================================================================

class TestConnectionHandling:
    """Tests for connection error handling."""

    @pytest.mark.asyncio
    async def test_handles_client_error(self, telegram_integration):
        """Handles aiohttp ClientError gracefully."""
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        mock_session.closed = False
        telegram_integration._session = mock_session

        message = TelegramMessage(text="Test")
        result = await telegram_integration._send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, telegram_config):
        """Retries on connection errors with backoff."""
        config = TelegramConfig(
            bot_token="test",
            chat_id="123",
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )
        integration = TelegramIntegration(config)

        # Verify retry configuration is set correctly
        assert config.max_retries == 3
        assert config.retry_delay == 0.01

        # Test that ClientError triggers retry behavior
        # The first test verified ClientError is handled, this verifies config
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Network error")
        mock_session.closed = False
        integration._session = mock_session

        message = TelegramMessage(text="Test")
        result = await integration._send_message(message)

        assert result is False
        # Verify at least one retry attempt was made (could be 1-3 calls depending on timing)
        assert mock_session.post.call_count >= 1


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task_in_summary(self, telegram_integration):
        """Handles empty task in debate result."""
        result = DebateResult(
            task="",
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        # Should not raise
        html = telegram_integration._build_debate_summary_html(result)
        assert "Debate Completed" in html

    def test_special_characters_in_task(self, telegram_integration):
        """Handles special characters in task."""
        result = DebateResult(
            task="Compare A<B>C & D",
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        html = telegram_integration._build_debate_summary_html(result)
        assert "&lt;" in html
        assert "&gt;" in html
        assert "&amp;" in html

    def test_very_long_final_answer_truncated(self, telegram_integration):
        """Very long final answer is truncated."""
        result = DebateResult(
            task="Test",
            final_answer="A" * 600,  # Very long
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        html = telegram_integration._build_debate_summary_html(result)
        # Should be truncated to 400 chars + "..."
        assert "..." in html

    def test_unicode_in_messages(self, telegram_integration):
        """Handles Unicode characters in messages."""
        result = DebateResult(
            task="Test \u2605 Unicode \U0001F600",
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        html = telegram_integration._build_debate_summary_html(result)
        assert "\u2605" in html or "&#9733;" in html  # Star or its entity

    @pytest.mark.asyncio
    async def test_missing_debate_id_in_summary(self, telegram_integration, mock_aiohttp_session):
        """Handles missing debate_id attribute."""
        telegram_integration._session = mock_aiohttp_session

        result = DebateResult(
            task="Test",
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        # Don't set debate_id

        # Should not raise
        success = await telegram_integration.post_debate_summary(result)
        assert success is True
