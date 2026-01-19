"""
Tests for Matrix/Element integration.

Covers:
- MatrixConfig: environment loading, defaults, URL normalization
- MatrixIntegration: send_message, rate limiting, session management
- HTML escaping for XSS prevention
- Notification methods: debate_summary, consensus_alert, error_alert, leaderboard
- Connection verification and room joining
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from aragora.core import DebateResult
from aragora.integrations.matrix import (
    MatrixConfig,
    MatrixIntegration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp.ClientSession for testing."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value='{"event_id": "$abc123"}')
    mock_response.json = AsyncMock(return_value={"event_id": "$abc123"})
    mock_session.put.return_value.__aenter__.return_value = mock_response
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.closed = False
    return mock_session


@pytest.fixture
def matrix_config():
    """Default Matrix configuration for testing."""
    return MatrixConfig(
        homeserver_url="https://matrix.example.org",
        access_token="syt_test_token_xxx",
        user_id="@aragora-bot:matrix.example.org",
        room_id="!abc123:matrix.example.org",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
        notify_on_leaderboard=True,
        enable_commands=True,
        min_consensus_confidence=0.7,
        max_messages_per_minute=10,
        use_html=True,
    )


@pytest.fixture
def matrix_integration(matrix_config, mock_aiohttp_session):
    """MatrixIntegration instance with mocked session."""
    integration = MatrixIntegration(matrix_config)
    integration._session = mock_aiohttp_session
    return integration


@pytest.fixture
def sample_debate_result():
    """Sample DebateResult for testing."""
    result = DebateResult(
        task="What is the best approach to machine learning?",
        final_answer="A combination of supervised and unsupervised methods works best.",
        consensus_reached=True,
        rounds_used=4,
        winner="claude",
        confidence=0.90,
    )
    result.debate_id = "test-debate-456"
    result.question = "What is the best approach to machine learning?"
    result.answer = "A combination of supervised and unsupervised methods works best."
    result.total_rounds = 4
    result.consensus_confidence = 0.90
    result.participating_agents = ["claude", "gpt-4", "gemini", "mistral"]
    return result


# =============================================================================
# MatrixConfig Tests
# =============================================================================


class TestMatrixConfig:
    """Tests for MatrixConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MatrixConfig()
        assert config.homeserver_url == ""
        assert config.access_token == ""
        assert config.user_id == ""
        assert config.room_id == ""
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.notify_on_error is True
        assert config.notify_on_leaderboard is False
        assert config.enable_commands is True
        assert config.min_consensus_confidence == 0.7
        assert config.max_messages_per_minute == 10
        assert config.use_html is True

    def test_custom_values(self, matrix_config):
        """Test custom configuration values."""
        assert matrix_config.homeserver_url == "https://matrix.example.org"
        assert matrix_config.access_token == "syt_test_token_xxx"
        assert matrix_config.user_id == "@aragora-bot:matrix.example.org"
        assert matrix_config.room_id == "!abc123:matrix.example.org"
        assert matrix_config.notify_on_leaderboard is True

    def test_env_variable_loading(self, monkeypatch):
        """Test loading credentials from environment variables."""
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://env.matrix.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env_token")
        monkeypatch.setenv("MATRIX_USER_ID", "@env-bot:matrix.org")
        monkeypatch.setenv("MATRIX_ROOM_ID", "!envroom:matrix.org")

        config = MatrixConfig()
        assert config.homeserver_url == "https://env.matrix.org"
        assert config.access_token == "env_token"
        assert config.user_id == "@env-bot:matrix.org"
        assert config.room_id == "!envroom:matrix.org"

    def test_explicit_overrides_env(self, monkeypatch):
        """Test that explicit values override environment variables."""
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://env.matrix.org")
        config = MatrixConfig(homeserver_url="https://explicit.matrix.org")
        assert config.homeserver_url == "https://explicit.matrix.org"

    def test_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from homeserver URL."""
        config = MatrixConfig(homeserver_url="https://matrix.org/")
        assert config.homeserver_url == "https://matrix.org"

    def test_url_multiple_trailing_slashes_stripped(self):
        """Test that multiple trailing slashes are stripped."""
        config = MatrixConfig(homeserver_url="https://matrix.org///")
        assert config.homeserver_url == "https://matrix.org"


# =============================================================================
# MatrixIntegration Basic Tests
# =============================================================================


class TestMatrixIntegrationBasic:
    """Tests for MatrixIntegration basic functionality."""

    def test_default_config(self):
        """Test integration with default config."""
        integration = MatrixIntegration()
        assert integration.config is not None
        assert integration._session is None
        assert integration._message_count == 0
        assert integration._sync_token is None

    def test_custom_config(self, matrix_config):
        """Test integration with custom config."""
        integration = MatrixIntegration(matrix_config)
        assert integration.config.homeserver_url == "https://matrix.example.org"
        assert integration.config.room_id == "!abc123:matrix.example.org"

    def test_is_configured_true(self, matrix_config):
        """Test is_configured returns True when properly configured."""
        integration = MatrixIntegration(matrix_config)
        assert integration.is_configured is True

    def test_is_configured_false_no_url(self):
        """Test is_configured returns False when homeserver URL is empty."""
        config = MatrixConfig(
            access_token="token",
            room_id="!room:matrix.org",
        )
        integration = MatrixIntegration(config)
        assert integration.is_configured is False

    def test_is_configured_false_no_token(self):
        """Test is_configured returns False when access token is empty."""
        config = MatrixConfig(
            homeserver_url="https://matrix.org",
            room_id="!room:matrix.org",
        )
        integration = MatrixIntegration(config)
        assert integration.is_configured is False

    def test_is_configured_false_no_room(self):
        """Test is_configured returns False when room ID is empty."""
        config = MatrixConfig(
            homeserver_url="https://matrix.org",
            access_token="token",
        )
        integration = MatrixIntegration(config)
        assert integration.is_configured is False

    def test_api_url(self, matrix_integration):
        """Test API URL building."""
        url = matrix_integration._api_url("/rooms/!room:test/send/m.room.message/txn123")
        assert url == "https://matrix.example.org/_matrix/client/v3/rooms/!room:test/send/m.room.message/txn123"

    def test_get_headers(self, matrix_integration):
        """Test request headers include Bearer token."""
        headers = matrix_integration._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {matrix_integration.config.access_token}"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, matrix_config):
        """Test _get_session creates new session when none exists."""
        integration = MatrixIntegration(matrix_config)
        assert integration._session is None

        session = await integration._get_session()
        assert session is not None
        assert integration._session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_close_session(self, matrix_config):
        """Test close() closes the session."""
        integration = MatrixIntegration(matrix_config)
        await integration._get_session()
        assert integration._session is not None

        await integration.close()


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestMatrixRateLimiting:
    """Tests for Matrix rate limiting functionality."""

    def test_initial_state(self, matrix_integration):
        """Test initial rate limit state."""
        assert matrix_integration._message_count == 0
        assert matrix_integration._last_reset is not None

    def test_increments_count(self, matrix_integration):
        """Test that _check_rate_limit increments message count."""
        result = matrix_integration._check_rate_limit()
        assert result is True
        assert matrix_integration._message_count == 1

    def test_allows_up_to_limit(self, matrix_integration):
        """Test that messages are allowed up to the limit."""
        for i in range(matrix_integration.config.max_messages_per_minute):
            result = matrix_integration._check_rate_limit()
            assert result is True

    def test_blocks_over_limit(self, matrix_integration):
        """Test that messages are blocked over the limit."""
        for _ in range(matrix_integration.config.max_messages_per_minute):
            matrix_integration._check_rate_limit()

        result = matrix_integration._check_rate_limit()
        assert result is False

    def test_resets_after_minute(self, matrix_integration):
        """Test that rate limit resets after one minute."""
        for _ in range(matrix_integration.config.max_messages_per_minute):
            matrix_integration._check_rate_limit()

        # Simulate time passing
        matrix_integration._last_reset = datetime.now() - timedelta(seconds=61)

        result = matrix_integration._check_rate_limit()
        assert result is True
        assert matrix_integration._message_count == 1


# =============================================================================
# HTML Escaping Tests
# =============================================================================


class TestMatrixHTMLEscaping:
    """Tests for HTML escaping to prevent XSS."""

    def test_escape_ampersand(self, matrix_integration):
        """Test ampersand is escaped."""
        result = matrix_integration._escape_html("A & B")
        assert result == "A &amp; B"

    def test_escape_less_than(self, matrix_integration):
        """Test less than is escaped."""
        result = matrix_integration._escape_html("A < B")
        assert result == "A &lt; B"

    def test_escape_greater_than(self, matrix_integration):
        """Test greater than is escaped."""
        result = matrix_integration._escape_html("A > B")
        assert result == "A &gt; B"

    def test_escape_quotes(self, matrix_integration):
        """Test quotes are escaped."""
        result = matrix_integration._escape_html('He said "hello"')
        assert result == 'He said &quot;hello&quot;'

    def test_escape_script_tag(self, matrix_integration):
        """Test script tags are properly escaped."""
        result = matrix_integration._escape_html('<script>alert("xss")</script>')
        assert result == "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;"

    def test_escape_multiple_special_chars(self, matrix_integration):
        """Test multiple special characters are escaped."""
        result = matrix_integration._escape_html('<a href="test">A & B</a>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result


# =============================================================================
# Send Message Tests
# =============================================================================


class TestMatrixSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, matrix_integration, mock_aiohttp_session):
        """Test successful message sending."""
        result = await matrix_integration.send_message("Test message")
        assert result is True
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self):
        """Test send_message returns False when not configured."""
        config = MatrixConfig()
        integration = MatrixIntegration(config)

        result = await integration.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_rate_limited(self, matrix_integration, mock_aiohttp_session):
        """Test send_message returns False when rate limited."""
        for _ in range(matrix_integration.config.max_messages_per_minute):
            matrix_integration._check_rate_limit()

        result = await matrix_integration.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_uses_put(self, matrix_integration, mock_aiohttp_session):
        """Test that send_message uses PUT method (Matrix standard)."""
        await matrix_integration.send_message("Test")
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_url_format(self, matrix_integration, mock_aiohttp_session):
        """Test message URL format includes room and transaction ID."""
        await matrix_integration.send_message("Test")

        call_args = mock_aiohttp_session.put.call_args
        url = call_args.args[0]
        assert "/_matrix/client/v3/rooms/" in url
        assert matrix_integration.config.room_id in url
        assert "/send/m.room.message/" in url

    @pytest.mark.asyncio
    async def test_send_message_text_only(self, matrix_integration, mock_aiohttp_session):
        """Test sending plain text message."""
        await matrix_integration.send_message("Plain text")

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert payload["msgtype"] == "m.text"
        assert payload["body"] == "Plain text"
        assert "format" not in payload

    @pytest.mark.asyncio
    async def test_send_message_with_html(self, matrix_integration, mock_aiohttp_session):
        """Test sending message with HTML formatting."""
        await matrix_integration.send_message("Plain", html="<b>Bold</b>")

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert payload["body"] == "Plain"
        assert payload["format"] == "org.matrix.custom.html"
        assert payload["formatted_body"] == "<b>Bold</b>"

    @pytest.mark.asyncio
    async def test_send_message_html_disabled(self, matrix_config, mock_aiohttp_session):
        """Test that HTML is not included when use_html is False."""
        matrix_config.use_html = False
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        await integration.send_message("Plain", html="<b>Bold</b>")

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "format" not in payload
        assert "formatted_body" not in payload

    @pytest.mark.asyncio
    async def test_send_message_custom_room(self, matrix_integration, mock_aiohttp_session):
        """Test sending to a custom room."""
        await matrix_integration.send_message("Test", room_id="!customroom:matrix.org")

        call_args = mock_aiohttp_session.put.call_args
        url = call_args.args[0]
        assert "!customroom:matrix.org" in url

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, matrix_integration, mock_aiohttp_session):
        """Test send_message handles API errors."""
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value='{"errcode": "M_FORBIDDEN"}')
        mock_aiohttp_session.put.return_value.__aenter__.return_value = mock_response

        result = await matrix_integration.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_connection_error(self, matrix_integration, mock_aiohttp_session):
        """Test send_message handles connection errors."""
        mock_aiohttp_session.put.side_effect = aiohttp.ClientError("Connection failed")

        result = await matrix_integration.send_message("Test")
        assert result is False


# =============================================================================
# Debate Summary Tests
# =============================================================================


class TestMatrixDebateSummary:
    """Tests for post_debate_summary method."""

    @pytest.mark.asyncio
    async def test_post_debate_summary_success(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test successful debate summary posting."""
        result = await matrix_integration.post_debate_summary(sample_debate_result)
        assert result is True
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(self, matrix_config, sample_debate_result, mock_aiohttp_session):
        """Test debate summary not sent when disabled."""
        matrix_config.notify_on_debate_end = False
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        result = await integration.post_debate_summary(sample_debate_result)
        assert result is False
        mock_aiohttp_session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_question(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test debate summary includes the question."""
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "machine learning" in payload["body"]
        assert "machine learning" in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_answer(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test debate summary includes the answer."""
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "supervised and unsupervised" in payload["body"]

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_stats(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test debate summary includes statistics."""
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "Rounds: 4" in payload["body"]
        assert "90%" in payload["body"]
        assert "Agents: 4" in payload["body"]

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_participants(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test debate summary includes participating agents."""
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "claude" in payload["body"]
        assert "gpt-4" in payload["body"]

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_link(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test debate summary includes view link."""
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert f"https://aragora.ai/debate/{sample_debate_result.debate_id}" in payload["body"]
        assert sample_debate_result.debate_id in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_post_debate_summary_truncates_many_agents(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test that more than 5 agents shows '+X more'."""
        sample_debate_result.participating_agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        await matrix_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "+2 more" in payload["formatted_body"]


# =============================================================================
# Consensus Alert Tests
# =============================================================================


class TestMatrixConsensusAlert:
    """Tests for send_consensus_alert method."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(self, matrix_integration, mock_aiohttp_session):
        """Test successful consensus alert."""
        result = await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="The answer is clear",
            confidence=0.85,
            agents=["claude", "gpt-4"],
        )
        assert result is True
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, matrix_config, mock_aiohttp_session):
        """Test consensus alert not sent when disabled."""
        matrix_config.notify_on_consensus = False
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
        )
        assert result is False
        mock_aiohttp_session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, matrix_integration, mock_aiohttp_session):
        """Test consensus alert not sent when confidence below threshold."""
        result = await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.5,
        )
        assert result is False
        mock_aiohttp_session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_at_threshold(self, matrix_integration, mock_aiohttp_session):
        """Test consensus alert sent at exact threshold."""
        result = await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.7,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_high_confidence_green(self, matrix_integration, mock_aiohttp_session):
        """Test high confidence uses green color in HTML."""
        await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert 'color="green"' in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_consensus_alert_medium_confidence_orange(self, matrix_integration, mock_aiohttp_session):
        """Test medium confidence uses orange color in HTML."""
        await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.75,
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert 'color="orange"' in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_consensus_alert_includes_agents(self, matrix_integration, mock_aiohttp_session):
        """Test consensus alert includes agent names."""
        await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
            agents=["claude", "gpt-4", "gemini"],
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "claude" in payload["body"]
        assert "gpt-4" in payload["body"]

    @pytest.mark.asyncio
    async def test_send_consensus_alert_truncates_long_answer(self, matrix_integration, mock_aiohttp_session):
        """Test that long answers are truncated."""
        long_answer = "A" * 600
        await matrix_integration.send_consensus_alert(
            debate_id="test-123",
            answer=long_answer,
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "..." in payload["body"]


# =============================================================================
# Error Alert Tests
# =============================================================================


class TestMatrixErrorAlert:
    """Tests for send_error_alert method."""

    @pytest.mark.asyncio
    async def test_send_error_alert_success(self, matrix_integration, mock_aiohttp_session):
        """Test successful error alert."""
        result = await matrix_integration.send_error_alert(
            debate_id="test-123",
            error="Something went wrong",
        )
        assert result is True
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, matrix_config, mock_aiohttp_session):
        """Test error alert not sent when disabled."""
        matrix_config.notify_on_error = False
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
        )
        assert result is False
        mock_aiohttp_session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_header(self, matrix_integration, mock_aiohttp_session):
        """Test error alert includes ARAGORA ERROR header."""
        await matrix_integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "ARAGORA ERROR" in payload["body"]
        assert "ARAGORA ERROR" in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_error_alert_uses_red_color(self, matrix_integration, mock_aiohttp_session):
        """Test error alert uses red color in HTML."""
        await matrix_integration.send_error_alert(
            debate_id="test-123",
            error="Error",
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert 'color="red"' in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_error_alert_with_phase(self, matrix_integration, mock_aiohttp_session):
        """Test error alert includes phase information."""
        await matrix_integration.send_error_alert(
            debate_id="test-123",
            error="Error",
            phase="consensus",
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "consensus" in payload["body"]
        assert "consensus" in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_error_alert_escapes_html_in_error(self, matrix_integration, mock_aiohttp_session):
        """Test error message is HTML escaped."""
        await matrix_integration.send_error_alert(
            debate_id="test-123",
            error='<script>alert("xss")</script>',
        )

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        # Should be escaped in HTML
        assert "<script>" not in payload["formatted_body"]
        assert "&lt;script&gt;" in payload["formatted_body"]


# =============================================================================
# Leaderboard Update Tests
# =============================================================================


class TestMatrixLeaderboardUpdate:
    """Tests for send_leaderboard_update method."""

    @pytest.fixture
    def sample_rankings(self):
        """Sample leaderboard rankings."""
        return [
            {"name": "claude", "elo": 1650, "wins": 15, "losses": 5},
            {"name": "gpt-4", "elo": 1600, "wins": 12, "losses": 8},
            {"name": "gemini", "elo": 1550, "wins": 10, "losses": 10},
        ]

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_success(self, matrix_integration, sample_rankings, mock_aiohttp_session):
        """Test successful leaderboard update."""
        result = await matrix_integration.send_leaderboard_update(sample_rankings)
        assert result is True
        mock_aiohttp_session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_disabled(self, matrix_config, sample_rankings, mock_aiohttp_session):
        """Test leaderboard update not sent when disabled."""
        matrix_config.notify_on_leaderboard = False
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_leaderboard_update(sample_rankings)
        assert result is False
        mock_aiohttp_session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_includes_rankings(self, matrix_integration, sample_rankings, mock_aiohttp_session):
        """Test leaderboard includes ranking information."""
        await matrix_integration.send_leaderboard_update(sample_rankings)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "claude" in payload["body"]
        assert "1650" in payload["body"]
        assert "15W/5L" in payload["body"]

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_with_domain(self, matrix_integration, sample_rankings, mock_aiohttp_session):
        """Test leaderboard update with domain filter."""
        await matrix_integration.send_leaderboard_update(sample_rankings, domain="math")

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "math" in payload["body"]
        assert "math" in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_uses_ordered_list(self, matrix_integration, sample_rankings, mock_aiohttp_session):
        """Test leaderboard uses ordered list in HTML."""
        await matrix_integration.send_leaderboard_update(sample_rankings)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "<ol>" in payload["formatted_body"]
        assert "<li>" in payload["formatted_body"]
        assert "</ol>" in payload["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_limits_to_10(self, matrix_integration, mock_aiohttp_session):
        """Test leaderboard is limited to top 10."""
        many_rankings = [{"name": f"agent{i}", "elo": 1500-i, "wins": 0, "losses": 0} for i in range(15)]
        await matrix_integration.send_leaderboard_update(many_rankings)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        # Count list items in HTML
        assert payload["formatted_body"].count("<li>") == 10


# =============================================================================
# Connection Verification Tests
# =============================================================================


class TestMatrixVerifyConnection:
    """Tests for verify_connection method."""

    @pytest.mark.asyncio
    async def test_verify_connection_success(self, matrix_integration, mock_aiohttp_session):
        """Test successful connection verification."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"user_id": "@bot:matrix.org"})
        mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

        result = await matrix_integration.verify_connection()
        assert result is True
        mock_aiohttp_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_connection_not_configured(self):
        """Test verify_connection returns False when not configured."""
        integration = MatrixIntegration(MatrixConfig())
        result = await integration.verify_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_connection_auth_failed(self, matrix_integration, mock_aiohttp_session):
        """Test verify_connection handles auth failure."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

        result = await matrix_integration.verify_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_connection_uses_whoami(self, matrix_integration, mock_aiohttp_session):
        """Test verify_connection calls /account/whoami endpoint."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"user_id": "@bot:matrix.org"})
        mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

        await matrix_integration.verify_connection()

        call_args = mock_aiohttp_session.get.call_args
        url = call_args.args[0]
        assert "/account/whoami" in url


# =============================================================================
# Join Room Tests
# =============================================================================


class TestMatrixJoinRoom:
    """Tests for join_room method."""

    @pytest.mark.asyncio
    async def test_join_room_success(self, matrix_integration, mock_aiohttp_session):
        """Test successful room join."""
        result = await matrix_integration.join_room("!newroom:matrix.org")
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_join_room_not_configured(self):
        """Test join_room returns False when not configured."""
        integration = MatrixIntegration(MatrixConfig())
        result = await integration.join_room("!room:matrix.org")
        assert result is False

    @pytest.mark.asyncio
    async def test_join_room_url_format(self, matrix_integration, mock_aiohttp_session):
        """Test join room URL format."""
        await matrix_integration.join_room("!testroom:matrix.org")

        call_args = mock_aiohttp_session.post.call_args
        url = call_args.args[0]
        assert "/rooms/!testroom:matrix.org/join" in url

    @pytest.mark.asyncio
    async def test_join_room_failure(self, matrix_integration, mock_aiohttp_session):
        """Test join_room handles failure."""
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value='{"errcode": "M_FORBIDDEN"}')
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        result = await matrix_integration.join_room("!room:matrix.org")
        assert result is False


# =============================================================================
# Session Management Tests
# =============================================================================


class TestMatrixSessionManagement:
    """Tests for session management functionality."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self, matrix_config):
        """Test that close() properly closes the session."""
        integration = MatrixIntegration(matrix_config)
        await integration._get_session()
        assert integration._session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, matrix_config):
        """Test that close() handles case when no session exists."""
        integration = MatrixIntegration(matrix_config)
        assert integration._session is None

        await integration.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestMatrixContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, matrix_config):
        """Test __aenter__ returns self."""
        integration = MatrixIntegration(matrix_config)
        async with integration as ctx:
            assert ctx is integration

    @pytest.mark.asyncio
    async def test_context_manager_exit_closes_session(self, matrix_config):
        """Test __aexit__ closes the session."""
        integration = MatrixIntegration(matrix_config)

        async with integration:
            await integration._get_session()
            assert integration._session is not None

    @pytest.mark.asyncio
    async def test_context_manager_handles_exception(self, matrix_config, mock_aiohttp_session):
        """Test context manager properly handles exceptions."""
        integration = MatrixIntegration(matrix_config)
        integration._session = mock_aiohttp_session

        with pytest.raises(ValueError):
            async with integration:
                raise ValueError("Test error")

        mock_aiohttp_session.close.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestMatrixIntegrationE2E:
    """End-to-end style tests for Matrix integration."""

    @pytest.mark.asyncio
    async def test_full_debate_workflow(self, matrix_integration, sample_debate_result, mock_aiohttp_session):
        """Test a full debate notification workflow."""
        # 1. Post debate summary
        result = await matrix_integration.post_debate_summary(sample_debate_result)
        assert result is True

        # 2. Send consensus alert
        result = await matrix_integration.send_consensus_alert(
            debate_id=sample_debate_result.debate_id,
            answer=sample_debate_result.answer,
            confidence=0.90,
            agents=sample_debate_result.participating_agents,
        )
        assert result is True

        # Verify both calls were made
        assert mock_aiohttp_session.put.call_count == 2

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, matrix_integration, mock_aiohttp_session):
        """Test error notification workflow."""
        result = await matrix_integration.send_error_alert(
            debate_id="test-123",
            error="Agent timeout",
            phase="round_2",
        )
        assert result is True

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        assert "ARAGORA ERROR" in payload["body"]
        assert "Agent timeout" in payload["body"]
        assert "round_2" in payload["body"]

    @pytest.mark.asyncio
    async def test_multiple_notifications_respect_rate_limit(self, matrix_integration, mock_aiohttp_session):
        """Test that multiple notifications respect rate limiting."""
        matrix_integration.config.max_messages_per_minute = 3

        results = []
        for i in range(5):
            result = await matrix_integration.send_message(f"Message {i}")
            results.append(result)

        # First 3 should succeed, last 2 should be rate limited
        assert results[:3] == [True, True, True]
        assert results[3:] == [False, False]

        # Only 3 API calls should have been made
        assert mock_aiohttp_session.put.call_count == 3

    @pytest.mark.asyncio
    async def test_xss_prevention_in_all_methods(self, matrix_integration, mock_aiohttp_session):
        """Test that XSS is prevented in all notification methods."""
        xss_payload = '<script>alert("xss")</script>'

        # Test in debate summary
        result = DebateResult(
            task=xss_payload,
            final_answer=xss_payload,
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.5,
        )
        result.debate_id = "test"
        result.question = xss_payload
        result.answer = xss_payload
        result.total_rounds = 1
        result.consensus_confidence = 0.5
        result.participating_agents = []

        await matrix_integration.post_debate_summary(result)

        call_args = mock_aiohttp_session.put.call_args
        payload = call_args.kwargs["json"]

        # HTML should be escaped
        assert "<script>" not in payload["formatted_body"]
        assert "&lt;script&gt;" in payload["formatted_body"]
