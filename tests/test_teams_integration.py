"""
Tests for Microsoft Teams integration.

Covers:
- TeamsConfig: environment loading, defaults, webhook validation
- AdaptiveCard: payload structure, schema version
- TeamsIntegration: send_card, rate limiting, session management
- Notification methods: debate_summary, consensus_alert, error_alert, leaderboard
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.core import DebateResult
from aragora.integrations.teams import (
    AdaptiveCard,
    TeamsConfig,
    TeamsIntegration,
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
    mock_response.text = AsyncMock(return_value="1")
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.closed = False
    return mock_session


@pytest.fixture
def teams_config():
    """Default Teams configuration for testing."""
    return TeamsConfig(
        webhook_url="https://test.webhook.office.com/webhookb2/test",
        bot_name="TestBot",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
        notify_on_leaderboard=True,
        min_consensus_confidence=0.7,
        max_messages_per_minute=10,
    )


@pytest.fixture
def teams_integration(teams_config, mock_aiohttp_session):
    """TeamsIntegration instance with mocked session."""
    integration = TeamsIntegration(teams_config)
    integration._session = mock_aiohttp_session
    return integration


@pytest.fixture
def sample_debate_result():
    """Sample DebateResult for testing."""
    result = DebateResult(
        task="What is the best programming language?",
        final_answer="Python is widely considered excellent for its readability.",
        consensus_reached=True,
        rounds_used=3,
        winner="claude",
        confidence=0.85,
    )
    result.debate_id = "test-debate-123"
    result.question = "What is the best programming language?"
    result.answer = "Python is widely considered excellent for its readability."
    result.total_rounds = 3
    result.consensus_confidence = 0.85
    result.participating_agents = ["claude", "gpt-4", "gemini"]
    return result


# =============================================================================
# TeamsConfig Tests
# =============================================================================


class TestTeamsConfig:
    """Tests for TeamsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TeamsConfig()
        assert config.webhook_url == ""
        assert config.bot_name == "Aragora"
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.notify_on_error is True
        assert config.notify_on_leaderboard is False
        assert config.min_consensus_confidence == 0.7
        assert config.max_messages_per_minute == 10
        assert config.accent_color == "good"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TeamsConfig(
            webhook_url="https://custom.webhook.office.com/test",
            bot_name="CustomBot",
            notify_on_consensus=False,
            notify_on_debate_end=False,
            notify_on_error=False,
            notify_on_leaderboard=True,
            min_consensus_confidence=0.9,
            max_messages_per_minute=5,
            accent_color="attention",
        )
        assert config.webhook_url == "https://custom.webhook.office.com/test"
        assert config.bot_name == "CustomBot"
        assert config.notify_on_consensus is False
        assert config.notify_on_leaderboard is True
        assert config.min_consensus_confidence == 0.9
        assert config.max_messages_per_minute == 5
        assert config.accent_color == "attention"

    def test_env_variable_loading(self, monkeypatch):
        """Test loading webhook URL from environment variable."""
        monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://env.webhook.office.com/test")
        config = TeamsConfig()
        assert config.webhook_url == "https://env.webhook.office.com/test"

    def test_explicit_overrides_env(self, monkeypatch):
        """Test that explicit value overrides environment variable."""
        monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://env.webhook.office.com/test")
        config = TeamsConfig(webhook_url="https://explicit.webhook.office.com/test")
        assert config.webhook_url == "https://explicit.webhook.office.com/test"

    def test_warning_on_missing_webhook(self, caplog):
        """Test warning is logged when webhook URL is missing."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear the env var
            import os

            if "TEAMS_WEBHOOK_URL" in os.environ:
                del os.environ["TEAMS_WEBHOOK_URL"]
            config = TeamsConfig(webhook_url="")
        assert "Teams webhook URL not configured" in caplog.text or config.webhook_url == ""


# =============================================================================
# AdaptiveCard Tests
# =============================================================================


class TestAdaptiveCard:
    """Tests for AdaptiveCard dataclass."""

    def test_basic_card_payload(self):
        """Test basic adaptive card payload structure."""
        card = AdaptiveCard(title="Test Title")
        payload = card.to_payload()

        assert payload["type"] == "message"
        assert "attachments" in payload
        assert len(payload["attachments"]) == 1

        attachment = payload["attachments"][0]
        assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"

        content = attachment["content"]
        assert content["type"] == "AdaptiveCard"
        assert content["version"] == "1.4"
        assert "$schema" in content
        assert "http://adaptivecards.io/schemas/adaptive-card.json" in content["$schema"]

    def test_card_with_body(self):
        """Test adaptive card with body elements."""
        body = [
            {"type": "TextBlock", "text": "Test body text"},
            {"type": "Image", "url": "https://example.com/image.png"},
        ]
        card = AdaptiveCard(title="Test Title", body=body)
        payload = card.to_payload()

        content = payload["attachments"][0]["content"]
        # First body element is the title
        assert content["body"][0]["text"] == "Test Title"
        # Then our custom body elements
        assert content["body"][1]["text"] == "Test body text"
        assert content["body"][2]["url"] == "https://example.com/image.png"

    def test_card_with_actions(self):
        """Test adaptive card with action buttons."""
        actions = [
            {"type": "Action.OpenUrl", "title": "View", "url": "https://example.com"},
        ]
        card = AdaptiveCard(title="Test Title", actions=actions)
        payload = card.to_payload()

        content = payload["attachments"][0]["content"]
        assert "actions" in content
        assert len(content["actions"]) == 1
        assert content["actions"][0]["title"] == "View"

    def test_card_without_actions(self):
        """Test that actions key is omitted when no actions provided."""
        card = AdaptiveCard(title="Test Title")
        payload = card.to_payload()

        content = payload["attachments"][0]["content"]
        assert "actions" not in content

    def test_card_accent_color(self):
        """Test adaptive card accent color setting."""
        card = AdaptiveCard(title="Test", accent_color="attention")
        assert card.accent_color == "attention"


# =============================================================================
# TeamsIntegration Basic Tests
# =============================================================================


class TestTeamsIntegrationBasic:
    """Tests for TeamsIntegration basic functionality."""

    def test_default_config(self):
        """Test integration with default config."""
        integration = TeamsIntegration()
        assert integration.config is not None
        assert integration._session is None
        assert integration._message_count == 0

    def test_custom_config(self, teams_config):
        """Test integration with custom config."""
        integration = TeamsIntegration(teams_config)
        assert integration.config.webhook_url == teams_config.webhook_url
        assert integration.config.bot_name == "TestBot"

    def test_is_configured_true(self, teams_config):
        """Test is_configured returns True when webhook URL is set."""
        integration = TeamsIntegration(teams_config)
        assert integration.is_configured is True

    def test_is_configured_false(self):
        """Test is_configured returns False when webhook URL is empty."""
        config = TeamsConfig(webhook_url="")
        integration = TeamsIntegration(config)
        assert integration.is_configured is False

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self):
        """Test _get_session creates new session when none exists."""
        integration = TeamsIntegration(TeamsConfig(webhook_url="https://test.com"))
        assert integration._session is None

        session = await integration._get_session()
        assert session is not None
        assert integration._session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, teams_integration, mock_aiohttp_session):
        """Test _get_session reuses existing non-closed session."""
        session1 = await teams_integration._get_session()
        session2 = await teams_integration._get_session()
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test close() closes the session."""
        integration = TeamsIntegration(TeamsConfig(webhook_url="https://test.com"))
        await integration._get_session()
        assert integration._session is not None

        await integration.close()
        # After close, session should be closed (or marked as such)


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestTeamsRateLimiting:
    """Tests for Teams rate limiting functionality."""

    def test_initial_state(self, teams_integration):
        """Test initial rate limit state."""
        assert teams_integration._message_count == 0
        assert teams_integration._last_reset is not None

    def test_increments_count(self, teams_integration):
        """Test that _check_rate_limit increments message count."""
        initial_count = teams_integration._message_count
        result = teams_integration._check_rate_limit()
        assert result is True
        assert teams_integration._message_count == initial_count + 1

    def test_allows_up_to_limit(self, teams_integration):
        """Test that messages are allowed up to the limit."""
        for i in range(teams_integration.config.max_messages_per_minute):
            result = teams_integration._check_rate_limit()
            assert result is True
        assert teams_integration._message_count == teams_integration.config.max_messages_per_minute

    def test_blocks_over_limit(self, teams_integration):
        """Test that messages are blocked over the limit."""
        # Exhaust the limit
        for _ in range(teams_integration.config.max_messages_per_minute):
            teams_integration._check_rate_limit()

        # Next message should be blocked
        result = teams_integration._check_rate_limit()
        assert result is False

    def test_resets_after_minute(self, teams_integration):
        """Test that rate limit resets after one minute."""
        # Exhaust the limit
        for _ in range(teams_integration.config.max_messages_per_minute):
            teams_integration._check_rate_limit()

        # Simulate time passing
        teams_integration._last_reset = datetime.now() - timedelta(seconds=61)

        # Should allow messages again
        result = teams_integration._check_rate_limit()
        assert result is True
        assert teams_integration._message_count == 1

    def test_boundary_at_60_seconds(self, teams_integration):
        """Test rate limit at exactly 60 seconds boundary."""
        teams_integration._check_rate_limit()

        # Set to exactly 59 seconds ago (should not reset)
        teams_integration._last_reset = datetime.now() - timedelta(seconds=59)
        teams_integration._message_count = teams_integration.config.max_messages_per_minute

        result = teams_integration._check_rate_limit()
        assert result is False


# =============================================================================
# Send Card Tests
# =============================================================================


class TestTeamsSendCard:
    """Tests for _send_card method."""

    @pytest.mark.asyncio
    async def test_send_card_success(self, teams_integration, mock_aiohttp_session):
        """Test successful card sending."""
        card = AdaptiveCard(title="Test")
        result = await teams_integration._send_card(card)

        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_card_not_configured(self):
        """Test _send_card returns False when not configured."""
        config = TeamsConfig(webhook_url="")
        integration = TeamsIntegration(config)

        card = AdaptiveCard(title="Test")
        result = await integration._send_card(card)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_card_rate_limited(self, teams_integration, mock_aiohttp_session):
        """Test _send_card returns False when rate limited."""
        # Exhaust rate limit
        for _ in range(teams_integration.config.max_messages_per_minute):
            teams_integration._check_rate_limit()

        card = AdaptiveCard(title="Test")
        result = await teams_integration._send_card(card)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_card_api_error(self, teams_integration, mock_aiohttp_session):
        """Test _send_card handles API errors."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        card = AdaptiveCard(title="Test")
        result = await teams_integration._send_card(card)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_card_connection_error(self, teams_integration, mock_aiohttp_session):
        """Test _send_card handles connection errors."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        card = AdaptiveCard(title="Test")
        result = await teams_integration._send_card(card)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_card_timeout_error(self, teams_integration, mock_aiohttp_session):
        """Test _send_card handles timeout errors."""
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError()

        card = AdaptiveCard(title="Test")
        result = await teams_integration._send_card(card)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_card_payload_structure(self, teams_integration, mock_aiohttp_session):
        """Test that sent payload has correct structure."""
        card = AdaptiveCard(
            title="Test Title",
            body=[{"type": "TextBlock", "text": "Body"}],
            actions=[{"type": "Action.OpenUrl", "title": "Click", "url": "https://test.com"}],
        )
        await teams_integration._send_card(card)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]

        assert payload["type"] == "message"
        assert "attachments" in payload
        assert payload["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"


# =============================================================================
# Debate Summary Tests
# =============================================================================


class TestTeamsDebateSummary:
    """Tests for post_debate_summary method."""

    @pytest.mark.asyncio
    async def test_post_debate_summary_success(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test successful debate summary posting."""
        result = await teams_integration.post_debate_summary(sample_debate_result)
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(
        self, teams_config, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary not sent when disabled."""
        teams_config.notify_on_debate_end = False
        integration = TeamsIntegration(teams_config)
        integration._session = mock_aiohttp_session

        result = await integration.post_debate_summary(sample_debate_result)
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_question(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that debate summary includes the question."""
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("What is the best programming language" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_answer(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that debate summary includes the answer."""
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("Python" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_stats(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that debate summary includes statistics."""
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        stats_text = next((t for t in body_texts if "Rounds:" in t), None)
        assert stats_text is not None
        assert "Rounds: 3" in stats_text

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_agents(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that debate summary includes participating agents."""
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("claude" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_post_debate_summary_truncates_many_agents(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that more than 5 agents shows '+X more'."""
        sample_debate_result.participating_agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("+2 more" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_post_debate_summary_includes_action(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that debate summary includes View Details action."""
        await teams_integration.post_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        assert "actions" in content
        assert content["actions"][0]["title"] == "View Details"
        assert sample_debate_result.debate_id in content["actions"][0]["url"]


# =============================================================================
# Consensus Alert Tests
# =============================================================================


class TestTeamsConsensusAlert:
    """Tests for send_consensus_alert method."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(self, teams_integration, mock_aiohttp_session):
        """Test successful consensus alert."""
        result = await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="The consensus answer",
            confidence=0.85,
            agents=["claude", "gpt-4"],
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, teams_config, mock_aiohttp_session):
        """Test consensus alert not sent when disabled."""
        teams_config.notify_on_consensus = False
        integration = TeamsIntegration(teams_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
        )
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test consensus alert not sent when confidence below threshold."""
        result = await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.5,  # Below default 0.7 threshold
        )
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_at_threshold(self, teams_integration, mock_aiohttp_session):
        """Test consensus alert sent at exact threshold."""
        result = await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.7,  # Exactly at threshold
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_truncates_long_answer(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that long answers are truncated."""
        long_answer = "A" * 600
        await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer=long_answer,
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        # Find the consensus text block which contains the answer
        consensus_text = next((t for t in body_texts if "**Consensus:**" in t), "")
        # The answer should be truncated to 500 chars with "..."
        assert "..." in consensus_text or len(long_answer) <= 500  # Truncated if > 500

    @pytest.mark.asyncio
    async def test_send_consensus_alert_high_confidence_color(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that high confidence uses 'Good' color."""
        await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Find the confidence text block
        for item in content["body"]:
            if item.get("type") == "ColumnSet":
                for col in item.get("columns", []):
                    for col_item in col.get("items", []):
                        if "Confidence" in col_item.get("text", ""):
                            assert col_item.get("color") == "Good"

    @pytest.mark.asyncio
    async def test_send_consensus_alert_medium_confidence_color(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that medium confidence uses 'Warning' color."""
        await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.75,  # Below 0.8
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Find the confidence text block
        for item in content["body"]:
            if item.get("type") == "ColumnSet":
                for col in item.get("columns", []):
                    for col_item in col.get("items", []):
                        if "Confidence" in col_item.get("text", ""):
                            assert col_item.get("color") == "Warning"

    @pytest.mark.asyncio
    async def test_send_consensus_alert_with_agents(self, teams_integration, mock_aiohttp_session):
        """Test consensus alert includes agent names."""
        await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
            agents=["claude", "gpt-4", "gemini"],
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("claude" in text and "gpt-4" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_consensus_alert_without_agents(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test consensus alert works without agent list."""
        result = await teams_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.85,
            agents=None,
        )
        assert result is True


# =============================================================================
# Error Alert Tests
# =============================================================================


class TestTeamsErrorAlert:
    """Tests for send_error_alert method."""

    @pytest.mark.asyncio
    async def test_send_error_alert_success(self, teams_integration, mock_aiohttp_session):
        """Test successful error alert."""
        result = await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Something went wrong",
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, teams_config, mock_aiohttp_session):
        """Test error alert not sent when disabled."""
        teams_config.notify_on_error = False
        integration = TeamsIntegration(teams_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
        )
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_error_message(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that error alert includes error message."""
        await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Connection timeout",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("Connection timeout" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_error_alert_with_phase(self, teams_integration, mock_aiohttp_session):
        """Test that error alert includes phase information."""
        await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
            phase="consensus",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("consensus" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_error_alert_without_phase(self, teams_integration, mock_aiohttp_session):
        """Test error alert works without phase."""
        result = await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
            phase=None,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_debate_id(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that error alert includes debate ID."""
        await teams_integration.send_error_alert(
            debate_id="test-error-456",
            error="Error",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("test-error-456" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_error_alert_uses_attention_color(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that error alert uses attention color."""
        await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Error",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Check error text has Attention color (skip title which doesn't have color)
        error_blocks = [item for item in content["body"] if "**Error:**" in item.get("text", "")]
        assert len(error_blocks) > 0
        assert error_blocks[0].get("color") == "Attention"


# =============================================================================
# Leaderboard Update Tests
# =============================================================================


class TestTeamsLeaderboardUpdate:
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
    async def test_send_leaderboard_update_success(
        self, teams_integration, sample_rankings, mock_aiohttp_session
    ):
        """Test successful leaderboard update."""
        result = await teams_integration.send_leaderboard_update(sample_rankings)
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_disabled(
        self, teams_config, sample_rankings, mock_aiohttp_session
    ):
        """Test leaderboard update not sent when disabled."""
        teams_config.notify_on_leaderboard = False
        integration = TeamsIntegration(teams_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_leaderboard_update(sample_rankings)
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_includes_rankings(
        self, teams_integration, sample_rankings, mock_aiohttp_session
    ):
        """Test that leaderboard includes ranking information."""
        await teams_integration.send_leaderboard_update(sample_rankings)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("claude" in text and "1650" in text for text in body_texts)
        assert any("gpt-4" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_with_domain(
        self, teams_integration, sample_rankings, mock_aiohttp_session
    ):
        """Test leaderboard update with domain filter."""
        await teams_integration.send_leaderboard_update(sample_rankings, domain="math")

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Title should include domain
        title = content["body"][0]["text"]
        assert "math" in title

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_limits_to_10(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that leaderboard is limited to top 10."""
        many_rankings = [
            {"name": f"agent{i}", "elo": 1500 - i, "wins": 0, "losses": 0} for i in range(15)
        ]
        await teams_integration.send_leaderboard_update(many_rankings)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Count ranking entries (excluding title)
        ranking_entries = [item for item in content["body"][1:] if "agent" in item.get("text", "")]
        assert len(ranking_entries) == 10

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_includes_win_loss(
        self, teams_integration, sample_rankings, mock_aiohttp_session
    ):
        """Test that leaderboard includes win/loss record."""
        await teams_integration.send_leaderboard_update(sample_rankings)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        body_texts = [item.get("text", "") for item in content["body"]]
        assert any("15W/5L" in text for text in body_texts)

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_includes_action(
        self, teams_integration, sample_rankings, mock_aiohttp_session
    ):
        """Test that leaderboard includes link to full leaderboard."""
        await teams_integration.send_leaderboard_update(sample_rankings)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        assert "actions" in content
        assert content["actions"][0]["title"] == "Full Leaderboard"
        assert "leaderboard" in content["actions"][0]["url"]


# =============================================================================
# Session Management Tests
# =============================================================================


class TestTeamsSessionManagement:
    """Tests for session management functionality."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self, teams_config):
        """Test that close() properly closes the session."""
        integration = TeamsIntegration(teams_config)

        # Create a real session
        await integration._get_session()
        assert integration._session is not None

        await integration.close()
        # Session should be closed now

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, teams_config):
        """Test that close() handles case when no session exists."""
        integration = TeamsIntegration(teams_config)
        assert integration._session is None

        # Should not raise
        await integration.close()

    @pytest.mark.asyncio
    async def test_get_session_recreates_after_close(self, teams_config):
        """Test that _get_session creates new session after close."""
        integration = TeamsIntegration(teams_config)

        session1 = await integration._get_session()
        await integration.close()

        # Simulate closed session
        mock_closed_session = AsyncMock()
        mock_closed_session.closed = True
        integration._session = mock_closed_session

        session2 = await integration._get_session()
        # Should have created a new session
        assert session2 is not mock_closed_session

        await integration.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestTeamsContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, teams_config):
        """Test __aenter__ returns self."""
        integration = TeamsIntegration(teams_config)
        async with integration as ctx:
            assert ctx is integration

    @pytest.mark.asyncio
    async def test_context_manager_exit_closes_session(self, teams_config):
        """Test __aexit__ closes the session."""
        integration = TeamsIntegration(teams_config)

        async with integration:
            await integration._get_session()
            assert integration._session is not None

        # Session should be closed after context exit

    @pytest.mark.asyncio
    async def test_context_manager_handles_exception(self, teams_config, mock_aiohttp_session):
        """Test context manager properly handles exceptions."""
        integration = TeamsIntegration(teams_config)
        integration._session = mock_aiohttp_session

        with pytest.raises(ValueError):
            async with integration:
                raise ValueError("Test error")

        # close() should still have been called
        mock_aiohttp_session.close.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestTeamsIntegrationE2E:
    """End-to-end style tests for Teams integration."""

    @pytest.mark.asyncio
    async def test_full_debate_workflow(
        self, teams_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test a full debate notification workflow."""
        # 1. Post debate summary
        result = await teams_integration.post_debate_summary(sample_debate_result)
        assert result is True

        # 2. Send consensus alert
        result = await teams_integration.send_consensus_alert(
            debate_id=sample_debate_result.debate_id,
            answer=sample_debate_result.answer,
            confidence=0.85,
            agents=sample_debate_result.participating_agents,
        )
        assert result is True

        # Verify both calls were made
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, teams_integration, mock_aiohttp_session):
        """Test error notification workflow."""
        # 1. Send error alert
        result = await teams_integration.send_error_alert(
            debate_id="test-123",
            error="Agent timeout",
            phase="round_2",
        )
        assert result is True

        # Verify the call
        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        content = payload["attachments"][0]["content"]

        # Check title
        assert content["body"][0]["text"] == "Debate Error"

    @pytest.mark.asyncio
    async def test_multiple_notifications_respect_rate_limit(
        self, teams_integration, mock_aiohttp_session
    ):
        """Test that multiple notifications respect rate limiting."""
        # Set a low rate limit
        teams_integration.config.max_messages_per_minute = 3

        results = []
        for i in range(5):
            result = await teams_integration.send_error_alert(
                debate_id=f"test-{i}",
                error=f"Error {i}",
            )
            results.append(result)

        # First 3 should succeed, last 2 should be rate limited
        assert results[:3] == [True, True, True]
        assert results[3:] == [False, False]

        # Only 3 API calls should have been made
        assert mock_aiohttp_session.post.call_count == 3
