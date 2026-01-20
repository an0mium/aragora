"""
Tests for WhatsApp integration.

Covers:
- WhatsAppConfig: environment loading, defaults, provider detection
- WhatsAppIntegration: send_message, rate limiting (per-minute and per-day), session management
- Provider routing: Meta WhatsApp Business API and Twilio API
- Notification methods: debate_summary, consensus_alert, error_alert
- Message formatting and truncation
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.core import DebateResult
from aragora.integrations.whatsapp import (
    WhatsAppConfig,
    WhatsAppIntegration,
    WhatsAppProvider,
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
    mock_response.text = AsyncMock(return_value='{"messages": [{"id": "wamid.xxx"}]}')
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.closed = False
    return mock_session


@pytest.fixture
def meta_config():
    """WhatsApp configuration for Meta Business API."""
    return WhatsAppConfig(
        phone_number_id="123456789",
        access_token="EAAtest_token_xxx",
        recipient="+1234567890",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
        min_consensus_confidence=0.7,
        max_messages_per_minute=5,
        max_messages_per_day=100,
    )


@pytest.fixture
def twilio_config():
    """WhatsApp configuration for Twilio API."""
    return WhatsAppConfig(
        twilio_account_sid="ACtest_account_sid",
        twilio_auth_token="test_auth_token",
        twilio_whatsapp_number="+14155238886",
        recipient="+1234567890",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
    )


@pytest.fixture
def whatsapp_meta_integration(meta_config, mock_aiohttp_session):
    """WhatsAppIntegration instance with Meta API and mocked session."""
    integration = WhatsAppIntegration(meta_config)
    integration._session = mock_aiohttp_session
    return integration


@pytest.fixture
def whatsapp_twilio_integration(twilio_config, mock_aiohttp_session):
    """WhatsAppIntegration instance with Twilio API and mocked session."""
    # Twilio returns 201 for created messages
    mock_response = AsyncMock()
    mock_response.status = 201
    mock_response.text = AsyncMock(return_value='{"sid": "SMxxx"}')
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    integration = WhatsAppIntegration(twilio_config)
    integration._session = mock_aiohttp_session
    return integration


@pytest.fixture
def sample_debate_result():
    """Sample DebateResult for testing."""
    result = DebateResult(
        task="What is the meaning of life?",
        final_answer="42, according to Deep Thought.",
        consensus_reached=True,
        rounds_used=3,
        winner="claude",
        confidence=0.85,
    )
    result.debate_id = "test-debate-123"
    result.question = "What is the meaning of life?"
    result.answer = "42, according to Deep Thought."
    result.total_rounds = 3
    result.consensus_confidence = 0.85
    result.participating_agents = ["claude", "gpt-4", "gemini"]
    return result


# =============================================================================
# WhatsAppConfig Tests
# =============================================================================


class TestWhatsAppConfig:
    """Tests for WhatsAppConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WhatsAppConfig()
        assert config.phone_number_id == ""
        assert config.access_token == ""
        assert config.twilio_account_sid == ""
        assert config.twilio_auth_token == ""
        assert config.twilio_whatsapp_number == ""
        assert config.recipient == ""
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.notify_on_error is True
        assert config.min_consensus_confidence == 0.7
        assert config.max_messages_per_minute == 5
        assert config.max_messages_per_day == 100
        assert config.api_version == "v18.0"

    def test_meta_config(self, meta_config):
        """Test Meta API configuration."""
        assert meta_config.phone_number_id == "123456789"
        assert meta_config.access_token == "EAAtest_token_xxx"
        assert meta_config.recipient == "+1234567890"
        assert meta_config.provider == WhatsAppProvider.META

    def test_twilio_config(self, twilio_config):
        """Test Twilio API configuration."""
        assert twilio_config.twilio_account_sid == "ACtest_account_sid"
        assert twilio_config.twilio_auth_token == "test_auth_token"
        assert twilio_config.twilio_whatsapp_number == "+14155238886"
        assert twilio_config.provider == WhatsAppProvider.TWILIO

    def test_provider_none_when_unconfigured(self):
        """Test provider is None when no credentials provided."""
        config = WhatsAppConfig()
        assert config.provider is None

    def test_meta_takes_precedence(self):
        """Test that Meta provider takes precedence when both configured."""
        config = WhatsAppConfig(
            phone_number_id="123",
            access_token="token",
            twilio_account_sid="AC123",
            twilio_auth_token="token",
            twilio_whatsapp_number="+1234",
        )
        assert config.provider == WhatsAppProvider.META

    def test_env_variable_loading_meta(self, monkeypatch):
        """Test loading Meta credentials from environment variables."""
        monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "env_phone_id")
        monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "env_access_token")
        config = WhatsAppConfig()
        assert config.phone_number_id == "env_phone_id"
        assert config.access_token == "env_access_token"

    def test_env_variable_loading_twilio(self, monkeypatch):
        """Test loading Twilio credentials from environment variables."""
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "env_sid")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "env_token")
        monkeypatch.setenv("TWILIO_WHATSAPP_NUMBER", "+1env")
        config = WhatsAppConfig()
        assert config.twilio_account_sid == "env_sid"
        assert config.twilio_auth_token == "env_token"
        assert config.twilio_whatsapp_number == "+1env"

    def test_explicit_overrides_env(self, monkeypatch):
        """Test that explicit values override environment variables."""
        monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "env_phone_id")
        config = WhatsAppConfig(phone_number_id="explicit_phone_id")
        assert config.phone_number_id == "explicit_phone_id"


# =============================================================================
# WhatsAppIntegration Basic Tests
# =============================================================================


class TestWhatsAppIntegrationBasic:
    """Tests for WhatsAppIntegration basic functionality."""

    def test_default_config(self):
        """Test integration with default config."""
        integration = WhatsAppIntegration()
        assert integration.config is not None
        assert integration._session is None
        assert integration._message_count_minute == 0
        assert integration._message_count_day == 0

    def test_is_configured_true_meta(self, meta_config):
        """Test is_configured returns True for Meta config with recipient."""
        integration = WhatsAppIntegration(meta_config)
        assert integration.is_configured is True

    def test_is_configured_true_twilio(self, twilio_config):
        """Test is_configured returns True for Twilio config with recipient."""
        integration = WhatsAppIntegration(twilio_config)
        assert integration.is_configured is True

    def test_is_configured_false_no_provider(self):
        """Test is_configured returns False when no provider configured."""
        config = WhatsAppConfig(recipient="+1234567890")
        integration = WhatsAppIntegration(config)
        assert integration.is_configured is False

    def test_is_configured_false_no_recipient(self, monkeypatch):
        """Test is_configured returns False when no recipient configured."""
        monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
        monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
        config = WhatsAppConfig(
            phone_number_id="123",
            access_token="token",
            recipient="",
        )
        integration = WhatsAppIntegration(config)
        assert integration.is_configured is False

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, meta_config):
        """Test _get_session creates new session when none exists."""
        integration = WhatsAppIntegration(meta_config)
        assert integration._session is None

        session = await integration._get_session()
        assert session is not None
        assert integration._session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_close_session(self, meta_config):
        """Test close() closes the session."""
        integration = WhatsAppIntegration(meta_config)
        await integration._get_session()
        assert integration._session is not None

        await integration.close()

    def test_format_phone_number(self, whatsapp_meta_integration):
        """Test phone number formatting removes + prefix."""
        assert whatsapp_meta_integration._format_phone_number("+1234567890") == "1234567890"
        assert whatsapp_meta_integration._format_phone_number("1234567890") == "1234567890"


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestWhatsAppRateLimiting:
    """Tests for WhatsApp rate limiting functionality."""

    def test_initial_state(self, whatsapp_meta_integration):
        """Test initial rate limit state."""
        assert whatsapp_meta_integration._message_count_minute == 0
        assert whatsapp_meta_integration._message_count_day == 0

    def test_increments_both_counters(self, whatsapp_meta_integration):
        """Test that _check_rate_limit increments both counters."""
        result = whatsapp_meta_integration._check_rate_limit()
        assert result is True
        assert whatsapp_meta_integration._message_count_minute == 1
        assert whatsapp_meta_integration._message_count_day == 1

    def test_allows_up_to_per_minute_limit(self, whatsapp_meta_integration):
        """Test messages allowed up to per-minute limit."""
        for i in range(whatsapp_meta_integration.config.max_messages_per_minute):
            result = whatsapp_meta_integration._check_rate_limit()
            assert result is True

        # Next should be blocked
        result = whatsapp_meta_integration._check_rate_limit()
        assert result is False

    def test_blocks_over_per_minute_limit(self, whatsapp_meta_integration):
        """Test messages blocked over per-minute limit."""
        for _ in range(whatsapp_meta_integration.config.max_messages_per_minute):
            whatsapp_meta_integration._check_rate_limit()

        result = whatsapp_meta_integration._check_rate_limit()
        assert result is False

    def test_blocks_over_per_day_limit(self, whatsapp_meta_integration):
        """Test messages blocked over per-day limit."""
        # Set up to exceed daily limit
        whatsapp_meta_integration._message_count_day = (
            whatsapp_meta_integration.config.max_messages_per_day
        )

        result = whatsapp_meta_integration._check_rate_limit()
        assert result is False

    def test_per_minute_resets_after_60_seconds(self, whatsapp_meta_integration):
        """Test per-minute counter resets after 60 seconds."""
        # Exhaust per-minute limit
        for _ in range(whatsapp_meta_integration.config.max_messages_per_minute):
            whatsapp_meta_integration._check_rate_limit()

        # Simulate time passing
        whatsapp_meta_integration._last_minute_reset = datetime.now() - timedelta(seconds=61)

        result = whatsapp_meta_integration._check_rate_limit()
        assert result is True
        assert whatsapp_meta_integration._message_count_minute == 1

    def test_per_day_resets_after_24_hours(self, whatsapp_meta_integration):
        """Test per-day counter resets after 24 hours."""
        # Set to daily limit
        whatsapp_meta_integration._message_count_day = (
            whatsapp_meta_integration.config.max_messages_per_day
        )

        # Simulate time passing
        whatsapp_meta_integration._last_day_reset = datetime.now() - timedelta(hours=25)

        result = whatsapp_meta_integration._check_rate_limit()
        assert result is True
        assert whatsapp_meta_integration._message_count_day == 1

    def test_minute_blocks_before_day(self, whatsapp_meta_integration):
        """Test per-minute limit blocks before per-day is reached."""
        # Set a low per-minute limit
        whatsapp_meta_integration.config.max_messages_per_minute = 2
        whatsapp_meta_integration.config.max_messages_per_day = 100

        # Use up per-minute limit
        whatsapp_meta_integration._check_rate_limit()
        whatsapp_meta_integration._check_rate_limit()

        result = whatsapp_meta_integration._check_rate_limit()
        assert result is False
        assert whatsapp_meta_integration._message_count_day == 2  # Still under daily


# =============================================================================
# Meta API Tests
# =============================================================================


class TestWhatsAppMetaAPI:
    """Tests for Meta WhatsApp Business API."""

    @pytest.mark.asyncio
    async def test_send_via_meta_success(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test successful message sending via Meta API."""
        result = await whatsapp_meta_integration._send_via_meta("Test message")
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_via_meta_url_format(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test Meta API URL format."""
        await whatsapp_meta_integration._send_via_meta("Test")

        call_args = mock_aiohttp_session.post.call_args
        url = call_args.args[0]
        assert "graph.facebook.com" in url
        assert "v18.0" in url
        assert whatsapp_meta_integration.config.phone_number_id in url
        assert "/messages" in url

    @pytest.mark.asyncio
    async def test_send_via_meta_payload(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test Meta API payload structure."""
        await whatsapp_meta_integration._send_via_meta("Hello World")

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]

        assert payload["messaging_product"] == "whatsapp"
        assert payload["recipient_type"] == "individual"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "Hello World"
        # Phone number should have + stripped
        assert payload["to"] == "1234567890"

    @pytest.mark.asyncio
    async def test_send_via_meta_headers(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test Meta API headers include Bearer token."""
        await whatsapp_meta_integration._send_via_meta("Test")

        call_args = mock_aiohttp_session.post.call_args
        headers = call_args.kwargs["headers"]

        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {whatsapp_meta_integration.config.access_token}"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_via_meta_api_error(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test Meta API error handling."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value='{"error": "Invalid token"}')
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        result = await whatsapp_meta_integration._send_via_meta("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_via_meta_connection_error(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test Meta API connection error handling."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        result = await whatsapp_meta_integration._send_via_meta("Test")
        assert result is False


# =============================================================================
# Twilio API Tests
# =============================================================================


class TestWhatsAppTwilioAPI:
    """Tests for Twilio WhatsApp API."""

    @pytest.mark.asyncio
    async def test_send_via_twilio_success(self, whatsapp_twilio_integration, mock_aiohttp_session):
        """Test successful message sending via Twilio API."""
        result = await whatsapp_twilio_integration._send_via_twilio("Test message")
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_via_twilio_url_format(
        self, whatsapp_twilio_integration, mock_aiohttp_session
    ):
        """Test Twilio API URL format."""
        await whatsapp_twilio_integration._send_via_twilio("Test")

        call_args = mock_aiohttp_session.post.call_args
        url = call_args.args[0]
        assert "api.twilio.com" in url
        assert whatsapp_twilio_integration.config.twilio_account_sid in url
        assert "/Messages.json" in url

    @pytest.mark.asyncio
    async def test_send_via_twilio_payload(self, whatsapp_twilio_integration, mock_aiohttp_session):
        """Test Twilio API payload structure."""
        await whatsapp_twilio_integration._send_via_twilio("Hello World")

        call_args = mock_aiohttp_session.post.call_args
        data = call_args.kwargs["data"]

        assert (
            data["From"] == f"whatsapp:{whatsapp_twilio_integration.config.twilio_whatsapp_number}"
        )
        assert data["To"] == f"whatsapp:{whatsapp_twilio_integration.config.recipient}"
        assert data["Body"] == "Hello World"

    @pytest.mark.asyncio
    async def test_send_via_twilio_auth(self, whatsapp_twilio_integration, mock_aiohttp_session):
        """Test Twilio API uses Basic Auth."""
        await whatsapp_twilio_integration._send_via_twilio("Test")

        call_args = mock_aiohttp_session.post.call_args
        auth = call_args.kwargs["auth"]

        assert isinstance(auth, aiohttp.BasicAuth)
        assert auth.login == whatsapp_twilio_integration.config.twilio_account_sid
        assert auth.password == whatsapp_twilio_integration.config.twilio_auth_token

    @pytest.mark.asyncio
    async def test_send_via_twilio_accepts_201(self, twilio_config, mock_aiohttp_session):
        """Test Twilio API accepts 201 status code."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.text = AsyncMock(return_value='{"sid": "SMxxx"}')
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        integration = WhatsAppIntegration(twilio_config)
        integration._session = mock_aiohttp_session

        result = await integration._send_via_twilio("Test")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_via_twilio_api_error(
        self, whatsapp_twilio_integration, mock_aiohttp_session
    ):
        """Test Twilio API error handling."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value='{"message": "Invalid number"}')
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        result = await whatsapp_twilio_integration._send_via_twilio("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_via_twilio_connection_error(
        self, whatsapp_twilio_integration, mock_aiohttp_session
    ):
        """Test Twilio API connection error handling."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        result = await whatsapp_twilio_integration._send_via_twilio("Test")
        assert result is False


# =============================================================================
# Send Message Tests
# =============================================================================


class TestWhatsAppSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_routes_to_meta(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test send_message routes to Meta API."""
        result = await whatsapp_meta_integration.send_message("Test")
        assert result is True

        call_args = mock_aiohttp_session.post.call_args
        url = call_args.args[0]
        assert "graph.facebook.com" in url

    @pytest.mark.asyncio
    async def test_send_message_routes_to_twilio(
        self, whatsapp_twilio_integration, mock_aiohttp_session
    ):
        """Test send_message routes to Twilio API."""
        result = await whatsapp_twilio_integration.send_message("Test")
        assert result is True

        call_args = mock_aiohttp_session.post.call_args
        url = call_args.args[0]
        assert "api.twilio.com" in url

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self):
        """Test send_message returns False when not configured."""
        config = WhatsAppConfig()
        integration = WhatsAppIntegration(config)

        result = await integration.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_rate_limited(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test send_message returns False when rate limited."""
        # Exhaust rate limit
        for _ in range(whatsapp_meta_integration.config.max_messages_per_minute):
            whatsapp_meta_integration._check_rate_limit()

        result = await whatsapp_meta_integration.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_truncates_long_messages(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test that long messages are truncated to 4000 chars."""
        long_message = "A" * 5000
        await whatsapp_meta_integration.send_message(long_message)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        sent_message = payload["text"]["body"]

        assert len(sent_message) == 4000
        assert sent_message.endswith("...")

    @pytest.mark.asyncio
    async def test_send_message_preserves_short_messages(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test that short messages are not truncated."""
        short_message = "Short message"
        await whatsapp_meta_integration.send_message(short_message)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        sent_message = payload["text"]["body"]

        assert sent_message == short_message


# =============================================================================
# Debate Summary Tests
# =============================================================================


class TestWhatsAppDebateSummary:
    """Tests for send_debate_summary method."""

    @pytest.mark.asyncio
    async def test_send_debate_summary_success(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test successful debate summary sending."""
        result = await whatsapp_meta_integration.send_debate_summary(sample_debate_result)
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_debate_summary_disabled(
        self, meta_config, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary not sent when disabled."""
        meta_config.notify_on_debate_end = False
        integration = WhatsAppIntegration(meta_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_debate_summary(sample_debate_result)
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_debate_summary_includes_header(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary includes ARAGORA header."""
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "ARAGORA DEBATE COMPLETE" in message

    @pytest.mark.asyncio
    async def test_send_debate_summary_includes_question(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary includes the question."""
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "What is the meaning of life" in message

    @pytest.mark.asyncio
    async def test_send_debate_summary_includes_answer(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary includes the answer."""
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "42" in message

    @pytest.mark.asyncio
    async def test_send_debate_summary_includes_stats(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary includes statistics."""
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "Rounds: 3" in message
        assert "85%" in message  # Confidence
        assert "Agents: 3" in message

    @pytest.mark.asyncio
    async def test_send_debate_summary_includes_link(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test debate summary includes view link."""
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert f"https://aragora.ai/debate/{sample_debate_result.debate_id}" in message

    @pytest.mark.asyncio
    async def test_send_debate_summary_truncates_long_question(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test that long questions are truncated."""
        sample_debate_result.question = "A" * 300
        await whatsapp_meta_integration.send_debate_summary(sample_debate_result)

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        # Question should be truncated to 200 chars
        assert "A" * 200 in message
        assert "A" * 201 not in message.split("Answer:")[0]  # Only check question part


# =============================================================================
# Consensus Alert Tests
# =============================================================================


class TestWhatsAppConsensusAlert:
    """Tests for send_consensus_alert method."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test successful consensus alert."""
        result = await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer="The answer is 42",
            confidence=0.85,
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, meta_config, mock_aiohttp_session):
        """Test consensus alert not sent when disabled."""
        meta_config.notify_on_consensus = False
        integration = WhatsAppIntegration(meta_config)
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
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test consensus alert not sent when confidence below threshold."""
        result = await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.5,  # Below default 0.7 threshold
        )
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_at_threshold(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test consensus alert sent at exact threshold."""
        result = await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer="Answer",
            confidence=0.7,  # Exactly at threshold
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_consensus_alert_includes_header(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test consensus alert includes CONSENSUS REACHED header."""
        await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer="The answer",
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "CONSENSUS REACHED" in message

    @pytest.mark.asyncio
    async def test_send_consensus_alert_includes_confidence(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test consensus alert includes confidence percentage."""
        await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer="The answer",
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "85%" in message

    @pytest.mark.asyncio
    async def test_send_consensus_alert_truncates_long_answer(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test that long answers are truncated to 400 chars."""
        long_answer = "A" * 500
        await whatsapp_meta_integration.send_consensus_alert(
            debate_id="test-123",
            answer=long_answer,
            confidence=0.85,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        # Answer should be truncated with "..."
        assert "..." in message


# =============================================================================
# Error Alert Tests
# =============================================================================


class TestWhatsAppErrorAlert:
    """Tests for send_error_alert method."""

    @pytest.mark.asyncio
    async def test_send_error_alert_success(self, whatsapp_meta_integration, mock_aiohttp_session):
        """Test successful error alert."""
        result = await whatsapp_meta_integration.send_error_alert(
            debate_id="test-123",
            error="Something went wrong",
        )
        assert result is True
        mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, meta_config, mock_aiohttp_session):
        """Test error alert not sent when disabled."""
        meta_config.notify_on_error = False
        integration = WhatsAppIntegration(meta_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
        )
        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_header(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test error alert includes ARAGORA ERROR header."""
        await whatsapp_meta_integration.send_error_alert(
            debate_id="test-123",
            error="Error message",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "ARAGORA ERROR" in message

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_debate_id(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test error alert includes debate ID."""
        await whatsapp_meta_integration.send_error_alert(
            debate_id="test-error-456",
            error="Error",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "test-error-456" in message

    @pytest.mark.asyncio
    async def test_send_error_alert_includes_error_message(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test error alert includes error message."""
        await whatsapp_meta_integration.send_error_alert(
            debate_id="test-123",
            error="Connection timeout to agent",
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        assert "Connection timeout to agent" in message

    @pytest.mark.asyncio
    async def test_send_error_alert_truncates_long_error(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test that long error messages are truncated to 500 chars."""
        long_error = "E" * 600
        await whatsapp_meta_integration.send_error_alert(
            debate_id="test-123",
            error=long_error,
        )

        call_args = mock_aiohttp_session.post.call_args
        payload = call_args.kwargs["json"]
        message = payload["text"]["body"]

        # Error should be truncated
        assert "E" * 500 in message
        assert "E" * 501 not in message


# =============================================================================
# Session Management Tests
# =============================================================================


class TestWhatsAppSessionManagement:
    """Tests for session management functionality."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self, meta_config):
        """Test that close() properly closes the session."""
        integration = WhatsAppIntegration(meta_config)

        await integration._get_session()
        assert integration._session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, meta_config):
        """Test that close() handles case when no session exists."""
        integration = WhatsAppIntegration(meta_config)
        assert integration._session is None

        # Should not raise
        await integration.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestWhatsAppContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, meta_config):
        """Test __aenter__ returns self."""
        integration = WhatsAppIntegration(meta_config)
        async with integration as ctx:
            assert ctx is integration

    @pytest.mark.asyncio
    async def test_context_manager_exit_closes_session(self, meta_config):
        """Test __aexit__ closes the session."""
        integration = WhatsAppIntegration(meta_config)

        async with integration:
            await integration._get_session()
            assert integration._session is not None

    @pytest.mark.asyncio
    async def test_context_manager_handles_exception(self, meta_config, mock_aiohttp_session):
        """Test context manager properly handles exceptions."""
        integration = WhatsAppIntegration(meta_config)
        integration._session = mock_aiohttp_session

        with pytest.raises(ValueError):
            async with integration:
                raise ValueError("Test error")

        mock_aiohttp_session.close.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestWhatsAppIntegrationE2E:
    """End-to-end style tests for WhatsApp integration."""

    @pytest.mark.asyncio
    async def test_full_debate_workflow_meta(
        self, whatsapp_meta_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test a full debate notification workflow via Meta API."""
        # 1. Post debate summary
        result = await whatsapp_meta_integration.send_debate_summary(sample_debate_result)
        assert result is True

        # 2. Send consensus alert
        result = await whatsapp_meta_integration.send_consensus_alert(
            debate_id=sample_debate_result.debate_id,
            answer=sample_debate_result.answer,
            confidence=0.85,
        )
        assert result is True

        # Verify both calls were made
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_full_debate_workflow_twilio(
        self, whatsapp_twilio_integration, sample_debate_result, mock_aiohttp_session
    ):
        """Test a full debate notification workflow via Twilio API."""
        # 1. Post debate summary
        result = await whatsapp_twilio_integration.send_debate_summary(sample_debate_result)
        assert result is True

        # 2. Send error alert
        result = await whatsapp_twilio_integration.send_error_alert(
            debate_id=sample_debate_result.debate_id,
            error="Test error",
        )
        assert result is True

        # Verify both calls were made to Twilio
        assert mock_aiohttp_session.post.call_count == 2
        for call in mock_aiohttp_session.post.call_args_list:
            assert "api.twilio.com" in call.args[0]

    @pytest.mark.asyncio
    async def test_multiple_messages_rate_limiting(
        self, whatsapp_meta_integration, mock_aiohttp_session
    ):
        """Test that multiple messages respect rate limiting."""
        # Set a low per-minute limit
        whatsapp_meta_integration.config.max_messages_per_minute = 2

        results = []
        for i in range(4):
            result = await whatsapp_meta_integration.send_message(f"Message {i}")
            results.append(result)

        # First 2 should succeed, rest should be rate limited
        assert results[:2] == [True, True]
        assert results[2:] == [False, False]

        # Only 2 API calls should have been made
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_provider_fallback_not_implemented(self, meta_config, mock_aiohttp_session):
        """Test that integration uses configured provider only."""
        # If we configure Meta but Meta fails, we don't fall back to Twilio
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='{"error": "Server error"}')
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

        integration = WhatsAppIntegration(meta_config)
        integration._session = mock_aiohttp_session

        result = await integration.send_message("Test")
        assert result is False

        # Only one call (to Meta), no fallback
        assert mock_aiohttp_session.post.call_count == 1
