"""
Tests for chat connector resilience features.

Validates circuit breaker, retry logic, and timeout handling across all connectors.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# Reset circuit breakers between tests
@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear all circuit breakers before each test to ensure isolation."""
    from aragora.resilience import _circuit_breakers, _circuit_breakers_lock

    # Clear the registry completely (not just reset state)
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    # Clear again after test
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


class TestBaseConnectorCircuitBreaker:
    """Test circuit breaker in base ChatPlatformConnector using Discord connector."""

    def test_circuit_breaker_lazy_initialization(self):
        """Circuit breaker should be initialized lazily."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=True)
        # Circuit breaker not initialized yet
        assert connector._circuit_breaker is None
        assert not connector._circuit_breaker_initialized

        # First call initializes it
        cb = connector._get_circuit_breaker()
        assert cb is not None
        assert connector._circuit_breaker_initialized

    def test_circuit_breaker_disabled(self):
        """Should return None when circuit breaker disabled."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=False)
        cb = connector._get_circuit_breaker()
        assert cb is None

    def test_check_circuit_breaker_allows_when_closed(self):
        """Should allow requests when circuit is closed."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=True)
        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is True
        assert error is None

    def test_check_circuit_breaker_blocks_when_open(self):
        """Should block requests when circuit is open."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=2,
        )
        # Record failures to open the circuit
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert error is not None
        assert "Circuit breaker open" in error

    def test_record_success_resets_failures(self):
        """Recording success should reset failure count."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )
        cb = connector._get_circuit_breaker()

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        # Record success
        connector._record_success()
        assert cb.failures == 0

    def test_record_failure_increments_count(self):
        """Recording failure should increment failure count."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )
        cb = connector._get_circuit_breaker()

        connector._record_failure()
        assert cb.failures == 1
        connector._record_failure()
        assert cb.failures == 2


class TestBaseConnectorRetry:
    """Test retry logic in base ChatPlatformConnector."""

    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self):
        """Should succeed on first attempt without retry."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=False)

        async def success_func():
            return "success"

        result = await connector._with_retry("test_op", success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_retry_retries_on_failure(self):
        """Should retry on transient failures."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=False)
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = await connector._with_retry(
            "test_op",
            failing_then_success,
            max_retries=3,
            base_delay=0.01,  # Fast for tests
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_fails_after_max_retries(self):
        """Should fail after exhausting retries."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test", enable_circuit_breaker=False)

        async def always_fail():
            raise ConnectionError("Persistent error")

        with pytest.raises(ConnectionError, match="Persistent error"):
            await connector._with_retry(
                "test_op",
                always_fail,
                max_retries=3,
                base_delay=0.01,
            )

    @pytest.mark.asyncio
    async def test_with_retry_respects_circuit_breaker(self):
        """Should fail fast if circuit breaker is open."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=2,
        )

        # Open the circuit
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        async def should_not_be_called():
            pytest.fail("Function should not be called when circuit is open")

        with pytest.raises(ConnectionError, match="Circuit breaker open"):
            await connector._with_retry("test_op", should_not_be_called)


class TestRetryableStatusCodes:
    """Test status code retry behavior."""

    def test_429_is_retryable(self):
        """429 Too Many Requests should be retryable."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")
        assert connector._is_retryable_status_code(429) is True

    def test_500_is_retryable(self):
        """500 Internal Server Error should be retryable."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")
        assert connector._is_retryable_status_code(500) is True

    def test_503_is_retryable(self):
        """503 Service Unavailable should be retryable."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")
        assert connector._is_retryable_status_code(503) is True

    def test_400_is_not_retryable(self):
        """400 Bad Request should not be retryable."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")
        assert connector._is_retryable_status_code(400) is False

    def test_401_is_not_retryable(self):
        """401 Unauthorized should not be retryable."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")
        assert connector._is_retryable_status_code(401) is False


class TestDiscordConnectorResilience:
    """Test resilience features of Discord connector."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Discord circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test_token",
            circuit_breaker_threshold=2,
        )

        # Get circuit breaker and record failures
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert "Circuit breaker open" in error

    @pytest.mark.asyncio
    async def test_send_message_blocked_when_circuit_open(self):
        """Send message should fail when circuit is open."""
        from aragora.connectors.chat.discord import DiscordConnector
        from aragora.connectors.chat.models import SendMessageResponse

        connector = DiscordConnector(
            bot_token="test_token",
            circuit_breaker_threshold=2,
        )

        # Open the circuit
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        result = await connector.send_message(
            channel_id="123",
            text="test",
        )
        assert isinstance(result, SendMessageResponse)
        assert not result.success
        # Circuit breaker error may be in error or handled differently
        assert result.error is not None


class TestTeamsConnectorResilience:
    """Test resilience features of Teams connector."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Teams circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            bot_id="test_bot",
            tenant_id="test_tenant",
            circuit_breaker_threshold=2,
        )

        # Get circuit breaker and record failures
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert "Circuit breaker open" in error


class TestWhatsAppConnectorResilience:
    """Test resilience features of WhatsApp connector."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """WhatsApp circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.whatsapp import WhatsAppConnector

        connector = WhatsAppConnector(
            access_token="test_token",
            phone_number_id="123",
            circuit_breaker_threshold=2,
        )

        # Get circuit breaker and record failures
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert "Circuit breaker open" in error


class TestTelegramConnectorResilience:
    """Test resilience features of Telegram connector."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Telegram circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.telegram import TelegramConnector

        connector = TelegramConnector(
            bot_token="123:ABC",
            circuit_breaker_threshold=2,
        )

        # Get circuit breaker and record failures
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert "Circuit breaker open" in error


class TestGoogleChatConnectorResilience:
    """Test resilience features of Google Chat connector."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Google Chat circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        connector = GoogleChatConnector(circuit_breaker_threshold=2)

        # Get circuit breaker and record failures
        cb = connector._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is False
        assert "Circuit breaker open" in error


class TestCascadingFailurePrevention:
    """Test that failures in one connector don't cascade to others."""

    @pytest.mark.asyncio
    async def test_connectors_have_independent_circuit_breakers(self):
        """Each connector should have its own circuit breaker."""
        from aragora.connectors.chat.discord import DiscordConnector
        from aragora.connectors.chat.telegram import TelegramConnector

        discord = DiscordConnector(
            bot_token="discord_token",
            circuit_breaker_threshold=2,
        )
        telegram = TelegramConnector(
            bot_token="123:ABC",
            circuit_breaker_threshold=2,
        )

        # Open Discord circuit
        discord_cb = discord._get_circuit_breaker()
        discord_cb.record_failure()
        discord_cb.record_failure()

        # Discord should be blocked
        discord_can_proceed, _ = discord._check_circuit_breaker()
        assert discord_can_proceed is False

        # Telegram should still work
        telegram_can_proceed, _ = telegram._check_circuit_breaker()
        assert telegram_can_proceed is True

    @pytest.mark.asyncio
    async def test_multiple_connector_instances_share_circuit_breaker(self):
        """Multiple instances of same connector should share circuit breaker."""
        from aragora.connectors.chat.discord import DiscordConnector

        discord1 = DiscordConnector(
            bot_token="discord_token_1",
            circuit_breaker_threshold=3,
        )
        discord2 = DiscordConnector(
            bot_token="discord_token_2",
            circuit_breaker_threshold=3,
        )

        # Record failures through first instance
        discord1_cb = discord1._get_circuit_breaker()
        discord1_cb.record_failure()
        discord1_cb.record_failure()
        discord1_cb.record_failure()

        # Second instance should see open circuit
        discord2_can_proceed, _ = discord2._check_circuit_breaker()
        assert discord2_can_proceed is False


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and monitoring."""

    def test_get_circuit_breaker_metrics(self):
        """Should return comprehensive metrics."""
        from aragora.resilience import get_circuit_breaker_metrics
        from aragora.connectors.chat.discord import DiscordConnector

        # Create connector to register circuit breaker
        connector = DiscordConnector(bot_token="test")
        connector._get_circuit_breaker()

        metrics = get_circuit_breaker_metrics()
        assert "timestamp" in metrics
        assert "registry_size" in metrics
        assert "summary" in metrics
        assert "health" in metrics
        assert metrics["summary"]["total"] >= 1

    def test_health_status_degraded_when_circuit_open(self):
        """Health status should be degraded when circuit is open."""
        from aragora.resilience import get_circuit_breaker_metrics
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(
            bot_token="test",
            circuit_breaker_threshold=2,
        )
        cb = connector._get_circuit_breaker()

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        assert metrics["health"]["status"] in ("degraded", "critical")
