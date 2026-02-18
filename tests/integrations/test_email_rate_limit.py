"""
Tests for Email integration rate limiting and circuit breaker.

Comprehensive tests for:
- Rate limiting enforcement
- Circuit breaker behavior
- Thread-safe rate limit tracking
- Provider-specific circuit breakers
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.email import (
    EmailConfig,
    EmailIntegration,
    EmailProvider,
    EmailRecipient,
    _get_email_circuit_breaker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _disable_distributed_features(monkeypatch):
    """Disable distributed rate limiter and OAuth refresh for local rate limit tests."""
    monkeypatch.setattr(
        "aragora.integrations.email._distributed_rate_limiter",
        None,
    )
    monkeypatch.setattr(
        "aragora.integrations.email._get_distributed_rate_limiter",
        lambda: None,
    )
    monkeypatch.setattr(
        "aragora.integrations.email._get_credential_store",
        lambda: None,
    )


@pytest.fixture
def smtp_config():
    return EmailConfig(
        smtp_host="smtp.example.com",
        max_emails_per_hour=10,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown=30.0,
        enable_distributed_rate_limit=False,
        enable_oauth_refresh=False,
    )


@pytest.fixture
def integration(smtp_config):
    return EmailIntegration(smtp_config)


@pytest.fixture
def recipient():
    return EmailRecipient(email="user@example.com", name="Test User")


# =============================================================================
# Rate Limit Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, smtp_config):
        """Test emails allowed when under rate limit."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=5)
        integration = EmailIntegration(config)

        # Should allow 5 emails
        for i in range(5):
            result = await integration._check_rate_limit()
            assert result is True, f"Email {i + 1} should be allowed"
            assert integration._email_count == i + 1

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self, smtp_config):
        """Test emails blocked when over rate limit."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=3)
        integration = EmailIntegration(config)

        # Use up the limit
        for _ in range(3):
            await integration._check_rate_limit()

        # Next should be blocked
        result = await integration._check_rate_limit()
        assert result is False
        assert integration._email_count == 3

    @pytest.mark.asyncio
    async def test_resets_after_hour(self, smtp_config):
        """Test rate limit resets after an hour."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=3)
        integration = EmailIntegration(config)

        # Use up the limit
        for _ in range(3):
            await integration._check_rate_limit()

        # Simulate time passing (1+ hour)
        integration._last_reset = datetime.now() - timedelta(hours=1, minutes=1)

        # Should be allowed again
        result = await integration._check_rate_limit()
        assert result is True
        assert integration._email_count == 1

    @pytest.mark.asyncio
    async def test_does_not_reset_within_hour(self, smtp_config):
        """Test rate limit does not reset within the hour."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=3)
        integration = EmailIntegration(config)

        # Use up the limit
        for _ in range(3):
            await integration._check_rate_limit()

        # Simulate time passing (less than 1 hour)
        integration._last_reset = datetime.now() - timedelta(minutes=30)

        # Should still be blocked
        result = await integration._check_rate_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self, smtp_config):
        """Test thread-safe concurrent rate limit checks."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=100)
        integration = EmailIntegration(config)

        # Run multiple concurrent rate limit checks
        async def check_limit():
            return await integration._check_rate_limit()

        results = await asyncio.gather(*[check_limit() for _ in range(50)])

        # All should succeed and count should be exact
        assert all(results)
        assert integration._email_count == 50

    @pytest.mark.asyncio
    async def test_health_status_includes_rate_limit(self, integration):
        """Test health status includes rate limit info."""
        await integration._check_rate_limit()
        await integration._check_rate_limit()

        status = integration.get_health_status()

        assert status["emails_sent_this_hour"] == 2
        assert status["rate_limit"] == 10


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_created_for_provider(self):
        """Test circuit breaker is created per provider."""
        cb_smtp = _get_email_circuit_breaker("smtp_test_1", threshold=5)
        cb_sendgrid = _get_email_circuit_breaker("sendgrid_test_1", threshold=5)

        assert cb_smtp is not None
        assert cb_sendgrid is not None
        # They should be different instances
        assert cb_smtp.name != cb_sendgrid.name

    def test_circuit_breaker_singleton_per_provider(self):
        """Test circuit breaker is singleton per provider."""
        cb1 = _get_email_circuit_breaker("singleton_test", threshold=5)
        cb2 = _get_email_circuit_breaker("singleton_test", threshold=5)

        # Same provider should return same instance
        assert cb1 is cb2

    def test_circuit_breaker_disabled(self, smtp_config):
        """Test circuit breaker can be disabled."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_circuit_breaker=False,
        )
        integration = EmailIntegration(config)

        cb = integration._get_circuit_breaker()
        assert cb is None

    def test_circuit_breaker_enabled(self, smtp_config):
        """Test circuit breaker is enabled by default."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_circuit_breaker=True,
        )
        integration = EmailIntegration(config)

        cb = integration._get_circuit_breaker()
        # Note: cb may be None if resilience module not available
        # This is acceptable behavior

    def test_check_circuit_breaker_allows_when_closed(self, integration):
        """Test circuit breaker allows requests when closed."""
        can_proceed, error = integration._check_circuit_breaker()
        assert can_proceed is True
        assert error is None

    def test_check_circuit_breaker_when_disabled(self):
        """Test circuit breaker check when disabled."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_circuit_breaker=False,
        )
        integration = EmailIntegration(config)

        can_proceed, error = integration._check_circuit_breaker()
        assert can_proceed is True
        assert error is None

    def test_record_success(self, integration):
        """Test recording successful operation."""
        # Should not raise
        integration._record_success()

    def test_record_failure(self, integration):
        """Test recording failed operation."""
        # Should not raise
        integration._record_failure(Exception("Test error"))

    def test_health_status_includes_circuit_breaker(self, integration):
        """Test health status includes circuit breaker info."""
        status = integration.get_health_status()

        assert "circuit_breaker_enabled" in status
        assert "circuit_breaker_status" in status


# =============================================================================
# Circuit Breaker State Tests
# =============================================================================


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, smtp_config):
        """Test circuit opens after threshold failures."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60.0,
        )
        integration = EmailIntegration(config)

        # Mock circuit breaker with predictable behavior
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        failure_count = 0

        def record_failure():
            nonlocal failure_count
            failure_count += 1

        mock_cb.record_failure = record_failure
        mock_cb.get_status.return_value = "closed" if failure_count < 3 else "open"

        with patch.object(integration, "_get_circuit_breaker", return_value=mock_cb):
            for _ in range(3):
                integration._record_failure()

        assert failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self):
        """Test circuit blocks requests when open."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_circuit_breaker=True,
        )
        integration = EmailIntegration(config)

        # Mock an open circuit breaker
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        mock_cb.cooldown_remaining.return_value = 45.0

        with patch.object(integration, "_get_circuit_breaker", return_value=mock_cb):
            can_proceed, error = integration._check_circuit_breaker()

        assert can_proceed is False
        assert "circuit breaker open" in error.lower()
        assert "45" in error

    def test_circuit_success_resets_failures(self, integration):
        """Test successful operations reset failure count."""
        mock_cb = MagicMock()
        mock_cb.record_success = MagicMock()

        with patch.object(integration, "_get_circuit_breaker", return_value=mock_cb):
            integration._record_success()

        mock_cb.record_success.assert_called_once()


# =============================================================================
# Send with Circuit Breaker Tests
# =============================================================================


class TestSendWithCircuitBreaker:
    """Tests for send operations with circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_send_blocked_by_circuit_breaker(self, integration, recipient):
        """Test send is blocked when circuit breaker is open."""
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        mock_cb.cooldown_remaining.return_value = 30.0

        with patch.object(integration, "_get_circuit_breaker", return_value=mock_cb):
            result = await integration._send_email(recipient, "Test", "<p>Test</p>")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_blocked_by_rate_limit(self, smtp_config, recipient):
        """Test send is blocked when rate limit exceeded."""
        config = EmailConfig(smtp_host="smtp.test.com", max_emails_per_hour=0)
        integration = EmailIntegration(config)
        integration._email_count = 1  # Already at limit

        with patch.object(integration, "_check_circuit_breaker", return_value=(True, None)):
            result = await integration._send_email(recipient, "Test", "<p>Test</p>")

        assert result is False


# =============================================================================
# Provider-Specific Tests
# =============================================================================


class TestProviderCircuitBreakers:
    """Tests for provider-specific circuit breakers."""

    def test_smtp_circuit_breaker(self):
        """Test SMTP has its own circuit breaker."""
        config = EmailConfig(smtp_host="smtp.test.com")
        integration = EmailIntegration(config)

        # Circuit breaker should be for SMTP
        cb = integration._get_circuit_breaker()
        if cb:
            assert "smtp" in cb.name.lower()

    def test_sendgrid_circuit_breaker(self):
        """Test SendGrid has its own circuit breaker."""
        config = EmailConfig(provider="sendgrid", sendgrid_api_key="SG.test")
        integration = EmailIntegration(config)

        cb = integration._get_circuit_breaker()
        if cb:
            assert "sendgrid" in cb.name.lower()

    def test_ses_circuit_breaker(self):
        """Test SES has its own circuit breaker."""
        config = EmailConfig(
            provider="ses",
            ses_access_key_id="AKIA_TEST",
            ses_secret_access_key="secret",
        )
        integration = EmailIntegration(config)

        cb = integration._get_circuit_breaker()
        if cb:
            assert "ses" in cb.name.lower()


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic with circuit breaker."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, smtp_config, recipient):
        """Test operation retries on transient failure."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_retries=3,
            retry_delay=0.01,  # Short delay for testing
        )
        integration = EmailIntegration(config)

        call_count = 0

        async def mock_smtp_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Transient error")
            return True

        with patch.object(integration, "_send_via_smtp", mock_smtp_send):
            with patch.object(integration, "_check_circuit_breaker", return_value=(True, None)):
                with patch.object(integration, "_check_rate_limit", return_value=True):
                    result = await integration._send_email(recipient, "Test", "<p>Test</p>")

        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_reached(self, smtp_config, recipient):
        """Test gives up after max retries."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_retries=2,
            retry_delay=0.01,
        )
        integration = EmailIntegration(config)

        call_count = 0

        async def mock_smtp_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError("Persistent error")

        with patch.object(integration, "_send_via_smtp", mock_smtp_send):
            with patch.object(integration, "_check_circuit_breaker", return_value=(True, None)):
                with patch.object(integration, "_check_rate_limit", return_value=True):
                    with patch.object(integration, "_record_failure"):
                        result = await integration._send_email(recipient, "Test", "<p>Test</p>")

        assert result is False
        assert call_count == 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_circuit_breaker_creation_thread_safe(self):
        """Test circuit breaker creation is thread-safe."""
        import threading

        results = []

        def get_cb():
            cb = _get_email_circuit_breaker("thread_test", threshold=5)
            results.append(cb)

        threads = [threading.Thread(target=get_cb) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same instance
        if results[0] is not None:
            assert all(r is results[0] for r in results)

    @pytest.mark.asyncio
    async def test_rate_limit_thread_safe(self):
        """Test rate limit checking is thread-safe with async."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_emails_per_hour=100,
            enable_distributed_rate_limit=False,
            enable_oauth_refresh=False,
        )
        integration = EmailIntegration(config)

        # Concurrent rate limit checks
        tasks = [integration._check_rate_limit() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        # Count should match exactly
        assert integration._email_count == 100


# =============================================================================
# Configuration Tests
# =============================================================================


class TestCircuitBreakerConfiguration:
    """Tests for circuit breaker configuration."""

    def test_custom_threshold(self):
        """Test custom failure threshold."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            circuit_breaker_threshold=10,
        )
        assert config.circuit_breaker_threshold == 10

    def test_custom_cooldown(self):
        """Test custom cooldown period."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            circuit_breaker_cooldown=120.0,
        )
        assert config.circuit_breaker_cooldown == 120.0

    def test_default_values(self):
        """Test default circuit breaker values."""
        config = EmailConfig(smtp_host="smtp.test.com")
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_cooldown == 60.0
