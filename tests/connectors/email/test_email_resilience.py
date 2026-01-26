"""
Tests for email resilience patterns.

Tests circuit breaker, retry executor, and OAuth token store
functionality for production email connectors.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.email.resilience import (
    CircuitBreakerConfig,
    CircuitState,
    EmailCircuitBreaker,
    RetryConfig,
    RetryExecutor,
)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestEmailCircuitBreaker:
    """Tests for EmailCircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test config."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=1.0,
            half_open_max_calls=2,
            success_threshold=2,
        )
        return EmailCircuitBreaker(name="test", config=config)

    @pytest.mark.asyncio
    async def test_closed_state_allows_execution(self, circuit_breaker):
        """Circuit in CLOSED state should allow execution."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert await circuit_breaker.can_execute() is True

    @pytest.mark.asyncio
    async def test_circuit_opens_at_failure_threshold(self, circuit_breaker):
        """Circuit should open after reaching failure threshold."""
        # Record failures up to threshold
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open is True
        assert await circuit_breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_circuit_stays_closed_below_threshold(self, circuit_breaker):
        """Circuit should stay closed below failure threshold."""
        for _ in range(2):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert await circuit_breaker.can_execute() is True

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, circuit_breaker):
        """Success should reset failure count."""
        await circuit_breaker.record_failure()
        await circuit_breaker.record_failure()
        await circuit_breaker.record_success()

        # Failure count should be reset
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self, circuit_breaker):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should transition to HALF_OPEN on next can_execute check
        assert await circuit_breaker.can_execute() is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self, circuit_breaker):
        """Circuit should close after success threshold in HALF_OPEN."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        await circuit_breaker.can_execute()  # Transition to HALF_OPEN

        # Record successes to close
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self, circuit_breaker):
        """Circuit should reopen on failure in HALF_OPEN."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        await circuit_breaker.can_execute()  # Transition to HALF_OPEN

        # Failure should reopen
        await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_limits_concurrent_calls(self, circuit_breaker):
        """HALF_OPEN should limit number of concurrent calls."""
        # Open and wait for half-open
        for _ in range(3):
            await circuit_breaker.record_failure()

        await asyncio.sleep(1.1)

        # First calls should be allowed
        assert await circuit_breaker.can_execute() is True
        assert await circuit_breaker.can_execute() is True

        # Third call should be rejected (half_open_max_calls=2)
        assert await circuit_breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics(self, circuit_breaker):
        """Circuit breaker should provide metrics."""
        await circuit_breaker.record_failure()
        await circuit_breaker.record_success()
        await circuit_breaker.record_failure()

        metrics = circuit_breaker.get_metrics()

        assert "state" in metrics
        assert "failure_count" in metrics
        assert "success_count" in metrics
        assert metrics["name"] == "test"


# =============================================================================
# Retry Executor Tests
# =============================================================================


class TestRetryExecutor:
    """Tests for RetryExecutor."""

    @pytest.fixture
    def retry_executor(self):
        """Create retry executor with test config."""
        config = RetryConfig(
            max_retries=3,
            initial_delay_seconds=0.1,
            max_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,
        )
        return RetryExecutor(config=config)

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self, retry_executor):
        """Successful execution should not retry."""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_executor.execute(successful_operation)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_exception(self, retry_executor):
        """Should retry on retryable exceptions."""
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = await retry_executor.execute(flaky_operation)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, retry_executor):
        """Should raise after max retries exceeded."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            await retry_executor.execute(always_fails)

        assert call_count == 4  # Initial + 3 retries

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self, retry_executor):
        """Non-retryable exceptions should not be retried."""
        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await retry_executor.execute(raises_value_error)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, retry_executor):
        """Should apply exponential backoff between retries."""
        delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            delays.append(delay)

        call_count = 0

        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Transient")
            return "success"

        with patch("asyncio.sleep", mock_sleep):
            await retry_executor.execute(fails_twice)

        # Delays should follow exponential pattern: 0.1, 0.2
        assert len(delays) == 2
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)

    @pytest.mark.asyncio
    async def test_delay_capped_at_max(self):
        """Delay should be capped at max_delay_seconds."""
        config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=1.0,
            max_delay_seconds=2.0,
            exponential_base=2.0,
            jitter=False,
        )
        executor = RetryExecutor(config=config)

        delays = []

        async def mock_sleep(delay):
            delays.append(delay)

        call_count = 0

        async def fails_often():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise ConnectionError("Transient")
            return "success"

        with patch("asyncio.sleep", mock_sleep):
            await executor.execute(fails_often)

        # Delays should be capped at 2.0
        assert all(d <= 2.0 for d in delays)

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self):
        """Jitter should add randomness to delays."""
        config = RetryConfig(
            max_retries=10,
            initial_delay_seconds=1.0,
            max_delay_seconds=10.0,
            exponential_base=2.0,
            jitter=True,
        )
        executor = RetryExecutor(config=config)

        delays = []

        async def mock_sleep(delay):
            delays.append(delay)

        call_count = 0

        async def fails_many():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise ConnectionError("Transient")
            return "success"

        with patch("asyncio.sleep", mock_sleep):
            await executor.execute(fails_many)

        # With jitter, delays should not be exactly equal to base values
        # This is a probabilistic test, might occasionally fail
        base_delays = [1.0, 2.0, 4.0, 8.0, 10.0]
        differences = [abs(d - b) for d, b in zip(delays, base_delays)]
        assert any(diff > 0.01 for diff in differences)


# =============================================================================
# Integration Tests
# =============================================================================


class TestResilienceIntegration:
    """Integration tests for resilience patterns working together."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Circuit breaker and retry should work together."""
        config = CircuitBreakerConfig(failure_threshold=5)
        circuit = EmailCircuitBreaker(name="test", config=config)

        retry_config = RetryConfig(max_retries=2, initial_delay_seconds=0.01)
        executor = RetryExecutor(config=retry_config)

        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if not await circuit.can_execute():
                raise RuntimeError("Circuit open")
            try:
                raise ConnectionError("API down")
            except Exception:
                await circuit.record_failure()
                raise

        # Should retry and record failures
        with pytest.raises(ConnectionError):
            await executor.execute(operation)

        # Circuit should still be closed (3 failures < 5 threshold)
        assert circuit.state == CircuitState.CLOSED


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_access(self):
        """Circuit breaker should handle concurrent access safely."""
        config = CircuitBreakerConfig(failure_threshold=10)
        circuit = EmailCircuitBreaker(name="concurrent", config=config)

        async def record_failure():
            await circuit.record_failure()

        # Record failures concurrently
        await asyncio.gather(*[record_failure() for _ in range(10)])

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_retry_with_sync_function(self):
        """RetryExecutor should handle sync functions wrapped in async."""
        config = RetryConfig(max_retries=2, initial_delay_seconds=0.01)
        executor = RetryExecutor(config=config)

        call_count = 0

        def sync_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient")
            return "success"

        async def async_wrapper():
            return sync_operation()

        result = await executor.execute(async_wrapper)

        assert result == "success"
        assert call_count == 2
