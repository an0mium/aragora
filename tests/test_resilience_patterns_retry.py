"""
Tests for resilience_patterns.retry module.

Tests cover:
- Retry strategies (exponential, linear, fibonacci, constant)
- Jitter modes (none, additive, multiplicative, full)
- Exception filtering
- Callback invocation
- Max attempts
- Both async and sync decorators
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.resilience_patterns import (
    RetryStrategy,
    RetryConfig,
    ExponentialBackoff,
    with_retry,
    with_retry_sync,
    calculate_backoff_delay,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.LINEAR,
        )
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.strategy == RetryStrategy.LINEAR


class TestCalculateBackoffDelay:
    """Tests for backoff delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        # attempt 0: 1.0 * 2^0 = 1.0
        delay = calculate_backoff_delay(0, 1.0, 60.0, RetryStrategy.EXPONENTIAL)
        assert delay == 1.0

        # attempt 1: 1.0 * 2^1 = 2.0
        delay = calculate_backoff_delay(1, 1.0, 60.0, RetryStrategy.EXPONENTIAL)
        assert delay == 2.0

        # attempt 2: 1.0 * 2^2 = 4.0
        delay = calculate_backoff_delay(2, 1.0, 60.0, RetryStrategy.EXPONENTIAL)
        assert delay == 4.0

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        # attempt 0: 1.0 * 1 = 1.0
        delay = calculate_backoff_delay(0, 1.0, 60.0, RetryStrategy.LINEAR)
        assert delay == 1.0

        # attempt 1: 1.0 * 2 = 2.0
        delay = calculate_backoff_delay(1, 1.0, 60.0, RetryStrategy.LINEAR)
        assert delay == 2.0

        # attempt 5: 1.0 * 6 = 6.0
        delay = calculate_backoff_delay(5, 1.0, 60.0, RetryStrategy.LINEAR)
        assert delay == 6.0

    def test_fibonacci_backoff(self):
        """Test fibonacci backoff strategy."""
        delays = [calculate_backoff_delay(i, 1.0, 60.0, RetryStrategy.FIBONACCI) for i in range(6)]
        # Fibonacci: 1, 1, 2, 3, 5, 8
        assert delays == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]

    def test_constant_backoff(self):
        """Test constant backoff strategy."""
        for attempt in range(5):
            delay = calculate_backoff_delay(attempt, 2.0, 60.0, RetryStrategy.CONSTANT)
            assert delay == 2.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        # Very high attempt with exponential should be capped
        delay = calculate_backoff_delay(20, 1.0, 60.0, RetryStrategy.EXPONENTIAL)
        assert delay == 60.0


class TestExponentialBackoff:
    """Tests for ExponentialBackoff iterator."""

    def test_iteration(self):
        """Test iteration through backoff delays."""
        backoff = ExponentialBackoff(base=1.0, max_delay=60.0, max_attempts=5)
        delays = list(backoff)
        assert len(delays) == 5
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0

    def test_reset(self):
        """Test reset functionality."""
        backoff = ExponentialBackoff(base=1.0, max_delay=60.0, max_attempts=3)
        list(backoff)  # Exhaust iterator
        backoff.reset()
        delays = list(backoff)
        assert len(delays) == 3


class TestWithRetryAsync:
    """Tests for async with_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call without retries."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        async def success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = await fail_twice()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test exception when max attempts exceeded."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError, match="Permanent failure"):
            await always_fail()
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, retryable_exceptions=(ConnectionError,)))
        async def value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            await value_error()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_callback(self):
        """Test retry callback invocation."""
        callback_calls = []

        def on_retry(attempt, exc, delay):
            callback_calls.append((attempt, str(exc), delay))

        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01, on_retry=on_retry))
        async def fail_once():
            if len(callback_calls) == 0:
                raise ConnectionError("First failure")
            return "success"

        result = await fail_once()
        assert result == "success"
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1  # Attempt number
        assert "First failure" in callback_calls[0][1]


class TestWithRetrySync:
    """Tests for sync with_retry_sync decorator."""

    def test_success_no_retry(self):
        """Test successful call without retries."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_attempts=3))
        def success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_attempts=3, base_delay=0.01))
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Transient failure")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Test exception when max attempts exceeded."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Permanent failure")

        with pytest.raises(TimeoutError, match="Permanent failure"):
            always_fail()
        assert call_count == 2


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"
        assert RetryStrategy.CONSTANT.value == "constant"

    def test_strategy_from_string(self):
        """Test strategy creation from string."""
        assert RetryStrategy("exponential") == RetryStrategy.EXPONENTIAL
        assert RetryStrategy("linear") == RetryStrategy.LINEAR
        assert RetryStrategy("fibonacci") == RetryStrategy.FIBONACCI
        assert RetryStrategy("constant") == RetryStrategy.CONSTANT


class TestRetryWithDifferentStrategies:
    """Test retry decorator with different backoff strategies."""

    @pytest.mark.asyncio
    async def test_linear_strategy(self):
        """Test retry with linear backoff."""
        delays = []

        def capture_delay(attempt, exc, delay):
            delays.append(delay)

        @with_retry(
            RetryConfig(
                max_attempts=4,
                base_delay=1.0,
                strategy=RetryStrategy.LINEAR,
                on_retry=capture_delay,
            )
        )
        async def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await always_fail()

        # Linear: 1, 2, 3 delays before max_attempts exceeded
        assert len(delays) == 3
        # Delays should increase linearly (with potential jitter)
        assert all(d > 0 for d in delays)

    @pytest.mark.asyncio
    async def test_constant_strategy(self):
        """Test retry with constant backoff."""
        delays = []

        def capture_delay(attempt, exc, delay):
            delays.append(delay)

        @with_retry(
            RetryConfig(
                max_attempts=4,
                base_delay=0.5,
                strategy=RetryStrategy.CONSTANT,
                on_retry=capture_delay,
                jitter_factor=0.0,  # Disable jitter for predictable testing
            )
        )
        async def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await always_fail()

        # All delays should be constant
        assert len(delays) == 3
        assert all(d == 0.5 for d in delays)
