"""
Tests for Retry Pattern.

Tests the retry implementation including:
- Different retry strategies (exponential, linear, constant, fibonacci)
- Jitter modes
- Retry configuration
- Sync and async decorators
- Exception handling
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from aragora.resilience_patterns.retry import (
    DEFAULT_RETRYABLE_EXCEPTIONS,
    JitterMode,
    RetryConfig,
    RetryStrategy,
    calculate_backoff_delay,
    with_retry,
    with_retry_sync,
)


# =============================================================================
# RetryStrategy Tests
# =============================================================================


class TestRetryStrategy:
    """Test RetryStrategy enum."""

    def test_strategy_values(self):
        """Test retry strategy values."""
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.CONSTANT.value == "constant"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"

    def test_strategy_is_string_enum(self):
        """Test RetryStrategy is a string enum."""
        assert isinstance(RetryStrategy.EXPONENTIAL, str)


# =============================================================================
# JitterMode Tests
# =============================================================================


class TestJitterMode:
    """Test JitterMode enum."""

    def test_jitter_mode_values(self):
        """Test jitter mode values."""
        assert JitterMode.NONE.value == "none"
        assert JitterMode.ADDITIVE.value == "additive"
        assert JitterMode.MULTIPLICATIVE.value == "multiplicative"
        assert JitterMode.FULL.value == "full"


# =============================================================================
# RetryConfig Tests
# =============================================================================


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.jitter_mode == JitterMode.ADDITIVE
        assert config.jitter_factor == 0.25
        assert config.retryable_exceptions == DEFAULT_RETRYABLE_EXCEPTIONS
        assert config.on_retry is None
        assert config.should_retry is None

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            strategy=RetryStrategy.LINEAR,
            jitter_mode=JitterMode.NONE,
            jitter_factor=0.1,
            retryable_exceptions=(ValueError,),
            on_retry=callback,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.strategy == RetryStrategy.LINEAR
        assert config.jitter_mode == JitterMode.NONE
        assert config.jitter_factor == 0.1
        assert config.retryable_exceptions == (ValueError,)
        assert config.on_retry == callback


# =============================================================================
# calculate_backoff_delay Tests
# =============================================================================


class TestCalculateDelay:
    """Test delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            jitter_mode=JitterMode.NONE,
        )
        # Exponential: 2^n * base_delay
        assert calculate_backoff_delay(config, 0) == 1.0  # 2^0 * 1 = 1
        assert calculate_backoff_delay(config, 1) == 2.0  # 2^1 * 1 = 2
        assert calculate_backoff_delay(config, 2) == 4.0  # 2^2 * 1 = 4
        assert calculate_backoff_delay(config, 3) == 8.0  # 2^3 * 1 = 8

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            base_delay=1.0,
            jitter_mode=JitterMode.NONE,
        )
        # Linear: (n + 1) * base_delay
        assert calculate_backoff_delay(config, 0) == 1.0
        assert calculate_backoff_delay(config, 1) == 2.0
        assert calculate_backoff_delay(config, 2) == 3.0
        assert calculate_backoff_delay(config, 3) == 4.0

    def test_constant_backoff(self):
        """Test constant backoff calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay=5.0,
            jitter_mode=JitterMode.NONE,
        )
        # Constant: always base_delay
        assert calculate_backoff_delay(config, 0) == 5.0
        assert calculate_backoff_delay(config, 1) == 5.0
        assert calculate_backoff_delay(config, 2) == 5.0
        assert calculate_backoff_delay(config, 10) == 5.0

    def test_fibonacci_backoff(self):
        """Test fibonacci backoff calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.FIBONACCI,
            base_delay=1.0,
            jitter_mode=JitterMode.NONE,
        )
        # Fibonacci: fib(n+1) * base_delay
        assert calculate_backoff_delay(config, 0) == 1.0  # fib(1) = 1
        assert calculate_backoff_delay(config, 1) == 1.0  # fib(2) = 1
        assert calculate_backoff_delay(config, 2) == 2.0  # fib(3) = 2
        assert calculate_backoff_delay(config, 3) == 3.0  # fib(4) = 3
        assert calculate_backoff_delay(config, 4) == 5.0  # fib(5) = 5

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=10.0,
            jitter_mode=JitterMode.NONE,
        )
        # 2^10 * 1 = 1024, but should be capped at 10
        assert calculate_backoff_delay(config, 10) == 10.0

    def test_jitter_additive(self):
        """Test additive jitter."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay=1.0,
            jitter_mode=JitterMode.ADDITIVE,
            jitter_factor=0.25,
        )
        # With additive jitter, delay should be between base and base + jitter
        delays = [calculate_backoff_delay(config, 0) for _ in range(100)]
        assert all(1.0 <= d <= 1.25 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1

    def test_jitter_multiplicative(self):
        """Test multiplicative jitter."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay=1.0,
            jitter_mode=JitterMode.MULTIPLICATIVE,
            jitter_factor=0.25,
        )
        # With multiplicative jitter: base * (1 +/- factor)
        delays = [calculate_backoff_delay(config, 0) for _ in range(100)]
        assert all(0.75 <= d <= 1.25 for d in delays)

    def test_jitter_full(self):
        """Test full jitter."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay=1.0,
            jitter_mode=JitterMode.FULL,
        )
        # With full jitter: random(0, delay)
        delays = [calculate_backoff_delay(config, 0) for _ in range(100)]
        assert all(0 <= d <= 1.0 for d in delays)


# =============================================================================
# with_retry Decorator Tests
# =============================================================================


class TestWithRetryAsync:
    """Test async retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=2, base_delay=0.01))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await always_fail()
        # Initial call + 2 retries = 3 total
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exception is not retried."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, retryable_exceptions=(ValueError,)))
        async def raise_runtime_error():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("not retryable")

        with pytest.raises(RuntimeError):
            await raise_runtime_error()
        # Should only be called once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback = MagicMock()
        call_count = 0

        @with_retry(RetryConfig(max_retries=2, base_delay=0.01, on_retry=callback))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry me")
            return "done"

        await flaky()
        # Callback should be called once (before second attempt)
        assert callback.call_count == 1
        args = callback.call_args[0]
        assert args[0] == 1  # attempt number
        assert isinstance(args[1], ConnectionError)  # exception
        # args[2] is delay

    @pytest.mark.asyncio
    async def test_should_retry_function(self):
        """Test custom should_retry function."""

        def only_retry_value_error(exc):
            return isinstance(exc, ValueError)

        call_count = 0

        @with_retry(
            RetryConfig(
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(Exception,),
                should_retry=only_retry_value_error,
            )
        )
        async def selective_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("retry this")
            if call_count == 2:
                raise RuntimeError("don't retry this")
            return "done"

        with pytest.raises(RuntimeError):
            await selective_retry()
        assert call_count == 2


# =============================================================================
# with_retry_sync Decorator Tests
# =============================================================================


class TestWithRetrySync:
    """Test sync retry decorator."""

    def test_sync_success_no_retry(self):
        """Test successful sync call doesn't retry."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_retries=3))
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_sync_retry_on_failure(self):
        """Test sync function is retried on failure."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_retries=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_sync_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        call_count = 0

        @with_retry_sync(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            always_fail()
        assert call_count == 3

    def test_sync_actual_delay(self):
        """Test sync decorator actually delays between retries."""
        call_times = []

        @with_retry_sync(
            RetryConfig(
                max_retries=2,
                base_delay=0.05,
                strategy=RetryStrategy.CONSTANT,
                jitter_mode=JitterMode.NONE,
            )
        )
        def track_time():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ConnectionError("retry")
            return "done"

        track_time()
        # Should have delays between calls
        if len(call_times) >= 2:
            delay = call_times[1] - call_times[0]
            assert delay >= 0.04  # Allow some tolerance


# =============================================================================
# Default Exceptions Tests
# =============================================================================


class TestDefaultRetryableExceptions:
    """Test default retryable exceptions."""

    def test_default_exceptions_includes_connection_error(self):
        """Test ConnectionError is retryable by default."""
        assert ConnectionError in DEFAULT_RETRYABLE_EXCEPTIONS

    def test_default_exceptions_includes_timeout_error(self):
        """Test TimeoutError is retryable by default."""
        assert TimeoutError in DEFAULT_RETRYABLE_EXCEPTIONS

    def test_default_exceptions_includes_os_error(self):
        """Test OSError is retryable by default."""
        assert OSError in DEFAULT_RETRYABLE_EXCEPTIONS
