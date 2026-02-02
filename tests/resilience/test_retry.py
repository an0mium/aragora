"""
Tests for the unified retry module.

Tests cover:
- RetryStrategy and JitterMode enum values
- RetryConfig defaults, __post_init__ jitter backward compat
- RetryConfig.calculate_delay() for each strategy
- RetryConfig.is_retryable() with default and custom should_retry
- RetryConfig.check_circuit_breaker() with and without circuit breaker
- RetryConfig.record_success/record_failure with circuit breaker
- calculate_backoff_delay for all strategies and jitter modes
- calculate_backoff_delay max_delay cap
- _fibonacci helper function
- ExponentialBackoff iterator and reset
- _is_rate_limit_exception, _is_server_error, _is_transient_error helpers
- PROVIDER_RETRY_POLICIES completeness
- get_provider_retry_config with known/unknown providers and overrides
- create_provider_config factory
- with_retry async decorator (success, retries, max_retries, circuit breaker, on_retry)
- with_retry_sync decorator (success, retries, max_retries, circuit breaker, on_retry)
- CircuitOpenError attributes
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import MagicMock, AsyncMock, call, patch

import pytest

from aragora.resilience.retry import (
    CircuitOpenError,
    ExponentialBackoff,
    JitterMode,
    PROVIDER_RETRY_POLICIES,
    RetryConfig,
    RetryStrategy,
    calculate_backoff_delay,
    create_provider_config,
    get_provider_retry_config,
    with_retry,
    with_retry_sync,
    _fibonacci,
    _is_rate_limit_exception,
    _is_server_error,
    _is_transient_error,
)


# ============================================================================
# RetryStrategy Enum Tests
# ============================================================================


class TestRetryStrategy:
    """Tests for RetryStrategy enum values."""

    def test_exponential_value(self):
        assert RetryStrategy.EXPONENTIAL == "exponential"
        assert RetryStrategy.EXPONENTIAL.value == "exponential"

    def test_linear_value(self):
        assert RetryStrategy.LINEAR == "linear"
        assert RetryStrategy.LINEAR.value == "linear"

    def test_constant_value(self):
        assert RetryStrategy.CONSTANT == "constant"
        assert RetryStrategy.CONSTANT.value == "constant"

    def test_fibonacci_value(self):
        assert RetryStrategy.FIBONACCI == "fibonacci"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"

    def test_is_str_enum(self):
        """RetryStrategy members are also strings."""
        assert isinstance(RetryStrategy.EXPONENTIAL, str)

    def test_all_members(self):
        members = set(RetryStrategy)
        assert members == {
            RetryStrategy.EXPONENTIAL,
            RetryStrategy.LINEAR,
            RetryStrategy.CONSTANT,
            RetryStrategy.FIBONACCI,
        }


# ============================================================================
# JitterMode Enum Tests
# ============================================================================


class TestJitterMode:
    """Tests for JitterMode enum values."""

    def test_none_value(self):
        assert JitterMode.NONE == "none"
        assert JitterMode.NONE.value == "none"

    def test_additive_value(self):
        assert JitterMode.ADDITIVE == "additive"
        assert JitterMode.ADDITIVE.value == "additive"

    def test_multiplicative_value(self):
        assert JitterMode.MULTIPLICATIVE == "multiplicative"
        assert JitterMode.MULTIPLICATIVE.value == "multiplicative"

    def test_full_value(self):
        assert JitterMode.FULL == "full"
        assert JitterMode.FULL.value == "full"

    def test_is_str_enum(self):
        assert isinstance(JitterMode.NONE, str)

    def test_all_members(self):
        members = set(JitterMode)
        assert members == {
            JitterMode.NONE,
            JitterMode.ADDITIVE,
            JitterMode.MULTIPLICATIVE,
            JitterMode.FULL,
        }


# ============================================================================
# RetryConfig Defaults Tests
# ============================================================================


class TestRetryConfigDefaults:
    """Tests for RetryConfig default values."""

    def test_default_max_retries(self):
        config = RetryConfig()
        assert config.max_retries == 3

    def test_default_base_delay(self):
        config = RetryConfig()
        assert config.base_delay == 0.1

    def test_default_max_delay(self):
        config = RetryConfig()
        assert config.max_delay == 30.0

    def test_default_strategy(self):
        config = RetryConfig()
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_default_jitter_mode(self):
        config = RetryConfig()
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_default_jitter_factor(self):
        config = RetryConfig()
        assert config.jitter_factor == 0.25

    def test_default_retryable_exceptions(self):
        config = RetryConfig()
        assert config.retryable_exceptions == (
            ConnectionError,
            TimeoutError,
            OSError,
            IOError,
        )

    def test_default_on_retry_is_none(self):
        config = RetryConfig()
        assert config.on_retry is None

    def test_default_should_retry_is_none(self):
        config = RetryConfig()
        assert config.should_retry is None

    def test_default_jitter_is_none(self):
        config = RetryConfig()
        assert config.jitter is None

    def test_default_circuit_breaker_is_none(self):
        config = RetryConfig()
        assert config.circuit_breaker is None

    def test_default_provider_name_is_none(self):
        config = RetryConfig()
        assert config.provider_name is None

    def test_default_non_retryable_status_codes(self):
        config = RetryConfig()
        assert config.non_retryable_status_codes == (400, 401, 403, 404, 422)


# ============================================================================
# RetryConfig __post_init__ Backward Compat Tests
# ============================================================================


class TestRetryConfigPostInit:
    """Tests for RetryConfig __post_init__ jitter backward compatibility."""

    def test_jitter_true_sets_multiplicative(self):
        config = RetryConfig(jitter=True)
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_jitter_false_sets_none_mode(self):
        config = RetryConfig(jitter=False)
        assert config.jitter_mode == JitterMode.NONE

    def test_jitter_none_preserves_default(self):
        config = RetryConfig(jitter=None)
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE  # default

    def test_jitter_none_preserves_explicit_mode(self):
        config = RetryConfig(jitter_mode=JitterMode.ADDITIVE, jitter=None)
        assert config.jitter_mode == JitterMode.ADDITIVE

    def test_jitter_true_overrides_explicit_mode(self):
        """When jitter=True is set, it overrides jitter_mode."""
        config = RetryConfig(jitter_mode=JitterMode.NONE, jitter=True)
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_jitter_false_overrides_explicit_mode(self):
        """When jitter=False is set, it overrides jitter_mode."""
        config = RetryConfig(jitter_mode=JitterMode.ADDITIVE, jitter=False)
        assert config.jitter_mode == JitterMode.NONE


# ============================================================================
# RetryConfig.calculate_delay() Tests
# ============================================================================


class TestRetryConfigCalculateDelay:
    """Tests for RetryConfig.calculate_delay() method."""

    def test_exponential_strategy_delay(self):
        """Exponential: delay = base_delay * 2^attempt."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter_mode=JitterMode.NONE,
        )
        assert config.calculate_delay(0) == 1.0  # 1.0 * 2^0 = 1.0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2^1 = 2.0
        assert config.calculate_delay(2) == 4.0  # 1.0 * 2^2 = 4.0
        assert config.calculate_delay(3) == 8.0  # 1.0 * 2^3 = 8.0

    def test_linear_strategy_delay(self):
        """Linear: delay = base_delay * (attempt + 1)."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR,
            jitter_mode=JitterMode.NONE,
        )
        assert config.calculate_delay(0) == 1.0  # 1.0 * 1 = 1.0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2 = 2.0
        assert config.calculate_delay(2) == 3.0  # 1.0 * 3 = 3.0

    def test_constant_strategy_delay(self):
        """Constant: delay = base_delay always."""
        config = RetryConfig(
            base_delay=1.5,
            strategy=RetryStrategy.CONSTANT,
            jitter_mode=JitterMode.NONE,
        )
        assert config.calculate_delay(0) == 1.5
        assert config.calculate_delay(1) == 1.5
        assert config.calculate_delay(5) == 1.5

    def test_fibonacci_strategy_delay(self):
        """Fibonacci: delay = base_delay * fib(attempt + 2)."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.FIBONACCI,
            jitter_mode=JitterMode.NONE,
        )
        # fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8
        assert config.calculate_delay(0) == 1.0  # fib(2) = 1
        assert config.calculate_delay(1) == 2.0  # fib(3) = 2
        assert config.calculate_delay(2) == 3.0  # fib(4) = 3
        assert config.calculate_delay(3) == 5.0  # fib(5) = 5
        assert config.calculate_delay(4) == 8.0  # fib(6) = 8

    def test_delay_respects_max_delay(self):
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter_mode=JitterMode.NONE,
        )
        # 2^10 * 1.0 = 1024.0, capped at 5.0
        assert config.calculate_delay(10) == 5.0


# ============================================================================
# RetryConfig.is_retryable() Tests
# ============================================================================


class TestRetryConfigIsRetryable:
    """Tests for RetryConfig.is_retryable() method."""

    def test_default_retryable_connection_error(self):
        config = RetryConfig()
        assert config.is_retryable(ConnectionError("failed")) is True

    def test_default_retryable_timeout_error(self):
        config = RetryConfig()
        assert config.is_retryable(TimeoutError("timed out")) is True

    def test_default_retryable_os_error(self):
        config = RetryConfig()
        assert config.is_retryable(OSError("os error")) is True

    def test_default_retryable_io_error(self):
        config = RetryConfig()
        assert config.is_retryable(IOError("io error")) is True

    def test_default_non_retryable_value_error(self):
        config = RetryConfig()
        assert config.is_retryable(ValueError("bad value")) is False

    def test_default_non_retryable_runtime_error(self):
        config = RetryConfig()
        assert config.is_retryable(RuntimeError("runtime")) is False

    def test_custom_should_retry_overrides_default(self):
        """When should_retry is provided, it overrides isinstance check."""

        def custom_check(exc: Exception) -> bool:
            return "retry_me" in str(exc)

        config = RetryConfig(should_retry=custom_check)
        # ValueError is not in retryable_exceptions, but should_retry says yes
        assert config.is_retryable(ValueError("retry_me")) is True
        # ConnectionError is in retryable_exceptions, but should_retry says no
        assert config.is_retryable(ConnectionError("dont retry")) is False

    def test_custom_retryable_exceptions(self):
        config = RetryConfig(retryable_exceptions=(ValueError, KeyError))
        assert config.is_retryable(ValueError("val")) is True
        assert config.is_retryable(KeyError("key")) is True
        assert config.is_retryable(ConnectionError("conn")) is False


# ============================================================================
# RetryConfig.check_circuit_breaker() Tests
# ============================================================================


class TestRetryConfigCheckCircuitBreaker:
    """Tests for RetryConfig.check_circuit_breaker() method."""

    def test_no_circuit_breaker_returns_true(self):
        config = RetryConfig()
        assert config.check_circuit_breaker() is True

    def test_circuit_breaker_allows_returns_true(self):
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = RetryConfig(circuit_breaker=mock_cb)
        assert config.check_circuit_breaker() is True
        mock_cb.can_proceed.assert_called_once()

    def test_circuit_breaker_denies_returns_false(self):
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        config = RetryConfig(circuit_breaker=mock_cb)
        assert config.check_circuit_breaker() is False
        mock_cb.can_proceed.assert_called_once()


# ============================================================================
# RetryConfig.record_success / record_failure Tests
# ============================================================================


class TestRetryConfigRecordSuccessFailure:
    """Tests for RetryConfig.record_success() and record_failure()."""

    def test_record_success_with_circuit_breaker(self):
        mock_cb = MagicMock()
        config = RetryConfig(circuit_breaker=mock_cb)
        config.record_success()
        mock_cb.record_success.assert_called_once()

    def test_record_success_without_circuit_breaker(self):
        config = RetryConfig()
        # Should not raise
        config.record_success()

    def test_record_failure_with_circuit_breaker(self):
        mock_cb = MagicMock()
        config = RetryConfig(circuit_breaker=mock_cb)
        exc = ConnectionError("fail")
        config.record_failure(exc)
        mock_cb.record_failure.assert_called_once()

    def test_record_failure_without_circuit_breaker(self):
        config = RetryConfig()
        # Should not raise
        config.record_failure(ConnectionError("fail"))

    def test_record_failure_without_exception(self):
        mock_cb = MagicMock()
        config = RetryConfig(circuit_breaker=mock_cb)
        config.record_failure()
        mock_cb.record_failure.assert_called_once()


# ============================================================================
# calculate_backoff_delay() Tests
# ============================================================================


class TestCalculateBackoffDelay:
    """Tests for the standalone calculate_backoff_delay function."""

    # --- Strategy tests with NONE jitter ---

    def test_exponential_strategy(self):
        """EXPONENTIAL: delay = base_delay * 2^attempt."""
        assert (
            calculate_backoff_delay(0, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 1.0
        )
        assert (
            calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 2.0
        )
        assert (
            calculate_backoff_delay(2, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 4.0
        )
        assert (
            calculate_backoff_delay(3, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 8.0
        )

    def test_exponential_with_custom_base(self):
        assert (
            calculate_backoff_delay(0, 0.5, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 0.5
        )
        assert (
            calculate_backoff_delay(1, 0.5, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 1.0
        )
        assert (
            calculate_backoff_delay(2, 0.5, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            == 2.0
        )

    def test_linear_strategy(self):
        """LINEAR: delay = base_delay * (attempt + 1)."""
        assert calculate_backoff_delay(0, 1.0, 100.0, RetryStrategy.LINEAR, JitterMode.NONE) == 1.0
        assert calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.LINEAR, JitterMode.NONE) == 2.0
        assert calculate_backoff_delay(2, 1.0, 100.0, RetryStrategy.LINEAR, JitterMode.NONE) == 3.0
        assert calculate_backoff_delay(3, 1.0, 100.0, RetryStrategy.LINEAR, JitterMode.NONE) == 4.0

    def test_constant_strategy(self):
        """CONSTANT: delay = base_delay always."""
        assert (
            calculate_backoff_delay(0, 2.0, 100.0, RetryStrategy.CONSTANT, JitterMode.NONE) == 2.0
        )
        assert (
            calculate_backoff_delay(1, 2.0, 100.0, RetryStrategy.CONSTANT, JitterMode.NONE) == 2.0
        )
        assert (
            calculate_backoff_delay(5, 2.0, 100.0, RetryStrategy.CONSTANT, JitterMode.NONE) == 2.0
        )
        assert (
            calculate_backoff_delay(10, 2.0, 100.0, RetryStrategy.CONSTANT, JitterMode.NONE) == 2.0
        )

    def test_fibonacci_strategy(self):
        """FIBONACCI: delay = base_delay * fib(attempt + 2)."""
        # fib sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
        # fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8
        assert (
            calculate_backoff_delay(0, 1.0, 100.0, RetryStrategy.FIBONACCI, JitterMode.NONE) == 1.0
        )
        assert (
            calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.FIBONACCI, JitterMode.NONE) == 2.0
        )
        assert (
            calculate_backoff_delay(2, 1.0, 100.0, RetryStrategy.FIBONACCI, JitterMode.NONE) == 3.0
        )
        assert (
            calculate_backoff_delay(3, 1.0, 100.0, RetryStrategy.FIBONACCI, JitterMode.NONE) == 5.0
        )
        assert (
            calculate_backoff_delay(4, 1.0, 100.0, RetryStrategy.FIBONACCI, JitterMode.NONE) == 8.0
        )

    # --- max_delay cap ---

    def test_max_delay_cap_exponential(self):
        delay = calculate_backoff_delay(20, 1.0, 5.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
        assert delay == 5.0

    def test_max_delay_cap_linear(self):
        delay = calculate_backoff_delay(100, 1.0, 10.0, RetryStrategy.LINEAR, JitterMode.NONE)
        assert delay == 10.0

    def test_max_delay_cap_fibonacci(self):
        delay = calculate_backoff_delay(20, 1.0, 5.0, RetryStrategy.FIBONACCI, JitterMode.NONE)
        assert delay == 5.0

    # --- Jitter mode tests ---

    def test_jitter_none_no_variation(self):
        """NONE jitter should produce consistent results."""
        results = set()
        for _ in range(20):
            d = calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE)
            results.add(d)
        assert len(results) == 1
        assert 2.0 in results

    @patch("aragora.resilience.retry.random.random", return_value=0.5)
    def test_jitter_additive(self, mock_rand):
        """ADDITIVE: delay = delay + random() * base_delay * jitter_factor."""
        # base_delay=1.0, attempt=1, EXPONENTIAL => base=2.0
        # jitter = 0.5 * 1.0 * 0.25 = 0.125
        # delay = 2.0 + 0.125 = 2.125
        delay = calculate_backoff_delay(
            1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.ADDITIVE, 0.25
        )
        assert delay == pytest.approx(2.125)

    @patch("aragora.resilience.retry.random.random", return_value=0.5)
    def test_jitter_multiplicative(self, mock_rand):
        """MULTIPLICATIVE: delay = delay * (1.0 + (random() * 2 - 1) * jitter_factor)."""
        # base_delay=1.0, attempt=1, EXPONENTIAL => base=2.0
        # factor = 1.0 + (0.5 * 2 - 1) * 0.25 = 1.0 + 0.0 = 1.0
        # delay = 2.0 * 1.0 = 2.0
        delay = calculate_backoff_delay(
            1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.MULTIPLICATIVE, 0.25
        )
        assert delay == pytest.approx(2.0)

    @patch("aragora.resilience.retry.random.random", return_value=0.0)
    def test_jitter_multiplicative_low(self, mock_rand):
        """MULTIPLICATIVE with random()=0: factor = 1.0 + (-1) * 0.25 = 0.75."""
        # delay = 2.0 * 0.75 = 1.5
        delay = calculate_backoff_delay(
            1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.MULTIPLICATIVE, 0.25
        )
        assert delay == pytest.approx(1.5)

    @patch("aragora.resilience.retry.random.random", return_value=1.0)
    def test_jitter_multiplicative_high(self, mock_rand):
        """MULTIPLICATIVE with random()=1: factor = 1.0 + 1.0 * 0.25 = 1.25."""
        # delay = 2.0 * 1.25 = 2.5
        delay = calculate_backoff_delay(
            1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.MULTIPLICATIVE, 0.25
        )
        assert delay == pytest.approx(2.5)

    @patch("aragora.resilience.retry.random.random", return_value=0.5)
    def test_jitter_full(self, mock_rand):
        """FULL: delay = random() * delay."""
        # base = 2.0, delay = 0.5 * 2.0 = 1.0
        delay = calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.FULL)
        assert delay == pytest.approx(1.0)

    @patch("aragora.resilience.retry.random.random", return_value=0.0)
    def test_jitter_full_zero(self, mock_rand):
        """FULL jitter with random()=0 should give 0."""
        delay = calculate_backoff_delay(1, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.FULL)
        assert delay == 0.0

    def test_delay_always_non_negative(self):
        """Delay should always be >= 0."""
        random.seed(42)
        for _ in range(100):
            for strategy in RetryStrategy:
                for jitter_mode in JitterMode:
                    d = calculate_backoff_delay(
                        attempt=random.randint(0, 10),
                        base_delay=random.uniform(0.01, 5.0),
                        max_delay=random.uniform(1.0, 60.0),
                        strategy=strategy,
                        jitter_mode=jitter_mode,
                        jitter_factor=random.uniform(0.0, 1.0),
                    )
                    assert d >= 0, f"Negative delay {d} for {strategy}/{jitter_mode}"


# ============================================================================
# _fibonacci() Tests
# ============================================================================


class TestFibonacci:
    """Tests for the _fibonacci helper function."""

    def test_fibonacci_zero(self):
        assert _fibonacci(0) == 0

    def test_fibonacci_one(self):
        assert _fibonacci(1) == 1

    def test_fibonacci_two(self):
        assert _fibonacci(2) == 1

    def test_fibonacci_three(self):
        assert _fibonacci(3) == 2

    def test_fibonacci_four(self):
        assert _fibonacci(4) == 3

    def test_fibonacci_five(self):
        assert _fibonacci(5) == 5

    def test_fibonacci_six(self):
        assert _fibonacci(6) == 8

    def test_fibonacci_seven(self):
        assert _fibonacci(7) == 13

    def test_fibonacci_ten(self):
        assert _fibonacci(10) == 55

    def test_fibonacci_negative(self):
        assert _fibonacci(-1) == 0
        assert _fibonacci(-10) == 0

    def test_fibonacci_large(self):
        assert _fibonacci(20) == 6765


# ============================================================================
# ExponentialBackoff Tests
# ============================================================================


class TestExponentialBackoff:
    """Tests for the ExponentialBackoff iterator class."""

    def test_iteration_count(self):
        """Should yield exactly max_retries values."""
        backoff = ExponentialBackoff(max_retries=3, jitter=False)
        delays = list(backoff)
        assert len(delays) == 3

    def test_iteration_zero_retries(self):
        backoff = ExponentialBackoff(max_retries=0, jitter=False)
        delays = list(backoff)
        assert len(delays) == 0

    def test_delays_are_exponential(self):
        """Delays should follow exponential pattern without jitter."""
        backoff = ExponentialBackoff(max_retries=4, base_delay=1.0, max_delay=100.0, jitter=False)
        delays = list(backoff)
        assert delays[0] == pytest.approx(1.0)  # 1.0 * 2^0
        assert delays[1] == pytest.approx(2.0)  # 1.0 * 2^1
        assert delays[2] == pytest.approx(4.0)  # 1.0 * 2^2
        assert delays[3] == pytest.approx(8.0)  # 1.0 * 2^3

    def test_delays_respect_max_delay(self):
        backoff = ExponentialBackoff(max_retries=10, base_delay=1.0, max_delay=5.0, jitter=False)
        delays = list(backoff)
        for d in delays:
            assert d <= 5.0

    def test_jitter_produces_variation(self):
        """With jitter=True, delays should have some variation."""
        random.seed(12345)
        backoff = ExponentialBackoff(max_retries=5, base_delay=1.0, jitter=True)
        delays = list(backoff)
        # With jitter, not all delays will be exact powers of 2
        # At least one should differ from exact exponential
        exact = [1.0 * (2**i) for i in range(5)]
        has_variation = any(abs(delays[i] - exact[i]) > 0.001 for i in range(5))
        assert has_variation, "Jitter should produce some variation"

    def test_reset(self):
        """After reset, iteration should start from beginning."""
        backoff = ExponentialBackoff(max_retries=3, base_delay=1.0, jitter=False)
        first_run = list(backoff)
        backoff.reset()
        second_run = list(backoff)
        assert first_run == second_run

    def test_iter_returns_self(self):
        backoff = ExponentialBackoff(max_retries=3)
        assert iter(backoff) is backoff

    def test_iter_resets_attempt(self):
        """Calling __iter__ should reset the attempt counter."""
        backoff = ExponentialBackoff(max_retries=2, jitter=False)
        # Exhaust the iterator
        list(backoff)
        # Re-iterate
        delays = list(backoff)
        assert len(delays) == 2

    def test_default_values(self):
        backoff = ExponentialBackoff()
        assert backoff.max_retries == 3
        assert backoff.base_delay == 0.1
        assert backoff.max_delay == 30.0
        assert backoff.jitter is True

    def test_stop_iteration_raised(self):
        backoff = ExponentialBackoff(max_retries=1, jitter=False)
        it = iter(backoff)
        next(it)  # first value
        with pytest.raises(StopIteration):
            next(it)


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestIsRateLimitException:
    """Tests for _is_rate_limit_exception helper."""

    def test_rate_limit_keywords(self):
        assert _is_rate_limit_exception(Exception("Rate limit exceeded")) is True

    def test_429_code(self):
        assert _is_rate_limit_exception(Exception("Error 429: too many")) is True

    def test_too_many_requests(self):
        assert _is_rate_limit_exception(Exception("too many requests")) is True

    def test_quota_exceeded(self):
        assert _is_rate_limit_exception(Exception("Quota exceeded")) is True

    def test_non_rate_limit(self):
        assert _is_rate_limit_exception(Exception("Connection refused")) is False

    def test_empty_message(self):
        assert _is_rate_limit_exception(Exception("")) is False

    def test_partial_rate_no_limit(self):
        """'rate' alone without 'limit' should not match (unless other keywords present)."""
        assert _is_rate_limit_exception(Exception("rate of change")) is False


class TestIsServerError:
    """Tests for _is_server_error helper."""

    def test_500_error(self):
        assert _is_server_error(Exception("Server returned 500")) is True

    def test_502_error(self):
        assert _is_server_error(Exception("502 Bad Gateway")) is True

    def test_503_error(self):
        assert _is_server_error(Exception("503 Service Unavailable")) is True

    def test_504_error(self):
        assert _is_server_error(Exception("504 Gateway Timeout")) is True

    def test_400_error_not_server(self):
        assert _is_server_error(Exception("400 Bad Request")) is False

    def test_404_error_not_server(self):
        assert _is_server_error(Exception("404 Not Found")) is False

    def test_no_error_code(self):
        assert _is_server_error(Exception("Something went wrong")) is False


class TestIsTransientError:
    """Tests for _is_transient_error helper."""

    def test_connection_error(self):
        assert _is_transient_error(ConnectionError("refused")) is True

    def test_timeout_error(self):
        assert _is_transient_error(TimeoutError("timed out")) is True

    def test_os_error(self):
        assert _is_transient_error(OSError("network error")) is True

    def test_io_error(self):
        assert _is_transient_error(IOError("io error")) is True

    def test_rate_limit_error(self):
        assert _is_transient_error(Exception("Rate limit exceeded")) is True

    def test_server_500_error(self):
        assert _is_transient_error(Exception("500 Internal Server Error")) is True

    def test_non_transient_error(self):
        assert _is_transient_error(ValueError("bad value")) is False

    def test_non_transient_key_error(self):
        assert _is_transient_error(KeyError("missing key")) is False


# ============================================================================
# PROVIDER_RETRY_POLICIES Tests
# ============================================================================


class TestProviderRetryPolicies:
    """Tests for PROVIDER_RETRY_POLICIES dictionary."""

    EXPECTED_PROVIDERS = [
        "anthropic",
        "openai",
        "mistral",
        "grok",
        "openrouter",
        "gemini",
        "knowledge_mound",
        "control_plane",
        "memory",
    ]

    def test_has_expected_providers(self):
        for provider in self.EXPECTED_PROVIDERS:
            assert provider in PROVIDER_RETRY_POLICIES, f"Missing provider: {provider}"

    def test_provider_count(self):
        assert len(PROVIDER_RETRY_POLICIES) == 9

    def test_all_values_are_retry_configs(self):
        for name, config in PROVIDER_RETRY_POLICIES.items():
            assert isinstance(config, RetryConfig), f"{name} is not a RetryConfig"

    def test_all_have_provider_name(self):
        for name, config in PROVIDER_RETRY_POLICIES.items():
            assert config.provider_name == name, (
                f"Provider name mismatch: config.provider_name={config.provider_name}, key={name}"
            )

    def test_anthropic_conservative(self):
        """Anthropic should have conservative retry settings."""
        config = PROVIDER_RETRY_POLICIES["anthropic"]
        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0

    def test_openai_moderate(self):
        config = PROVIDER_RETRY_POLICIES["openai"]
        assert config.max_retries == 4
        assert config.base_delay == 1.0

    def test_memory_fast(self):
        """Memory operations should have shorter delays."""
        config = PROVIDER_RETRY_POLICIES["memory"]
        assert config.base_delay == 0.3
        assert config.max_delay == 15.0

    def test_all_use_exponential_strategy(self):
        for name, config in PROVIDER_RETRY_POLICIES.items():
            assert config.strategy == RetryStrategy.EXPONENTIAL, (
                f"{name} does not use EXPONENTIAL strategy"
            )

    def test_all_use_multiplicative_jitter(self):
        for name, config in PROVIDER_RETRY_POLICIES.items():
            assert config.jitter_mode == JitterMode.MULTIPLICATIVE, (
                f"{name} does not use MULTIPLICATIVE jitter"
            )

    def test_all_have_should_retry(self):
        for name, config in PROVIDER_RETRY_POLICIES.items():
            assert config.should_retry is _is_transient_error, (
                f"{name} does not use _is_transient_error as should_retry"
            )


# ============================================================================
# get_provider_retry_config() Tests
# ============================================================================


class TestGetProviderRetryConfig:
    """Tests for get_provider_retry_config function."""

    def test_known_provider(self):
        config = get_provider_retry_config("anthropic")
        assert config.provider_name == "anthropic"
        assert config.base_delay == 2.0

    def test_unknown_provider_returns_default(self):
        config = get_provider_retry_config("unknown_provider")
        assert config.provider_name == "unknown_provider"
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0

    def test_with_circuit_breaker(self):
        mock_cb = MagicMock()
        config = get_provider_retry_config("openai", circuit_breaker=mock_cb)
        assert config.circuit_breaker is mock_cb

    def test_with_overrides(self):
        config = get_provider_retry_config("anthropic", max_retries=10, base_delay=5.0)
        assert config.max_retries == 10
        assert config.base_delay == 5.0
        # Other fields should come from the anthropic policy
        assert config.provider_name == "anthropic"

    def test_override_strategy(self):
        config = get_provider_retry_config("openai", strategy=RetryStrategy.LINEAR)
        assert config.strategy == RetryStrategy.LINEAR

    def test_unknown_provider_with_overrides(self):
        config = get_provider_retry_config("custom", max_retries=7)
        assert config.provider_name == "custom"
        assert config.max_retries == 7

    def test_without_circuit_breaker_is_none(self):
        config = get_provider_retry_config("openai")
        assert config.circuit_breaker is None


# ============================================================================
# create_provider_config() Tests
# ============================================================================


class TestCreateProviderConfig:
    """Tests for create_provider_config factory function."""

    def test_basic_creation(self):
        config = create_provider_config("test_provider")
        assert config.provider_name == "test_provider"
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE
        assert config.jitter_factor == 0.25

    def test_custom_values(self):
        config = create_provider_config(
            "custom",
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.LINEAR,
            jitter_factor=0.5,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.strategy == RetryStrategy.LINEAR
        assert config.jitter_factor == 0.5

    def test_with_circuit_breaker(self):
        mock_cb = MagicMock()
        config = create_provider_config("test", circuit_breaker=mock_cb)
        assert config.circuit_breaker is mock_cb

    def test_should_retry_is_transient(self):
        config = create_provider_config("test")
        assert config.should_retry is _is_transient_error

    def test_retryable_exceptions(self):
        config = create_provider_config("test")
        assert config.retryable_exceptions == (ConnectionError, TimeoutError, OSError, IOError)


# ============================================================================
# CircuitOpenError Tests
# ============================================================================


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_default_message(self):
        err = CircuitOpenError()
        assert str(err) == "Circuit breaker is open"

    def test_custom_message(self):
        err = CircuitOpenError("Custom message")
        assert str(err) == "Custom message"

    def test_provider_attribute(self):
        err = CircuitOpenError(provider="anthropic")
        assert err.provider == "anthropic"

    def test_cooldown_remaining_attribute(self):
        err = CircuitOpenError(cooldown_remaining=15.5)
        assert err.cooldown_remaining == 15.5

    def test_all_attributes(self):
        err = CircuitOpenError(
            "Circuit open for openai",
            provider="openai",
            cooldown_remaining=30.0,
        )
        assert str(err) == "Circuit open for openai"
        assert err.provider == "openai"
        assert err.cooldown_remaining == 30.0

    def test_default_provider_is_none(self):
        err = CircuitOpenError()
        assert err.provider is None

    def test_default_cooldown_is_none(self):
        err = CircuitOpenError()
        assert err.cooldown_remaining is None

    def test_is_exception(self):
        assert issubclass(CircuitOpenError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(CircuitOpenError) as exc_info:
            raise CircuitOpenError("test", provider="test_provider", cooldown_remaining=10.0)
        assert exc_info.value.provider == "test_provider"
        assert exc_info.value.cooldown_remaining == 10.0


# ============================================================================
# with_retry Async Decorator Tests
# ============================================================================


class TestWithRetryAsync:
    """Tests for the with_retry async decorator."""

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_success_on_first_try(self, mock_sleep):
        """Should return result without retrying on success."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_retryable_exception(self, mock_sleep):
        """Should retry on retryable exceptions."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, jitter_mode=JitterMode.NONE))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection failed")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_raises_after_max_retries(self, mock_sleep):
        """Should raise the exception after exhausting retries."""

        @with_retry(RetryConfig(max_retries=2))
        async def always_fail():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError, match="always fails"):
            await always_fail()
        # Initial try + 2 retries = 3 total calls, 2 sleeps
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_non_retryable_exception_raises_immediately(self, mock_sleep):
        """Non-retryable exceptions should not trigger retries."""

        @with_retry(RetryConfig(max_retries=3))
        async def bad_value():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await bad_value()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_circuit_breaker_open_raises(self, mock_sleep):
        """Should raise CircuitOpenError when circuit breaker is open."""
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        config = RetryConfig(max_retries=3, circuit_breaker=mock_cb, provider_name="test")

        @with_retry(config)
        async def protected():
            return "ok"

        with pytest.raises(CircuitOpenError):
            await protected()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_circuit_breaker_records_success(self, mock_sleep):
        """Circuit breaker should record success on successful call."""
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = RetryConfig(max_retries=3, circuit_breaker=mock_cb)

        @with_retry(config)
        async def succeed():
            return "ok"

        await succeed()
        mock_cb.record_success.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_circuit_breaker_records_failure(self, mock_sleep):
        """Circuit breaker should record failure when retries are exhausted."""
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = RetryConfig(max_retries=1, circuit_breaker=mock_cb)

        @with_retry(config)
        async def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await always_fail()
        mock_cb.record_failure.assert_called()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_on_retry_callback(self, mock_sleep):
        """on_retry callback should be called on each retry."""
        callback = MagicMock()
        call_count = 0

        config = RetryConfig(
            max_retries=3,
            on_retry=callback,
            jitter_mode=JitterMode.NONE,
        )

        @with_retry(config)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("fail")
            return "ok"

        await flaky()
        assert callback.call_count == 2
        # Verify callback receives (attempt, exception, delay)
        for c in callback.call_args_list:
            args = c[0]
            assert isinstance(args[0], int)  # attempt
            assert isinstance(args[1], Exception)  # exception
            assert isinstance(args[2], float)  # delay

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_preserves_function_metadata(self, mock_sleep):
        """Decorated function should preserve __name__ and __doc__."""

        @with_retry(RetryConfig())
        async def my_function():
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_with_keyword_arguments(self, mock_sleep):
        """Should work with keyword config arguments."""

        @with_retry(max_retries=2, base_delay=0.5)
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_with_provider_argument(self, mock_sleep):
        """Should use provider-specific config."""
        call_count = 0

        @with_retry(provider="anthropic")
        async def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "ok"

        result = await api_call()
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_passes_arguments_to_function(self, mock_sleep):
        """Decorated function should receive all args and kwargs."""

        @with_retry(RetryConfig(max_retries=1))
        async def add(a, b, extra=0):
            return a + b + extra

        result = await add(1, 2, extra=3)
        assert result == 6

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_is_retryable_check(self, mock_sleep):
        """should_retry returning False should prevent retry even for retryable exception types."""
        config = RetryConfig(
            max_retries=3,
            should_retry=lambda e: False,  # Never retry
        )

        @with_retry(config)
        async def flaky():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await flaky()
        mock_sleep.assert_not_called()


# ============================================================================
# with_retry_sync Decorator Tests
# ============================================================================


class TestWithRetrySync:
    """Tests for the with_retry_sync synchronous decorator."""

    @patch("aragora.resilience.retry.time.sleep")
    def test_success_on_first_try(self, mock_sleep):
        call_count = 0

        @with_retry_sync(RetryConfig(max_retries=3))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1
        mock_sleep.assert_not_called()

    @patch("aragora.resilience.retry.time.sleep")
    def test_retries_on_retryable_exception(self, mock_sleep):
        call_count = 0

        @with_retry_sync(RetryConfig(max_retries=3, jitter_mode=JitterMode.NONE))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timed out")
            return "recovered"

        result = flaky()
        assert result == "recovered"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("aragora.resilience.retry.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        @with_retry_sync(RetryConfig(max_retries=2))
        def always_fail():
            raise OSError("always fails")

        with pytest.raises(OSError, match="always fails"):
            always_fail()
        assert mock_sleep.call_count == 2

    @patch("aragora.resilience.retry.time.sleep")
    def test_non_retryable_exception_raises_immediately(self, mock_sleep):
        @with_retry_sync(RetryConfig(max_retries=3))
        def bad():
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            bad()
        mock_sleep.assert_not_called()

    @patch("aragora.resilience.retry.time.sleep")
    def test_circuit_breaker_open_raises(self, mock_sleep):
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        config = RetryConfig(max_retries=3, circuit_breaker=mock_cb, provider_name="test_sync")

        @with_retry_sync(config)
        def protected():
            return "ok"

        with pytest.raises(CircuitOpenError):
            protected()

    @patch("aragora.resilience.retry.time.sleep")
    def test_circuit_breaker_records_success(self, mock_sleep):
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = RetryConfig(max_retries=3, circuit_breaker=mock_cb)

        @with_retry_sync(config)
        def succeed():
            return "ok"

        succeed()
        mock_cb.record_success.assert_called_once()

    @patch("aragora.resilience.retry.time.sleep")
    def test_circuit_breaker_records_failure(self, mock_sleep):
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = RetryConfig(max_retries=1, circuit_breaker=mock_cb)

        @with_retry_sync(config)
        def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            always_fail()
        mock_cb.record_failure.assert_called()

    @patch("aragora.resilience.retry.time.sleep")
    def test_on_retry_callback(self, mock_sleep):
        callback = MagicMock()
        call_count = 0

        config = RetryConfig(
            max_retries=3,
            on_retry=callback,
            jitter_mode=JitterMode.NONE,
        )

        @with_retry_sync(config)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("fail")
            return "ok"

        flaky()
        assert callback.call_count == 2
        for c in callback.call_args_list:
            args = c[0]
            assert isinstance(args[0], int)
            assert isinstance(args[1], Exception)
            assert isinstance(args[2], float)

    @patch("aragora.resilience.retry.time.sleep")
    def test_preserves_function_metadata(self, mock_sleep):
        @with_retry_sync(RetryConfig())
        def my_sync_function():
            """Sync docstring."""
            return "ok"

        assert my_sync_function.__name__ == "my_sync_function"
        assert my_sync_function.__doc__ == "Sync docstring."

    @patch("aragora.resilience.retry.time.sleep")
    def test_with_keyword_arguments(self, mock_sleep):
        @with_retry_sync(max_retries=2, base_delay=0.5)
        def succeed():
            return "ok"

        result = succeed()
        assert result == "ok"

    @patch("aragora.resilience.retry.time.sleep")
    def test_with_provider_argument(self, mock_sleep):
        call_count = 0

        @with_retry_sync(provider="openai")
        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "ok"

        result = api_call()
        assert result == "ok"
        assert call_count == 2

    @patch("aragora.resilience.retry.time.sleep")
    def test_passes_arguments_to_function(self, mock_sleep):
        @with_retry_sync(RetryConfig(max_retries=1))
        def multiply(a, b, factor=1):
            return a * b * factor

        result = multiply(3, 4, factor=2)
        assert result == 24

    @patch("aragora.resilience.retry.time.sleep")
    def test_is_retryable_check(self, mock_sleep):
        """should_retry returning False should prevent retry."""
        config = RetryConfig(
            max_retries=3,
            should_retry=lambda e: False,
        )

        @with_retry_sync(config)
        def flaky():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            flaky()
        mock_sleep.assert_not_called()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_provider_config_with_retry_decorator(self, mock_sleep):
        """get_provider_retry_config result used with with_retry."""
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True
        config = get_provider_retry_config("anthropic", circuit_breaker=mock_cb)
        call_count = 0

        @with_retry(config)
        async def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await api_call()
        assert result == "success"
        mock_cb.record_success.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.resilience.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_create_provider_config_with_retry(self, mock_sleep):
        """create_provider_config result used with with_retry."""
        config = create_provider_config("custom", max_retries=2)
        call_count = 0

        @with_retry(config)
        async def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "done"

        result = await api_call()
        assert result == "done"

    def test_exponential_backoff_with_calculate_delay(self):
        """ExponentialBackoff should match calculate_backoff_delay for same params."""
        backoff = ExponentialBackoff(max_retries=5, base_delay=1.0, max_delay=100.0, jitter=False)
        delays = list(backoff)
        for i, delay in enumerate(delays):
            expected = calculate_backoff_delay(
                i, 1.0, 100.0, RetryStrategy.EXPONENTIAL, JitterMode.NONE
            )
            assert delay == pytest.approx(expected), f"Mismatch at attempt {i}"

    @patch("aragora.resilience.retry.time.sleep")
    def test_sync_retry_with_circuit_breaker_full_flow(self, mock_sleep):
        """Full flow: circuit open -> error, circuit closed -> retry -> success."""
        mock_cb = MagicMock()

        # First call: circuit is open
        mock_cb.can_proceed.return_value = False
        config1 = RetryConfig(max_retries=3, circuit_breaker=mock_cb, provider_name="test")

        @with_retry_sync(config1)
        def protected():
            return "ok"

        with pytest.raises(CircuitOpenError):
            protected()

        # Second call: circuit allows, operation succeeds
        mock_cb.can_proceed.return_value = True
        result = protected()
        assert result == "ok"
        mock_cb.record_success.assert_called()
