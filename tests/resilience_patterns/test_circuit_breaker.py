"""
Tests for Circuit Breaker Pattern.

Tests the circuit breaker implementation including:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure counting and thresholds
- Cooldown and recovery periods
- Thread safety
- Decorators for sync and async functions
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from aragora.resilience_patterns.circuit_breaker import (
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
    with_circuit_breaker,
    with_circuit_breaker_sync,
)


# =============================================================================
# CircuitState Tests
# =============================================================================


class TestCircuitState:
    """Test CircuitState enum."""

    def test_state_values(self):
        """Test circuit state values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_state_string_enum(self):
        """Test CircuitState is a string enum."""
        assert isinstance(CircuitState.CLOSED, str)
        # The value attribute contains the actual string
        assert CircuitState.CLOSED.value == "closed"


# =============================================================================
# CircuitBreakerOpenError Tests
# =============================================================================


class TestCircuitBreakerOpenError:
    """Test CircuitBreakerOpenError exception."""

    def test_error_creation(self):
        """Test error creation with default message."""
        error = CircuitBreakerOpenError()
        assert str(error) == "Circuit breaker is open"
        assert error.circuit_name is None
        assert error.cooldown_remaining is None

    def test_error_with_details(self):
        """Test error creation with details."""
        error = CircuitBreakerOpenError(
            message="Service unavailable",
            circuit_name="my_service",
            cooldown_remaining=30.5,
        )
        assert str(error) == "Service unavailable"
        assert error.circuit_name == "my_service"
        assert error.cooldown_remaining == 30.5


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.cooldown_seconds == 60.0
        assert config.half_open_max_requests == 3
        assert config.failure_rate_threshold is None
        assert config.window_size == 60.0
        assert config.excluded_exceptions == ()
        assert config.on_state_change is None

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            cooldown_seconds=120.0,
            half_open_max_requests=5,
            failure_rate_threshold=0.5,
            window_size=30.0,
            excluded_exceptions=(ValueError,),
            on_state_change=callback,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.cooldown_seconds == 120.0
        assert config.half_open_max_requests == 5
        assert config.failure_rate_threshold == 0.5
        assert config.window_size == 30.0
        assert config.excluded_exceptions == (ValueError,)
        assert config.on_state_change == callback


# =============================================================================
# CircuitBreakerStats Tests
# =============================================================================


class TestCircuitBreakerStats:
    """Test CircuitBreakerStats dataclass."""

    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = CircuitBreakerStats(
            state=CircuitState.CLOSED,
            failure_count=2,
            success_count=10,
            last_failure_time=1000.0,
            last_success_time=1001.0,
            consecutive_failures=0,
            consecutive_successes=5,
            total_requests=12,
            total_failures=2,
            cooldown_remaining=None,
        )
        result = stats.to_dict()

        assert result["state"] == "closed"
        assert result["failure_count"] == 2
        assert result["success_count"] == 10
        assert result["last_failure_time"] == 1000.0
        assert result["total_requests"] == 12


# =============================================================================
# BaseCircuitBreaker Tests
# =============================================================================


class TestBaseCircuitBreakerInitialization:
    """Test BaseCircuitBreaker initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        cb = BaseCircuitBreaker("test_service")
        assert cb.name == "test_service"
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = BaseCircuitBreaker("test_service", config)
        assert cb.config.failure_threshold == 10


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions."""

    def test_stays_closed_on_success(self):
        """Test circuit stays closed on successful operations."""
        cb = BaseCircuitBreaker("test")
        cb.record_success()
        cb.record_success()
        cb.record_success()
        assert cb.is_closed is True

    def test_opens_on_failure_threshold(self):
        """Test circuit opens when failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = BaseCircuitBreaker("test", config)

        # Record failures up to threshold
        cb.record_failure(Exception("fail 1"))
        assert cb.is_closed is True
        cb.record_failure(Exception("fail 2"))
        assert cb.is_closed is True
        cb.record_failure(Exception("fail 3"))
        assert cb.is_open is True

    def test_transitions_to_half_open_after_cooldown(self):
        """Test circuit transitions to half-open after cooldown."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1)
        cb = BaseCircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure(Exception("fail"))
        assert cb.is_open is True

        # Wait for cooldown
        time.sleep(0.15)

        # Check can_execute (should transition to half-open)
        assert cb.can_execute() is True
        assert cb.is_half_open is True

    def test_closes_from_half_open_on_success(self):
        """Test circuit closes from half-open after successes."""
        config = CircuitBreakerConfig(
            failure_threshold=1, cooldown_seconds=0.1, success_threshold=2
        )
        cb = BaseCircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure(Exception("fail"))
        time.sleep(0.15)

        # Transition to half-open
        cb.can_execute()
        assert cb.is_half_open is True

        # Record successes to close
        cb.record_success()
        assert cb.is_half_open is True  # Still half-open
        cb.record_success()
        assert cb.is_closed is True  # Now closed

    def test_reopens_from_half_open_on_failure(self):
        """Test circuit reopens from half-open on failure."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1)
        cb = BaseCircuitBreaker("test", config)

        # Open and transition to half-open
        cb.record_failure(Exception("fail"))
        time.sleep(0.15)
        cb.can_execute()
        assert cb.is_half_open is True

        # Failure in half-open should reopen
        cb.record_failure(Exception("fail again"))
        assert cb.is_open is True


class TestCircuitBreakerCanExecute:
    """Test can_execute method."""

    def test_can_execute_when_closed(self):
        """Test can_execute returns True when closed."""
        cb = BaseCircuitBreaker("test")
        assert cb.can_execute() is True

    def test_cannot_execute_when_open(self):
        """Test can_execute returns False when open."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=60.0)
        cb = BaseCircuitBreaker("test", config)
        cb.record_failure(Exception("fail"))
        assert cb.can_execute() is False

    def test_can_execute_when_half_open_below_limit(self):
        """Test can_execute works in half-open with request limit."""
        config = CircuitBreakerConfig(
            failure_threshold=1, cooldown_seconds=0.1, half_open_max_requests=2
        )
        cb = BaseCircuitBreaker("test", config)

        # Open and wait for half-open
        cb.record_failure(Exception("fail"))
        time.sleep(0.15)

        # First two requests should be allowed
        assert cb.can_execute() is True
        assert cb.can_execute() is True


class TestCircuitBreakerGetStats:
    """Test get_stats method."""

    def test_get_stats_initial(self):
        """Test initial stats."""
        cb = BaseCircuitBreaker("test")
        stats = cb.get_stats()

        assert stats.state == CircuitState.CLOSED
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.total_requests == 0

    def test_get_stats_after_operations(self):
        """Test stats after operations."""
        cb = BaseCircuitBreaker("test")
        cb.record_success()
        cb.record_success()
        cb.record_failure(Exception("fail"))

        stats = cb.get_stats()
        assert stats.success_count >= 2
        assert stats.failure_count >= 1


class TestCircuitBreakerReset:
    """Test reset method."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = BaseCircuitBreaker("test", config)

        # Accumulate state
        cb.record_failure(Exception("fail"))
        cb.record_failure(Exception("fail"))
        assert cb.is_open is True

        # Reset
        cb.reset()
        assert cb.is_closed is True
        stats = cb.get_stats()
        assert stats.failure_count == 0
        assert stats.consecutive_failures == 0


class TestCircuitBreakerExcludedExceptions:
    """Test excluded exceptions handling."""

    def test_excluded_exception_not_counted(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(failure_threshold=2, excluded_exceptions=(ValueError,))
        cb = BaseCircuitBreaker("test", config)

        # ValueError should be excluded
        cb.record_failure(ValueError("excluded"))
        cb.record_failure(ValueError("excluded"))
        assert cb.is_closed is True

        # Non-excluded exception should count
        cb.record_failure(RuntimeError("not excluded"))
        cb.record_failure(RuntimeError("not excluded"))
        assert cb.is_open is True


class TestCircuitBreakerCallback:
    """Test state change callback."""

    def test_callback_on_state_change(self):
        """Test callback is called on state change."""
        callback = MagicMock()
        config = CircuitBreakerConfig(failure_threshold=1, on_state_change=callback)
        cb = BaseCircuitBreaker("test", config)

        # Trigger state change
        cb.record_failure(Exception("fail"))

        # Callback should be called
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"  # name
        assert args[1] == CircuitState.CLOSED  # old state
        assert args[2] == CircuitState.OPEN  # new state


class TestCircuitBreakerThreadSafety:
    """Test thread safety."""

    def test_concurrent_operations(self):
        """Test circuit breaker handles concurrent operations."""
        config = CircuitBreakerConfig(failure_threshold=100)
        cb = BaseCircuitBreaker("test", config)
        errors = []

        def record_operations():
            try:
                for _ in range(50):
                    if cb.can_execute():
                        cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_operations) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All operations should be tracked
        stats = cb.get_stats()
        assert stats.total_requests == 500


# =============================================================================
# Decorator Tests
# =============================================================================


class TestWithCircuitBreaker:
    """Test async circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through_success(self):
        """Test decorator passes through successful calls."""
        call_count = 0

        @with_circuit_breaker("test_async", CircuitBreakerConfig())
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_handles_failure(self):
        """Test decorator handles and records failures."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @with_circuit_breaker("test_async_fail", config)
        async def failing_func():
            raise ValueError("intentional failure")

        # First two calls should raise ValueError
        with pytest.raises(ValueError):
            await failing_func()
        with pytest.raises(ValueError):
            await failing_func()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await failing_func()

    @pytest.mark.asyncio
    async def test_decorator_excluded_exceptions(self):
        """Test decorator respects excluded exceptions."""
        config = CircuitBreakerConfig(failure_threshold=1, excluded_exceptions=(ValueError,))

        @with_circuit_breaker("test_async_excluded", config)
        async def excluded_exception_func():
            raise ValueError("excluded")

        # ValueError should not open circuit
        with pytest.raises(ValueError):
            await excluded_exception_func()
        with pytest.raises(ValueError):
            await excluded_exception_func()

        # Should still be able to call (circuit not open)
        with pytest.raises(ValueError):
            await excluded_exception_func()


class TestWithCircuitBreakerSync:
    """Test sync circuit breaker decorator."""

    def test_sync_decorator_passes_through_success(self):
        """Test sync decorator passes through successful calls."""

        @with_circuit_breaker_sync("test_sync", CircuitBreakerConfig())
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_sync_decorator_handles_failure(self):
        """Test sync decorator handles and records failures."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @with_circuit_breaker_sync("test_sync_fail", config)
        def failing_func():
            raise ValueError("intentional failure")

        # First two calls should raise ValueError
        with pytest.raises(ValueError):
            failing_func()
        with pytest.raises(ValueError):
            failing_func()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            failing_func()
