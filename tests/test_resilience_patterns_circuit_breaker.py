"""
Tests for resilience_patterns.circuit_breaker module.

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold triggering
- Cooldown period
- Half-open recovery
- State change callbacks
- Decorator usage
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.resilience_patterns import (
    CircuitState,
    CircuitBreakerConfig,
    BaseCircuitBreaker,
    with_circuit_breaker,
)
from aragora.resilience_patterns.circuit_breaker import CircuitBreakerOpenError


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self):
        """Test all state values exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.cooldown_seconds == 60.0
        assert config.half_open_max_calls == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            cooldown_seconds=30.0,
            half_open_max_calls=2,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.cooldown_seconds == 30.0
        assert config.half_open_max_calls == 2


class TestBaseCircuitBreaker:
    """Tests for BaseCircuitBreaker class."""

    def test_initial_state(self):
        """Test initial state is CLOSED."""
        cb = BaseCircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_record_success_in_closed_state(self):
        """Test recording success in closed state."""
        cb = BaseCircuitBreaker("test")
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_record_failure_below_threshold(self):
        """Test recording failures below threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    def test_transition_to_open(self):
        """Test transition from CLOSED to OPEN when threshold reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = BaseCircuitBreaker("test", config)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_allow_request_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        cb = BaseCircuitBreaker("test")
        assert cb.allow_request() is True

    def test_block_request_when_open(self):
        """Test that requests are blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=60.0)
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()  # Opens circuit
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_transition_to_half_open_after_cooldown(self):
        """Test transition from OPEN to HALF_OPEN after cooldown."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1)
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()  # Opens circuit
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)  # Wait for cooldown

        # Next allow_request check should transition to HALF_OPEN
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_close_circuit_on_success_in_half_open(self):
        """Test closing circuit after successful calls in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, cooldown_seconds=0.01
        )
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()  # CLOSED -> OPEN
        time.sleep(0.02)  # Wait for cooldown
        cb.allow_request()  # OPEN -> HALF_OPEN

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()  # First success
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()  # Second success - should close
        assert cb.state == CircuitState.CLOSED

    def test_reopen_on_failure_in_half_open(self):
        """Test reopening circuit on failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.01)
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()  # CLOSED -> OPEN
        time.sleep(0.02)  # Wait for cooldown
        cb.allow_request()  # OPEN -> HALF_OPEN

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()  # Should reopen
        assert cb.state == CircuitState.OPEN

    def test_state_change_callback(self):
        """Test that state change callback is invoked."""
        callback_calls = []

        def on_state_change(old_state, new_state):
            callback_calls.append((old_state, new_state))

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = BaseCircuitBreaker("test", config, on_state_change=on_state_change)

        cb.record_failure()  # CLOSED -> OPEN

        assert len(callback_calls) == 1
        assert callback_calls[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_success_resets_failure_count(self):
        """Test that success resets consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = BaseCircuitBreaker("test", config)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0

    def test_get_stats(self):
        """Test getting circuit breaker stats."""
        cb = BaseCircuitBreaker("test")
        cb.record_success()
        cb.record_failure()

        stats = cb.get_stats()
        assert stats.name == "test"
        assert stats.state == CircuitState.CLOSED
        assert stats.failure_count == 1
        assert stats.total_requests >= 2


class TestWithCircuitBreakerDecorator:
    """Tests for with_circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_success_passes_through(self):
        """Test successful calls pass through."""

        @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=3))
        async def success():
            return "success"

        result = await success()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_failure_recorded(self):
        """Test failures are recorded."""
        call_count = 0

        @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=3))
        async def fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await fail()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        call_count = 0

        @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60.0))
        async def fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        # First two calls fail and open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await fail()

        # Third call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await fail()

        # Should only have called the function twice
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_error_has_cooldown(self):
        """Test CircuitBreakerOpenError includes cooldown info."""

        @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=30.0))
        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await fail()

        try:
            await fail()
        except CircuitBreakerOpenError as e:
            assert e.cooldown_remaining > 0
            assert e.cooldown_remaining <= 30.0


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_error_message(self):
        """Test error message format."""
        error = CircuitBreakerOpenError("test_service", 25.5)
        assert "test_service" in str(error)
        assert "25.5" in str(error) or "25" in str(error)

    def test_error_attributes(self):
        """Test error attributes."""
        error = CircuitBreakerOpenError("test_service", 25.5)
        assert error.service_name == "test_service"
        assert error.cooldown_remaining == 25.5


class TestCircuitBreakerConcurrency:
    """Tests for thread safety and concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test concurrent calls are handled correctly."""
        call_count = 0
        success_count = 0

        @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=10))
        async def concurrent_op():
            nonlocal call_count, success_count
            call_count += 1
            await asyncio.sleep(0.01)
            success_count += 1
            return "success"

        # Run many concurrent calls
        tasks = [concurrent_op() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(r == "success" for r in results)
        assert call_count == 20
        assert success_count == 20

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Test concurrent failures are counted correctly."""
        config = CircuitBreakerConfig(failure_threshold=5, cooldown_seconds=60.0)

        failure_count = 0

        @with_circuit_breaker(config)
        async def concurrent_fail():
            nonlocal failure_count
            failure_count += 1
            raise ValueError("fail")

        # Run concurrent failing calls
        tasks = []
        for _ in range(10):
            tasks.append(concurrent_fail())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have some ValueError and some CircuitBreakerOpenError
        value_errors = sum(1 for r in results if isinstance(r, ValueError))
        breaker_errors = sum(1 for r in results if isinstance(r, CircuitBreakerOpenError))

        # First 5 should be ValueError, rest should be blocked
        assert value_errors >= 5
        assert breaker_errors > 0
        assert value_errors + breaker_errors == 10
