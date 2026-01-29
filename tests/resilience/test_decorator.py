"""
Tests for the @with_resilience decorator.

Tests cover:
- Retry with exponential/linear/constant backoff
- Circuit breaker integration
- Exception propagation after retries exhausted
- Disabled circuit breaker mode
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aragora.resilience.circuit_breaker import CircuitOpenError
from aragora.resilience.decorator import with_resilience
from aragora.resilience.registry import _circuit_breakers, _circuit_breakers_lock


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


# ============================================================================
# Basic Decorator Tests
# ============================================================================


class TestDecoratorBasics:
    """Tests for basic decorator behavior."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test decorator passes through successful calls."""

        @with_resilience(circuit_name="test-success")
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Test decorator preserves original function name."""

        @with_resilience(circuit_name="name-test")
        async def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    @pytest.mark.asyncio
    async def test_auto_generated_circuit_name(self):
        """Test circuit name is auto-generated from function name."""

        @with_resilience()
        async def my_auto_func():
            return "ok"

        await my_auto_func()
        # Should have created a circuit breaker with func_ prefix
        cbs = dict(_circuit_breakers)
        assert any("my_auto_func" in name for name in cbs)


# ============================================================================
# Retry Tests
# ============================================================================


class TestRetry:
    """Tests for retry behavior."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0

        @with_resilience(circuit_name="retry-test", retries=3, use_circuit_breaker=False)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        with patch("aragora.resilience.decorator.asyncio.sleep", new_callable=AsyncMock):
            result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self):
        """Test raises last exception after all retries fail."""

        @with_resilience(circuit_name="exhaust-test", retries=2, use_circuit_breaker=False)
        async def always_fail():
            raise RuntimeError("permanent error")

        with patch("aragora.resilience.decorator.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="permanent error"):
                await always_fail()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        @with_resilience(
            circuit_name="exp-backoff",
            retries=4,
            backoff="exponential",
            use_circuit_breaker=False,
        )
        async def always_fail():
            raise ValueError("fail")

        with patch("aragora.resilience.decorator.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ValueError):
                await always_fail()

        # Exponential: 2^0=1, 2^1=2, 2^2=4 (3 retries = 3 sleeps)
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1
        assert sleep_calls[1] == 2
        assert sleep_calls[2] == 4

    @pytest.mark.asyncio
    async def test_linear_backoff(self):
        """Test linear backoff delay calculation."""
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        @with_resilience(
            circuit_name="linear-backoff",
            retries=4,
            backoff="linear",
            use_circuit_breaker=False,
        )
        async def always_fail():
            raise ValueError("fail")

        with patch("aragora.resilience.decorator.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ValueError):
                await always_fail()

        # Linear: 1, 2, 3 (3 sleeps for 4 retries)
        assert sleep_calls == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_constant_backoff(self):
        """Test constant backoff delay."""
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        @with_resilience(
            circuit_name="const-backoff",
            retries=3,
            backoff="constant",
            use_circuit_breaker=False,
        )
        async def always_fail():
            raise ValueError("fail")

        with patch("aragora.resilience.decorator.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ValueError):
                await always_fail()

        # Constant: 1.0, 1.0
        assert sleep_calls == [1.0, 1.0]


# ============================================================================
# Circuit Breaker Integration Tests
# ============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration in decorator."""

    @pytest.mark.asyncio
    async def test_records_success(self):
        """Test success is recorded in circuit breaker."""

        @with_resilience(circuit_name="cb-success", retries=1)
        async def good_func():
            return "ok"

        await good_func()
        cbs = dict(_circuit_breakers)
        cb = cbs["cb-success"]
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_records_failure(self):
        """Test failure is recorded in circuit breaker."""

        @with_resilience(
            circuit_name="cb-failure",
            retries=1,
            failure_threshold=10,
        )
        async def bad_func():
            raise RuntimeError("error")

        with patch("aragora.resilience.decorator.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await bad_func()

        cbs = dict(_circuit_breakers)
        cb = cbs["cb-failure"]
        assert cb.failures > 0

    @pytest.mark.asyncio
    async def test_open_circuit_raises_immediately(self):
        """Test open circuit raises CircuitOpenError without calling function."""
        call_count = 0

        @with_resilience(
            circuit_name="cb-open",
            failure_threshold=2,
            cooldown_seconds=60.0,
        )
        async def tracked_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        # Exhaust the circuit breaker
        with patch("aragora.resilience.decorator.asyncio.sleep", new_callable=AsyncMock):
            for _ in range(3):
                try:
                    await tracked_func()
                except (RuntimeError, CircuitOpenError):
                    pass

        call_count = 0
        with pytest.raises(CircuitOpenError):
            await tracked_func()

        assert call_count == 0  # Function was not called

    @pytest.mark.asyncio
    async def test_disabled_circuit_breaker(self):
        """Test with use_circuit_breaker=False."""
        call_count = 0

        @with_resilience(
            circuit_name="no-cb",
            retries=1,
            use_circuit_breaker=False,
        )
        async def my_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        await my_func()
        assert call_count == 1
        # No circuit breaker should be created for this name
        cbs = dict(_circuit_breakers)
        assert "no-cb" not in cbs
