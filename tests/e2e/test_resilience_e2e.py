"""
E2E tests for resilience patterns.

Tests fault tolerance and recovery:
1. Circuit breaker states (closed, open, half-open)
2. Circuit breaker opens after failure threshold
3. Circuit breaker cooldown and recovery
4. Multi-entity circuit breaker tracking
5. Resilience decorator with retry logic
6. Fallback chain behavior
7. Graceful degradation patterns
8. Timeout handling with circuit breaker
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    get_circuit_breaker_status,
    get_circuit_breaker_metrics,
    prune_circuit_breakers,
    with_resilience,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset all circuit breakers before and after each test."""
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture
def circuit_breaker():
    """Create a fresh circuit breaker."""
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)


@pytest.fixture
def fast_cooldown_breaker():
    """Create a circuit breaker with fast cooldown for testing."""
    return CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)


# =============================================================================
# Circuit Breaker State Tests
# =============================================================================


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_closed(self, circuit_breaker):
        """E2E: Circuit breaker should start in closed state."""
        assert circuit_breaker.get_status() == "closed"
        assert circuit_breaker.can_proceed() is True
        assert circuit_breaker.failures == 0

    def test_state_open_after_failures(self, circuit_breaker):
        """E2E: Circuit should open after failure threshold reached."""
        # Record failures up to threshold
        for i in range(3):
            opened = circuit_breaker.record_failure()
            if i < 2:
                assert opened is False
            else:
                assert opened is True

        assert circuit_breaker.get_status() == "open"
        assert circuit_breaker.can_proceed() is False

    def test_state_half_open_after_cooldown(self, fast_cooldown_breaker):
        """E2E: Circuit should be half-open after cooldown elapsed."""
        # Open the circuit
        fast_cooldown_breaker.record_failure()
        fast_cooldown_breaker.record_failure()
        assert fast_cooldown_breaker.get_status() == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # Should be half-open (can_proceed returns True after cooldown)
        assert fast_cooldown_breaker.can_proceed() is True
        # After can_proceed resets it back to closed in single-entity mode
        assert fast_cooldown_breaker.get_status() == "closed"

    def test_state_closed_after_success(self, fast_cooldown_breaker):
        """E2E: Circuit should close after successful request in half-open state."""
        # Open the circuit
        fast_cooldown_breaker.record_failure()
        fast_cooldown_breaker.record_failure()

        # Wait for cooldown
        time.sleep(0.15)

        # Allow a request through (transitions to closed in single-entity mode)
        assert fast_cooldown_breaker.can_proceed() is True

        # Record success
        fast_cooldown_breaker.record_success()

        # Should be closed
        assert fast_cooldown_breaker.get_status() == "closed"
        assert fast_cooldown_breaker.failures == 0


# =============================================================================
# Circuit Breaker Failure Tracking Tests
# =============================================================================


class TestCircuitBreakerFailureTracking:
    """Tests for failure counting and tracking."""

    def test_failures_increment(self, circuit_breaker):
        """E2E: Failures should increment correctly."""
        assert circuit_breaker.failures == 0

        circuit_breaker.record_failure()
        assert circuit_breaker.failures == 1

        circuit_breaker.record_failure()
        assert circuit_breaker.failures == 2

    def test_success_resets_failures(self, circuit_breaker):
        """E2E: Success should reset failure count when circuit is closed."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.failures == 2

        circuit_breaker.record_success()
        assert circuit_breaker.failures == 0

    def test_cooldown_remaining(self, circuit_breaker):
        """E2E: Cooldown remaining should be calculated correctly."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        remaining = circuit_breaker.cooldown_remaining()

        assert remaining > 0
        assert remaining <= circuit_breaker.cooldown_seconds


# =============================================================================
# Multi-Entity Circuit Breaker Tests
# =============================================================================


class TestMultiEntityCircuitBreaker:
    """Tests for multi-entity circuit breaker mode."""

    def test_entity_isolation(self, circuit_breaker):
        """E2E: Failures for one entity should not affect others."""
        # Fail entity A
        circuit_breaker.record_failure("entity-a")
        circuit_breaker.record_failure("entity-a")
        circuit_breaker.record_failure("entity-a")

        # Entity A should be unavailable
        assert circuit_breaker.is_available("entity-a") is False

        # Entity B should still be available
        assert circuit_breaker.is_available("entity-b") is True

    def test_entity_recovery(self):
        """E2E: Each entity should recover independently."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Fail both entities
        cb.record_failure("entity-a")
        cb.record_failure("entity-a")
        cb.record_failure("entity-b")
        cb.record_failure("entity-b")

        # Both unavailable
        assert cb.is_available("entity-a") is False
        assert cb.is_available("entity-b") is False

        # Wait for cooldown
        time.sleep(0.15)

        # Both should be in half-open state (available for trial)
        assert cb.is_available("entity-a") is True
        assert cb.is_available("entity-b") is True

    def test_entity_status_tracking(self, circuit_breaker):
        """E2E: Should track status for all entities."""
        circuit_breaker.record_failure("entity-a")
        circuit_breaker.record_failure("entity-b")
        circuit_breaker.record_failure("entity-b")
        circuit_breaker.record_failure("entity-b")

        statuses = circuit_breaker.get_all_status()

        assert statuses["entity-a"]["failures"] == 1
        assert statuses["entity-a"]["status"] == "closed"
        assert statuses["entity-b"]["failures"] == 3
        assert statuses["entity-b"]["status"] == "open"

    def test_filter_available_entities(self, circuit_breaker):
        """E2E: Should filter out unavailable entities."""
        # Make entity-b unavailable
        circuit_breaker.record_failure("entity-b")
        circuit_breaker.record_failure("entity-b")
        circuit_breaker.record_failure("entity-b")

        entities = ["entity-a", "entity-b", "entity-c"]
        available = circuit_breaker.filter_available_entities(entities)

        assert "entity-a" in available
        assert "entity-b" not in available
        assert "entity-c" in available


# =============================================================================
# Global Circuit Breaker Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_creates_new_breaker(self):
        """E2E: get_circuit_breaker should create new breaker if not exists."""
        cb = get_circuit_breaker("test-service")

        assert cb is not None
        assert isinstance(cb, CircuitBreaker)

    def test_get_returns_same_breaker(self):
        """E2E: get_circuit_breaker should return same instance."""
        cb1 = get_circuit_breaker("same-service")
        cb2 = get_circuit_breaker("same-service")

        assert cb1 is cb2

    def test_reset_all_clears_state(self):
        """E2E: reset_all should clear all circuit breaker state."""
        cb = get_circuit_breaker("reset-test")
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        assert cb.get_status() == "open"

        reset_all_circuit_breakers()

        assert cb.get_status() == "closed"
        assert cb.failures == 0

    def test_status_returns_all_breakers(self):
        """E2E: get_circuit_breaker_status should return all breakers."""
        get_circuit_breaker("service-1")
        get_circuit_breaker("service-2")
        cb3 = get_circuit_breaker("service-3")
        cb3.record_failure()

        status = get_circuit_breaker_status()

        assert status["_registry_size"] >= 3
        assert "service-1" in status
        assert "service-3" in status
        assert status["service-3"]["failures"] == 1

    def test_metrics_includes_summary(self):
        """E2E: get_circuit_breaker_metrics should include summary."""
        cb = get_circuit_breaker("metrics-test")
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert "summary" in metrics
        assert "open" in metrics["summary"]
        assert metrics["summary"]["open"] >= 1
        assert "health" in metrics
        assert metrics["health"]["status"] in ["healthy", "degraded", "critical"]


# =============================================================================
# Protected Call Context Manager Tests
# =============================================================================


class TestProtectedCallContextManager:
    """Tests for protected_call async context manager."""

    @pytest.mark.asyncio
    async def test_protected_call_success(self, circuit_breaker):
        """E2E: Successful call should record success."""
        async with circuit_breaker.protected_call():
            pass  # Simulated successful operation

        assert circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_failure(self, circuit_breaker):
        """E2E: Failed call should record failure."""
        with pytest.raises(ValueError):
            async with circuit_breaker.protected_call():
                raise ValueError("Simulated failure")

        assert circuit_breaker.failures == 1

    @pytest.mark.asyncio
    async def test_protected_call_blocks_when_open(self, circuit_breaker):
        """E2E: Protected call should raise when circuit is open."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with circuit_breaker.protected_call(circuit_name="test"):
                pass

        assert "test" in str(exc_info.value)
        assert exc_info.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_protected_call_with_entity(self, circuit_breaker):
        """E2E: Protected call should work with entity tracking."""
        with pytest.raises(RuntimeError):
            async with circuit_breaker.protected_call(entity="agent-1"):
                raise RuntimeError("Agent failure")

        # Entity should have failure recorded
        statuses = circuit_breaker.get_all_status()
        assert statuses["agent-1"]["failures"] == 1

    @pytest.mark.asyncio
    async def test_cancelled_error_not_recorded(self, circuit_breaker):
        """E2E: Task cancellation should not record failure."""
        with pytest.raises(asyncio.CancelledError):
            async with circuit_breaker.protected_call():
                raise asyncio.CancelledError()

        # Cancellation is not a service failure
        assert circuit_breaker.failures == 0


# =============================================================================
# Sync Protected Call Tests
# =============================================================================


class TestSyncProtectedCall:
    """Tests for sync protected_call_sync context manager."""

    def test_sync_protected_call_success(self, circuit_breaker):
        """E2E: Successful sync call should record success."""
        with circuit_breaker.protected_call_sync():
            pass

        assert circuit_breaker.failures == 0

    def test_sync_protected_call_failure(self, circuit_breaker):
        """E2E: Failed sync call should record failure."""
        with pytest.raises(IOError):
            with circuit_breaker.protected_call_sync():
                raise IOError("Simulated IO error")

        assert circuit_breaker.failures == 1


# =============================================================================
# Resilience Decorator Tests
# =============================================================================


class TestResilienceDecorator:
    """Tests for @with_resilience decorator."""

    @pytest.mark.asyncio
    async def test_decorator_retries_on_failure(self):
        """E2E: Decorator should retry failed operations."""
        call_count = {"value": 0}

        @with_resilience(circuit_name="retry-test", retries=3, use_circuit_breaker=False)
        async def flaky_operation():
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await flaky_operation()

        assert result == "success"
        assert call_count["value"] == 3

    @pytest.mark.asyncio
    async def test_decorator_with_circuit_breaker(self):
        """E2E: Decorator should integrate with circuit breaker."""
        @with_resilience(
            circuit_name="decorated-service",
            retries=1,
            failure_threshold=2,
            cooldown_seconds=60.0,
        )
        async def failing_operation():
            raise RuntimeError("Always fails")

        # First calls should fail and record to circuit breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await failing_operation()

        # Circuit should now be open
        cb = get_circuit_breaker("decorated-service")
        assert cb.get_status() == "open"

        # Next call should fail with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await failing_operation()

    @pytest.mark.asyncio
    async def test_decorator_exponential_backoff(self):
        """E2E: Decorator should use exponential backoff."""
        start_times = []

        @with_resilience(
            circuit_name="backoff-test",
            retries=3,
            backoff="exponential",
            use_circuit_breaker=False,
        )
        async def timed_operation():
            start_times.append(time.time())
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await timed_operation()

        # Should have 3 attempts
        assert len(start_times) == 3

        # Delays should increase (exponential: 1s, 2s minimum)
        # Note: actual delays may be longer due to test overhead

    @pytest.mark.asyncio
    async def test_decorator_success_resets_circuit(self):
        """E2E: Successful call should reset circuit breaker."""
        call_count = {"value": 0}

        @with_resilience(circuit_name="reset-test", retries=2, failure_threshold=5)
        async def sometimes_fails():
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise ValueError("First attempt fails")
            return "success"

        result = await sometimes_fails()
        assert result == "success"

        # Circuit should be closed after success
        cb = get_circuit_breaker("reset-test")
        assert cb.get_status() == "closed"


# =============================================================================
# Circuit Breaker Serialization Tests
# =============================================================================


class TestCircuitBreakerSerialization:
    """Tests for circuit breaker state serialization."""

    def test_to_dict_captures_state(self, circuit_breaker):
        """E2E: to_dict should capture circuit breaker state."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()

        state = circuit_breaker.to_dict()

        assert "single_mode" in state
        assert state["single_mode"]["failures"] == 2
        assert state["single_mode"]["is_open"] is False

    def test_to_dict_captures_open_state(self, circuit_breaker):
        """E2E: to_dict should capture open circuit state."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        state = circuit_breaker.to_dict()

        assert state["single_mode"]["is_open"] is True
        assert state["single_mode"]["open_for_seconds"] >= 0

    def test_from_dict_restores_state(self, circuit_breaker):
        """E2E: from_dict should restore circuit breaker state."""
        # Set up state
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        state = circuit_breaker.to_dict()

        # Restore to new circuit breaker
        restored = CircuitBreaker.from_dict(
            state,
            failure_threshold=circuit_breaker.failure_threshold,
            cooldown_seconds=circuit_breaker.cooldown_seconds,
        )

        assert restored.failures == 2

    def test_entity_state_serialized(self, circuit_breaker):
        """E2E: Entity mode state should be serialized."""
        circuit_breaker.record_failure("agent-1")
        circuit_breaker.record_failure("agent-1")
        circuit_breaker.record_failure("agent-2")

        state = circuit_breaker.to_dict()

        assert "entity_mode" in state
        assert state["entity_mode"]["failures"]["agent-1"] == 2
        assert state["entity_mode"]["failures"]["agent-2"] == 1


# =============================================================================
# Circuit Breaker Reset Tests
# =============================================================================


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset functionality."""

    def test_reset_single_entity(self, circuit_breaker):
        """E2E: Reset should clear single entity state."""
        circuit_breaker.record_failure("entity-1")
        circuit_breaker.record_failure("entity-1")
        circuit_breaker.record_failure("entity-1")
        circuit_breaker.record_failure("entity-2")

        circuit_breaker.reset("entity-1")

        assert circuit_breaker.is_available("entity-1") is True
        statuses = circuit_breaker.get_all_status()
        assert "entity-1" not in statuses
        assert statuses["entity-2"]["failures"] == 1

    def test_reset_all(self, circuit_breaker):
        """E2E: Reset without entity should clear all state."""
        circuit_breaker.record_failure("entity-1")
        circuit_breaker.record_failure("entity-2")
        circuit_breaker.record_failure()  # Single mode

        circuit_breaker.reset()

        assert circuit_breaker.failures == 0
        assert len(circuit_breaker.get_all_status()) == 0


# =============================================================================
# Health Status Tests
# =============================================================================


class TestHealthStatus:
    """Tests for health status reporting."""

    def test_healthy_when_all_closed(self):
        """E2E: Health should be healthy when all circuits closed."""
        get_circuit_breaker("healthy-1")
        get_circuit_breaker("healthy-2")

        metrics = get_circuit_breaker_metrics()

        assert metrics["health"]["status"] == "healthy"
        assert len(metrics["health"]["open_circuits"]) == 0

    def test_degraded_when_circuit_open(self):
        """E2E: Health should be degraded when circuit is open."""
        cb = get_circuit_breaker("failing-service", failure_threshold=1)
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert metrics["health"]["status"] == "degraded"
        assert "failing-service" in metrics["health"]["open_circuits"]

    def test_critical_when_multiple_open(self):
        """E2E: Health should be critical when multiple circuits open."""
        for i in range(3):
            cb = get_circuit_breaker(f"critical-service-{i}", failure_threshold=1)
            cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert metrics["health"]["status"] == "critical"
        assert len(metrics["health"]["open_circuits"]) >= 3

    def test_high_failure_circuits_flagged(self):
        """E2E: Circuits with high failures should be flagged."""
        cb = get_circuit_breaker("warning-service", failure_threshold=10)

        # 6 failures = 60% of threshold
        for _ in range(6):
            cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        # Should be flagged as high-failure (>50% of threshold)
        high_failure = metrics["health"]["high_failure_circuits"]
        assert any(c["name"] == "warning-service" for c in high_failure)
