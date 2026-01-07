"""
Integration tests for CircuitBreaker resilience pattern.

Tests circuit breaker state transitions, multi-entity tracking,
protected call context managers, and recovery behavior.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    get_circuit_breaker_status,
)


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    def test_initial_state_is_closed(self) -> None:
        """Circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.get_status() == "closed"
        assert cb.can_proceed() is True
        assert cb.failures == 0

    def test_failure_increments_counter(self) -> None:
        """Each failure increments the counter."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        assert cb.failures == 1

        cb.record_failure()
        assert cb.failures == 2

    def test_success_resets_failure_counter(self) -> None:
        """Success resets failure count to zero."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        cb.record_success()
        assert cb.failures == 0

    def test_circuit_opens_after_threshold(self) -> None:
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "closed"

        opened = cb.record_failure()  # 3rd failure
        assert opened is True
        assert cb.get_status() == "open"
        assert cb.can_proceed() is False

    def test_open_circuit_blocks_requests(self) -> None:
        """Open circuit should block can_proceed()."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

        cb.record_failure()
        cb.record_failure()

        assert cb.can_proceed() is False
        assert cb.get_status() == "open"


class TestCircuitBreakerCooldown:
    """Test cooldown and half-open behavior."""

    def test_circuit_reopens_after_cooldown(self) -> None:
        """Circuit allows requests after cooldown expires."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False

        time.sleep(0.15)  # Wait for cooldown
        assert cb.can_proceed() is True

    def test_single_success_closes_single_entity_circuit(self) -> None:
        """In single-entity mode, one success closes the circuit."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "open"

        time.sleep(0.15)
        cb.record_success()

        assert cb.get_status() == "closed"
        assert cb.failures == 0


class TestCircuitBreakerMultiEntity:
    """Test multi-entity tracking mode."""

    def test_separate_tracking_per_entity(self) -> None:
        """Each entity has independent failure tracking."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        cb.record_failure("agent-2")

        assert cb.is_available("agent-1") is False
        assert cb.is_available("agent-2") is True

    def test_entity_circuit_opens_independently(self) -> None:
        """Opening one entity's circuit doesn't affect others."""
        cb = CircuitBreaker(failure_threshold=2)

        # Open circuit for agent-1
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        # agent-2 should still work
        assert cb.can_proceed("agent-2") is True
        assert cb.get_status("agent-2") == "closed"

    def test_entity_recovers_independently(self) -> None:
        """Each entity recovers independently after cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuits for both
        for _ in range(2):
            cb.record_failure("agent-1")
            cb.record_failure("agent-2")

        assert cb.is_available("agent-1") is False
        assert cb.is_available("agent-2") is False

        time.sleep(0.15)

        # Record success for agent-1 only
        cb.record_success("agent-1")
        cb.record_success("agent-1")  # Need 2 for half_open_success_threshold

        # Only agent-1 should be closed
        assert cb.get_status("agent-1") == "closed"
        assert cb.get_status("agent-2") == "half-open"

    def test_filter_available_entities(self) -> None:
        """filter_available_entities returns only available entities."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("agent-bad")
        cb.record_failure("agent-bad")

        entities = ["agent-good", "agent-bad", "agent-unknown"]
        available = cb.filter_available_entities(entities)

        assert "agent-good" in available
        assert "agent-unknown" in available
        assert "agent-bad" not in available

    def test_get_all_status(self) -> None:
        """get_all_status returns status for all tracked entities."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        cb.record_failure("agent-2")

        status = cb.get_all_status()

        assert "agent-1" in status
        assert status["agent-1"]["status"] == "open"
        assert status["agent-1"]["failures"] == 2

        assert "agent-2" in status
        assert status["agent-2"]["status"] == "closed"
        assert status["agent-2"]["failures"] == 1


class TestProtectedCallContextManager:
    """Test protected_call async context manager."""

    @pytest.mark.asyncio
    async def test_protected_call_records_success(self) -> None:
        """Successful call is recorded."""
        cb = CircuitBreaker(failure_threshold=3)

        async with cb.protected_call("test-agent"):
            pass  # Success

        # Failure count should be 0
        assert cb._failures.get("test-agent", 0) == 0

    @pytest.mark.asyncio
    async def test_protected_call_records_failure(self) -> None:
        """Exception in protected call records failure."""
        cb = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            async with cb.protected_call("test-agent"):
                raise ValueError("Test error")

        # Should have recorded failure
        assert cb._failures.get("test-agent", 0) == 1

    @pytest.mark.asyncio
    async def test_protected_call_raises_circuit_open_error(self) -> None:
        """Protected call raises CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

        # Open the circuit
        cb.record_failure("test-agent")
        cb.record_failure("test-agent")

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call("test-agent"):
                pass

        assert "test-agent" in str(exc_info.value)
        assert exc_info.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_protected_call_single_entity_mode(self) -> None:
        """Protected call works in single-entity mode."""
        cb = CircuitBreaker(failure_threshold=2)

        with pytest.raises(RuntimeError):
            async with cb.protected_call():  # No entity
                raise RuntimeError("Test")

        assert cb._single_failures == 1


class TestProtectedCallSync:
    """Test protected_call_sync sync context manager."""

    def test_sync_protected_call_records_success(self) -> None:
        """Successful sync call is recorded."""
        cb = CircuitBreaker(failure_threshold=3)

        with cb.protected_call_sync("test-agent"):
            pass

        assert cb._failures.get("test-agent", 0) == 0

    def test_sync_protected_call_records_failure(self) -> None:
        """Exception in sync protected call records failure."""
        cb = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            with cb.protected_call_sync("test-agent"):
                raise ValueError("Test error")

        assert cb._failures.get("test-agent", 0) == 1

    def test_sync_protected_call_raises_circuit_open(self) -> None:
        """Sync protected call raises CircuitOpenError when open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

        cb.record_failure("test-agent")
        cb.record_failure("test-agent")

        with pytest.raises(CircuitOpenError):
            with cb.protected_call_sync("test-agent"):
                pass


class TestCircuitBreakerReset:
    """Test reset functionality."""

    def test_reset_clears_all_state(self) -> None:
        """Reset clears all single and multi-entity state."""
        cb = CircuitBreaker(failure_threshold=2)

        # Build up some state
        cb.record_failure()
        cb.record_failure()
        cb.record_failure("entity-1")
        cb.record_failure("entity-1")

        cb.reset()

        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert len(cb._failures) == 0
        assert len(cb._circuit_open_at) == 0

    def test_reset_single_entity(self) -> None:
        """Reset with entity only clears that entity."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("entity-1")
        cb.record_failure("entity-1")
        cb.record_failure("entity-2")
        cb.record_failure("entity-2")

        cb.reset("entity-1")

        assert cb.is_available("entity-1") is True
        assert cb.is_available("entity-2") is False


class TestCircuitBreakerSerialization:
    """Test serialization and deserialization."""

    def test_to_dict_captures_state(self) -> None:
        """to_dict captures current state."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure()
        cb.record_failure("entity-1")
        cb.record_failure("entity-1")

        data = cb.to_dict()

        assert data["single_mode"]["failures"] == 1
        assert data["entity_mode"]["failures"]["entity-1"] == 2
        assert "entity-1" in data["entity_mode"]["open_circuits"]

    def test_from_dict_restores_state(self) -> None:
        """from_dict restores circuit state."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

        cb.record_failure("entity-1")
        cb.record_failure("entity-1")

        data = cb.to_dict()

        # Create new circuit breaker from data
        cb2 = CircuitBreaker.from_dict(data, failure_threshold=2, cooldown_seconds=60)

        # State should be restored
        assert cb2._failures.get("entity-1", 0) == 2
        assert cb2.is_available("entity-1") is False


class TestGlobalCircuitBreakerRegistry:
    """Test global circuit breaker registry functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_all_circuit_breakers()

    def test_get_circuit_breaker_creates_new(self) -> None:
        """get_circuit_breaker creates a new breaker if not exists."""
        cb = get_circuit_breaker("test-service")
        assert cb is not None
        assert isinstance(cb, CircuitBreaker)

    def test_get_circuit_breaker_returns_same_instance(self) -> None:
        """get_circuit_breaker returns same instance for same name."""
        cb1 = get_circuit_breaker("test-service")
        cb2 = get_circuit_breaker("test-service")

        assert cb1 is cb2

    def test_get_circuit_breaker_status(self) -> None:
        """get_circuit_breaker_status returns status of all breakers."""
        cb = get_circuit_breaker("test-service-status", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        status = get_circuit_breaker_status()

        assert "test-service-status" in status
        # Single-entity mode uses failures count
        assert status["test-service-status"]["failures"] == 2

    def test_reset_all_circuit_breakers(self) -> None:
        """reset_all_circuit_breakers resets all registered breakers."""
        cb1 = get_circuit_breaker("service-1", failure_threshold=1)
        cb2 = get_circuit_breaker("service-2", failure_threshold=1)

        cb1.record_failure()
        cb2.record_failure()

        reset_all_circuit_breakers()

        assert cb1.failures == 0
        assert cb2.failures == 0


class TestCircuitBreakerCascadingFailures:
    """Test circuit breaker behavior under cascading failure scenarios."""

    @pytest.mark.asyncio
    async def test_cascading_failures_open_circuit(self) -> None:
        """Multiple rapid failures correctly open the circuit."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=1)

        # Simulate cascading failures
        for i in range(5):
            try:
                async with cb.protected_call("failing-service"):
                    raise ConnectionError(f"Connection failed {i}")
            except CircuitOpenError:
                # Circuit should open after 3 failures
                assert i >= 3, "Circuit opened too early"
            except ConnectionError:
                pass

    @pytest.mark.asyncio
    async def test_recovery_after_cascading_failures(self) -> None:
        """System recovers after cascading failures and cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Cause cascading failures using multi-entity mode
        for _ in range(3):
            try:
                async with cb.protected_call("failing-service"):
                    raise ConnectionError("Failed")
            except (ConnectionError, CircuitOpenError):
                pass

        assert cb.get_status("failing-service") == "open"

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Should be able to proceed now (half-open)
        assert cb.can_proceed("failing-service") is True

        # Successful calls should close circuit (need 2 for half_open_success_threshold)
        async with cb.protected_call("failing-service"):
            pass
        async with cb.protected_call("failing-service"):
            pass

        assert cb.get_status("failing-service") == "closed"
