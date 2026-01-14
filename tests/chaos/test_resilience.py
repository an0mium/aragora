"""
Chaos tests for resilience patterns.

Tests failure scenarios to verify graceful degradation:
- Circuit breaker activation and recovery
- Agent failure during debate
- Database timeout handling
- Concurrent failure handling
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark chaos tests as slow (involve failure recovery scenarios)
pytestmark = [pytest.mark.slow, pytest.mark.integration]

from aragora.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)


# =============================================================================
# Circuit Breaker Chaos Tests
# =============================================================================


class TestCircuitBreakerActivation:
    """Tests for circuit breaker opening under failure conditions."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Circuit should open after failure_threshold consecutive failures."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        # Record failures up to threshold
        assert cb.can_proceed() is True
        cb.record_failure()
        assert cb.can_proceed() is True
        cb.record_failure()
        assert cb.can_proceed() is True
        cb.record_failure()  # Third failure opens circuit

        # Circuit should now be open
        assert cb.can_proceed() is False
        assert cb.get_status() == "open"

    def test_circuit_blocks_requests_when_open(self) -> None:
        """Open circuit should block all requests until cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Verify blocked
        assert cb.can_proceed() is False

        # Multiple check attempts should all be blocked
        for _ in range(10):
            assert cb.can_proceed() is False

    def test_circuit_reopens_after_cooldown(self) -> None:
        """Circuit should allow requests after cooldown expires."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False

        # Wait for cooldown
        time.sleep(0.15)

        # Should be allowed now (half-open or closed)
        assert cb.can_proceed() is True

    def test_success_closes_circuit(self) -> None:
        """Recording success after cooldown should close the circuit."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for cooldown
        time.sleep(0.15)

        # Record success
        cb.record_success()

        # Circuit should be closed
        assert cb.get_status() == "closed"
        assert cb.failures == 0


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker under concurrent access."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit(self) -> None:
        """Concurrent failures should correctly open the circuit."""
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0)

        async def record_failure() -> None:
            cb.record_failure()
            await asyncio.sleep(0.01)

        # Record failures concurrently
        await asyncio.gather(*[record_failure() for _ in range(10)])

        # Circuit should be open
        assert cb.get_status() == "open"

    @pytest.mark.asyncio
    async def test_concurrent_checks_return_consistent_state(self) -> None:
        """Concurrent can_proceed checks should return consistent results."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        async def check_proceed() -> bool:
            return cb.can_proceed()

        # All concurrent checks should return False
        results = await asyncio.gather(*[check_proceed() for _ in range(100)])
        assert all(r is False for r in results)


class TestCircuitBreakerProtectedCall:
    """Tests for the protected_call context manager."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_protected_call_records_success(self) -> None:
        """Successful call should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        # Successful protected call
        async with cb.protected_call():
            pass

        # Failures should be reset
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_records_failure(self) -> None:
        """Failed call should increment failure count."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        with pytest.raises(ValueError):
            async with cb.protected_call():
                raise ValueError("Test error")

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_protected_call_raises_circuit_open(self) -> None:
        """Protected call should raise CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call():
                pass

        assert exc_info.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_cancelled_task_not_recorded_as_failure(self) -> None:
        """Task cancellation should not be recorded as a failure."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        async def cancellable_operation() -> None:
            async with cb.protected_call():
                await asyncio.sleep(10)  # Long operation

        task = asyncio.create_task(cancellable_operation())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Cancellation should not count as failure
        assert cb.failures == 0


# =============================================================================
# Multi-Entity Circuit Breaker Tests
# =============================================================================


class TestMultiEntityCircuitBreaker:
    """Tests for circuit breaker with multiple entities (agents)."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    def test_entities_tracked_independently(self) -> None:
        """Each entity should have independent failure tracking."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Fail entity A
        cb.record_failure("agent-a")
        cb.record_failure("agent-a")

        # Entity A should be blocked, B should be available
        assert cb.is_available("agent-a") is False
        assert cb.is_available("agent-b") is True

    def test_filter_available_agents(self) -> None:
        """Should correctly filter available agents."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Create mock agents with name attribute set correctly
        agent_a = MagicMock()
        agent_a.name = "agent-a"
        agent_b = MagicMock()
        agent_b.name = "agent-b"
        agent_c = MagicMock()
        agent_c.name = "agent-c"
        agents = [agent_a, agent_b, agent_c]

        # Block agent-b
        cb.record_failure("agent-b")
        cb.record_failure("agent-b")

        # Filter should exclude agent-b
        available = cb.filter_available_agents(agents)
        assert len(available) == 2
        assert agent_a in available
        assert agent_c in available
        assert agent_b not in available

    @pytest.mark.asyncio
    async def test_concurrent_entity_failures(self) -> None:
        """Concurrent failures on different entities should be isolated."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        async def fail_entity(entity: str, count: int) -> None:
            for _ in range(count):
                cb.record_failure(entity)
                await asyncio.sleep(0.01)

        # Fail multiple entities concurrently
        await asyncio.gather(
            fail_entity("agent-a", 3),
            fail_entity("agent-b", 2),
            fail_entity("agent-c", 1),
        )

        # Check statuses
        assert cb.is_available("agent-a") is False  # 3 failures = open
        assert cb.is_available("agent-b") is True  # 2 failures = still closed
        assert cb.is_available("agent-c") is True  # 1 failure = still closed


# =============================================================================
# Agent Failure During Debate Simulation
# =============================================================================


class TestAgentFailureDuringDebate:
    """Tests simulating agent failures during debate execution."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_debate_continues_with_remaining_agents(self) -> None:
        """Debate should continue if one agent fails but others are available."""
        cb = get_circuit_breaker("debate_agents", failure_threshold=2)

        agents = ["claude", "gpt4", "gemini"]
        available_agents = agents.copy()

        # Simulate claude failing
        cb.record_failure("claude")
        cb.record_failure("claude")

        # Filter available agents
        remaining = [a for a in available_agents if cb.is_available(a)]

        # Debate can continue with remaining agents
        assert len(remaining) == 2
        assert "gpt4" in remaining
        assert "gemini" in remaining
        assert "claude" not in remaining

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_agents_fail(self) -> None:
        """Should handle all agents failing gracefully."""
        cb = get_circuit_breaker("all_fail_test", failure_threshold=1)

        agents = ["claude", "gpt4"]

        # All agents fail
        for agent in agents:
            cb.record_failure(agent)

        # No agents available
        remaining = [a for a in agents if cb.is_available(a)]
        assert len(remaining) == 0

        # Verify circuit breaker status reflects this
        status = cb.get_all_status()
        assert all(s["status"] == "open" for s in status.values())


# =============================================================================
# Global Circuit Breaker Registry Tests
# =============================================================================


class TestCircuitBreakerRegistry:
    """Tests for the global circuit breaker registry."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    def test_get_circuit_breaker_returns_same_instance(self) -> None:
        """get_circuit_breaker should return same instance for same name."""
        cb1 = get_circuit_breaker("test-service")
        cb2 = get_circuit_breaker("test-service")

        assert cb1 is cb2

    def test_different_names_different_instances(self) -> None:
        """Different names should return different instances."""
        cb1 = get_circuit_breaker("service-a")
        cb2 = get_circuit_breaker("service-b")

        assert cb1 is not cb2

    def test_reset_all_clears_state(self) -> None:
        """reset_all_circuit_breakers should reset all circuit states."""
        cb1 = get_circuit_breaker("service-1", failure_threshold=2)
        cb2 = get_circuit_breaker("service-2", failure_threshold=2)

        # Open both circuits
        cb1.record_failure()
        cb1.record_failure()
        cb2.record_failure()
        cb2.record_failure()

        assert cb1.is_open is True
        assert cb2.is_open is True

        # Reset all
        reset_all_circuit_breakers()

        # Both should be closed
        assert cb1.is_open is False
        assert cb2.is_open is False


# =============================================================================
# Rapid Failure Injection Tests
# =============================================================================


class TestRapidFailureInjection:
    """Tests for rapid failure scenarios."""

    def setup_method(self) -> None:
        """Reset circuit breakers before each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_burst_failures(self) -> None:
        """Circuit should handle burst of failures correctly."""
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0)

        # Rapid burst of failures
        for _ in range(100):
            cb.record_failure()

        # Circuit should be open
        assert cb.is_open is True
        assert cb.failures >= 5  # At least threshold

    @pytest.mark.asyncio
    async def test_intermittent_failures(self) -> None:
        """Circuit should handle intermittent failure patterns."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        # Pattern: fail, succeed, fail, succeed, fail, fail, fail
        cb.record_failure()
        cb.record_success()  # Resets failures
        cb.record_failure()
        cb.record_success()  # Resets failures
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Third consecutive failure

        # Circuit should be open after 3 consecutive failures
        assert cb.is_open is True

    @pytest.mark.asyncio
    async def test_recovery_under_load(self) -> None:
        """Circuit should recover correctly even under continued load."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Record success while other failures might be happening
        async def record_success() -> None:
            if cb.can_proceed():
                cb.record_success()

        await record_success()

        # Circuit should be closed
        assert cb.is_open is False
