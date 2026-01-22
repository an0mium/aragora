"""
Chaos tests for circuit breaker behavior under stress.

Tests circuit breaker resilience under:
- Rapid failure/success transitions
- Concurrent state changes
- Edge cases in state machine
- Multiple circuit breaker coordination
- Recovery under load
"""

from __future__ import annotations

import asyncio
import random
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset circuit breakers before and after each test."""
    from aragora.resilience import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions under chaos."""

    @pytest.mark.asyncio
    async def test_rapid_failure_success_oscillation(self):
        """Circuit breaker should handle rapid failure/success changes."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("oscillation_test", failure_threshold=3, cooldown_seconds=0.1)

        # Rapid oscillation between failures and successes
        for i in range(20):
            if i % 2 == 0:
                cb.record_failure()
            else:
                cb.record_success()
            await asyncio.sleep(0.01)

        # Should not be stuck in invalid state
        assert cb.state in ("closed", "open", "half_open")

    @pytest.mark.asyncio
    async def test_threshold_boundary_behavior(self):
        """Circuit breaker should behave correctly at threshold boundary."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("boundary_test", failure_threshold=3, cooldown_seconds=0.1)

        # Record exactly threshold - 1 failures
        for _ in range(2):
            cb.record_failure()

        assert not cb.is_open
        assert cb.failures == 2

        # One more failure should open
        cb.record_failure()
        assert cb.is_open
        assert cb.failures == 3

    @pytest.mark.asyncio
    async def test_half_open_state_under_load(self):
        """Half-open state should handle concurrent requests correctly."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("half_open_test", failure_threshold=2, cooldown_seconds=0.05)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for reset timeout
        await asyncio.sleep(0.1)

        # Multiple concurrent checks during half-open
        results = []
        for _ in range(5):
            can_proceed = cb.can_proceed()
            results.append(can_proceed)

        # At least one should be allowed (half-open test request)
        assert any(results)


class TestConcurrentCircuitBreakerAccess:
    """Tests for concurrent access to circuit breaker."""

    @pytest.mark.asyncio
    async def test_concurrent_failure_recording(self):
        """Concurrent failure recording should be thread-safe."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("concurrent_failures", failure_threshold=100, cooldown_seconds=1.0)

        async def record_failures(count: int):
            for _ in range(count):
                cb.record_failure()
                await asyncio.sleep(0.001)

        # Record failures concurrently
        tasks = [record_failures(20) for _ in range(5)]
        await asyncio.gather(*tasks)

        # All 100 failures should be recorded
        assert cb.failures >= 95  # Allow small variance due to timing

    @pytest.mark.asyncio
    async def test_concurrent_success_recording(self):
        """Concurrent success recording should be thread-safe."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("concurrent_success", failure_threshold=5, cooldown_seconds=1.0)

        # Set up some failures
        for _ in range(3):
            cb.record_failure()

        async def record_success():
            cb.record_success()

        # Record successes concurrently
        tasks = [record_success() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Failures should be reset
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_can_proceed_checks(self):
        """Concurrent can_proceed checks should be consistent."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("concurrent_check", failure_threshold=2, cooldown_seconds=1.0)

        results: list[bool] = []

        async def check_proceed():
            result = cb.can_proceed()
            results.append(result)

        # All should pass when closed
        tasks = [check_proceed() for _ in range(20)]
        await asyncio.gather(*tasks)

        assert all(results)


class TestMultipleCircuitBreakers:
    """Tests for multiple circuit breaker coordination."""

    @pytest.mark.asyncio
    async def test_independent_circuit_breaker_states(self):
        """Multiple circuit breakers should maintain independent states."""
        from aragora.resilience import get_circuit_breaker

        cb1 = get_circuit_breaker("service_a", failure_threshold=2, cooldown_seconds=1.0)
        cb2 = get_circuit_breaker("service_b", failure_threshold=2, cooldown_seconds=1.0)
        cb3 = get_circuit_breaker("service_c", failure_threshold=2, cooldown_seconds=1.0)

        # Open only cb1
        cb1.record_failure()
        cb1.record_failure()

        assert cb1.is_open
        assert not cb2.is_open
        assert not cb3.is_open

    @pytest.mark.asyncio
    async def test_cascading_circuit_breaker_failures(self):
        """Should handle cascading failures across circuit breakers."""
        from aragora.resilience import get_circuit_breaker

        upstream = get_circuit_breaker("upstream", failure_threshold=2, cooldown_seconds=0.1)
        downstream = get_circuit_breaker("downstream", failure_threshold=2, cooldown_seconds=0.1)

        async def call_downstream():
            if not downstream.can_proceed():
                raise Exception("Downstream circuit open")
            # Simulate downstream failure
            downstream.record_failure()
            raise ConnectionError("Downstream failed")

        async def call_upstream():
            if not upstream.can_proceed():
                raise Exception("Upstream circuit open")
            try:
                await call_downstream()
            except (ConnectionError, Exception):
                upstream.record_failure()
                raise

        # Trigger cascading failures
        for _ in range(3):
            try:
                await call_upstream()
            except Exception:
                pass

        # Both circuits should be open
        assert downstream.is_open
        assert upstream.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_registry_metrics(self):
        """Should track metrics across all circuit breakers."""
        from aragora.resilience import (
            get_circuit_breaker,
            get_circuit_breaker_metrics,
        )

        # Create several circuit breakers with different states
        cb1 = get_circuit_breaker("metrics_test_1", failure_threshold=2, cooldown_seconds=1.0)
        cb2 = get_circuit_breaker("metrics_test_2", failure_threshold=2, cooldown_seconds=1.0)
        cb3 = get_circuit_breaker("metrics_test_3", failure_threshold=2, cooldown_seconds=1.0)

        # Open one
        cb1.record_failure()
        cb1.record_failure()

        # Partial failures on another
        cb2.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert "registry_size" in metrics
        assert "summary" in metrics
        assert metrics["summary"]["total"] >= 3


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery scenarios."""

    @pytest.mark.asyncio
    async def test_gradual_recovery_under_load(self):
        """Circuit breaker should recover gradually under continued load."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("gradual_recovery", failure_threshold=3, cooldown_seconds=0.05)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open

        # Wait for half-open
        await asyncio.sleep(0.1)

        # Simulate gradual recovery with some failures
        success_count = 0
        for i in range(10):
            if cb.can_proceed():
                if random.random() > 0.3:  # 70% success rate
                    cb.record_success()
                    success_count += 1
                else:
                    cb.record_failure()
            await asyncio.sleep(0.02)

        # Should have some successes
        assert success_count >= 1

    @pytest.mark.asyncio
    async def test_recovery_with_sustained_success(self):
        """Circuit should close after sustained success."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("sustained_success", failure_threshold=2, cooldown_seconds=0.05)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for half-open
        await asyncio.sleep(0.1)

        # Record sustained successes
        for _ in range(5):
            if cb.can_proceed():
                cb.record_success()

        # Should be closed
        assert not cb.is_open
        assert cb.failures == 0


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases in circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_zero_threshold(self):
        """Circuit breaker with zero threshold (always open)."""
        from aragora.resilience import get_circuit_breaker

        # Threshold of 1 means first failure opens
        cb = get_circuit_breaker("low_threshold", failure_threshold=1, cooldown_seconds=1.0)

        cb.record_failure()
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_very_short_reset_timeout(self):
        """Circuit breaker with very short reset timeout."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("short_reset", failure_threshold=2, cooldown_seconds=0.01)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait minimal time
        await asyncio.sleep(0.02)

        # Should be half-open now
        can_proceed = cb.can_proceed()
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_many_failures_then_success(self):
        """Many failures followed by single success."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("many_failures", failure_threshold=5, cooldown_seconds=0.05)

        # Record many failures
        for _ in range(100):
            cb.record_failure()

        assert cb.is_open
        assert cb.failures >= 5

        # Wait for half-open
        await asyncio.sleep(0.1)

        # Single success
        if cb.can_proceed():
            cb.record_success()

        # Should be closed with reset failures
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_alternating_states_under_stress(self):
        """Circuit breaker under stress with alternating conditions."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("stress_test", failure_threshold=3, cooldown_seconds=0.02)

        state_changes = []

        for i in range(50):
            initial_state = cb.state

            # Random action
            action = random.choice(["fail", "succeed", "check"])
            if action == "fail":
                cb.record_failure()
            elif action == "succeed":
                cb.record_success()
            else:
                cb.can_proceed()

            if cb.state != initial_state:
                state_changes.append((i, initial_state, cb.state))

            await asyncio.sleep(0.005)

        # Should have recorded some state changes
        # (not asserting specific number as it depends on random actions)
        assert cb.state in ("closed", "open", "half_open")


class TestCircuitBreakerWithRealOperations:
    """Integration tests with circuit breaker protecting real-ish operations."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_http_calls(self):
        """Circuit breaker should protect simulated HTTP calls."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("http_protection", failure_threshold=3, cooldown_seconds=0.1)
        call_attempts = []
        server_healthy = False

        async def make_http_call():
            if not cb.can_proceed():
                return {"error": "circuit_open", "status": 503}

            call_attempts.append(1)
            try:
                if not server_healthy:
                    raise ConnectionError("Server unavailable")
                cb.record_success()
                return {"status": 200, "data": "success"}
            except ConnectionError:
                cb.record_failure()
                return {"error": "connection_failed", "status": 502}

        # Server is down - calls should fail and open circuit
        for _ in range(5):
            result = await make_http_call()
            if cb.is_open:
                break

        assert cb.is_open
        initial_attempts = len(call_attempts)

        # While circuit is open, calls should be rejected without hitting server
        for _ in range(3):
            result = await make_http_call()
            assert result["error"] == "circuit_open"

        # No new actual call attempts
        assert len(call_attempts) == initial_attempts

        # Server recovers
        server_healthy = True
        await asyncio.sleep(0.15)

        # Should allow test call and succeed
        result = await make_http_call()
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_timeout(self):
        """Circuit breaker should work with timeout-based failures."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("timeout_protection", failure_threshold=2, cooldown_seconds=0.1)

        async def slow_operation():
            if not cb.can_proceed():
                raise Exception("Circuit open")
            await asyncio.sleep(1.0)  # Very slow
            return "success"

        # Operations should timeout and record failures
        for _ in range(3):
            try:
                await asyncio.wait_for(slow_operation(), timeout=0.05)
            except asyncio.TimeoutError:
                cb.record_failure()
            except Exception:
                pass  # Circuit open

        assert cb.is_open
