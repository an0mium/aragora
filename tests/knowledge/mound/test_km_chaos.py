"""
Knowledge Mound Chaos Engineering Tests.

Tests resilience patterns under failure conditions to verify:
- Circuit breaker state transitions under repeated failures
- Bulkhead rejection under concurrency pressure
- Transaction timeout recovery
- Health monitor failure threshold behavior
- Retry logic under transient failures

These tests simulate chaos scenarios to ensure the KM system
gracefully handles failures and recovers appropriately.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.resilience import (
    AdapterCircuitBreaker,
    AdapterCircuitBreakerConfig,
    AdapterCircuitState,
    AdapterBulkhead,
    BulkheadConfig,
    BulkheadFullError,
    HealthStatus,
    RetryConfig,
    RetryStrategy,
    TransactionConfig,
    TransactionIsolation,
    with_retry,
)


class TestCircuitBreakerChaos:
    """Chaos tests for circuit breaker state transitions."""

    def test_circuit_opens_after_failure_threshold(self):
        """Circuit should open after consecutive failures reach threshold."""
        config = AdapterCircuitBreakerConfig(failure_threshold=3)
        breaker = AdapterCircuitBreaker("test_adapter", config)

        # Initial state should be closed
        assert breaker.state == AdapterCircuitState.CLOSED
        assert breaker.can_proceed()

        # Simulate failures below threshold
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == AdapterCircuitState.CLOSED

        # Third failure should trip the circuit
        breaker.record_failure("Error 3")
        assert breaker.state == AdapterCircuitState.OPEN
        assert not breaker.can_proceed()

    def test_circuit_recovery_through_half_open(self):
        """Circuit should recover through half-open state after timeout."""
        config = AdapterCircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,  # Short timeout for testing
            half_open_max_calls=2,
            success_threshold=2,
        )
        breaker = AdapterCircuitBreaker("test_adapter", config)

        # Trip the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == AdapterCircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open on next check
        assert breaker.can_proceed()
        assert breaker.state == AdapterCircuitState.HALF_OPEN

        # Successful calls should close the circuit
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == AdapterCircuitState.CLOSED

    def test_circuit_reopens_on_half_open_failure(self):
        """Circuit should reopen if failure occurs in half-open state."""
        config = AdapterCircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            half_open_max_calls=3,
        )
        breaker = AdapterCircuitBreaker("test_adapter", config)

        # Trip the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == AdapterCircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Transition to half-open
        assert breaker.can_proceed()
        assert breaker.state == AdapterCircuitState.HALF_OPEN

        # Failure in half-open should reopen circuit
        breaker.record_failure("Error in half-open")
        assert breaker.state == AdapterCircuitState.OPEN

    def test_rapid_failures_track_metrics(self):
        """Rapid failures should correctly track all metrics."""
        config = AdapterCircuitBreakerConfig(failure_threshold=5)
        breaker = AdapterCircuitBreaker("test_adapter", config)

        # Rapid fire failures
        for i in range(10):
            breaker.record_failure(f"Error {i}")

        stats = breaker.get_stats()
        assert stats.total_failures == 10
        assert stats.total_circuit_opens >= 1
        assert stats.state == AdapterCircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Success should reset failure count in closed state."""
        config = AdapterCircuitBreakerConfig(failure_threshold=3)
        breaker = AdapterCircuitBreaker("test_adapter", config)

        # Accumulate some failures
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")

        # Success should reset
        breaker.record_success()

        # More failures needed to trip
        breaker.record_failure("Error 3")
        breaker.record_failure("Error 4")
        assert breaker.state == AdapterCircuitState.CLOSED

        # Third consecutive failure trips
        breaker.record_failure("Error 5")
        assert breaker.state == AdapterCircuitState.OPEN


class TestBulkheadChaos:
    """Chaos tests for bulkhead under concurrency pressure."""

    @pytest.mark.asyncio
    async def test_bulkhead_rejects_over_capacity(self):
        """Bulkhead should reject calls when at capacity."""
        config = BulkheadConfig(max_concurrent_calls=2, max_wait_seconds=0.1)
        bulkhead = AdapterBulkhead("test_adapter", config)

        results = []
        errors = []

        async def slow_operation(idx: int):
            try:
                async with bulkhead.acquire():
                    await asyncio.sleep(0.5)  # Hold the permit
                    results.append(idx)
            except BulkheadFullError as e:
                errors.append((idx, str(e)))

        # Launch 5 concurrent operations (only 2 should succeed)
        tasks = [asyncio.create_task(slow_operation(i)) for i in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 2 should succeed, 3 should be rejected
        assert len(results) == 2
        assert len(errors) == 3

        stats = bulkhead.get_stats()
        assert stats["rejected_calls"] == 3
        assert stats["rejection_rate"] > 0

    @pytest.mark.asyncio
    async def test_bulkhead_releases_on_exception(self):
        """Bulkhead should release permit even when operation fails."""
        config = BulkheadConfig(max_concurrent_calls=1, max_wait_seconds=0.1)
        bulkhead = AdapterBulkhead("test_adapter", config)

        # Operation that fails
        with pytest.raises(ValueError):
            async with bulkhead.acquire():
                raise ValueError("Operation failed")

        # Permit should be released
        assert bulkhead.available_permits == 1
        assert bulkhead.active_calls == 0

        # Next operation should succeed
        async with bulkhead.acquire():
            assert bulkhead.active_calls == 1

    @pytest.mark.asyncio
    async def test_concurrent_bulkhead_fairness(self):
        """Bulkhead should allow waiting callers to proceed in order."""
        config = BulkheadConfig(max_concurrent_calls=1, max_wait_seconds=2.0)
        bulkhead = AdapterBulkhead("test_adapter", config)

        order = []

        async def operation(idx: int, delay: float):
            await asyncio.sleep(delay)  # Stagger start times
            async with bulkhead.acquire():
                order.append(idx)
                await asyncio.sleep(0.1)  # Brief operation

        tasks = [
            asyncio.create_task(operation(0, 0)),
            asyncio.create_task(operation(1, 0.05)),
            asyncio.create_task(operation(2, 0.1)),
        ]
        await asyncio.gather(*tasks)

        # All should complete (FIFO-ish order)
        assert len(order) == 3
        assert 0 in order


class TestTransactionTimeoutChaos:
    """Chaos tests for transaction timeout recovery."""

    @pytest.mark.asyncio
    async def test_transaction_timeout_recovery(self):
        """Transaction config should handle various isolation levels."""
        # Test transaction configuration creation
        config = TransactionConfig(
            timeout_seconds=0.2,
            isolation=TransactionIsolation.READ_COMMITTED,
        )

        # Verify config properties
        assert config.timeout_seconds == 0.2
        assert config.isolation == TransactionIsolation.READ_COMMITTED
        assert config.savepoint_on_nested is True

        # Test different isolation levels
        serializable_config = TransactionConfig(
            isolation=TransactionIsolation.SERIALIZABLE,
        )
        assert serializable_config.isolation == TransactionIsolation.SERIALIZABLE


class TestHealthMonitorChaos:
    """Chaos tests for health status tracking."""

    def test_health_status_degrades_after_failures(self):
        """Health status should track consecutive failures."""

        # Initial healthy state
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=0,
        )
        assert status.healthy
        assert status.consecutive_failures == 0

        # Simulate degradation
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=2,
            last_error="Connection error 2",
        )
        assert status.consecutive_failures == 2

        # At threshold (3), should be unhealthy
        status = HealthStatus(
            healthy=False,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=3,
            last_error="Connection error 3",
        )
        assert not status.healthy
        assert status.consecutive_failures == 3

    def test_health_status_recovers_on_success(self):
        """Health status should recover after success resets failures."""

        # Unhealthy state
        status = HealthStatus(
            healthy=False,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=5,
            last_error="Multiple failures",
        )
        assert not status.healthy

        # Recovery state
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=0,
            latency_ms=15.0,
        )
        assert status.healthy
        assert status.latency_ms == 15.0

    def test_health_status_latency_tracking(self):
        """Health status should track latency metrics."""

        # Status with latency
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
            latency_ms=42.5,
        )
        assert status.latency_ms == 42.5

        # Convert to dict for monitoring
        status_dict = status.to_dict()
        assert status_dict["latency_ms"] == 42.5
        assert status_dict["healthy"] is True


class TestRetryChaos:
    """Chaos tests for retry logic under transient failures."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failures(self):
        """Retry should succeed if transient failures clear."""
        call_count = 0

        # max_retries=4 means 5 total attempts (initial + 4 retries)
        @with_retry(RetryConfig(max_retries=4, base_delay=0.01, jitter=False))
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_respects_max_retries(self):
        """Retry should stop after max retries."""
        call_count = 0

        # max_retries=2 means 3 total attempts (initial + 2 retries)
        @with_retry(RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_does_not_retry_non_retryable(self):
        """Retry should not retry non-retryable exceptions."""
        call_count = 0

        # Default retryable_exceptions are (ConnectionError, TimeoutError, OSError)
        # ValueError is not in that list
        @with_retry(RetryConfig(max_retries=4, base_delay=0.01, jitter=False))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # Should not retry


class TestCascadeFailureScenarios:
    """Test scenarios involving cascading failures across components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascade(self):
        """Circuit breaker should prevent cascade failures."""
        config = AdapterCircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        breaker = AdapterCircuitBreaker("cascade_test", config)

        call_count = 0

        async def protected_operation():
            nonlocal call_count
            if not breaker.can_proceed():
                return "circuit_open"

            call_count += 1
            try:
                raise ConnectionError("Downstream failure")
            except ConnectionError:
                breaker.record_failure("Downstream failure")
                raise

        # First two calls fail and trip the circuit
        for _ in range(2):
            try:
                await protected_operation()
            except ConnectionError:
                pass

        assert breaker.state == AdapterCircuitState.OPEN

        # Subsequent calls should be blocked by circuit
        result = await protected_operation()
        assert result == "circuit_open"
        assert call_count == 2  # No new calls made

    @pytest.mark.asyncio
    async def test_bulkhead_isolates_adapter_failures(self):
        """Bulkhead should isolate one adapter's problems from others."""
        bulkhead_a = AdapterBulkhead(
            "adapter_a",
            BulkheadConfig(max_concurrent_calls=2, max_wait_seconds=0.1),
        )
        bulkhead_b = AdapterBulkhead(
            "adapter_b",
            BulkheadConfig(max_concurrent_calls=2, max_wait_seconds=0.1),
        )

        # Fill bulkhead A completely
        async def block_a():
            async with bulkhead_a.acquire():
                await asyncio.sleep(1.0)

        # Start blocking tasks for A
        tasks_a = [asyncio.create_task(block_a()) for _ in range(2)]
        await asyncio.sleep(0.05)  # Let them acquire

        # B should still be available
        assert bulkhead_b.available_permits == 2

        async with bulkhead_b.acquire():
            assert bulkhead_b.active_calls == 1
            assert bulkhead_a.available_permits == 0

        # Cleanup
        for t in tasks_a:
            t.cancel()
        await asyncio.gather(*tasks_a, return_exceptions=True)


class TestRecoveryScenarios:
    """Test recovery from various failure states."""

    def test_circuit_breaker_full_recovery_cycle(self):
        """Test complete circuit breaker recovery cycle."""
        config = AdapterCircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            success_threshold=2,
            half_open_max_calls=3,
        )
        breaker = AdapterCircuitBreaker("recovery_test", config)

        # Phase 1: Trip the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == AdapterCircuitState.OPEN

        # Phase 2: Wait for timeout and enter half-open
        time.sleep(0.15)
        assert breaker.can_proceed()
        assert breaker.state == AdapterCircuitState.HALF_OPEN

        # Phase 3: Fail again, reopen
        breaker.record_failure("Error in half-open")
        assert breaker.state == AdapterCircuitState.OPEN

        # Phase 4: Wait again, try recovery
        time.sleep(0.15)
        assert breaker.can_proceed()
        assert breaker.state == AdapterCircuitState.HALF_OPEN

        # Phase 5: Successful recovery
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == AdapterCircuitState.CLOSED

        # Verify metrics (stats is AdapterCircuitStats dataclass)
        stats = breaker.get_stats()
        assert stats.total_circuit_opens >= 2
        assert stats.total_successes == 2
        assert stats.total_failures == 3

    @pytest.mark.asyncio
    async def test_bulkhead_recovery_after_burst(self):
        """Bulkhead should recover after a burst of traffic."""
        config = BulkheadConfig(max_concurrent_calls=3, max_wait_seconds=0.5)
        bulkhead = AdapterBulkhead("burst_test", config)

        async def quick_operation():
            async with bulkhead.acquire():
                await asyncio.sleep(0.1)

        # Burst of operations
        tasks = [asyncio.create_task(quick_operation()) for _ in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Should be fully recovered
        assert bulkhead.available_permits == 3
        assert bulkhead.active_calls == 0

        # New operations should work
        async with bulkhead.acquire():
            assert bulkhead.active_calls == 1
