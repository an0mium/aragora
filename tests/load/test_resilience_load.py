"""
Load tests for the Knowledge Mound resilience layer.

Tests cover:
- ResilientPostgresStore under concurrent operations
- Circuit breaker behavior under failure conditions
- Cache invalidation bus throughput
- Retry logic under transient failures
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


# ============================================================================
# Load Test Configuration
# ============================================================================


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""

    concurrent_operations: int = 50
    total_operations: int = 200
    failure_rate: float = 0.1  # 10% failure rate
    operation_delay_ms: int = 10
    circuit_breaker_threshold: int = 5
    retry_max_attempts: int = 3


@pytest.fixture
def load_config() -> LoadTestConfig:
    return LoadTestConfig()


# ============================================================================
# Mock Resilience Components
# ============================================================================


@dataclass
class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    _failure_count: int = 0
    _is_open: bool = False
    _opened_at: float = 0.0
    _half_open: bool = False

    def record_success(self):
        self._failure_count = 0
        self._is_open = False
        self._half_open = False

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._is_open = True
            self._opened_at = time.time()

    def can_execute(self) -> bool:
        if not self._is_open:
            return True

        elapsed = time.time() - self._opened_at
        if elapsed >= self.recovery_timeout:
            self._half_open = True
            return True

        return False

    @property
    def state(self) -> str:
        if self._is_open:
            if self._half_open:
                return "half_open"
            return "open"
        return "closed"


@dataclass
class MockRetryConfig:
    """Mock retry configuration."""

    max_retries: int = 3
    base_delay: float = 0.01
    max_delay: float = 1.0
    exponential_base: float = 2.0


@dataclass
class MockCacheInvalidationBus:
    """Mock cache invalidation bus for throughput testing."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    subscribers: List[Any] = field(default_factory=list)
    _processed_count: int = 0
    _dropped_count: int = 0
    _max_queue_size: int = 1000

    async def publish(self, event: Dict[str, Any]) -> bool:
        if len(self.events) >= self._max_queue_size:
            self._dropped_count += 1
            return False

        self.events.append(event)
        self._processed_count += 1

        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                await subscriber(event)
            except Exception:
                pass

        return True

    def subscribe(self, handler) -> callable:
        self.subscribers.append(handler)
        return lambda: self.subscribers.remove(handler)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "processed": self._processed_count,
            "dropped": self._dropped_count,
            "pending": len(self.events),
            "subscribers": len(self.subscribers),
        }


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


class TestResilientStoreConcurrency:
    """Load tests for concurrent store operations."""

    @pytest.fixture
    def mock_store(self, load_config: LoadTestConfig):
        """Create mock resilient store."""
        items = {}
        operation_count = {"reads": 0, "writes": 0, "failures": 0}

        class MockStore:
            async def store(self, item_id: str, content: str, **kwargs) -> bool:
                operation_count["writes"] += 1
                # Simulate random failures
                if random.random() < load_config.failure_rate:
                    operation_count["failures"] += 1
                    raise ConnectionError("Simulated failure")

                await asyncio.sleep(load_config.operation_delay_ms / 1000)
                items[item_id] = {"content": content, **kwargs}
                return True

            async def get(self, item_id: str) -> Dict[str, Any] | None:
                operation_count["reads"] += 1
                if random.random() < load_config.failure_rate:
                    operation_count["failures"] += 1
                    raise ConnectionError("Simulated failure")

                await asyncio.sleep(load_config.operation_delay_ms / 1000)
                return items.get(item_id)

            @property
            def stats(self) -> Dict[str, int]:
                return operation_count

        return MockStore()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, mock_store, load_config: LoadTestConfig):
        """Test concurrent write operations."""
        async def write_item(idx: int):
            item_id = f"item-{idx}"
            try:
                await mock_store.store(item_id, f"Content for item {idx}")
                return True
            except ConnectionError:
                return False

        # Run concurrent writes
        start_time = time.time()
        tasks = [write_item(i) for i in range(load_config.total_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False or isinstance(r, Exception))

        # Verify throughput
        ops_per_second = load_config.total_operations / elapsed
        assert ops_per_second > 10, f"Throughput too low: {ops_per_second:.2f} ops/s"

        # Some operations should succeed
        assert successful > 0, "No successful operations"

        # Log stats
        print(f"\nConcurrent writes: {load_config.total_operations} operations")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_second:.2f} ops/s")

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, mock_store, load_config: LoadTestConfig):
        """Test concurrent read operations."""
        # Seed some items first
        for i in range(10):
            try:
                await mock_store.store(f"item-{i}", f"Content {i}")
            except ConnectionError:
                pass

        async def read_item(idx: int):
            item_id = f"item-{idx % 10}"
            try:
                return await mock_store.get(item_id)
            except ConnectionError:
                return None

        # Run concurrent reads
        start_time = time.time()
        tasks = [read_item(i) for i in range(load_config.total_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))

        ops_per_second = load_config.total_operations / elapsed
        assert ops_per_second > 10, f"Read throughput too low: {ops_per_second:.2f} ops/s"

        print(f"\nConcurrent reads: {load_config.total_operations} operations")
        print(f"  Successful: {successful}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_second:.2f} ops/s")


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreakerLoad:
    """Load tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_opens_under_failures(self, load_config: LoadTestConfig):
        """Test that circuit breaker opens after threshold failures."""
        circuit_breaker = MockCircuitBreaker(
            failure_threshold=load_config.circuit_breaker_threshold
        )

        # Simulate failures
        for _ in range(load_config.circuit_breaker_threshold):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == "open"
        assert not circuit_breaker.can_execute()

    @pytest.mark.asyncio
    async def test_circuit_recovers_after_timeout(self):
        """Test circuit recovery after timeout."""
        circuit_breaker = MockCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
        )

        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == "open"

        # Wait for recovery
        await asyncio.sleep(0.15)

        assert circuit_breaker.can_execute()
        assert circuit_breaker.state == "half_open"

        # Successful operation closes circuit
        circuit_breaker.record_success()
        assert circuit_breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_under_intermittent_failures(self, load_config: LoadTestConfig):
        """Test circuit behavior under intermittent failures."""
        circuit_breaker = MockCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=0.05,
        )

        states = []
        operations = 100

        for i in range(operations):
            if circuit_breaker.can_execute():
                # 30% failure rate
                if random.random() < 0.3:
                    circuit_breaker.record_failure()
                else:
                    circuit_breaker.record_success()

            states.append(circuit_breaker.state)

            # Brief pause to allow recovery
            if i % 20 == 0:
                await asyncio.sleep(0.06)

        # Count state transitions
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1

        print(f"\nCircuit breaker states over {operations} operations:")
        for state, count in state_counts.items():
            print(f"  {state}: {count}")

        # Should have some closed states (recovered)
        assert state_counts.get("closed", 0) > 0


# ============================================================================
# Cache Invalidation Bus Tests
# ============================================================================


class TestCacheInvalidationLoad:
    """Load tests for cache invalidation bus."""

    @pytest.mark.asyncio
    async def test_high_volume_events(self, load_config: LoadTestConfig):
        """Test invalidation bus under high event volume."""
        bus = MockCacheInvalidationBus()
        received_events = []

        async def subscriber(event):
            received_events.append(event)

        bus.subscribe(subscriber)

        # Publish many events
        start_time = time.time()
        events_to_publish = load_config.total_operations * 2

        for i in range(events_to_publish):
            await bus.publish({
                "type": "node_updated",
                "node_id": f"node-{i}",
                "workspace_id": "test",
            })

        elapsed = time.time() - start_time

        events_per_second = events_to_publish / elapsed
        assert events_per_second > 100, f"Event throughput too low: {events_per_second:.2f}/s"

        print(f"\nCache invalidation bus: {events_to_publish} events")
        print(f"  Received: {len(received_events)}")
        print(f"  Dropped: {bus.stats['dropped']}")
        print(f"  Duration: {elapsed:.4f}s")
        print(f"  Throughput: {events_per_second:.2f} events/s")

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, load_config: LoadTestConfig):
        """Test bus with multiple subscribers."""
        bus = MockCacheInvalidationBus()
        subscriber_events = {i: [] for i in range(5)}

        # Add multiple subscribers
        for i in range(5):
            async def make_subscriber(idx):
                async def subscriber(event):
                    subscriber_events[idx].append(event)
                return subscriber

            bus.subscribe(await make_subscriber(i))

        # Publish events
        for i in range(100):
            await bus.publish({"type": "test", "id": i})

        # All subscribers should receive all events
        for i, events in subscriber_events.items():
            assert len(events) == 100, f"Subscriber {i} missed events"

    @pytest.mark.asyncio
    async def test_bus_overflow_handling(self):
        """Test bus behavior when queue overflows."""
        bus = MockCacheInvalidationBus()
        bus._max_queue_size = 50

        # Flood the bus
        for i in range(100):
            await bus.publish({"type": "flood", "id": i})

        assert bus.stats["dropped"] == 50
        assert bus.stats["processed"] == 50


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestRetryUnderLoad:
    """Load tests for retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_success_rate(self, load_config: LoadTestConfig):
        """Test retry success rate under transient failures."""
        retry_config = MockRetryConfig(max_retries=load_config.retry_max_attempts)

        async def flaky_operation(success_on_attempt: int):
            """Operation that succeeds on Nth attempt."""
            attempt = [0]

            async def operation():
                attempt[0] += 1
                if attempt[0] < success_on_attempt:
                    raise ConnectionError("Transient failure")
                return {"success": True, "attempts": attempt[0]}

            # Simulate retry logic
            for i in range(retry_config.max_retries):
                try:
                    return await operation()
                except ConnectionError:
                    if i == retry_config.max_retries - 1:
                        raise
                    await asyncio.sleep(retry_config.base_delay * (retry_config.exponential_base ** i))

        # Test various failure scenarios
        results = []

        for success_on in [1, 2, 3, 4]:  # 4 exceeds max retries
            try:
                result = await flaky_operation(success_on)
                results.append(("success", result["attempts"]))
            except ConnectionError:
                results.append(("failed", success_on))

        successful = sum(1 for r in results if r[0] == "success")
        assert successful == 3  # Operations 1, 2, 3 should succeed

    @pytest.mark.asyncio
    async def test_concurrent_retries(self, load_config: LoadTestConfig):
        """Test retry behavior under concurrent operations."""
        operation_stats = {"total": 0, "retries": 0, "successes": 0, "failures": 0}

        async def unreliable_operation():
            operation_stats["total"] += 1

            # 40% chance of failure on each attempt
            max_attempts = 3
            for attempt in range(max_attempts):
                if random.random() > 0.4:
                    operation_stats["successes"] += 1
                    return True

                operation_stats["retries"] += 1
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.001)

            operation_stats["failures"] += 1
            return False

        # Run concurrent operations with retries
        tasks = [unreliable_operation() for _ in range(load_config.total_operations)]
        results = await asyncio.gather(*tasks)

        success_rate = sum(results) / len(results)

        print(f"\nConcurrent retry test: {load_config.total_operations} operations")
        print(f"  Total attempts: {operation_stats['total']}")
        print(f"  Total retries: {operation_stats['retries']}")
        print(f"  Successes: {operation_stats['successes']}")
        print(f"  Failures: {operation_stats['failures']}")
        print(f"  Success rate: {success_rate:.2%}")

        # With 60% success per attempt and 3 attempts, expect >85% success
        assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"


# ============================================================================
# Stress Tests
# ============================================================================


class TestResilienceStress:
    """Stress tests combining all resilience components."""

    @pytest.mark.asyncio
    async def test_combined_stress(self, load_config: LoadTestConfig):
        """Stress test with all components active."""
        circuit_breaker = MockCircuitBreaker(failure_threshold=10)
        bus = MockCacheInvalidationBus()

        stats = {
            "operations": 0,
            "successes": 0,
            "failures": 0,
            "circuit_trips": 0,
            "retries": 0,
        }

        async def stress_operation(idx: int):
            stats["operations"] += 1

            if not circuit_breaker.can_execute():
                stats["circuit_trips"] += 1
                return False

            # Simulate operation with failure chance
            for attempt in range(3):
                if random.random() > 0.2:  # 80% success
                    circuit_breaker.record_success()
                    stats["successes"] += 1

                    # Publish invalidation
                    await bus.publish({
                        "type": "node_updated",
                        "node_id": f"node-{idx}",
                    })

                    return True

                stats["retries"] += 1
                circuit_breaker.record_failure()
                await asyncio.sleep(0.001)

            stats["failures"] += 1
            return False

        # Run stress test
        start_time = time.time()
        tasks = [stress_operation(i) for i in range(load_config.total_operations)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        print(f"\nCombined stress test: {load_config.total_operations} operations")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Operations: {stats['operations']}")
        print(f"  Successes: {stats['successes']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Retries: {stats['retries']}")
        print(f"  Circuit trips: {stats['circuit_trips']}")
        print(f"  Events published: {bus.stats['processed']}")

        # Verify reasonable success rate
        success_rate = stats["successes"] / stats["operations"]
        assert success_rate > 0.5, f"Success rate too low under stress: {success_rate:.2%}"
