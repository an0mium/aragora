"""
Chaos tests for concurrent load scenarios.

Tests system resilience under:
- High concurrent request volume
- Resource contention
- Deadlock detection
- Memory pressure
- Queue overflow
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def seed_random():
    """Seed random for reproducible chaos tests."""
    random.seed(42)
    yield


class LoadGenerator:
    """Generates configurable load for testing."""

    def __init__(
        self,
        requests_per_second: int = 100,
        duration_seconds: float = 1.0,
        error_rate: float = 0.0,
    ):
        self.requests_per_second = requests_per_second
        self.duration_seconds = duration_seconds
        self.error_rate = error_rate
        self.completed_requests = 0
        self.failed_requests = 0
        self.latencies: list[float] = []

    async def generate_load(self, handler) -> dict[str, Any]:
        """Generate load and collect metrics."""
        interval = 1.0 / self.requests_per_second
        end_time = time.time() + self.duration_seconds

        tasks = []
        while time.time() < end_time:
            task = asyncio.create_task(self._make_request(handler))
            tasks.append(task)
            await asyncio.sleep(interval)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "total_requests": len(tasks),
            "completed": self.completed_requests,
            "failed": self.failed_requests,
            "avg_latency_ms": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95)]
            if self.latencies
            else 0,
        }

    async def _make_request(self, handler):
        """Make a single request and record metrics."""
        start = time.time()
        try:
            if random.random() < self.error_rate:
                raise Exception("Simulated error")
            await handler()
            self.completed_requests += 1
        except Exception:
            self.failed_requests += 1
        finally:
            self.latencies.append((time.time() - start) * 1000)


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset circuit breakers before each test."""
    from aragora.resilience import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


class TestHighConcurrency:
    """Tests for high concurrent request volume."""

    @pytest.mark.asyncio
    async def test_100_concurrent_requests(self):
        """Should handle 100 concurrent requests."""
        results = []

        async def handler():
            await asyncio.sleep(0.01)
            return "success"

        tasks = [handler() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert all(r == "success" for r in results)

    @pytest.mark.asyncio
    async def test_request_queuing_under_load(self):
        """Should queue requests when under heavy load."""
        queue: deque[Any] = deque(maxlen=100)
        processed = []

        async def producer(count: int):
            for i in range(count):
                if len(queue) < queue.maxlen:
                    queue.append(i)
                await asyncio.sleep(0.001)

        async def consumer():
            while True:
                if queue:
                    item = queue.popleft()
                    processed.append(item)
                await asyncio.sleep(0.002)
                if len(processed) >= 50:
                    break

        # Run producer and consumer concurrently
        await asyncio.gather(
            producer(100),
            consumer(),
        )

        assert len(processed) >= 50

    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Should apply backpressure when overwhelmed."""
        max_in_flight = 10
        in_flight = [0]
        rejected = [0]
        completed = [0]

        async def request_with_backpressure():
            if in_flight[0] >= max_in_flight:
                rejected[0] += 1
                return "rejected"

            in_flight[0] += 1
            try:
                await asyncio.sleep(0.02)
                completed[0] += 1
                return "success"
            finally:
                in_flight[0] -= 1

        # Send more requests than can be handled
        tasks = [request_with_backpressure() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # Some should be rejected due to backpressure
        rejected_count = sum(1 for r in results if r == "rejected")
        assert rejected_count > 0
        assert completed[0] > 0

    @pytest.mark.asyncio
    async def test_concurrent_debate_operations(self):
        """Should handle concurrent debate operations."""
        debates: dict[str, dict[str, Any]] = {}
        lock = asyncio.Lock()

        async def create_debate(debate_id: str):
            async with lock:
                debates[debate_id] = {"id": debate_id, "rounds": []}
            await asyncio.sleep(0.01)
            return debate_id

        async def add_round(debate_id: str, round_num: int):
            async with lock:
                if debate_id in debates:
                    debates[debate_id]["rounds"].append(round_num)
            await asyncio.sleep(0.005)

        # Create debates concurrently
        create_tasks = [create_debate(f"debate_{i}") for i in range(10)]
        await asyncio.gather(*create_tasks)

        # Add rounds concurrently
        round_tasks = []
        for i in range(10):
            for r in range(3):
                round_tasks.append(add_round(f"debate_{i}", r))
        await asyncio.gather(*round_tasks)

        # All debates should have all rounds
        for i in range(10):
            assert len(debates[f"debate_{i}"]["rounds"]) == 3


class TestResourceContention:
    """Tests for resource contention scenarios."""

    @pytest.mark.asyncio
    async def test_lock_contention(self):
        """Should handle lock contention gracefully."""
        lock = asyncio.Lock()
        counter = [0]
        contention_count = [0]

        async def increment_with_lock():
            if lock.locked():
                contention_count[0] += 1

            async with lock:
                current = counter[0]
                await asyncio.sleep(0.001)
                counter[0] = current + 1

        tasks = [increment_with_lock() for _ in range(100)]
        await asyncio.gather(*tasks)

        assert counter[0] == 100
        assert contention_count[0] > 0  # Should have some contention

    @pytest.mark.asyncio
    async def test_semaphore_limiting(self):
        """Should limit concurrent access via semaphore."""
        semaphore = asyncio.Semaphore(5)
        max_concurrent = [0]
        current_concurrent = [0]

        async def limited_operation():
            async with semaphore:
                current_concurrent[0] += 1
                max_concurrent[0] = max(max_concurrent[0], current_concurrent[0])
                await asyncio.sleep(0.01)
                current_concurrent[0] -= 1

        tasks = [limited_operation() for _ in range(50)]
        await asyncio.gather(*tasks)

        assert max_concurrent[0] <= 5

    @pytest.mark.asyncio
    async def test_reader_writer_lock_pattern(self):
        """Should handle reader-writer lock pattern correctly."""
        data = {"value": 0}
        read_count = [0]
        write_count = [0]
        readers = [0]
        writers = [0]
        lock = asyncio.Lock()

        async def read_operation():
            # Multiple readers allowed
            read_count[0] += 1
            readers[0] += 1
            await asyncio.sleep(0.005)
            value = data["value"]
            readers[0] -= 1
            return value

        async def write_operation(new_value: int):
            async with lock:
                # Exclusive write
                write_count[0] += 1
                writers[0] += 1
                await asyncio.sleep(0.01)
                data["value"] = new_value
                writers[0] -= 1

        # Mix of reads and writes
        tasks = []
        for i in range(30):
            if i % 5 == 0:
                tasks.append(write_operation(i))
            else:
                tasks.append(read_operation())

        await asyncio.gather(*tasks)

        assert write_count[0] == 6
        assert read_count[0] == 24


class TestDeadlockPrevention:
    """Tests for deadlock detection and prevention."""

    @pytest.mark.asyncio
    async def test_lock_ordering_prevents_deadlock(self):
        """Consistent lock ordering should prevent deadlock."""
        lock_a = asyncio.Lock()
        lock_b = asyncio.Lock()
        completed = [0]

        async def task_1():
            # Always acquire in same order: A then B
            async with lock_a:
                await asyncio.sleep(0.001)
                async with lock_b:
                    await asyncio.sleep(0.001)
                    completed[0] += 1

        async def task_2():
            # Same order: A then B
            async with lock_a:
                await asyncio.sleep(0.001)
                async with lock_b:
                    await asyncio.sleep(0.001)
                    completed[0] += 1

        # Both tasks should complete (no deadlock)
        await asyncio.wait_for(
            asyncio.gather(task_1(), task_2()),
            timeout=1.0,
        )

        assert completed[0] == 2

    @pytest.mark.asyncio
    async def test_timeout_prevents_deadlock(self):
        """Timeout should prevent indefinite blocking."""
        lock = asyncio.Lock()
        timed_out = [False]

        async def hold_lock():
            async with lock:
                await asyncio.sleep(1.0)

        async def try_acquire():
            try:
                await asyncio.wait_for(lock.acquire(), timeout=0.1)
                lock.release()
            except asyncio.TimeoutError:
                timed_out[0] = True

        # Start holder, then try to acquire
        holder = asyncio.create_task(hold_lock())
        await asyncio.sleep(0.01)
        await try_acquire()

        holder.cancel()
        try:
            await holder
        except asyncio.CancelledError:
            pass

        assert timed_out[0]


class TestMemoryPressure:
    """Tests for memory pressure scenarios."""

    @pytest.mark.asyncio
    async def test_bounded_queue_overflow(self):
        """Should handle queue overflow gracefully."""
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=10)
        overflow_count = [0]

        async def producer():
            for i in range(50):
                try:
                    queue.put_nowait(i)
                except asyncio.QueueFull:
                    overflow_count[0] += 1
                await asyncio.sleep(0.001)

        async def consumer():
            consumed = 0
            while consumed < 30:
                try:
                    await asyncio.wait_for(queue.get(), timeout=0.1)
                    consumed += 1
                except asyncio.TimeoutError:
                    break
                await asyncio.sleep(0.005)

        await asyncio.gather(producer(), consumer())

        assert overflow_count[0] > 0

    @pytest.mark.asyncio
    async def test_cache_eviction_under_pressure(self):
        """Should evict cache entries under memory pressure."""
        max_size = 100
        cache: dict[str, str] = {}
        evictions = [0]

        def put_with_eviction(key: str, value: str):
            if len(cache) >= max_size:
                # Evict oldest (FIFO)
                oldest = next(iter(cache))
                del cache[oldest]
                evictions[0] += 1
            cache[key] = value

        # Add more than max_size items
        for i in range(200):
            put_with_eviction(f"key_{i}", f"value_{i}")

        assert len(cache) == max_size
        assert evictions[0] == 100

    @pytest.mark.asyncio
    async def test_large_payload_handling(self):
        """Should handle large payloads without memory issues."""
        payloads: list[bytes] = []

        async def process_payload(size: int) -> int:
            payload = b"x" * size
            payloads.append(payload)
            await asyncio.sleep(0.001)
            result = len(payload)
            payloads.remove(payload)  # Release memory
            return result

        # Process multiple large payloads concurrently
        sizes = [10000, 50000, 100000, 10000, 50000]
        results = await asyncio.gather(*[process_payload(s) for s in sizes])

        assert results == sizes
        assert len(payloads) == 0  # All released


class TestQueueOverflow:
    """Tests for queue overflow handling."""

    @pytest.mark.asyncio
    async def test_priority_queue_under_load(self):
        """Priority queue should maintain order under load."""
        import heapq

        queue: list[tuple[int, str]] = []
        processed: list[str] = []

        async def producer():
            for i in range(100):
                priority = random.randint(1, 10)
                heapq.heappush(queue, (priority, f"item_{i}"))
                await asyncio.sleep(0.001)

        async def consumer():
            while len(processed) < 100:
                if queue:
                    _, item = heapq.heappop(queue)
                    processed.append(item)
                await asyncio.sleep(0.002)

        await asyncio.gather(producer(), consumer())

        assert len(processed) == 100

    @pytest.mark.asyncio
    async def test_rate_limited_queue(self):
        """Should enforce rate limiting on queue processing."""
        queue: asyncio.Queue[int] = asyncio.Queue()
        processed: list[tuple[float, int]] = []
        rate_limit = 10  # items per second

        async def producer():
            for i in range(20):
                await queue.put(i)

        async def rate_limited_consumer():
            interval = 1.0 / rate_limit
            while len(processed) < 20:
                item = await queue.get()
                processed.append((time.time(), item))
                await asyncio.sleep(interval)

        start = time.time()
        await asyncio.gather(producer(), rate_limited_consumer())
        duration = time.time() - start

        # Should take at least 2 seconds for 20 items at 10/sec
        assert duration >= 1.9
        assert len(processed) == 20


class TestStressScenarios:
    """Combined stress test scenarios."""

    @pytest.mark.asyncio
    async def test_debate_simulation_under_load(self):
        """Simulate multiple concurrent debates under load."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("debate_stress", failure_threshold=10, cooldown_seconds=1.0)

        active_debates: dict[str, dict[str, Any]] = {}
        completed_debates: list[str] = []
        lock = asyncio.Lock()

        async def run_debate(debate_id: str, rounds: int = 3):
            if not cb.can_proceed():
                cb.record_failure()
                return None

            async with lock:
                active_debates[debate_id] = {"rounds_completed": 0}

            try:
                for r in range(rounds):
                    # Simulate agent responses
                    await asyncio.sleep(random.uniform(0.005, 0.02))

                    # Random failure
                    if random.random() < 0.1:
                        cb.record_failure()
                        raise Exception("Debate failed")

                    async with lock:
                        active_debates[debate_id]["rounds_completed"] = r + 1

                cb.record_success()
                async with lock:
                    completed_debates.append(debate_id)
                    del active_debates[debate_id]

                return debate_id
            except Exception:
                async with lock:
                    if debate_id in active_debates:
                        del active_debates[debate_id]
                return None

        # Run 50 concurrent debates
        tasks = [run_debate(f"debate_{i}") for i in range(50)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]

        # Should have some successful debates
        assert len(successful) > 0
        assert len(active_debates) == 0  # All cleaned up

    @pytest.mark.asyncio
    async def test_mixed_workload_resilience(self):
        """Test resilience under mixed workload types."""
        stats = {
            "reads": 0,
            "writes": 0,
            "compute": 0,
            "errors": 0,
        }
        lock = asyncio.Lock()

        async def read_operation():
            await asyncio.sleep(0.005)
            async with lock:
                stats["reads"] += 1

        async def write_operation():
            await asyncio.sleep(0.01)
            async with lock:
                stats["writes"] += 1

        async def compute_operation():
            # Simulate CPU-bound work
            _ = sum(i * i for i in range(1000))
            async with lock:
                stats["compute"] += 1

        async def failing_operation():
            async with lock:
                stats["errors"] += 1
            if random.random() < 0.3:
                raise Exception("Random failure")

        # Generate mixed workload
        tasks = []
        for _ in range(100):
            op = random.choice(
                [
                    read_operation,
                    write_operation,
                    compute_operation,
                    failing_operation,
                ]
            )
            tasks.append(op())

        await asyncio.gather(*tasks, return_exceptions=True)

        # All operation types should have been executed
        assert stats["reads"] > 0
        assert stats["writes"] > 0
        assert stats["compute"] > 0
        # Some errors expected
        total_ops = sum(stats.values())
        assert total_ops == 100
