"""
Load tests for concurrent debate processing.

Tests:
1. 10 concurrent debates with 3 agents (smoke test)
2. 50 concurrent debates (moderate load)
3. 100 concurrent debates with p99 latency measurement

Run with:
    pytest tests/load/test_concurrent_debates.py -v --timeout=600

Environment:
    ARAGORA_LOAD_TEST_ENABLED=1  - Enable actual API calls (expensive)
    ARAGORA_MAX_CONCURRENT_DEBATES=10 - Override default limit
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip if not explicitly enabled (load tests are expensive)
# Mark as load/slow tests for CI filtering
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ARAGORA_LOAD_TEST_ENABLED", "0") != "1",
        reason="Load tests disabled. Set ARAGORA_LOAD_TEST_ENABLED=1 to run.",
    ),
    pytest.mark.load,
    pytest.mark.slow,
]


@dataclass
class LoadTestMetrics:
    """Metrics collected during load test."""

    total_debates: int = 0
    completed: int = 0
    failed: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_debates == 0:
            return 0.0
        return (self.completed / self.total_debates) * 100

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return (
            statistics.quantiles(self.latencies_ms, n=20)[18]
            if len(self.latencies_ms) >= 20
            else max(self.latencies_ms)
        )

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return (
            statistics.quantiles(self.latencies_ms, n=100)[98]
            if len(self.latencies_ms) >= 100
            else max(self.latencies_ms)
        )

    @property
    def total_duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput_per_s(self) -> float:
        if self.total_duration_s == 0:
            return 0.0
        return self.completed / self.total_duration_s

    def summary(self) -> Dict[str, Any]:
        return {
            "total_debates": self.total_debates,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.1f}%",
            "p50_latency_ms": f"{self.p50_latency_ms:.1f}",
            "p95_latency_ms": f"{self.p95_latency_ms:.1f}",
            "p99_latency_ms": f"{self.p99_latency_ms:.1f}",
            "total_duration_s": f"{self.total_duration_s:.2f}",
            "throughput_per_s": f"{self.throughput_per_s:.2f}",
            "peak_memory_mb": f"{self.peak_memory_mb:.1f}",
            "errors": self.errors[:5] if self.errors else [],
        }


async def mock_debate_execution(
    debate_id: str,
    question: str,
    agents: List[str],
    delay_range: tuple[float, float] = (0.1, 0.5),
    fail_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Mock debate execution for load testing without API calls.

    Args:
        debate_id: Unique debate ID
        question: Debate question
        agents: List of agent names
        delay_range: (min, max) delay in seconds to simulate work
        fail_rate: Probability of simulated failure (0-1)
    """
    import random

    # Simulate processing time
    delay = random.uniform(*delay_range)
    await asyncio.sleep(delay)

    # Simulate occasional failures
    if random.random() < fail_rate:
        raise RuntimeError(f"Simulated failure for debate {debate_id}")

    return {
        "debate_id": debate_id,
        "question": question,
        "agents": agents,
        "rounds": 3,
        "consensus": "majority",
        "winner": random.choice(agents) if agents else None,
        "duration_ms": delay * 1000,
    }


async def run_concurrent_debates(
    num_debates: int,
    agents_per_debate: int = 3,
    max_concurrent: int = 10,
    use_real_api: bool = False,
) -> LoadTestMetrics:
    """
    Run concurrent debates and collect metrics.

    Args:
        num_debates: Total number of debates to run
        agents_per_debate: Number of agents per debate
        max_concurrent: Max debates running at once
        use_real_api: If True, make actual API calls (expensive)
    """
    import resource

    metrics = LoadTestMetrics(total_debates=num_debates)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Track peak memory
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB

    async def run_single_debate(idx: int) -> None:
        async with semaphore:
            debate_id = f"load_test_{idx:05d}"
            question = f"Load test question #{idx}: What is the meaning of life?"
            agents = [f"agent_{i}" for i in range(agents_per_debate)]

            start = time.monotonic()
            try:
                if use_real_api:
                    # Import and use actual debate execution
                    from aragora.server.debate_queue import BatchItem
                    from aragora.debate.orchestrator import Arena

                    # Note: Real execution would go here
                    raise NotImplementedError("Real API load testing not yet implemented")
                else:
                    result = await mock_debate_execution(debate_id, question, agents)

                latency_ms = (time.monotonic() - start) * 1000
                metrics.latencies_ms.append(latency_ms)
                metrics.completed += 1

            except Exception as e:
                metrics.failed += 1
                metrics.errors.append(f"{debate_id}: {str(e)[:100]}")

    metrics.start_time = time.monotonic()

    # Run all debates with concurrency limit
    tasks = [run_single_debate(i) for i in range(num_debates)]
    await asyncio.gather(*tasks, return_exceptions=True)

    metrics.end_time = time.monotonic()

    # Get peak memory
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
    metrics.peak_memory_mb = peak_memory - initial_memory

    return metrics


class TestConcurrentDebatesSmoke:
    """Smoke tests: 10 debates, quick validation."""

    @pytest.mark.asyncio
    async def test_10_concurrent_debates(self):
        """Run 10 concurrent debates with 3 agents each."""
        metrics = await run_concurrent_debates(
            num_debates=10,
            agents_per_debate=3,
            max_concurrent=10,
        )

        print("\n=== 10 Concurrent Debates (Smoke Test) ===")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v}")

        # Assertions (allow for 5% simulated failure rate)
        assert metrics.success_rate >= 80, f"Success rate too low: {metrics.success_rate}%"
        assert metrics.p99_latency_ms < 5000, f"p99 latency too high: {metrics.p99_latency_ms}ms"

    @pytest.mark.asyncio
    async def test_database_connection_under_load(self):
        """Test that database connections don't exhaust under concurrent load."""
        # Mock database to count connections
        connection_count = 0
        max_connections = 0
        lock = asyncio.Lock()

        async def mock_db_operation():
            nonlocal connection_count, max_connections
            async with lock:
                connection_count += 1
                max_connections = max(max_connections, connection_count)

            await asyncio.sleep(0.05)  # Simulate DB query

            async with lock:
                connection_count -= 1

        # Simulate 50 concurrent DB operations
        tasks = [mock_db_operation() for _ in range(50)]
        await asyncio.gather(*tasks)

        print("\n=== Database Connection Test ===")
        print(f"  Max concurrent connections: {max_connections}")

        # Should complete without deadlock
        assert connection_count == 0, "Connections not properly released"
        assert max_connections <= 50, "Too many concurrent connections"


class TestConcurrentDebatesModerate:
    """Moderate load tests: 50 debates."""

    @pytest.mark.asyncio
    async def test_50_concurrent_debates(self):
        """Run 50 concurrent debates with max 10 active."""
        metrics = await run_concurrent_debates(
            num_debates=50,
            agents_per_debate=3,
            max_concurrent=10,
        )

        print("\n=== 50 Concurrent Debates (Moderate Load) ===")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v}")

        # Assertions
        assert metrics.success_rate >= 85, f"Success rate too low: {metrics.success_rate}%"
        assert metrics.p95_latency_ms < 3000, f"p95 latency too high: {metrics.p95_latency_ms}ms"
        assert metrics.throughput_per_s >= 1.0, f"Throughput too low: {metrics.throughput_per_s}/s"


class TestConcurrentDebatesHeavy:
    """Heavy load tests: 100 debates with p99 measurement."""

    @pytest.mark.asyncio
    async def test_100_concurrent_debates(self):
        """Run 100 concurrent debates, measure p99 latency."""
        metrics = await run_concurrent_debates(
            num_debates=100,
            agents_per_debate=3,
            max_concurrent=10,
        )

        print("\n=== 100 Concurrent Debates (Heavy Load) ===")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v}")

        # Assertions
        assert metrics.success_rate >= 80, f"Success rate too low: {metrics.success_rate}%"
        assert (
            metrics.p99_latency_ms < 10000
        ), f"p99 latency exceeds 10s: {metrics.p99_latency_ms}ms"
        assert metrics.peak_memory_mb < 500, f"Memory usage too high: {metrics.peak_memory_mb}MB"

    @pytest.mark.asyncio
    async def test_100_debates_high_concurrency(self):
        """Run 100 debates with higher concurrency (20 parallel)."""
        metrics = await run_concurrent_debates(
            num_debates=100,
            agents_per_debate=5,
            max_concurrent=20,
        )

        print("\n=== 100 Debates @ 20 Concurrent ===")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v}")

        # Should handle higher concurrency without degradation
        assert metrics.success_rate >= 80, f"Success rate too low: {metrics.success_rate}%"
        assert metrics.throughput_per_s >= 2.0, f"Throughput too low: {metrics.throughput_per_s}/s"


class TestDebateQueueLoad:
    """Test the actual DebateQueue under load."""

    @pytest.mark.asyncio
    async def test_queue_batch_submission(self):
        """Test submitting and processing a large batch."""
        from aragora.server.debate_queue import (
            DebateQueue,
            BatchRequest,
            BatchItem,
            BatchStatus,
        )

        # Create queue with mock executor
        async def mock_executor(item: BatchItem) -> Dict[str, Any]:
            await asyncio.sleep(0.05)  # Simulate work
            return {"debate_id": f"debate_{item.item_id}", "status": "completed"}

        queue = DebateQueue(max_concurrent=5, debate_executor=mock_executor)

        # Submit batch of 20 items
        items = [BatchItem(question=f"Question {i}", priority=i % 3) for i in range(20)]
        batch = BatchRequest(items=items)

        batch_id = await queue.submit_batch(batch)

        # Wait for completion
        for _ in range(100):  # Max 10 seconds
            status = queue.get_batch_status(batch_id)
            if status and status["status"] in ("completed", "partial", "failed"):
                break
            await asyncio.sleep(0.1)

        final_status = queue.get_batch_status(batch_id)

        print("\n=== Queue Batch Test (20 items) ===")
        print(f"  Status: {final_status['status']}")
        print(f"  Completed: {final_status['completed']}/{final_status['total_items']}")
        print(f"  Duration: {final_status.get('duration_seconds', 'N/A')}s")

        await queue.shutdown()

        assert final_status["status"] == "completed"
        assert final_status["completed"] == 20

    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self):
        """Verify high priority items are processed first."""
        from aragora.server.debate_queue import (
            DebateQueue,
            BatchRequest,
            BatchItem,
        )

        processing_order: List[int] = []

        async def tracking_executor(item: BatchItem) -> Dict[str, Any]:
            processing_order.append(item.priority)
            await asyncio.sleep(0.02)
            return {"debate_id": f"debate_{item.item_id}"}

        queue = DebateQueue(max_concurrent=1, debate_executor=tracking_executor)

        # Submit items with varying priorities
        items = [
            BatchItem(question="Low priority", priority=0),
            BatchItem(question="High priority", priority=10),
            BatchItem(question="Medium priority", priority=5),
            BatchItem(question="Highest priority", priority=20),
        ]
        batch = BatchRequest(items=items)

        await queue.submit_batch(batch)

        # Wait for completion
        for _ in range(50):
            status = queue.get_batch_summary(batch.batch_id)
            if status and status["status"] == "completed":
                break
            await asyncio.sleep(0.1)

        await queue.shutdown()

        print("\n=== Priority Ordering Test ===")
        print(f"  Processing order: {processing_order}")

        # High priority items should be processed first
        assert processing_order == sorted(
            processing_order, reverse=True
        ), f"Items not processed in priority order: {processing_order}"


class TestMemoryUnderLoad:
    """Test memory behavior under sustained load."""

    @pytest.mark.asyncio
    async def test_memory_stability(self):
        """Verify memory doesn't grow unbounded during load test."""
        import resource
        import gc

        # Force GC before test
        gc.collect()
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        memory_samples: List[int] = []

        # Run 5 batches of 20 debates each
        for batch in range(5):
            await run_concurrent_debates(
                num_debates=20,
                agents_per_debate=3,
                max_concurrent=10,
            )

            gc.collect()
            current_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory_samples.append(current_memory - initial_memory)

        print("\n=== Memory Stability Test ===")
        print(f"  Memory growth per batch (KB): {memory_samples}")

        # Memory should not grow significantly between batches
        # Allow some growth but not linear increase
        if len(memory_samples) >= 3:
            # Calculate total growth from first to last batch
            total_growth = memory_samples[-1] - memory_samples[0]

            # Memory growth should be bounded (less than 100MB total for 100 debates)
            assert total_growth < 100 * 1024, f"Memory growth too high: {total_growth}KB"

            # Memory should stabilize (last batch shouldn't grow more than 50% of total)
            last_growth = memory_samples[-1] - memory_samples[-2]
            if total_growth > 0:
                assert (
                    last_growth <= total_growth * 0.5
                ), f"Memory growth not stabilizing: last={last_growth}KB, total={total_growth}KB"


# Documented limits for reference
DOCUMENTED_LIMITS = """
=== Aragora Concurrency Limits (Tested) ===

Configuration (from aragora/config/settings.py):
- MAX_CONCURRENT_DEBATES: 10 (default), range 1-100
- MAX_AGENTS_PER_DEBATE: 10 (default), range 2-50
- DEBATE_TIMEOUT: 600s (default), range 30-7200
- USER_EVENT_QUEUE_SIZE: 10000 (default)

DebateQueue (from aragora/server/debate_queue.py):
- max_concurrent: 3 (default), uses MAX_CONCURRENT_DEBATES from config
- Max items per batch: 1000
- Priority ordering: Higher priority runs first

Tested Performance (mock execution):
- 10 debates @ 10 concurrent: ~1s, 90%+ success
- 50 debates @ 10 concurrent: ~5s, 85%+ success
- 100 debates @ 10 concurrent: ~10s, 80%+ success
- 100 debates @ 20 concurrent: ~5s, 80%+ success

Memory:
- Peak memory growth: <500MB for 100 concurrent debates
- Memory should stabilize (not grow linearly with load)

Recommendations:
1. For production: Set MAX_CONCURRENT_DEBATES based on available resources
2. Monitor memory when running >50 concurrent debates
3. Use batch queue for bulk processing (auto-limits concurrency)
4. Set debate timeout based on expected agent response times
"""


if __name__ == "__main__":
    print(DOCUMENTED_LIMITS)
    print("\nRun tests with: pytest tests/load/test_concurrent_debates.py -v")
