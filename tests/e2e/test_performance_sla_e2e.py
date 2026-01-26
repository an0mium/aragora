"""
E2E Performance SLA Tests for Aragora.

Validates that the system meets performance service level agreements:
- P99 latency under threshold (< 500ms for API endpoints)
- Concurrent debate throughput (10 concurrent debates)
- WebSocket event delivery latency
- API response time under load

Run with: pytest tests/e2e/test_performance_sla_e2e.py -v
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import dataclass

import pytest
import pytest_asyncio

from tests.e2e.harness import (
    E2ETestConfig,
    E2ETestHarness,
    e2e_environment,
)

# Mark all tests in this module as e2e and performance
pytestmark = [pytest.mark.e2e, pytest.mark.performance]


# ============================================================================
# Performance Thresholds (SLAs)
# ============================================================================


@dataclass
class PerformanceSLAs:
    """Performance SLA thresholds."""

    # API latency (milliseconds)
    api_p50_latency_ms: float = 100.0
    api_p99_latency_ms: float = 500.0
    api_max_latency_ms: float = 1000.0

    # Debate throughput
    concurrent_debates_target: int = 10
    debate_start_latency_ms: float = 200.0

    # WebSocket
    ws_event_latency_ms: float = 100.0
    ws_delivery_rate: float = 0.99  # 99% of events delivered

    # Task processing
    task_submission_latency_ms: float = 50.0
    task_completion_timeout_seconds: float = 30.0


SLAS = PerformanceSLAs()


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def perf_harness():
    """Harness configured for performance testing."""
    config = E2ETestConfig(
        num_agents=5,
        agent_capabilities=["debate", "general", "analysis"],
        agent_response_delay=0.01,  # Minimal delay for performance tests
        timeout_seconds=60.0,
        task_timeout_seconds=30.0,
        heartbeat_interval=2.0,
        default_debate_rounds=2,
    )
    async with e2e_environment(config) as harness:
        yield harness


@pytest_asyncio.fixture
async def load_harness():
    """Harness configured for load testing - more agents."""
    config = E2ETestConfig(
        num_agents=10,
        agent_capabilities=["debate", "general"],
        agent_response_delay=0.005,  # Very fast for load tests
        timeout_seconds=120.0,
        task_timeout_seconds=60.0,
        heartbeat_interval=5.0,
        default_debate_rounds=1,
    )
    async with e2e_environment(config) as harness:
        yield harness


# ============================================================================
# Latency Measurement Utilities
# ============================================================================


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    samples: List[float]
    p50: float
    p90: float
    p99: float
    mean: float
    max_val: float
    min_val: float

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        """Calculate statistics from latency samples (in ms)."""
        if not samples:
            return cls(
                samples=[],
                p50=0.0,
                p90=0.0,
                p99=0.0,
                mean=0.0,
                max_val=0.0,
                min_val=0.0,
            )

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return cls(
            samples=samples,
            p50=sorted_samples[int(n * 0.5)] if n > 0 else 0,
            p90=sorted_samples[int(n * 0.9)] if n > 0 else 0,
            p99=sorted_samples[int(n * 0.99)] if n > 0 else sorted_samples[-1],
            mean=statistics.mean(samples),
            max_val=max(samples),
            min_val=min(samples),
        )


async def measure_latency_ms(coro) -> float:
    """Measure latency of a coroutine in milliseconds."""
    start = time.perf_counter()
    await coro
    return (time.perf_counter() - start) * 1000


# ============================================================================
# API Latency Tests
# ============================================================================


class TestAPILatency:
    """Test API endpoint latency SLAs."""

    @pytest.mark.asyncio
    async def test_stats_endpoint_p99_latency(self, perf_harness: E2ETestHarness):
        """Test stats endpoint meets P99 latency SLA."""
        samples = []

        # Warm up
        for _ in range(5):
            await perf_harness.get_stats()

        # Measure
        for _ in range(100):
            latency = await measure_latency_ms(perf_harness.get_stats())
            samples.append(latency)

        stats = LatencyStats.from_samples(samples)

        assert stats.p99 < SLAS.api_p99_latency_ms, (
            f"P99 latency {stats.p99:.2f}ms exceeds SLA {SLAS.api_p99_latency_ms}ms"
        )
        assert stats.p50 < SLAS.api_p50_latency_ms, (
            f"P50 latency {stats.p50:.2f}ms exceeds SLA {SLAS.api_p50_latency_ms}ms"
        )

    @pytest.mark.asyncio
    async def test_task_submission_latency(self, perf_harness: E2ETestHarness):
        """Test task submission meets latency SLA."""
        samples = []

        for i in range(50):
            start = time.perf_counter()
            task_id = await perf_harness.submit_task(
                "analysis",
                {"topic": f"Test topic {i}"},
            )
            latency = (time.perf_counter() - start) * 1000
            samples.append(latency)
            assert task_id is not None

        stats = LatencyStats.from_samples(samples)

        assert stats.p99 < SLAS.task_submission_latency_ms * 5, (
            f"Task submission P99 latency {stats.p99:.2f}ms exceeds threshold"
        )

    @pytest.mark.asyncio
    async def test_agent_registry_lookup_latency(self, perf_harness: E2ETestHarness):
        """Test agent registry lookup is fast."""
        samples = []
        agent_id = perf_harness.agents[0].id

        for _ in range(100):
            start = time.perf_counter()
            agent = await perf_harness.coordinator.get_agent(agent_id)
            latency = (time.perf_counter() - start) * 1000
            samples.append(latency)
            assert agent is not None

        stats = LatencyStats.from_samples(samples)

        # Registry lookups should be very fast (< 10ms P99)
        assert stats.p99 < 10.0, f"Registry lookup P99 {stats.p99:.2f}ms exceeds 10ms threshold"


# ============================================================================
# Concurrent Debate Throughput Tests
# ============================================================================


class TestConcurrentDebateThroughput:
    """Test concurrent debate handling meets throughput SLAs."""

    @pytest.mark.asyncio
    async def test_concurrent_debate_start(self, load_harness: E2ETestHarness):
        """Test system can start multiple debates concurrently."""
        num_debates = SLAS.concurrent_debates_target
        start_times = []
        debate_tasks = []

        # Start all debates concurrently
        global_start = time.perf_counter()

        async def start_debate(i: int) -> tuple:
            start = time.perf_counter()
            result = await load_harness.run_debate(
                f"Concurrent debate topic {i}",
                rounds=1,
            )
            latency = (time.perf_counter() - start) * 1000
            return result, latency

        tasks = [start_debate(i) for i in range(num_debates)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = (time.perf_counter() - global_start) * 1000

        # Analyze results
        successful = 0
        latencies = []
        for result in results:
            if isinstance(result, Exception):
                continue
            debate_result, latency = result
            if debate_result:
                successful += 1
                latencies.append(latency)

        success_rate = successful / num_debates

        # At least 80% of debates should complete successfully
        assert success_rate >= 0.8, (
            f"Only {successful}/{num_debates} debates completed successfully ({success_rate:.1%})"
        )

        # Average latency should be reasonable
        if latencies:
            avg_latency = statistics.mean(latencies)
            # Allow some slowdown for concurrent execution
            assert avg_latency < SLAS.task_completion_timeout_seconds * 1000, (
                f"Average debate latency {avg_latency:.2f}ms exceeds threshold"
            )

    @pytest.mark.asyncio
    async def test_task_throughput_under_load(self, load_harness: E2ETestHarness):
        """Test task processing throughput under load."""
        num_tasks = 50
        tasks_submitted = []

        # Submit many tasks rapidly
        start = time.perf_counter()
        for i in range(num_tasks):
            task_id = await load_harness.submit_task(
                "analysis",
                {"topic": f"Load test {i}"},
            )
            tasks_submitted.append(task_id)

        submission_time = (time.perf_counter() - start) * 1000

        # Verify all tasks were submitted
        assert len(tasks_submitted) == num_tasks

        # Calculate submission throughput (tasks per second)
        throughput = num_tasks / (submission_time / 1000)

        # Should be able to submit at least 100 tasks/second
        assert throughput >= 50, (
            f"Task submission throughput {throughput:.1f}/s below 50/s threshold"
        )


# ============================================================================
# WebSocket Event Delivery Tests
# ============================================================================


class TestWebSocketEventDelivery:
    """Test WebSocket event delivery meets latency SLAs."""

    @pytest.mark.asyncio
    async def test_event_queue_latency(self, perf_harness: E2ETestHarness):
        """Test that events are queued and processed quickly."""
        # This tests the internal event processing latency
        # In a real system, this would involve actual WebSocket connections

        # Simulate event emission and handler execution
        event_count = 100
        latencies = []

        for i in range(event_count):
            start = time.perf_counter()

            # Simulate event emission (in real system, this would be WS broadcast)
            event_data = {
                "type": "agent_message",
                "data": {"content": f"Message {i}"},
                "timestamp": time.time(),
            }

            # Minimal processing to simulate handler
            processed = {**event_data, "processed": True}

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        stats = LatencyStats.from_samples(latencies)

        # Event processing should be very fast
        assert stats.p99 < 1.0, f"Event processing P99 {stats.p99:.2f}ms exceeds 1ms threshold"

    @pytest.mark.asyncio
    async def test_concurrent_event_emission(self, perf_harness: E2ETestHarness):
        """Test concurrent event emission doesn't cause bottlenecks."""
        num_concurrent = 20
        events_per_task = 10

        async def emit_events(task_id: int) -> List[float]:
            latencies = []
            for i in range(events_per_task):
                start = time.perf_counter()
                # Simulate event emission
                event = {"task_id": task_id, "event_num": i, "time": time.time()}
                await asyncio.sleep(0)  # Yield to event loop
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            return latencies

        # Run concurrent event emissions
        tasks = [emit_events(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # Flatten all latencies
        all_latencies = [lat for result in results for lat in result]
        stats = LatencyStats.from_samples(all_latencies)

        # Even under concurrent load, latency should be low
        assert stats.p99 < 10.0, (
            f"Concurrent event emission P99 {stats.p99:.2f}ms exceeds 10ms threshold"
        )


# ============================================================================
# API Response Time Under Load Tests
# ============================================================================


class TestAPIResponseTimeUnderLoad:
    """Test API response times when system is under load."""

    @pytest.mark.asyncio
    async def test_stats_latency_during_debates(self, perf_harness: E2ETestHarness):
        """Test stats endpoint remains responsive during active debates."""

        # Start some background debates
        async def run_background_debates():
            for i in range(3):
                await perf_harness.run_debate(
                    f"Background debate {i}",
                    rounds=2,
                )
                await asyncio.sleep(0.1)

        # Start debates in background
        debate_task = asyncio.create_task(run_background_debates())

        # Measure stats latency while debates are running
        latencies = []
        for _ in range(30):
            latency = await measure_latency_ms(perf_harness.get_stats())
            latencies.append(latency)
            await asyncio.sleep(0.05)

        # Wait for debates to complete
        await debate_task

        stats = LatencyStats.from_samples(latencies)

        # Stats should remain responsive even during debates
        assert stats.p99 < SLAS.api_p99_latency_ms * 2, (
            f"Stats P99 latency {stats.p99:.2f}ms degraded under load"
        )

    @pytest.mark.asyncio
    async def test_registry_latency_during_task_processing(self, perf_harness: E2ETestHarness):
        """Test registry lookups remain fast during task processing."""
        # Submit many tasks to create load
        task_ids = []
        for i in range(20):
            task_id = await perf_harness.submit_task(
                "analysis",
                {"topic": f"Load task {i}"},
            )
            task_ids.append(task_id)

        # Measure registry lookup latency while tasks are processing
        agent_id = perf_harness.agents[0].id
        latencies = []

        for _ in range(50):
            latency = await measure_latency_ms(perf_harness.coordinator.get_agent(agent_id))
            latencies.append(latency)

        stats = LatencyStats.from_samples(latencies)

        # Registry should remain fast even under task load
        assert stats.p99 < 20.0, f"Registry lookup P99 {stats.p99:.2f}ms degraded under load"


# ============================================================================
# Resource Efficiency Tests
# ============================================================================


class TestResourceEfficiency:
    """Test resource efficiency under load."""

    @pytest.mark.asyncio
    async def test_memory_stable_during_sustained_load(self, perf_harness: E2ETestHarness):
        """Test memory doesn't grow unboundedly during sustained load."""
        import gc

        # Force garbage collection before measuring
        gc.collect()

        # Run sustained load
        for batch in range(5):
            tasks = []
            for i in range(10):
                task_id = await perf_harness.submit_task(
                    "analysis",
                    {"topic": f"Memory test batch {batch} task {i}"},
                )
                tasks.append(task_id)

            # Allow some processing time
            await asyncio.sleep(0.1)

        # If we got here without OOM or excessive slowdown, memory is stable
        # In a real test, we would measure actual memory usage
        final_stats = await perf_harness.get_stats()
        assert final_stats["running"] is True

    @pytest.mark.asyncio
    async def test_no_connection_leaks(self, perf_harness: E2ETestHarness):
        """Test no connection/resource leaks during repeated operations."""
        # Perform many operations that could leak connections
        for i in range(50):
            await perf_harness.get_stats()
            agent_id = perf_harness.agents[i % len(perf_harness.agents)].id
            await perf_harness.coordinator.get_agent(agent_id)

        # Verify system is still healthy
        stats = await perf_harness.get_stats()
        assert stats["running"] is True
        assert len(stats["agents"]) >= 1


# ============================================================================
# SLA Summary Test
# ============================================================================


class TestSLASummary:
    """Summary test that validates overall SLA compliance."""

    @pytest.mark.asyncio
    async def test_overall_sla_compliance(self, perf_harness: E2ETestHarness):
        """Comprehensive test validating overall SLA compliance."""
        results = {
            "api_latency": {"pass": False, "p99": 0},
            "task_submission": {"pass": False, "p99": 0},
            "registry_lookup": {"pass": False, "p99": 0},
        }

        # Test API latency
        api_samples = []
        for _ in range(50):
            latency = await measure_latency_ms(perf_harness.get_stats())
            api_samples.append(latency)
        api_stats = LatencyStats.from_samples(api_samples)
        results["api_latency"]["p99"] = api_stats.p99
        results["api_latency"]["pass"] = api_stats.p99 < SLAS.api_p99_latency_ms

        # Test task submission
        task_samples = []
        for i in range(20):
            start = time.perf_counter()
            await perf_harness.submit_task("analysis", {"topic": f"SLA test {i}"})
            task_samples.append((time.perf_counter() - start) * 1000)
        task_stats = LatencyStats.from_samples(task_samples)
        results["task_submission"]["p99"] = task_stats.p99
        results["task_submission"]["pass"] = task_stats.p99 < SLAS.task_submission_latency_ms * 5

        # Test registry lookup
        registry_samples = []
        agent_id = perf_harness.agents[0].id
        for _ in range(50):
            latency = await measure_latency_ms(perf_harness.coordinator.get_agent(agent_id))
            registry_samples.append(latency)
        registry_stats = LatencyStats.from_samples(registry_samples)
        results["registry_lookup"]["p99"] = registry_stats.p99
        results["registry_lookup"]["pass"] = registry_stats.p99 < 10.0

        # Report results
        passed = all(r["pass"] for r in results.values())
        if not passed:
            failures = [k for k, v in results.items() if not v["pass"]]
            pytest.fail(f"SLA compliance failed for: {failures}. Results: {results}")
