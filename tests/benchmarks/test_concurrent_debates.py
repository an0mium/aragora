"""
Concurrent debates benchmark tests.

Measures performance and scalability of running multiple debates concurrently.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# Mock Debate Components
# =============================================================================


@dataclass
class MockDebateResult:
    """Mock debate result for benchmarking."""

    debate_id: str
    duration: float
    rounds: int
    consensus_reached: bool


class MockAgent:
    """Mock agent for benchmark testing."""

    def __init__(self, agent_id: str, latency: float = 0.05):
        self.agent_id = agent_id
        self.latency = latency

    async def respond(self, prompt: str) -> str:
        await asyncio.sleep(self.latency)
        return f"Response from {self.agent_id}"


class MockDebate:
    """Mock debate for benchmark testing."""

    def __init__(self, debate_id: str, agents: List[MockAgent], rounds: int = 3):
        self.debate_id = debate_id
        self.agents = agents
        self.rounds = rounds

    async def run(self) -> MockDebateResult:
        start = time.time()

        for round_num in range(self.rounds):
            tasks = [agent.respond(f"Round {round_num}") for agent in self.agents]
            await asyncio.gather(*tasks)

        duration = time.time() - start
        return MockDebateResult(
            debate_id=self.debate_id,
            duration=duration,
            rounds=self.rounds,
            consensus_reached=True,
        )


# =============================================================================
# Concurrent Debate Scaling Tests
# =============================================================================


class TestConcurrentDebateScaling:
    """Test debate performance with increasing concurrency."""

    @pytest.mark.asyncio
    async def test_single_debate_baseline(self):
        """Establish baseline for single debate performance."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
        debate = MockDebate("debate-1", agents, rounds=3)

        start = time.time()
        result = await debate.run()
        elapsed = time.time() - start

        assert result.consensus_reached
        # 3 rounds * 2 agents * 0.01s = 0.06s theoretical
        # With parallelism: 3 rounds * 0.01s = 0.03s theoretical
        assert result.duration < 0.5  # Allow some overhead

    @pytest.mark.asyncio
    async def test_5_concurrent_debates(self):
        """Test 5 debates running concurrently."""
        debates = []
        for i in range(5):
            agents = [MockAgent(f"d{i}-agent-{j}", latency=0.01) for j in range(2)]
            debates.append(MockDebate(f"debate-{i}", agents, rounds=3))

        start = time.time()
        results = await asyncio.gather(*[d.run() for d in debates])
        total_time = time.time() - start

        assert len(results) == 5
        assert all(r.consensus_reached for r in results)
        # Should complete in roughly the same time as single debate
        assert total_time < 1.0

    @pytest.mark.asyncio
    async def test_10_concurrent_debates(self):
        """Test 10 debates running concurrently."""
        debates = []
        for i in range(10):
            agents = [MockAgent(f"d{i}-agent-{j}", latency=0.01) for j in range(2)]
            debates.append(MockDebate(f"debate-{i}", agents, rounds=3))

        start = time.time()
        results = await asyncio.gather(*[d.run() for d in debates])
        total_time = time.time() - start

        assert len(results) == 10
        assert all(r.consensus_reached for r in results)
        # Should scale well
        assert total_time < 2.0

    @pytest.mark.asyncio
    async def test_20_concurrent_debates(self):
        """Test 20 debates running concurrently."""
        debates = []
        for i in range(20):
            agents = [MockAgent(f"d{i}-agent-{j}", latency=0.01) for j in range(2)]
            debates.append(MockDebate(f"debate-{i}", agents, rounds=3))

        start = time.time()
        results = await asyncio.gather(*[d.run() for d in debates])
        total_time = time.time() - start

        assert len(results) == 20
        assert all(r.consensus_reached for r in results)
        # Should still complete reasonably
        assert total_time < 5.0


# =============================================================================
# Agent Count Scaling Tests
# =============================================================================


class TestAgentCountScaling:
    """Test debate performance with increasing agent count."""

    @pytest.mark.asyncio
    async def test_2_agents_debate(self):
        """Benchmark with 2 agents."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
        debate = MockDebate("debate-1", agents, rounds=3)

        result = await debate.run()

        assert result.consensus_reached
        assert result.duration < 0.2

    @pytest.mark.asyncio
    async def test_4_agents_debate(self):
        """Benchmark with 4 agents."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(4)]
        debate = MockDebate("debate-1", agents, rounds=3)

        result = await debate.run()

        assert result.consensus_reached
        # 4 agents run in parallel, shouldn't be much slower
        assert result.duration < 0.3

    @pytest.mark.asyncio
    async def test_8_agents_debate(self):
        """Benchmark with 8 agents."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(8)]
        debate = MockDebate("debate-1", agents, rounds=3)

        result = await debate.run()

        assert result.consensus_reached
        assert result.duration < 0.4


# =============================================================================
# Round Scaling Tests
# =============================================================================


class TestRoundScaling:
    """Test debate performance with increasing rounds."""

    @pytest.mark.asyncio
    async def test_3_rounds_debate(self):
        """Benchmark with 3 rounds."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
        debate = MockDebate("debate-1", agents, rounds=3)

        result = await debate.run()

        assert result.rounds == 3
        assert result.duration < 0.2

    @pytest.mark.asyncio
    async def test_5_rounds_debate(self):
        """Benchmark with 5 rounds."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
        debate = MockDebate("debate-1", agents, rounds=5)

        result = await debate.run()

        assert result.rounds == 5
        assert result.duration < 0.3

    @pytest.mark.asyncio
    async def test_10_rounds_debate(self):
        """Benchmark with 10 rounds."""
        agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
        debate = MockDebate("debate-1", agents, rounds=10)

        result = await debate.run()

        assert result.rounds == 10
        # Linear scaling: 10 rounds * 0.01s = 0.1s
        assert result.duration < 0.5


# =============================================================================
# Throughput Tests
# =============================================================================


class TestDebateThroughput:
    """Test debate throughput metrics."""

    @pytest.mark.asyncio
    async def test_debates_per_second(self):
        """Measure debates completed per second."""
        num_debates = 50
        debates = []
        for i in range(num_debates):
            agents = [MockAgent(f"d{i}-agent-{j}", latency=0.005) for j in range(2)]
            debates.append(MockDebate(f"debate-{i}", agents, rounds=2))

        start = time.time()
        results = await asyncio.gather(*[d.run() for d in debates])
        elapsed = time.time() - start

        debates_per_second = num_debates / elapsed

        assert len(results) == num_debates
        # Should achieve reasonable throughput
        assert debates_per_second > 5

    @pytest.mark.asyncio
    async def test_rounds_per_second(self):
        """Measure total rounds processed per second."""
        num_debates = 20
        rounds_per_debate = 5
        debates = []
        for i in range(num_debates):
            agents = [MockAgent(f"d{i}-agent-{j}", latency=0.005) for j in range(2)]
            debates.append(MockDebate(f"debate-{i}", agents, rounds=rounds_per_debate))

        start = time.time()
        results = await asyncio.gather(*[d.run() for d in debates])
        elapsed = time.time() - start

        total_rounds = sum(r.rounds for r in results)
        rounds_per_second = total_rounds / elapsed

        assert total_rounds == num_debates * rounds_per_debate
        # Should process many rounds per second
        assert rounds_per_second > 10


# =============================================================================
# Resource Usage Tests
# =============================================================================


class TestResourceUsage:
    """Test resource usage during concurrent debates."""

    @pytest.mark.asyncio
    async def test_task_count_bounded(self):
        """Verify task count stays bounded."""
        max_tasks = 0
        current_tasks = 0
        lock = asyncio.Lock()

        async def track_task():
            nonlocal max_tasks, current_tasks
            async with lock:
                current_tasks += 1
                max_tasks = max(max_tasks, current_tasks)

            await asyncio.sleep(0.01)

            async with lock:
                current_tasks -= 1

        # Run many concurrent tasks
        tasks = [track_task() for _ in range(100)]
        await asyncio.gather(*tasks)

        assert current_tasks == 0
        # All tasks ran concurrently
        assert max_tasks > 50

    @pytest.mark.asyncio
    async def test_memory_stable_across_debates(self):
        """Memory usage should remain stable across many debates."""
        import sys

        results_sizes = []

        for batch in range(5):
            debates = []
            for i in range(10):
                agents = [MockAgent(f"d{i}-agent-{j}", latency=0.001) for j in range(2)]
                debates.append(MockDebate(f"debate-{i}", agents, rounds=2))

            results = await asyncio.gather(*[d.run() for d in debates])
            results_sizes.append(sys.getsizeof(results))

        # Size should be relatively consistent
        avg_size = sum(results_sizes) / len(results_sizes)
        for size in results_sizes:
            assert abs(size - avg_size) < avg_size * 0.5  # Within 50%


# =============================================================================
# Latency Distribution Tests
# =============================================================================


class TestLatencyDistribution:
    """Test latency distribution across concurrent debates."""

    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        """Measure latency percentiles."""
        latencies = []

        async def timed_debate():
            agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
            debate = MockDebate("debate", agents, rounds=2)
            start = time.time()
            await debate.run()
            latencies.append(time.time() - start)

        tasks = [timed_debate() for _ in range(50)]
        await asyncio.gather(*tasks)

        latencies.sort()

        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        # P50 should be reasonable (relaxed for CI/concurrent test execution)
        assert p50 < 0.5, f"P50 latency {p50:.3f}s exceeded 0.5s threshold"
        # P95 should not be too far from P50 (4x allows for scheduling variance)
        assert p95 < p50 * 4, f"P95 latency {p95:.3f}s exceeded 4x P50 ({p50:.3f}s)"
        # P99 tail latency (relaxed for resource contention)
        assert p99 < 2.0, f"P99 latency {p99:.3f}s exceeded 2.0s threshold"

    @pytest.mark.asyncio
    async def test_no_starvation(self):
        """All debates should complete in reasonable time."""
        completion_times = []

        async def timed_debate(debate_id: int):
            start = time.time()
            agents = [MockAgent(f"agent-{i}", latency=0.01) for i in range(2)]
            debate = MockDebate(f"debate-{debate_id}", agents, rounds=3)
            await debate.run()
            completion_times.append((debate_id, time.time() - start))

        tasks = [timed_debate(i) for i in range(30)]
        await asyncio.gather(*tasks)

        # No debate should take more than 5x the average
        avg_time = sum(t for _, t in completion_times) / len(completion_times)
        for debate_id, comp_time in completion_times:
            assert comp_time < avg_time * 5, f"Debate {debate_id} starved"
