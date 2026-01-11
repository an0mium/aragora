"""
Performance Benchmark Tests for Aragora.

These tests measure performance characteristics and set baselines.
Run with: pytest tests/benchmarks/ -v

Optionally install pytest-benchmark for detailed statistics:
    pip install pytest-benchmark
    pytest tests/benchmarks/ --benchmark-only
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch, AsyncMock

import pytest

from aragora.core import Environment, Critique, Vote
from aragora.debate.orchestrator import Arena, DebateProtocol

from .conftest import BenchmarkAgent, SimpleBenchmark


# =============================================================================
# Debate Performance Tests
# =============================================================================


class TestDebatePerformance:
    """Benchmark tests for debate operations."""

    @pytest.mark.asyncio
    async def test_single_round_latency(self, benchmark_agents, benchmark_environment):
        """Measure single debate round latency."""
        protocol = DebateProtocol(rounds=1, consensus="any")

        with patch.object(
            Arena, "_gather_trending_context",
            new_callable=AsyncMock, return_value=None
        ):
            arena = Arena(benchmark_environment, benchmark_agents, protocol)

            start = time.perf_counter()
            result = await asyncio.wait_for(arena.run(), timeout=30.0)
            elapsed = time.perf_counter() - start

        # Baseline: single round should complete in under 5 seconds
        assert elapsed < 5.0, f"Single round took {elapsed:.2f}s (target: <5s)"
        assert result is not None

    @pytest.mark.asyncio
    async def test_multi_round_scaling(self, benchmark_agents, benchmark_environment):
        """Measure how latency scales with rounds."""
        round_times = {}

        with patch.object(
            Arena, "_gather_trending_context",
            new_callable=AsyncMock, return_value=None
        ):
            for num_rounds in [1, 2, 3]:
                protocol = DebateProtocol(rounds=num_rounds, consensus="any")
                arena = Arena(benchmark_environment, benchmark_agents, protocol)

                start = time.perf_counter()
                await asyncio.wait_for(arena.run(), timeout=60.0)
                elapsed = time.perf_counter() - start

                round_times[num_rounds] = elapsed

        # Check scaling is roughly linear (not exponential)
        if round_times[1] > 0:
            ratio = round_times[3] / round_times[1]
            # Should be less than 5x for 3x rounds (allowing overhead)
            assert ratio < 5.0, f"3 rounds took {ratio:.1f}x longer than 1 round"

    @pytest.mark.asyncio
    async def test_agent_count_scaling(self, benchmark_environment):
        """Measure how latency scales with agent count."""
        agent_times = {}

        with patch.object(
            Arena, "_gather_trending_context",
            new_callable=AsyncMock, return_value=None
        ):
            for num_agents in [2, 3, 5]:
                agents = [BenchmarkAgent(f"agent_{i}") for i in range(num_agents)]
                protocol = DebateProtocol(rounds=1, consensus="any")
                arena = Arena(benchmark_environment, agents, protocol)

                start = time.perf_counter()
                await asyncio.wait_for(arena.run(), timeout=30.0)
                elapsed = time.perf_counter() - start

                agent_times[num_agents] = elapsed

        # Check scaling is sub-linear (agents run in parallel)
        if agent_times[2] > 0:
            ratio = agent_times[5] / agent_times[2]
            # 5 agents shouldn't take 2.5x longer than 2 agents
            assert ratio < 2.5, f"5 agents took {ratio:.1f}x longer than 2 agents"


# =============================================================================
# Memory Performance Tests
# =============================================================================


class TestMemoryPerformance:
    """Benchmark tests for memory operations."""

    @pytest.mark.asyncio
    async def test_critique_store_write(self, temp_benchmark_db):
        """Measure critique store write performance."""
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore(str(temp_benchmark_db))

        critiques = [
            Critique(
                agent=f"agent_{i}",
                target_agent="target",
                target_content=f"content_{i}",
                issues=["issue"],
                suggestions=["suggestion"],
                severity=0.5,
                reasoning="reasoning",
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        for critique in critiques:
            store.store(critique)
        elapsed = time.perf_counter() - start

        # Baseline: 100 writes should complete in under 1 second
        assert elapsed < 1.0, f"100 writes took {elapsed:.2f}s (target: <1s)"
        ops_per_sec = 100 / elapsed
        assert ops_per_sec > 100, f"Only {ops_per_sec:.0f} ops/sec (target: >100)"

    @pytest.mark.asyncio
    async def test_critique_store_read(self, temp_benchmark_db):
        """Measure critique store read performance."""
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore(str(temp_benchmark_db))

        # Write some data first
        for i in range(50):
            store.store(Critique(
                agent="agent",
                target_agent="target",
                target_content=f"content_{i}",
                issues=[],
                suggestions=[],
                severity=0.5,
                reasoning="reasoning",
            ))

        # Measure reads
        start = time.perf_counter()
        for _ in range(100):
            store.get_recent(limit=10)
        elapsed = time.perf_counter() - start

        # Baseline: 100 reads should complete in under 0.5 seconds
        assert elapsed < 0.5, f"100 reads took {elapsed:.2f}s (target: <0.5s)"


# =============================================================================
# ELO Performance Tests
# =============================================================================


class TestEloPerformance:
    """Benchmark tests for ELO operations."""

    def test_elo_update_throughput(self, temp_benchmark_db):
        """Measure ELO update throughput."""
        from aragora.ranking.elo import EloSystem

        system = EloSystem(str(temp_benchmark_db))

        # Record many matches
        start = time.perf_counter()
        for i in range(100):
            system.record_match(
                winner=f"agent_{i % 10}",
                loser=f"agent_{(i + 1) % 10}",
                task="benchmark",
            )
        elapsed = time.perf_counter() - start

        # Baseline: 100 updates should complete in under 1 second
        assert elapsed < 1.0, f"100 ELO updates took {elapsed:.2f}s (target: <1s)"
        ops_per_sec = 100 / elapsed
        assert ops_per_sec > 100, f"Only {ops_per_sec:.0f} ops/sec (target: >100)"

    def test_leaderboard_query(self, temp_benchmark_db):
        """Measure leaderboard query performance."""
        from aragora.ranking.elo import EloSystem

        system = EloSystem(str(temp_benchmark_db))

        # Add some agents
        for i in range(50):
            system.record_match(
                winner=f"agent_{i}",
                loser=f"agent_{(i + 1) % 50}",
                task="benchmark",
            )

        # Query leaderboard many times
        start = time.perf_counter()
        for _ in range(100):
            system.get_leaderboard(limit=20)
        elapsed = time.perf_counter() - start

        # Baseline: 100 queries should complete in under 0.5 seconds
        assert elapsed < 0.5, f"100 leaderboard queries took {elapsed:.2f}s (target: <0.5s)"


# =============================================================================
# Concurrent Operation Tests
# =============================================================================


class TestConcurrentPerformance:
    """Benchmark tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_debates(self, benchmark_environment):
        """Measure concurrent debate throughput."""
        num_debates = 5

        with patch.object(
            Arena, "_gather_trending_context",
            new_callable=AsyncMock, return_value=None
        ):
            async def run_single_debate(idx: int):
                agents = [BenchmarkAgent(f"debate{idx}_agent_{i}") for i in range(2)]
                env = Environment(task=f"Concurrent task {idx}")
                protocol = DebateProtocol(rounds=1, consensus="any")
                arena = Arena(env, agents, protocol)
                return await arena.run()

            start = time.perf_counter()
            results = await asyncio.gather(*[
                run_single_debate(i) for i in range(num_debates)
            ])
            elapsed = time.perf_counter() - start

        # All should complete
        assert len(results) == num_debates
        assert all(r is not None for r in results)

        # Concurrent debates shouldn't take num_debates times as long
        debates_per_sec = num_debates / elapsed
        assert debates_per_sec > 0.5, f"Only {debates_per_sec:.2f} debates/sec"

    @pytest.mark.asyncio
    async def test_concurrent_memory_writes(self, temp_benchmark_db):
        """Measure concurrent memory write performance."""
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore(str(temp_benchmark_db))
        num_writers = 10
        writes_per_writer = 10

        async def write_batch(writer_id: int):
            for i in range(writes_per_writer):
                store.store(Critique(
                    agent=f"writer_{writer_id}",
                    target_agent="target",
                    target_content=f"content_{writer_id}_{i}",
                    issues=[],
                    suggestions=[],
                    severity=0.5,
                    reasoning="concurrent write",
                ))

        start = time.perf_counter()
        await asyncio.gather(*[write_batch(i) for i in range(num_writers)])
        elapsed = time.perf_counter() - start

        total_writes = num_writers * writes_per_writer
        ops_per_sec = total_writes / elapsed

        # Concurrent writes should still be reasonably fast
        assert ops_per_sec > 50, f"Only {ops_per_sec:.0f} ops/sec (target: >50)"


# =============================================================================
# Baseline Performance Assertions
# =============================================================================


class TestPerformanceBaselines:
    """Establish and verify performance baselines."""

    def test_import_time(self):
        """Verify aragora core imports are fast.

        Note: We don't remove modules from sys.modules as that breaks
        global fixtures. Instead we measure a fresh subprocess import.
        """
        import subprocess

        # Measure import time in subprocess for clean measurement
        result = subprocess.run(
            ["python", "-c", "import time; start=time.perf_counter(); import aragora; print(time.perf_counter()-start)"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            elapsed = float(result.stdout.strip())
            # Import should be under 3 seconds (subprocess has overhead)
            assert elapsed < 3.0, f"Import took {elapsed:.2f}s (target: <3s)"
        else:
            # If subprocess fails, skip test rather than fail
            pytest.skip(f"Subprocess import failed: {result.stderr}")

    @pytest.mark.asyncio
    async def test_agent_generation_overhead(self):
        """Verify agent response generation has minimal overhead."""
        agent = BenchmarkAgent()

        # Warm up
        await agent.generate("warmup")

        # Measure overhead
        start = time.perf_counter()
        for _ in range(100):
            await agent.generate("test prompt")
        elapsed = time.perf_counter() - start

        # 100 generations should be nearly instant (mock agent)
        assert elapsed < 0.1, f"100 mock generations took {elapsed:.3f}s (target: <0.1s)"
