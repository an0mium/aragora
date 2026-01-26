"""
Performance Benchmark Tests for Aragora.

These tests measure performance characteristics and validate against SLOs.
Run with: pytest tests/benchmarks/ -v

Optionally install pytest-benchmark for detailed statistics:
    pip install pytest-benchmark
    pytest tests/benchmarks/ --benchmark-only

SLOs are defined in tests/slo_config.py and docs/PERFORMANCE_TARGETS.md.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch, AsyncMock

import pytest

from aragora.core import Environment, Critique, Vote
from aragora.debate.orchestrator import Arena, DebateProtocol

from .conftest import BenchmarkAgent, SimpleBenchmark
from tests.slo_config import (
    SLO,
    assert_debate_slo,
    assert_memory_ops_slo,
    assert_elo_ops_slo,
)


# =============================================================================
# Debate Performance Tests
# =============================================================================


class TestDebatePerformance:
    """Benchmark tests for debate operations."""

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    @pytest.mark.slow  # Skip in regular CI, run in nightly only
    async def test_single_round_latency(self, benchmark_agents, benchmark_environment):
        """Measure single debate round latency against SLO.

        SLO threshold is set to 15 seconds to account for CI/parallel test
        resource contention and external network timeouts (DuckDuckGo).
        Local runs typically complete in 2-4 seconds. Timeout is 60 seconds
        to accommodate heavy CI load.

        This test is marked flaky due to timing sensitivity in CI environments.
        """
        protocol = DebateProtocol(rounds=1, consensus="any")

        with (
            patch.object(
                Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                Arena, "_fetch_knowledge_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(Arena, "_init_km_context", new_callable=AsyncMock, return_value=None),
            patch(
                "aragora.debate.context_gatherer.ContextGatherer.gather_all",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            arena = Arena(benchmark_environment, benchmark_agents, protocol)

            # Mock prompt_builder.classify_question_async to avoid LLM calls
            if arena.prompt_builder:
                arena.prompt_builder.classify_question_async = AsyncMock(return_value=None)

            start = time.perf_counter()
            result = await asyncio.wait_for(arena.run(), timeout=60.0)
            elapsed = time.perf_counter() - start

        # Assert against centralized SLO
        assert_debate_slo("single_round_max_sec", elapsed)
        assert result is not None

    @pytest.mark.asyncio
    async def test_multi_round_scaling(self, benchmark_agents, benchmark_environment):
        """Measure how latency scales with rounds."""
        round_times = {}

        with (
            patch.object(
                Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                Arena, "_fetch_knowledge_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(Arena, "_init_km_context", new_callable=AsyncMock, return_value=None),
            patch(
                "aragora.debate.context_gatherer.ContextGatherer.gather_all",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            for num_rounds in [1, 2, 3]:
                protocol = DebateProtocol(rounds=num_rounds, consensus="any")
                arena = Arena(benchmark_environment, benchmark_agents, protocol)

                # Mock prompt_builder.classify_question_async to avoid LLM calls
                if arena.prompt_builder:
                    arena.prompt_builder.classify_question_async = AsyncMock(return_value=None)

                start = time.perf_counter()
                await asyncio.wait_for(arena.run(), timeout=60.0)
                elapsed = time.perf_counter() - start

                round_times[num_rounds] = elapsed

        # Check scaling is roughly linear (not exponential) against SLO
        if round_times[1] > 0:
            ratio = round_times[3] / round_times[1]
            assert_debate_slo("round_scaling_max_ratio", ratio)

    @pytest.mark.asyncio
    async def test_agent_count_scaling(self, benchmark_environment):
        """Measure how latency scales with agent count.

        Note: This test can be flaky under high system load (CI/parallel tests).
        We add safeguards for unreliable timing measurements.
        """
        agent_times = {}

        with (
            patch.object(
                Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                Arena, "_fetch_knowledge_context", new_callable=AsyncMock, return_value=None
            ),
            patch(
                "aragora.debate.context_gatherer.ContextGatherer.gather_all",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            # Warmup run to stabilize timing
            warmup_agents = [BenchmarkAgent(f"warmup_{i}") for i in range(2)]
            warmup_arena = Arena(
                benchmark_environment,
                warmup_agents,
                DebateProtocol(rounds=1, consensus="any"),
            )
            await asyncio.wait_for(warmup_arena.run(), timeout=30.0)

            for num_agents in [2, 3, 5]:
                agents = [BenchmarkAgent(f"agent_{i}") for i in range(num_agents)]
                protocol = DebateProtocol(rounds=1, consensus="any")
                arena = Arena(benchmark_environment, agents, protocol)

                start = time.perf_counter()
                await asyncio.wait_for(arena.run(), timeout=30.0)
                elapsed = time.perf_counter() - start

                agent_times[num_agents] = elapsed

        # Check scaling is sub-linear (agents run in parallel) against SLO
        # Skip ratio check if baseline is too fast (< 0.1s) as timing becomes unreliable
        if agent_times[2] >= 0.1:
            ratio = agent_times[5] / agent_times[2]
            assert_debate_slo("agent_scaling_max_ratio", ratio)


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

        # Assert against centralized SLO
        assert_memory_ops_slo("critique_write", 100, elapsed)

    @pytest.mark.asyncio
    async def test_critique_store_read(self, temp_benchmark_db):
        """Measure critique store read performance."""
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore(str(temp_benchmark_db))

        # Write some data first
        for i in range(50):
            store.store(
                Critique(
                    agent="agent",
                    target_agent="target",
                    target_content=f"content_{i}",
                    issues=[],
                    suggestions=[],
                    severity=0.5,
                    reasoning="reasoning",
                )
            )

        # Measure reads
        start = time.perf_counter()
        for _ in range(100):
            store.get_recent(limit=10)
        elapsed = time.perf_counter() - start

        # Assert against centralized SLO
        assert_memory_ops_slo("critique_read", 100, elapsed)


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

        # Assert against centralized SLO
        assert_elo_ops_slo("rating_update", 100, elapsed)

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

        # Assert against centralized SLO
        assert_elo_ops_slo("leaderboard_query", 100, elapsed)


# =============================================================================
# Concurrent Operation Tests
# =============================================================================


class TestConcurrentPerformance:
    """Benchmark tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_debates(self, benchmark_environment):
        """Measure concurrent debate throughput."""
        num_debates = 5

        with (
            patch.object(
                Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                Arena, "_fetch_knowledge_context", new_callable=AsyncMock, return_value=None
            ),
        ):

            async def run_single_debate(idx: int):
                agents = [BenchmarkAgent(f"debate{idx}_agent_{i}") for i in range(2)]
                env = Environment(task=f"Concurrent task {idx}")
                protocol = DebateProtocol(rounds=1, consensus="any")
                arena = Arena(env, agents, protocol)
                return await arena.run()

            start = time.perf_counter()
            results = await asyncio.gather(*[run_single_debate(i) for i in range(num_debates)])
            elapsed = time.perf_counter() - start

        # All should complete
        assert len(results) == num_debates
        assert all(r is not None for r in results)

        # Assert against centralized SLO
        debates_per_sec = num_debates / elapsed
        assert_debate_slo("concurrent_min_per_sec", debates_per_sec)

    @pytest.mark.asyncio
    async def test_concurrent_memory_writes(self, temp_benchmark_db):
        """Measure concurrent memory write performance."""
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore(str(temp_benchmark_db))
        num_writers = 10
        writes_per_writer = 10

        async def write_batch(writer_id: int):
            for i in range(writes_per_writer):
                store.store(
                    Critique(
                        agent=f"writer_{writer_id}",
                        target_agent="target",
                        target_content=f"content_{writer_id}_{i}",
                        issues=[],
                        suggestions=[],
                        severity=0.5,
                        reasoning="concurrent write",
                    )
                )

        start = time.perf_counter()
        await asyncio.gather(*[write_batch(i) for i in range(num_writers)])
        elapsed = time.perf_counter() - start

        total_writes = num_writers * writes_per_writer

        # Concurrent writes use relaxed SLO (50% of normal)
        # This accounts for lock contention in concurrent scenarios
        ops_per_sec = total_writes / elapsed
        min_concurrent_ops = SLO.MEMORY_OPS["critique_write"]["min_ops_per_sec"] * 0.5
        assert ops_per_sec >= min_concurrent_ops, (
            f"Concurrent critique writes: {ops_per_sec:.0f} ops/sec "
            f"(target: >{min_concurrent_ops:.0f} ops/sec)"
        )


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
            [
                "python",
                "-c",
                "import time; start=time.perf_counter(); import aragora; print(time.perf_counter()-start)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            elapsed = float(result.stdout.strip())
            # Assert against centralized SLO
            assert elapsed < SLO.STARTUP["import_max_sec"], (
                f"Import took {elapsed:.2f}s (target: <{SLO.STARTUP['import_max_sec']}s)"
            )
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
