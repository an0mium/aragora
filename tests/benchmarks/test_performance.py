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
import sys
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


async def _stub_synthesis(self, ctx) -> bool:
    """Fast synthesis stub for performance benchmarks."""
    synthesis = "Benchmark synthesis."
    ctx.result.synthesis = synthesis
    ctx.result.final_answer = synthesis
    return True


class TestDebatePerformance:
    """Benchmark tests for debate operations."""

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.slow  # Skip in regular CI, run in nightly only
    async def test_single_round_latency(self, benchmark_agents, benchmark_environment):
        """Measure single debate round latency against SLO.

        SLO threshold is set to 30 seconds to account for CI/parallel test
        resource contention, shared runners, and external network timeouts.
        Local runs typically complete in 2-4 seconds. Timeout is 60 seconds
        to accommodate heavy CI load.

        This test is marked flaky(reruns=3) due to timing sensitivity in CI
        environments where CPU/memory contention can cause outlier latencies.
        """
        # Warmup run to stabilise timing and JIT caches
        warmup_proto = DebateProtocol(
            rounds=1,
            consensus="any",
            enable_research=False,
            role_matching=False,
            role_rotation=False,
            enable_trickster=False,
            enable_rhetorical_observer=False,
            enable_breakpoints=False,
            enable_calibration=False,
        )
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
            patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator.generate_mandatory_synthesis",
                new=_stub_synthesis,
            ),
        ):
            warmup_arena = Arena(benchmark_environment, benchmark_agents, warmup_proto)
            if warmup_arena.prompt_builder:
                warmup_arena.prompt_builder.classify_question_async = AsyncMock(return_value=None)
            await asyncio.wait_for(warmup_arena.run(), timeout=30.0)

        protocol = DebateProtocol(
            rounds=1,
            consensus="any",
            enable_research=False,
            role_matching=False,
            role_rotation=False,
            enable_trickster=False,
            enable_rhetorical_observer=False,
            enable_breakpoints=False,
            enable_calibration=False,
        )

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
            patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator.generate_mandatory_synthesis",
                new=_stub_synthesis,
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
            patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator.generate_mandatory_synthesis",
                new=_stub_synthesis,
            ),
        ):
            for num_rounds in [1, 2, 3]:
                protocol = DebateProtocol(
                    rounds=num_rounds,
                    consensus="any",
                    enable_research=False,
                    role_matching=False,
                    role_rotation=False,
                    enable_trickster=False,
                    enable_rhetorical_observer=False,
                    enable_breakpoints=False,
                    enable_calibration=False,
                )
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

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.slow
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
            patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator.generate_mandatory_synthesis",
                new=_stub_synthesis,
            ),
        ):
            # Warmup run to stabilize timing
            warmup_agents = [BenchmarkAgent(f"warmup_{i}") for i in range(2)]
            warmup_arena = Arena(
                benchmark_environment,
                warmup_agents,
                DebateProtocol(
                    rounds=1,
                    consensus="any",
                    enable_research=False,
                    role_matching=False,
                    role_rotation=False,
                    enable_trickster=False,
                    enable_rhetorical_observer=False,
                    enable_breakpoints=False,
                    enable_calibration=False,
                ),
            )
            await asyncio.wait_for(warmup_arena.run(), timeout=30.0)

            for num_agents in [2, 3, 5]:
                agents = [BenchmarkAgent(f"agent_{i}") for i in range(num_agents)]
                protocol = DebateProtocol(
                    rounds=1,
                    consensus="any",
                    enable_research=False,
                    role_matching=False,
                    role_rotation=False,
                    enable_trickster=False,
                    enable_rhetorical_observer=False,
                    enable_breakpoints=False,
                    enable_calibration=False,
                )
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

        protocol = DebateProtocol(
            rounds=1,
            consensus="any",
            enable_research=False,
            role_matching=False,
            role_rotation=False,
            enable_trickster=False,
            enable_rhetorical_observer=False,
            enable_breakpoints=False,
            enable_calibration=False,
        )

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
            patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator.generate_mandatory_synthesis",
                new=_stub_synthesis,
            ),
        ):

            async def run_single_debate(idx: int):
                agents = [BenchmarkAgent(f"debate{idx}_agent_{i}") for i in range(2)]
                env = Environment(task=f"Concurrent task {idx}")
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
        # Use sys.executable to ensure same Python interpreter is used
        result = subprocess.run(
            [
                sys.executable,
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
            pytest.fail(f"Subprocess import failed: {result.stderr}")

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


# =============================================================================
# pytest-benchmark Based Performance Benchmarks
#
# These benchmarks use the pytest-benchmark `benchmark` fixture for precise
# measurement with statistical analysis (min, max, mean, stddev, rounds).
# Run with:
#   pytest tests/benchmarks/test_performance.py -k "Benchmarks" --benchmark-only
# Or verify they execute (no timing):
#   pytest tests/benchmarks/test_performance.py -k "Benchmarks" --benchmark-disable -v
# =============================================================================


class TestBenchmarks:
    """Precise benchmarks using pytest-benchmark fixture."""

    def test_bench_receipt_creation(self, benchmark):
        """Benchmark creating a DecisionReceipt with 10 findings and calling to_dict().

        Expected baseline: < 1ms per call.
        """
        from aragora.export.decision_receipt import DecisionReceipt, ReceiptFinding

        findings = [
            ReceiptFinding(
                id=f"finding-{i}",
                severity=["CRITICAL", "HIGH", "MEDIUM", "LOW"][i % 4],
                category=f"cat-{i % 4}",
                title=f"Finding {i}",
                description=f"Description of finding {i}",
                mitigation=f"Fix {i}",
                source=f"agent-{i % 3}",
                verified=i % 2 == 0,
            )
            for i in range(10)
        ]

        def create_and_serialize():
            receipt = DecisionReceipt(
                receipt_id="rcpt_bench",
                gauntlet_id="gauntlet_bench",
                timestamp="2026-02-11T00:00:00Z",
                findings=findings,
                critical_count=3,
                high_count=3,
                medium_count=2,
                low_count=2,
                agents_involved=["a0", "a1", "a2"],
                rounds_completed=3,
                duration_seconds=10.0,
            )
            return receipt.to_dict()

        result = benchmark.pedantic(create_and_serialize, rounds=500, iterations=1)
        assert isinstance(result, dict)
        assert len(result["findings"]) == 10

    def test_bench_receipt_to_markdown(self, benchmark, sample_receipt):
        """Benchmark converting a receipt to markdown.

        Expected baseline: < 1ms per call.
        """
        result = benchmark.pedantic(sample_receipt.to_markdown, rounds=500, iterations=1)
        assert "# Decision Receipt" in result
        assert sample_receipt.receipt_id in result

    def test_bench_receipt_to_sarif(self, benchmark, sample_receipt):
        """Benchmark converting a receipt to SARIF format via DecisionReceipt.to_sarif().

        Expected baseline: < 2ms per call.
        """
        result = benchmark.pedantic(sample_receipt.to_sarif, rounds=500, iterations=1)
        assert result["version"] == "2.1.0"
        assert len(result["runs"]) == 1

    def test_bench_findings_to_sarif(self, benchmark, mock_review_findings):
        """Benchmark converting review findings to SARIF using findings_to_sarif.

        Expected baseline: < 1ms per call.
        """
        from aragora.cli.review import findings_to_sarif

        def convert():
            return findings_to_sarif(mock_review_findings)

        result = benchmark.pedantic(convert, rounds=500, iterations=1)
        assert result["version"] == "2.1.0"
        assert len(result["runs"][0]["results"]) > 0

    def test_bench_verdict_enum_lookup(self, benchmark):
        """Benchmark Verdict enum value lookups (baseline for hot path).

        Expected baseline: < 0.01ms per call (pure Python enum access).
        """
        from aragora.core_types import Verdict

        def lookup_all_verdicts():
            # Access each verdict value (simulates hot-path lookups)
            _ = Verdict.APPROVED.value
            _ = Verdict.APPROVED_WITH_CONDITIONS.value
            _ = Verdict.NEEDS_REVIEW.value
            _ = Verdict.REJECTED.value
            # Also test string comparison (common in receipt code)
            assert Verdict.APPROVED == "approved"
            assert Verdict("approved") is Verdict.APPROVED
            return True

        result = benchmark.pedantic(lookup_all_verdicts, rounds=1000, iterations=1)
        assert result is True

    def test_bench_review_extract_findings(self, benchmark):
        """Benchmark extracting findings from a mock review result.

        Uses extract_review_findings from aragora.cli.review with a
        DebateResult that has votes, critiques, and messages populated.

        Expected baseline: < 5ms per call.
        """
        from aragora.cli.review import extract_review_findings
        from aragora.core import DebateResult, Critique, Message

        # Build a realistic DebateResult with critiques and votes
        critiques = [
            Critique(
                agent=f"agent-{i}",
                target_agent=f"agent-{(i + 1) % 3}",
                target_content="Some proposal text",
                issues=[f"Issue {j} from agent-{i}" for j in range(3)],
                suggestions=[f"Suggestion {j}" for j in range(3)],
                severity=0.5 + (i * 0.2),
                reasoning=f"Reasoning from agent-{i}",
            )
            for i in range(3)
        ]
        votes = [
            Vote(
                agent=f"agent-{i}",
                choice="agent-0",
                reasoning="Best proposal",
                confidence=0.7 + (i * 0.1),
                continue_debate=False,
            )
            for i in range(3)
        ]
        messages = [
            Message(
                role="proposer",
                agent=f"agent-{i}",
                content=f"Proposal from agent-{i}",
            )
            for i in range(3)
        ]

        mock_result = DebateResult(
            task="Review benchmark diff",
            final_answer="Synthesis of code review findings.",
            confidence=0.75,
            consensus_reached=True,
            rounds_used=2,
        )
        mock_result.critiques = critiques
        mock_result.votes = votes
        mock_result.messages = messages

        def extract():
            return extract_review_findings(mock_result)

        result = benchmark.pedantic(extract, rounds=100, iterations=1)
        assert "unanimous_critiques" in result
        assert "agreement_score" in result

    def test_bench_handler_registry_lookup(self, benchmark):
        """Benchmark O(1) route lookup in RouteIndex.

        Tests the fast-path dict lookup for exact route matches.
        This exercises RouteIndex._exact_routes without needing a full
        server initialization, by manually populating the index.

        Expected baseline: < 0.05ms per call.
        """
        from aragora.server.handler_registry.core import RouteIndex

        class MockHandler:
            ROUTES = ["/api/debates", "/api/agents", "/api/health"]
            ROUTE_PREFIXES = []

            @staticmethod
            def can_handle(path: str) -> bool:
                return path.startswith("/api/")

        index = RouteIndex()
        # Manually populate exact routes to avoid full server init
        for path in MockHandler.ROUTES:
            index._exact_routes[path] = ("_mock_handler", MockHandler())

        # Also add some prefix routes for the prefix lookup path
        index._prefix_routes.append(("/api/debates/", "_mock_handler", MockHandler()))

        def lookup():
            # Exact match (O(1) dict lookup)
            r1 = index._exact_routes.get("/api/debates")
            r2 = index._exact_routes.get("/api/agents")
            r3 = index._exact_routes.get("/api/health")
            # Miss (should return None)
            r4 = index._exact_routes.get("/api/nonexistent")
            return r1 is not None and r2 is not None and r3 is not None and r4 is None

        result = benchmark.pedantic(lookup, rounds=1000, iterations=1)
        assert result is True

    def test_bench_permission_check(self, benchmark):
        """Benchmark RBAC permission checks via PermissionChecker.

        Creates a checker with a user context that has the 'admin' role,
        then checks a permission. Cache is disabled to measure the full
        evaluation path each time.

        Expected baseline: < 1ms per call.
        """
        from aragora.rbac.checker import PermissionChecker
        from aragora.rbac.models import AuthorizationContext

        checker = PermissionChecker(enable_cache=False)
        ctx = AuthorizationContext(
            user_id="bench-user",
            org_id="bench-org",
            roles={"admin"},
        )

        def check():
            decision = checker.check_permission(ctx, "debates.create")
            return decision.allowed

        result = benchmark.pedantic(check, rounds=500, iterations=1)
        assert result is True

    def test_bench_import_time(self, benchmark):
        """Benchmark time to import aragora main package.

        Measured in a subprocess to avoid module cache.

        Expected baseline: < 3s for cold import.
        """
        import subprocess

        def import_aragora():
            # Use sys.executable to ensure same Python interpreter is used
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import time; s=time.perf_counter(); import aragora; "
                    "print(time.perf_counter()-s)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                pytest.fail(f"Subprocess import failed: {proc.stderr}")
            return float(proc.stdout.strip())

        elapsed = benchmark.pedantic(import_aragora, rounds=10, iterations=1)
        # Sanity: import should complete in under 5 seconds
        assert elapsed < 5.0, f"Import took {elapsed:.2f}s (target: < 5s)"
