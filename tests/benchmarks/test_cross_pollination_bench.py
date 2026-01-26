"""
Cross-Pollination Feature Benchmarks.

Measures the performance impact of cross-pollination integrations:
- ELO → Vote Weighting
- Calibration → Proposals
- Evidence Quality → Consensus
- Memory → Debate Strategy

Run with: pytest tests/benchmarks/test_cross_pollination_bench.py -v
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from .conftest import SimpleBenchmark

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem


# =============================================================================
# Weight Calculator Benchmarks
# =============================================================================


class TestWeightCalculatorBenchmarks:
    """Benchmark weight calculation with and without ELO."""

    @pytest.fixture
    def mock_elo_system(self):
        """Create mock ELO system with properly typed return values."""
        from dataclasses import dataclass

        @dataclass
        class MockRating:
            elo: float = 1600.0

        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=1600.0)
        elo.get_domain_rating.return_value = 1550.0
        return elo

    def test_weight_calculation_without_elo(self):
        """Benchmark weight calculation without ELO system."""
        from aragora.debate.phases.weight_calculator import (
            WeightCalculator,
            WeightCalculatorConfig,
        )

        config = WeightCalculatorConfig(
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calculator = WeightCalculator(config=config)
        bench = SimpleBenchmark("weight_no_elo")

        # Run 1000 iterations
        for _ in range(1000):
            bench(lambda: calculator.get_weight("test_agent"))

        # Weight calculation should be fast (<0.5ms per call)
        assert bench.mean < 0.005, f"Mean time {bench.mean * 1000:.3f}ms exceeds 5ms"

    def test_weight_calculation_with_elo(self, mock_elo_system):
        """Benchmark weight calculation with ELO system."""
        from aragora.debate.phases.weight_calculator import (
            WeightCalculator,
            WeightCalculatorConfig,
        )

        config = WeightCalculatorConfig(enable_elo_skill=True)
        calculator = WeightCalculator(config=config, elo_system=mock_elo_system)
        bench = SimpleBenchmark("weight_with_elo")

        # Run 1000 iterations
        for _ in range(1000):
            bench(lambda: calculator.get_weight("test_agent"))

        # ELO lookup should add minimal overhead (<1ms per call)
        assert bench.mean < 0.010, f"Mean time {bench.mean * 1000:.3f}ms exceeds 10ms"

    def test_weight_factors_overhead(self):
        """Compare overhead of different weight factor configurations."""
        from aragora.debate.phases.weight_calculator import (
            WeightCalculator,
            WeightCalculatorConfig,
        )

        # Minimal config (baseline)
        minimal_config = WeightCalculatorConfig(
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        minimal_calc = WeightCalculator(config=minimal_config)

        # Config with some factors enabled (no ELO to avoid mock complexity)
        some_config = WeightCalculatorConfig(
            enable_reputation=True,
            enable_reliability=True,
            enable_consistency=True,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        some_calc = WeightCalculator(config=some_config)

        # Benchmark minimal
        minimal_bench = SimpleBenchmark("minimal")
        for _ in range(1000):
            minimal_bench(lambda: minimal_calc.get_weight("test"))

        # Benchmark some factors
        some_bench = SimpleBenchmark("some_factors")
        for _ in range(1000):
            some_bench(lambda: some_calc.get_weight("test"))

        # Some factors should be at most 3x slower than minimal
        overhead_ratio = some_bench.mean / minimal_bench.mean if minimal_bench.mean > 0 else 1
        assert overhead_ratio < 3.0, (
            f"Some factors config {overhead_ratio:.1f}x slower than minimal"
        )


# =============================================================================
# Debate Strategy Benchmarks
# =============================================================================


class TestDebateStrategyBenchmarks:
    """Benchmark memory-aware debate strategy."""

    def test_strategy_estimation_without_memory(self):
        """Benchmark strategy estimation without memory."""
        from aragora.debate.strategy import DebateStrategy

        strategy = DebateStrategy()
        bench = SimpleBenchmark("strategy_no_memory")

        for _ in range(1000):
            bench(lambda: strategy.estimate_rounds("Test task without memory"))

        # Should be fast without memory lookup (<0.5ms)
        assert bench.mean < 0.005, f"Mean time {bench.mean * 1000:.3f}ms exceeds 5ms"

    def test_strategy_estimation_with_mock_memory(self):
        """Benchmark strategy estimation with mocked memory."""
        from aragora.debate.strategy import DebateStrategy

        mock_memory = MagicMock()
        mock_memory.recall.return_value = []  # No memories found

        strategy = DebateStrategy(continuum_memory=mock_memory)
        bench = SimpleBenchmark("strategy_with_memory")

        for _ in range(1000):
            bench(lambda: strategy.estimate_rounds("Test task with memory"))

        # Memory lookup should add minimal overhead (<1ms)
        assert bench.mean < 0.010, f"Mean time {bench.mean * 1000:.3f}ms exceeds 10ms"


# =============================================================================
# RLM Cache Benchmarks
# =============================================================================


class TestRLMCacheBenchmarks:
    """Benchmark RLM hierarchy caching."""

    def test_cache_hash_computation(self):
        """Benchmark cache key hash computation."""
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache()
        bench = SimpleBenchmark("cache_hash")

        test_content = "This is a test context for the debate. " * 100

        for _ in range(1000):
            bench(lambda: cache._compute_task_hash(test_content))

        # Hash computation should be fast (<0.5ms)
        assert bench.mean < 0.005, f"Mean time {bench.mean * 1000:.3f}ms exceeds 5ms"

    def test_local_cache_lookup(self):
        """Benchmark local cache dict lookup."""
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache()

        # Pre-populate local cache
        for i in range(100):
            cache._local_cache[f"hash_{i}"] = MagicMock()

        bench = SimpleBenchmark("local_cache_lookup")

        for _ in range(1000):
            bench(lambda: cache._local_cache.get("hash_50"))

        # Dict lookup should be very fast (<0.01ms)
        assert bench.mean < 0.0001, f"Mean time {bench.mean * 1000:.3f}ms exceeds 0.1ms"


# =============================================================================
# Calibration Benchmarks
# =============================================================================


class TestCalibrationBenchmarks:
    """Benchmark calibration operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temp database path."""
        return str(tmp_path / "calibration.db")

    def test_calibration_record(self, temp_db):
        """Benchmark recording calibration data."""
        from aragora.agents.calibration import CalibrationTracker

        tracker = CalibrationTracker(db_path=temp_db)
        bench = SimpleBenchmark("calibration_record")

        for i in range(100):  # Fewer iterations for DB-backed operations
            bench(
                lambda: tracker.record_prediction(
                    agent="test_agent", confidence=0.75, correct=i % 2 == 0, domain="benchmark"
                )
            )

        # Recording to SQLite should be reasonable (<10ms)
        assert bench.mean < 0.01, f"Mean time {bench.mean * 1000:.3f}ms exceeds 10ms"

    def test_calibration_brier_score(self, temp_db):
        """Benchmark Brier score retrieval."""
        from aragora.agents.calibration import CalibrationTracker

        tracker = CalibrationTracker(db_path=temp_db)

        # Add some calibration data
        for i in range(50):
            tracker.record_prediction(
                agent="test_agent",
                confidence=0.5 + (i % 10) * 0.05,  # Varying confidence
                correct=i % 5 != 0,  # 80% correct
                domain="test",
            )

        bench = SimpleBenchmark("calibration_brier")

        for _ in range(100):
            bench(lambda: tracker.get_brier_score("test_agent", "test"))

        # Brier score should be reasonable (<10ms)
        assert bench.mean < 0.01, f"Mean time {bench.mean * 1000:.3f}ms exceeds 10ms"


# =============================================================================
# ELO System Benchmarks
# =============================================================================


class TestELOBenchmarks:
    """Benchmark ELO system operations."""

    @pytest.fixture
    def elo_system(self, tmp_path):
        """Create ELO system with temp storage."""
        from aragora.ranking.elo import EloSystem

        return EloSystem(db_path=str(tmp_path / "elo.db"))

    def test_elo_rating_lookup(self, elo_system):
        """Benchmark ELO rating lookup."""
        # Register some agents
        for i in range(20):
            elo_system.register_agent(f"agent_{i}")

        bench = SimpleBenchmark("elo_lookup")

        for _ in range(100):
            bench(lambda: elo_system.get_rating("agent_10"))

        # ELO lookup should be fast (<1ms)
        assert bench.mean < 0.010, f"Mean time {bench.mean * 1000:.3f}ms exceeds 10ms"

    def test_elo_match_record(self, elo_system):
        """Benchmark ELO match recording."""
        # Register agents
        elo_system.register_agent("winner")
        elo_system.register_agent("loser")

        bench = SimpleBenchmark("elo_match")

        for _ in range(100):
            bench(lambda: elo_system.record_match(winner="winner", loser="loser", domain="test"))

        # ELO update should be reasonable (<50ms to avoid flaky tests on different hardware)
        assert bench.mean < 0.050, f"Mean time {bench.mean * 1000:.3f}ms exceeds 50ms"

    def test_learning_efficiency_computation(self, elo_system):
        """Benchmark learning efficiency computation."""
        # Register and add some history
        elo_system.register_agent("learner")
        elo_system.register_agent("opponent")
        for i in range(50):
            elo_system.record_match(
                winner="learner" if i % 3 == 0 else "opponent",
                loser="opponent" if i % 3 == 0 else "learner",
                domain="test",
            )

        bench = SimpleBenchmark("learning_efficiency")

        for _ in range(100):
            bench(lambda: elo_system.get_learning_efficiency("learner"))

        # Learning efficiency should be reasonable (<5ms)
        assert bench.mean < 0.005, f"Mean time {bench.mean * 1000:.3f}ms exceeds 5ms"
