"""Tests for OutcomeComplexityBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from aragora.debate.outcome_complexity_bridge import (
    OutcomeComplexityBridge,
    OutcomeComplexityBridgeConfig,
    ComplexityStats,
    TimeoutAdjustment,
    create_outcome_complexity_bridge,
)


@dataclass
class MockConsensusOutcome:
    """Mock consensus outcome for testing."""

    debate_id: str = "test-debate"
    consensus_text: str = "Design a simple cache system"
    implementation_succeeded: bool = True
    rounds_completed: int = 5
    failure_reason: Optional[str] = None
    time_to_failure: Optional[float] = None


class MockOutcomeTracker:
    """Mock outcome tracker."""

    def __init__(self):
        self._outcomes: List[MockConsensusOutcome] = []

    def get_recent_outcomes(self, limit: int = 100) -> List[MockConsensusOutcome]:
        """Get recent outcomes."""
        return self._outcomes[:limit]

    def add_outcome(self, outcome: MockConsensusOutcome) -> None:
        """Add an outcome."""
        self._outcomes.append(outcome)


class TestComplexityStats:
    """Tests for ComplexityStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = ComplexityStats()
        assert stats.total_debates == 0
        assert stats.successful_debates == 0
        assert stats.timeout_debates == 0

    def test_success_rate_empty(self):
        """Test success rate with no data."""
        stats = ComplexityStats()
        assert stats.success_rate == 0.5  # Neutral

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ComplexityStats(total_debates=10, successful_debates=8)
        assert stats.success_rate == 0.8

    def test_timeout_rate(self):
        """Test timeout rate calculation."""
        stats = ComplexityStats(total_debates=10, timeout_debates=2)
        assert stats.timeout_rate == 0.2

    def test_avg_time_to_failure(self):
        """Test average time to failure."""
        stats = ComplexityStats(
            total_debates=10,
            successful_debates=7,
            total_time_to_failure=30.0,
        )
        # 3 failures, 30 total seconds
        assert stats.avg_time_to_failure == 10.0

    def test_avg_time_to_failure_no_failures(self):
        """Test avg time to failure with no failures."""
        stats = ComplexityStats(total_debates=10, successful_debates=10)
        assert stats.avg_time_to_failure == float("inf")


class TestOutcomeComplexityBridge:
    """Tests for OutcomeComplexityBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = OutcomeComplexityBridge()
        assert bridge.outcome_tracker is None
        assert bridge.config is not None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = OutcomeComplexityBridgeConfig(
            min_debates_for_adjustment=5,
            learning_rate=0.2,
        )
        bridge = OutcomeComplexityBridge(config=config)
        assert bridge.config.min_debates_for_adjustment == 5
        assert bridge.config.learning_rate == 0.2

    def test_record_complexity_outcome_success(self):
        """Test recording successful outcome."""
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome(
            debate_id="test-1",
            complexity="simple",
            succeeded=True,
        )

        stats = bridge.get_complexity_stats("simple")
        assert stats is not None
        assert stats.total_debates == 1
        assert stats.successful_debates == 1

    def test_record_complexity_outcome_failure(self):
        """Test recording failed outcome."""
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome(
            debate_id="test-1",
            complexity="complex",
            succeeded=False,
            timeout=True,
            time_to_failure=45.0,
            task_text="Optimize a distributed system",
        )

        stats = bridge.get_complexity_stats("complex")
        assert stats.total_debates == 1
        assert stats.successful_debates == 0
        assert stats.timeout_debates == 1
        assert stats.total_time_to_failure == 45.0

    def test_get_adaptive_timeout_factor_no_data(self):
        """Test adaptive timeout with no historical data."""
        bridge = OutcomeComplexityBridge()
        factor = bridge.get_adaptive_timeout_factor("simple task")
        # Should return base factor from complexity governor
        assert factor > 0

    def test_get_adaptive_timeout_factor_with_history(self):
        """Test adaptive timeout with historical data."""
        bridge = OutcomeComplexityBridge(
            config=OutcomeComplexityBridgeConfig(min_debates_for_adjustment=5)
        )

        # Record outcomes
        for i in range(10):
            bridge.record_complexity_outcome(
                debate_id=f"test-{i}",
                complexity="complex",
                succeeded=i < 7,  # 70% success
                timeout=i >= 7,
            )

        factor = bridge.get_adaptive_timeout_factor(
            "Design a complex distributed system",
            base_complexity="complex",
        )

        # Factor should be adjusted based on history
        assert factor > 0

    def test_failure_signal_extraction(self):
        """Test extraction of failure signals."""
        bridge = OutcomeComplexityBridge()

        # Record failure with task text containing signals
        bridge.record_complexity_outcome(
            debate_id="test-1",
            complexity="complex",
            succeeded=False,
            task_text="Optimize distributed concurrent system with real-time processing",
        )

        signals = bridge.get_failure_signals()
        assert "distributed" in signals or "concurrent" in signals

    def test_signal_boost(self):
        """Test signal boost for tasks with failure-correlated signals."""
        bridge = OutcomeComplexityBridge()

        # Record multiple failures with similar signals
        for i in range(5):
            bridge.record_complexity_outcome(
                debate_id=f"test-{i}",
                complexity="complex",
                succeeded=False,
                task_text="Distributed system optimization",
            )

        # Task with similar signals should get boost
        factor1 = bridge.get_adaptive_timeout_factor("simple task")
        factor2 = bridge.get_adaptive_timeout_factor("distributed system design")

        # The distributed task should have higher factor
        assert factor2 >= factor1

    def test_get_all_stats(self):
        """Test getting all complexity stats."""
        bridge = OutcomeComplexityBridge()

        bridge.record_complexity_outcome("t1", "simple", True)
        bridge.record_complexity_outcome("t2", "complex", False)

        all_stats = bridge.get_all_stats()
        assert "simple" in all_stats
        assert "complex" in all_stats

    def test_get_factor_adjustments(self):
        """Test getting factor adjustments."""
        bridge = OutcomeComplexityBridge()
        adjustments = bridge.get_factor_adjustments()
        assert isinstance(adjustments, dict)

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome("t1", "simple", True)

        stats = bridge.get_stats()
        assert "complexity_levels_tracked" in stats
        assert "total_outcomes_processed" in stats
        assert stats["total_outcomes_processed"] >= 1

    def test_factory_function(self):
        """Test factory function."""
        bridge = create_outcome_complexity_bridge(
            min_debates_for_adjustment=3,
            learning_rate=0.15,
        )
        assert bridge.config.min_debates_for_adjustment == 3
        assert bridge.config.learning_rate == 0.15

    def test_maybe_recalibrate(self):
        """Test recalibration trigger."""
        bridge = OutcomeComplexityBridge(
            config=OutcomeComplexityBridgeConfig(
                min_debates_for_adjustment=10,
                timeout_threshold_increase=0.15,
            )
        )

        # Record enough outcomes to trigger recalibration
        for i in range(10):
            bridge.record_complexity_outcome(
                debate_id=f"test-{i}",
                complexity="moderate",
                succeeded=i < 5,  # 50% success
                timeout=i >= 8,  # 20% timeout
            )

        # Stats should show timeout rate above threshold
        stats = bridge.get_complexity_stats("moderate")
        assert stats.timeout_rate == 0.2


class TestTimeoutAdjustment:
    """Tests for TimeoutAdjustment dataclass."""

    def test_timeout_adjustment(self):
        """Test TimeoutAdjustment creation."""
        adj = TimeoutAdjustment(
            original_factor=1.0,
            adjusted_factor=1.2,
            complexity="complex",
            reason="high_timeout_rate",
            confidence=0.8,
        )
        assert adj.original_factor == 1.0
        assert adj.adjusted_factor == 1.2
        assert adj.complexity == "complex"
        assert adj.confidence == 0.8
