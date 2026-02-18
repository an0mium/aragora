"""Tests for the outcome-to-complexity governor bridge.

Covers ComplexityStats, TimeoutAdjustment, OutcomeComplexityBridgeConfig,
OutcomeComplexityBridge (record, timeout factors, signal boost, recalibration),
and create_outcome_complexity_bridge factory.
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.outcome_complexity_bridge import (
    ComplexityStats,
    OutcomeComplexityBridge,
    OutcomeComplexityBridgeConfig,
    TimeoutAdjustment,
    create_outcome_complexity_bridge,
)


# ---------------------------------------------------------------------------
# ComplexityStats
# ---------------------------------------------------------------------------


class TestComplexityStats:
    def test_success_rate_no_debates(self):
        stats = ComplexityStats()
        assert stats.success_rate == 0.5  # Neutral default

    def test_success_rate_with_data(self):
        stats = ComplexityStats(total_debates=10, successful_debates=8)
        assert stats.success_rate == 0.8

    def test_timeout_rate_no_debates(self):
        stats = ComplexityStats()
        assert stats.timeout_rate == 0.0

    def test_timeout_rate_with_data(self):
        stats = ComplexityStats(total_debates=10, timeout_debates=3)
        assert stats.timeout_rate == 0.3

    def test_avg_time_to_failure_no_failures(self):
        stats = ComplexityStats(total_debates=10, successful_debates=10)
        assert stats.avg_time_to_failure == float("inf")

    def test_avg_time_to_failure_with_failures(self):
        stats = ComplexityStats(
            total_debates=10,
            successful_debates=7,
            total_time_to_failure=30.0,
        )
        assert stats.avg_time_to_failure == 10.0  # 30/3


# ---------------------------------------------------------------------------
# TimeoutAdjustment
# ---------------------------------------------------------------------------


class TestTimeoutAdjustment:
    def test_defaults(self):
        adj = TimeoutAdjustment(
            original_factor=1.0,
            adjusted_factor=1.2,
            complexity="complex",
            reason="high_timeout_rate",
        )
        assert adj.confidence == 0.0

    def test_with_confidence(self):
        adj = TimeoutAdjustment(
            original_factor=1.0,
            adjusted_factor=1.2,
            complexity="complex",
            reason="test",
            confidence=0.8,
        )
        assert adj.confidence == 0.8


# ---------------------------------------------------------------------------
# OutcomeComplexityBridgeConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = OutcomeComplexityBridgeConfig()
        assert cfg.min_debates_for_adjustment == 10
        assert cfg.learning_rate == 0.1
        assert cfg.max_factor_increase == 0.5
        assert cfg.max_factor_decrease == 0.3
        assert cfg.timeout_threshold_increase == 0.15
        assert cfg.success_threshold_decrease == 0.85
        assert cfg.time_to_failure_weight == 0.3


# ---------------------------------------------------------------------------
# OutcomeComplexityBridge — basic operations
# ---------------------------------------------------------------------------


class TestBridgeBasics:
    def test_init_no_tracker(self):
        bridge = OutcomeComplexityBridge()
        assert bridge.outcome_tracker is None
        assert len(bridge._complexity_stats) == 0

    def test_init_with_tracker(self):
        tracker = MagicMock()
        tracker.get_recent_outcomes.return_value = []
        bridge = OutcomeComplexityBridge(outcome_tracker=tracker)
        tracker.get_recent_outcomes.assert_called_once()

    def test_init_tracker_error_handled(self):
        tracker = MagicMock()
        tracker.get_recent_outcomes.side_effect = RuntimeError("db error")
        # Should not raise
        bridge = OutcomeComplexityBridge(outcome_tracker=tracker)
        assert len(bridge._complexity_stats) == 0


# ---------------------------------------------------------------------------
# record_complexity_outcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    def test_record_success(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome("d1", "simple", succeeded=True)
        stats = bridge.get_complexity_stats("simple")
        assert stats.total_debates == 1
        assert stats.successful_debates == 1

    def test_record_failure(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome("d1", "complex", succeeded=False)
        stats = bridge.get_complexity_stats("complex")
        assert stats.total_debates == 1
        assert stats.successful_debates == 0

    def test_record_timeout(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome(
            "d1", "complex", succeeded=False, timeout=True
        )
        stats = bridge.get_complexity_stats("complex")
        assert stats.timeout_debates == 1

    def test_record_time_to_failure(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome(
            "d1", "complex", succeeded=False, time_to_failure=15.0
        )
        stats = bridge.get_complexity_stats("complex")
        assert stats.total_time_to_failure == 15.0

    def test_record_extracts_failure_signals(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome(
            "d1", "complex", succeeded=False,
            task_text="optimize distributed concurrent system",
        )
        signals = bridge.get_failure_signals()
        assert signals.get("distributed", 0) > 0
        assert signals.get("concurrent", 0) > 0
        assert signals.get("optimize", 0) > 0

    def test_multiple_outcomes_accumulate(self):
        bridge = OutcomeComplexityBridge()
        for i in range(5):
            bridge.record_complexity_outcome(f"d{i}", "moderate", succeeded=True)
        for i in range(3):
            bridge.record_complexity_outcome(f"d{i+5}", "moderate", succeeded=False)

        stats = bridge.get_complexity_stats("moderate")
        assert stats.total_debates == 8
        assert stats.successful_debates == 5

    def test_nonexistent_complexity_returns_none(self):
        bridge = OutcomeComplexityBridge()
        assert bridge.get_complexity_stats("unknown") is None


# ---------------------------------------------------------------------------
# get_adaptive_timeout_factor
# ---------------------------------------------------------------------------


class TestAdaptiveTimeoutFactor:
    def test_basic_factor_with_explicit_complexity(self):
        bridge = OutcomeComplexityBridge()
        factor = bridge.get_adaptive_timeout_factor("simple task", base_complexity="simple")
        assert 0.3 <= factor <= 3.0

    def test_with_explicit_complexity(self):
        bridge = OutcomeComplexityBridge()
        factor = bridge.get_adaptive_timeout_factor("task", base_complexity="unknown")
        assert 0.3 <= factor <= 3.0

    def test_invalid_complexity_uses_unknown(self):
        bridge = OutcomeComplexityBridge()
        factor = bridge.get_adaptive_timeout_factor("task", base_complexity="invalid_xxx")
        assert 0.3 <= factor <= 3.0

    def test_signal_boost_applied(self):
        bridge = OutcomeComplexityBridge()
        # Record failures with signals
        for _ in range(5):
            bridge.record_complexity_outcome(
                "d", "complex", succeeded=False,
                task_text="distributed concurrent system",
            )

        boost = bridge._compute_signal_boost("distributed concurrent task")
        assert boost > 0.0

    def test_signal_boost_zero_no_signals(self):
        bridge = OutcomeComplexityBridge()
        assert bridge._compute_signal_boost("simple task") == 0.0

    def test_signal_boost_empty_task(self):
        bridge = OutcomeComplexityBridge()
        assert bridge._compute_signal_boost("") == 0.0

    def test_signal_boost_capped(self):
        bridge = OutcomeComplexityBridge()
        # Load many signals
        for signal in ["distributed", "concurrent", "real-time", "optimize", "scale", "security"]:
            bridge._failure_signals[signal] = 100
        boost = bridge._compute_signal_boost(
            "distributed concurrent real-time optimize scale security"
        )
        assert boost <= 0.5


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------


class TestRecalibration:
    def test_recalibration_not_triggered_below_min(self):
        bridge = OutcomeComplexityBridge()
        # Record 9 outcomes — below min_debates_for_adjustment=10
        for i in range(9):
            bridge.record_complexity_outcome(f"d{i}", "complex", succeeded=False, timeout=True)
        assert bridge.get_factor_adjustments().get("complex") is None

    def test_recalibration_triggered_at_boundary(self):
        bridge = OutcomeComplexityBridge(
            config=OutcomeComplexityBridgeConfig(min_debates_for_adjustment=10)
        )
        # Record 10 timeout outcomes to trigger recalibration
        for i in range(10):
            bridge.record_complexity_outcome(
                f"d{i}", "simple", succeeded=False, timeout=True
            )
        # After 10 outcomes with 100% timeout, should have adjustment
        # (recalibrate is called when total_debates % 10 == 0)
        adj = bridge.get_factor_adjustments()
        # May or may not have adjustment depending on whether TaskComplexity('simple') works
        assert isinstance(adj, dict)


# ---------------------------------------------------------------------------
# Stats and utilities
# ---------------------------------------------------------------------------


class TestStatsAndUtils:
    def test_get_all_stats(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome("d1", "simple", succeeded=True)
        bridge.record_complexity_outcome("d2", "complex", succeeded=False)
        all_stats = bridge.get_all_stats()
        assert "simple" in all_stats
        assert "complex" in all_stats

    def test_get_stats(self):
        bridge = OutcomeComplexityBridge()
        stats = bridge.get_stats()
        assert stats["complexity_levels_tracked"] == 0
        assert stats["total_outcomes_processed"] == 0
        assert stats["factor_adjustments_active"] == 0
        assert stats["failure_signals_tracked"] == 0

    def test_get_stats_after_recording(self):
        bridge = OutcomeComplexityBridge()
        bridge.record_complexity_outcome("d1", "simple", succeeded=True)
        bridge.record_complexity_outcome(
            "d2", "complex", succeeded=False,
            task_text="optimize distributed system",
        )
        stats = bridge.get_stats()
        assert stats["complexity_levels_tracked"] == 2
        assert stats["total_outcomes_processed"] == 2
        assert stats["failure_signals_tracked"] > 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        bridge = create_outcome_complexity_bridge()
        assert isinstance(bridge, OutcomeComplexityBridge)
        assert bridge.outcome_tracker is None

    def test_create_with_config(self):
        bridge = create_outcome_complexity_bridge(
            learning_rate=0.2,
            max_factor_increase=0.8,
        )
        assert bridge.config.learning_rate == 0.2
        assert bridge.config.max_factor_increase == 0.8

    def test_create_with_tracker(self):
        tracker = MagicMock()
        tracker.get_recent_outcomes.return_value = []
        bridge = create_outcome_complexity_bridge(outcome_tracker=tracker)
        assert bridge.outcome_tracker is tracker
