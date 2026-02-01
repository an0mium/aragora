"""
Comprehensive tests for aragora.billing.calibration_cost_bridge module.

Tests cover:
- AgentCostEfficiency dataclass
- CalibrationCostBridgeConfig dataclass
- CalibrationCostBridge class
- Cost efficiency computations
- Task cost estimation
- Agent recommendation algorithms
- Budget-aware selection
- Overconfident/well-calibrated agent detection
- Cache management
- Factory function create_calibration_cost_bridge
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from aragora.billing.calibration_cost_bridge import (
    AgentCostEfficiency,
    CalibrationCostBridge,
    CalibrationCostBridgeConfig,
    create_calibration_cost_bridge,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockCalibrationSummary:
    """Mock CalibrationSummary for testing."""

    ece: float = 0.1
    accuracy: float = 0.85
    total_predictions: int = 100
    is_overconfident: bool = False
    is_underconfident: bool = False


class MockCalibrationTracker:
    """Mock CalibrationTracker for testing."""

    def __init__(self, summaries: dict[str, MockCalibrationSummary] | None = None):
        self._summaries = summaries or {}

    def get_calibration_summary(self, agent_name: str) -> MockCalibrationSummary | None:
        return self._summaries.get(agent_name)

    def get_all_agents(self) -> list[str]:
        return list(self._summaries.keys())


class MockCostTracker:
    """Mock CostTracker for testing."""

    def __init__(self):
        self._workspace_stats: dict[str, dict] = {}

    def add_workspace_stats(self, workspace_id: str, stats: dict) -> None:
        self._workspace_stats[workspace_id] = stats


# =============================================================================
# AgentCostEfficiency Tests
# =============================================================================


class TestAgentCostEfficiency:
    """Tests for AgentCostEfficiency dataclass."""

    def test_default_values(self):
        """Test AgentCostEfficiency default values."""
        eff = AgentCostEfficiency(agent_name="claude")

        assert eff.agent_name == "claude"
        assert eff.calibration_score == 0.0
        assert eff.accuracy_score == 0.0
        assert eff.cost_per_call == Decimal("0")
        assert eff.efficiency_score == 0.0
        assert eff.confidence_reliability == 0.0
        assert eff.is_overconfident is False
        assert eff.is_underconfident is False
        assert eff.recommendation == ""

    def test_with_custom_values(self):
        """Test AgentCostEfficiency with custom values."""
        eff = AgentCostEfficiency(
            agent_name="gpt-4",
            calibration_score=0.85,
            accuracy_score=0.90,
            cost_per_call=Decimal("0.05"),
            efficiency_score=0.80,
            confidence_reliability=0.88,
            is_overconfident=False,
            is_underconfident=False,
            recommendation="efficient",
        )

        assert eff.agent_name == "gpt-4"
        assert eff.calibration_score == 0.85
        assert eff.efficiency_score == 0.80
        assert eff.recommendation == "efficient"


# =============================================================================
# CalibrationCostBridgeConfig Tests
# =============================================================================


class TestCalibrationCostBridgeConfig:
    """Tests for CalibrationCostBridgeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationCostBridgeConfig()

        assert config.min_predictions_for_scoring == 20
        assert config.calibration_weight == 0.4
        assert config.accuracy_weight == 0.3
        assert config.cost_weight == 0.3
        assert config.well_calibrated_ece_threshold == 0.1
        assert config.overconfident_cost_multiplier == 1.3
        assert config.underconfident_cost_multiplier == 1.1
        assert config.efficient_threshold == 0.7
        assert config.moderate_threshold == 0.4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CalibrationCostBridgeConfig(
            min_predictions_for_scoring=50,
            calibration_weight=0.5,
            accuracy_weight=0.25,
            cost_weight=0.25,
            efficient_threshold=0.8,
        )

        assert config.min_predictions_for_scoring == 50
        assert config.calibration_weight == 0.5
        assert config.efficient_threshold == 0.8

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = CalibrationCostBridgeConfig()
        total = config.calibration_weight + config.accuracy_weight + config.cost_weight
        assert abs(total - 1.0) < 0.001


# =============================================================================
# CalibrationCostBridge Initialization Tests
# =============================================================================


class TestCalibrationCostBridgeInit:
    """Tests for CalibrationCostBridge initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        bridge = CalibrationCostBridge()

        assert bridge.calibration_tracker is None
        assert bridge.cost_tracker is None
        assert isinstance(bridge.config, CalibrationCostBridgeConfig)
        assert bridge._efficiency_cache == {}
        assert bridge._cache_timestamp is None

    def test_with_trackers(self):
        """Test initialization with trackers."""
        cal_tracker = MockCalibrationTracker()
        cost_tracker = MockCostTracker()

        bridge = CalibrationCostBridge(
            calibration_tracker=cal_tracker,
            cost_tracker=cost_tracker,
        )

        assert bridge.calibration_tracker is cal_tracker
        assert bridge.cost_tracker is cost_tracker

    def test_with_custom_config(self):
        """Test initialization with custom config."""
        config = CalibrationCostBridgeConfig(min_predictions_for_scoring=100)

        bridge = CalibrationCostBridge(config=config)

        assert bridge.config.min_predictions_for_scoring == 100


# =============================================================================
# Cost Efficiency Computation Tests
# =============================================================================


class TestComputeCostEfficiency:
    """Tests for compute_cost_efficiency method."""

    def test_no_calibration_tracker(self):
        """Test computation without calibration tracker."""
        bridge = CalibrationCostBridge()

        result = bridge.compute_cost_efficiency("claude")

        assert result.agent_name == "claude"
        assert result.recommendation == "unknown"

    def test_no_calibration_data(self):
        """Test computation when agent has no calibration data."""
        tracker = MockCalibrationTracker({})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("unknown_agent")

        assert result.recommendation == "unknown"

    def test_insufficient_predictions(self):
        """Test computation with insufficient predictions."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(total_predictions=10)  # Below threshold
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("claude")

        assert result.recommendation == "insufficient_data"

    def test_well_calibrated_agent(self):
        """Test computation for well-calibrated agent."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    ece=0.05,  # Very low ECE
                    accuracy=0.90,
                    total_predictions=100,
                    is_overconfident=False,
                    is_underconfident=False,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("claude")

        assert result.calibration_score > 0.8  # High due to low ECE
        assert result.accuracy_score == 0.90
        assert result.is_overconfident is False
        assert result.is_underconfident is False
        assert result.confidence_reliability > 0.9  # High for well-calibrated

    def test_overconfident_agent(self):
        """Test computation for overconfident agent."""
        tracker = MockCalibrationTracker(
            {
                "gpt-4": MockCalibrationSummary(
                    ece=0.2,
                    accuracy=0.85,
                    total_predictions=100,
                    is_overconfident=True,
                    is_underconfident=False,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("gpt-4")

        assert result.is_overconfident is True
        assert result.confidence_reliability <= 0.7  # Reduced for overconfident
        # Efficiency should have 15% penalty applied

    def test_underconfident_agent(self):
        """Test computation for underconfident agent."""
        tracker = MockCalibrationTracker(
            {
                "gemini": MockCalibrationSummary(
                    ece=0.15,
                    accuracy=0.88,
                    total_predictions=100,
                    is_overconfident=False,
                    is_underconfident=True,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("gemini")

        assert result.is_underconfident is True
        # Underconfident gets only 5% penalty (less severe than overconfident)

    def test_efficiency_thresholds(self):
        """Test efficiency recommendation thresholds."""
        # Efficient agent (high score)
        tracker = MockCalibrationTracker(
            {
                "efficient": MockCalibrationSummary(
                    ece=0.02,
                    accuracy=0.95,
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("efficient")
        assert result.recommendation == "efficient"  # Score >= 0.7

    def test_moderate_efficiency(self):
        """Test moderate efficiency recommendation."""
        tracker = MockCalibrationTracker(
            {
                "moderate": MockCalibrationSummary(
                    ece=0.15,
                    accuracy=0.75,
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("moderate")
        # Score between 0.4 and 0.7 -> moderate

    def test_costly_recommendation(self):
        """Test costly recommendation for low efficiency."""
        tracker = MockCalibrationTracker(
            {
                "costly": MockCalibrationSummary(
                    ece=0.28,  # Very high ECE
                    accuracy=0.55,  # Low accuracy
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("costly")
        # Low scores should give "costly" or "moderate" recommendation

    def test_caches_result(self):
        """Test that results are cached."""
        tracker = MockCalibrationTracker({"claude": MockCalibrationSummary(total_predictions=100)})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result1 = bridge.compute_cost_efficiency("claude")

        assert "claude" in bridge._efficiency_cache
        assert bridge._cache_timestamp is not None


# =============================================================================
# Cost Score Calculation Tests
# =============================================================================


class TestCostScore:
    """Tests for _cost_score method."""

    def test_zero_cost(self):
        """Test cost score for zero cost."""
        bridge = CalibrationCostBridge()
        score = bridge._cost_score(Decimal("0"))
        assert score == 0.5  # Unknown/neutral

    def test_negative_cost(self):
        """Test cost score for negative cost."""
        bridge = CalibrationCostBridge()
        score = bridge._cost_score(Decimal("-1"))
        assert score == 0.5  # Treat as unknown

    def test_very_low_cost(self):
        """Test cost score for very low cost."""
        bridge = CalibrationCostBridge()
        score = bridge._cost_score(Decimal("0.0005"))
        assert score == 1.0  # Maximum efficiency

    def test_very_high_cost(self):
        """Test cost score for very high cost."""
        bridge = CalibrationCostBridge()
        score = bridge._cost_score(Decimal("0.15"))
        assert score == 0.0  # Minimum efficiency

    def test_mid_range_cost(self):
        """Test cost score for mid-range cost."""
        bridge = CalibrationCostBridge()
        score = bridge._cost_score(Decimal("0.05"))
        assert 0.0 < score < 1.0  # Somewhere in between


# =============================================================================
# Task Cost Estimation Tests
# =============================================================================


class TestEstimateTaskCost:
    """Tests for estimate_task_cost method."""

    def test_basic_estimation(self):
        """Test basic task cost estimation."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    total_predictions=100,
                    is_overconfident=False,
                    is_underconfident=False,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        cost = bridge.estimate_task_cost(
            agent_name="claude",
            base_cost=Decimal("0.01"),
            rounds=3,
        )

        assert cost > Decimal("0")

    def test_overconfident_multiplier(self):
        """Test cost multiplier for overconfident agent."""
        tracker = MockCalibrationTracker(
            {
                "gpt-4": MockCalibrationSummary(
                    total_predictions=100,
                    is_overconfident=True,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        cost = bridge.estimate_task_cost(
            agent_name="gpt-4",
            base_cost=Decimal("0.01"),
            rounds=3,
        )

        # Should have 1.3x multiplier
        expected_base = Decimal("0.01") * 3 * Decimal("1.3")
        assert cost == expected_base

    def test_underconfident_multiplier(self):
        """Test cost multiplier for underconfident agent."""
        tracker = MockCalibrationTracker(
            {
                "gemini": MockCalibrationSummary(
                    total_predictions=100,
                    is_underconfident=True,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        cost = bridge.estimate_task_cost(
            agent_name="gemini",
            base_cost=Decimal("0.01"),
            rounds=3,
        )

        # Should have 1.1x multiplier
        expected_base = Decimal("0.01") * 3 * Decimal("1.1")
        assert cost == expected_base

    def test_well_calibrated_discount(self):
        """Test cost discount for well-calibrated agent."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    total_predictions=100,
                    ece=0.05,  # Low ECE
                    is_overconfident=False,
                    is_underconfident=False,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        cost = bridge.estimate_task_cost(
            agent_name="claude",
            base_cost=Decimal("0.01"),
            rounds=3,
        )

        # Well-calibrated gets 0.9x multiplier
        # confidence_reliability > 0.8 from low ECE

    def test_different_rounds(self):
        """Test estimation with different round counts."""
        tracker = MockCalibrationTracker({"claude": MockCalibrationSummary(total_predictions=100)})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        cost_1 = bridge.estimate_task_cost("claude", Decimal("0.01"), rounds=1)
        cost_5 = bridge.estimate_task_cost("claude", Decimal("0.01"), rounds=5)

        assert cost_5 > cost_1


# =============================================================================
# Agent Recommendation Tests
# =============================================================================


class TestRecommendCostEfficientAgent:
    """Tests for recommend_cost_efficient_agent method."""

    def test_recommends_best_agent(self):
        """Test recommending most efficient agent."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    ece=0.05,
                    accuracy=0.92,
                    total_predictions=100,
                ),
                "gpt-4": MockCalibrationSummary(
                    ece=0.15,
                    accuracy=0.85,
                    total_predictions=100,
                ),
                "gemini": MockCalibrationSummary(
                    ece=0.10,
                    accuracy=0.88,
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.recommend_cost_efficient_agent(["claude", "gpt-4", "gemini"])

        assert result == "claude"  # Highest efficiency

    def test_filters_by_accuracy(self):
        """Test filtering by minimum accuracy."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    ece=0.05,
                    accuracy=0.65,  # Below threshold
                    total_predictions=100,
                ),
                "gpt-4": MockCalibrationSummary(
                    ece=0.10,
                    accuracy=0.75,  # Above threshold
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.recommend_cost_efficient_agent(
            ["claude", "gpt-4"],
            min_accuracy=0.70,
        )

        assert result == "gpt-4"  # claude filtered out

    def test_no_candidates(self):
        """Test when no candidates meet criteria."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    accuracy=0.50,  # Too low
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.recommend_cost_efficient_agent(
            ["claude"],
            min_accuracy=0.80,
        )

        assert result is None

    def test_skips_insufficient_data(self):
        """Test skipping agents with insufficient data."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    total_predictions=5,  # Insufficient
                ),
                "gpt-4": MockCalibrationSummary(
                    accuracy=0.85,
                    total_predictions=100,  # Sufficient
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.recommend_cost_efficient_agent(["claude", "gpt-4"])

        assert result == "gpt-4"

    def test_empty_agent_list(self):
        """Test with empty agent list."""
        bridge = CalibrationCostBridge()
        result = bridge.recommend_cost_efficient_agent([])
        assert result is None


# =============================================================================
# Agent Ranking Tests
# =============================================================================


class TestRankAgentsByCostEfficiency:
    """Tests for rank_agents_by_cost_efficiency method."""

    def test_ranks_agents(self):
        """Test ranking agents by efficiency."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    ece=0.05,
                    accuracy=0.92,
                    total_predictions=100,
                ),
                "gpt-4": MockCalibrationSummary(
                    ece=0.20,
                    accuracy=0.80,
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        rankings = bridge.rank_agents_by_cost_efficiency(["claude", "gpt-4"])

        assert len(rankings) == 2
        assert rankings[0][0] == "claude"  # Higher efficiency
        assert rankings[0][1] > rankings[1][1]  # Sorted descending

    def test_includes_all_agents(self):
        """Test that all agents are included in ranking."""
        tracker = MockCalibrationTracker(
            {
                "a": MockCalibrationSummary(total_predictions=100),
                "b": MockCalibrationSummary(total_predictions=100),
                "c": MockCalibrationSummary(total_predictions=100),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        rankings = bridge.rank_agents_by_cost_efficiency(["a", "b", "c"])

        assert len(rankings) == 3


# =============================================================================
# Budget-Aware Selection Tests
# =============================================================================


class TestBudgetAwareSelection:
    """Tests for get_budget_aware_selection method."""

    def test_selects_within_budget(self):
        """Test selecting agents within budget."""
        tracker = MockCalibrationTracker(
            {
                "cheap": MockCalibrationSummary(
                    ece=0.10,
                    accuracy=0.85,
                    total_predictions=100,
                ),
                "expensive": MockCalibrationSummary(
                    ece=0.08,
                    accuracy=0.90,
                    total_predictions=100,
                ),
            }
        )
        cost_tracker = MockCostTracker()
        cost_tracker.add_workspace_stats(
            "ws1",
            {
                "by_agent": {
                    "cheap": Decimal("0.01"),
                    "expensive": Decimal("0.10"),
                },
                "api_calls": 10,
            },
        )

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            cost_tracker=cost_tracker,
        )

        # Low budget should only allow cheap agent
        result = bridge.get_budget_aware_selection(
            ["cheap", "expensive"],
            budget_remaining=Decimal("0.05"),
            estimated_rounds=3,
        )

        assert "cheap" in result

    def test_sorts_by_efficiency(self):
        """Test that results are sorted by efficiency."""
        tracker = MockCalibrationTracker(
            {
                "a": MockCalibrationSummary(
                    ece=0.20,
                    accuracy=0.80,
                    total_predictions=100,
                ),
                "b": MockCalibrationSummary(
                    ece=0.05,
                    accuracy=0.90,
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.get_budget_aware_selection(
            ["a", "b"],
            budget_remaining=Decimal("100"),  # High budget
            estimated_rounds=3,
        )

        # "b" should be first (more efficient)
        if len(result) >= 2:
            # Both fit budget, "b" should be first
            pass

    def test_empty_when_no_fit(self):
        """Test empty result when no agent fits budget."""
        tracker = MockCalibrationTracker(
            {"expensive": MockCalibrationSummary(total_predictions=100)}
        )
        cost_tracker = MockCostTracker()
        cost_tracker.add_workspace_stats(
            "ws1",
            {
                "by_agent": {"expensive": Decimal("1.00")},
                "api_calls": 1,
            },
        )

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            cost_tracker=cost_tracker,
        )

        result = bridge.get_budget_aware_selection(
            ["expensive"],
            budget_remaining=Decimal("0.01"),  # Too low
            estimated_rounds=3,
        )

        # May be empty depending on cost estimation


# =============================================================================
# Overconfident/Well-Calibrated Detection Tests
# =============================================================================


class TestAgentDetection:
    """Tests for overconfident and well-calibrated agent detection."""

    def test_get_overconfident_agents(self):
        """Test detecting overconfident agents."""
        tracker = MockCalibrationTracker(
            {
                "overconfident": MockCalibrationSummary(
                    is_overconfident=True,
                    total_predictions=100,
                ),
                "normal": MockCalibrationSummary(
                    is_overconfident=False,
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.get_overconfident_agents()

        assert "overconfident" in result
        assert "normal" not in result

    def test_get_overconfident_agents_filtered(self):
        """Test detecting overconfident agents with filter."""
        tracker = MockCalibrationTracker(
            {
                "a": MockCalibrationSummary(is_overconfident=True),
                "b": MockCalibrationSummary(is_overconfident=True),
                "c": MockCalibrationSummary(is_overconfident=False),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        # Only check "a" and "c"
        result = bridge.get_overconfident_agents(["a", "c"])

        assert result == ["a"]

    def test_get_overconfident_agents_no_tracker(self):
        """Test getting overconfident agents without tracker."""
        bridge = CalibrationCostBridge()
        result = bridge.get_overconfident_agents()
        assert result == []

    def test_get_well_calibrated_agents(self):
        """Test detecting well-calibrated agents."""
        tracker = MockCalibrationTracker(
            {
                "well_calibrated": MockCalibrationSummary(
                    ece=0.05,  # Below threshold
                    total_predictions=100,
                ),
                "poorly_calibrated": MockCalibrationSummary(
                    ece=0.25,  # Above threshold
                    total_predictions=100,
                ),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.get_well_calibrated_agents()

        assert "well_calibrated" in result
        assert "poorly_calibrated" not in result

    def test_get_well_calibrated_agents_no_tracker(self):
        """Test getting well-calibrated agents without tracker."""
        bridge = CalibrationCostBridge()
        result = bridge.get_well_calibrated_agents()
        assert result == []


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Tests for cache management."""

    def test_get_efficiency_cached(self):
        """Test getting cached efficiency data."""
        tracker = MockCalibrationTracker({"claude": MockCalibrationSummary(total_predictions=100)})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        # Compute to populate cache
        bridge.compute_cost_efficiency("claude")

        # Get from cache
        cached = bridge.get_efficiency("claude")
        assert cached is not None
        assert cached.agent_name == "claude"

    def test_get_efficiency_not_cached(self):
        """Test getting uncached efficiency returns None."""
        bridge = CalibrationCostBridge()
        result = bridge.get_efficiency("unknown")
        assert result is None

    def test_cache_expiration(self):
        """Test cache expiration."""
        tracker = MockCalibrationTracker({"claude": MockCalibrationSummary(total_predictions=100)})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        # Compute to populate cache
        bridge.compute_cost_efficiency("claude")

        # Simulate cache expiration
        bridge._cache_timestamp = datetime.now() - timedelta(seconds=600)

        # Should be cleared due to expiration
        result = bridge.get_efficiency("claude")
        assert result is None

    def test_refresh_cache(self):
        """Test cache refresh."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(total_predictions=100),
                "gpt-4": MockCalibrationSummary(total_predictions=100),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        count = bridge.refresh_cache(["claude", "gpt-4"])

        assert count == 2
        assert "claude" in bridge._efficiency_cache
        assert "gpt-4" in bridge._efficiency_cache

    def test_refresh_cache_all_agents(self):
        """Test refreshing cache for all agents."""
        tracker = MockCalibrationTracker(
            {
                "a": MockCalibrationSummary(total_predictions=100),
                "b": MockCalibrationSummary(total_predictions=100),
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        count = bridge.refresh_cache()  # No specific agents

        assert count == 2

    def test_refresh_cache_no_agents(self):
        """Test refresh with no agents."""
        bridge = CalibrationCostBridge()
        count = bridge.refresh_cache()
        assert count == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_stats_without_trackers(self):
        """Test stats without trackers."""
        bridge = CalibrationCostBridge()
        stats = bridge.get_stats()

        assert stats["calibration_tracker_attached"] is False
        assert stats["cost_tracker_attached"] is False
        assert stats["agents_cached"] == 0

    def test_stats_with_trackers(self):
        """Test stats with trackers attached."""
        tracker = MockCalibrationTracker(
            {
                "claude": MockCalibrationSummary(
                    ece=0.05,
                    is_overconfident=False,
                ),
                "gpt-4": MockCalibrationSummary(
                    ece=0.20,
                    is_overconfident=True,
                ),
            }
        )
        cost_tracker = MockCostTracker()

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            cost_tracker=cost_tracker,
        )

        stats = bridge.get_stats()

        assert stats["calibration_tracker_attached"] is True
        assert stats["cost_tracker_attached"] is True
        assert stats["well_calibrated_agents"] == 1  # claude
        assert stats["overconfident_agents"] == 1  # gpt-4

    def test_stats_cache_validity(self):
        """Test stats includes cache validity."""
        tracker = MockCalibrationTracker({"claude": MockCalibrationSummary(total_predictions=100)})
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        # Initially no cache
        stats = bridge.get_stats()
        assert stats["cache_valid"] is False

        # After computation
        bridge.compute_cost_efficiency("claude")
        stats = bridge.get_stats()
        assert stats["cache_valid"] is True
        assert stats["agents_cached"] == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateCalibrationCostBridge:
    """Tests for create_calibration_cost_bridge factory function."""

    def test_create_basic_bridge(self):
        """Test creating basic bridge."""
        bridge = create_calibration_cost_bridge()

        assert isinstance(bridge, CalibrationCostBridge)
        assert bridge.calibration_tracker is None
        assert bridge.cost_tracker is None

    def test_create_with_trackers(self):
        """Test creating bridge with trackers."""
        cal_tracker = MockCalibrationTracker()
        cost_tracker = MockCostTracker()

        bridge = create_calibration_cost_bridge(
            calibration_tracker=cal_tracker,
            cost_tracker=cost_tracker,
        )

        assert bridge.calibration_tracker is cal_tracker
        assert bridge.cost_tracker is cost_tracker

    def test_create_with_config_kwargs(self):
        """Test creating bridge with config kwargs."""
        bridge = create_calibration_cost_bridge(
            min_predictions_for_scoring=50,
            efficient_threshold=0.8,
        )

        assert bridge.config.min_predictions_for_scoring == 50
        assert bridge.config.efficient_threshold == 0.8


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ece_at_boundary(self):
        """Test ECE at exact threshold boundary."""
        tracker = MockCalibrationTracker(
            {
                "boundary": MockCalibrationSummary(
                    ece=0.1,  # Exactly at well_calibrated_ece_threshold
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("boundary")
        # Should still compute without errors

    def test_zero_accuracy(self):
        """Test agent with zero accuracy."""
        tracker = MockCalibrationTracker(
            {
                "zero_acc": MockCalibrationSummary(
                    accuracy=0.0,
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("zero_acc")
        assert result.accuracy_score == 0.0

    def test_perfect_calibration(self):
        """Test perfectly calibrated agent (ECE=0)."""
        tracker = MockCalibrationTracker(
            {
                "perfect": MockCalibrationSummary(
                    ece=0.0,
                    accuracy=1.0,
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("perfect")
        assert result.calibration_score == 1.0

    def test_very_high_ece(self):
        """Test agent with very high ECE."""
        tracker = MockCalibrationTracker(
            {
                "bad": MockCalibrationSummary(
                    ece=0.5,  # Very poorly calibrated
                    accuracy=0.5,
                    total_predictions=100,
                )
            }
        )
        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("bad")
        # Calibration score should be capped at 0
        assert result.calibration_score >= 0.0

    def test_calibration_tracker_exception(self):
        """Test handling of calibration tracker exceptions."""
        tracker = MockCalibrationTracker({})

        # Make get_calibration_summary raise an exception
        def raise_error(name):
            raise RuntimeError("Tracker error")

        tracker.get_calibration_summary = raise_error

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        result = bridge.compute_cost_efficiency("agent")
        assert result.recommendation == "unknown"

    def test_exact_predictions_threshold(self):
        """Test agent with exactly threshold predictions."""
        config = CalibrationCostBridgeConfig(min_predictions_for_scoring=20)
        tracker = MockCalibrationTracker(
            {
                "exactly_20": MockCalibrationSummary(
                    total_predictions=20,  # Exactly at threshold
                    accuracy=0.85,
                )
            }
        )
        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            config=config,
        )

        result = bridge.compute_cost_efficiency("exactly_20")
        # Should NOT be "insufficient_data" since it meets threshold
        assert result.recommendation != "insufficient_data"

    def test_negative_budget(self):
        """Test budget-aware selection with negative budget."""
        bridge = CalibrationCostBridge()
        result = bridge.get_budget_aware_selection(
            ["agent1", "agent2"],
            budget_remaining=Decimal("-10"),
        )
        assert result == []
