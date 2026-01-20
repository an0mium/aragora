"""Tests for CalibrationCostBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from aragora.billing.calibration_cost_bridge import (
    CalibrationCostBridge,
    CalibrationCostBridgeConfig,
    AgentCostEfficiency,
    create_calibration_cost_bridge,
)


@dataclass
class MockCalibrationBucket:
    """Mock calibration bucket."""

    range_start: float = 0.0
    range_end: float = 0.1
    total_predictions: int = 10
    correct_predictions: int = 8


@dataclass
class MockTemperatureParams:
    """Mock temperature params."""

    temperature: float = 1.0
    domain_temperatures: Dict[str, float] = field(default_factory=dict)


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary for testing."""

    agent: str = "test-agent"
    total_predictions: int = 100
    total_correct: int = 85
    brier_score: float = 0.1
    ece: float = 0.08
    buckets: List[MockCalibrationBucket] = field(default_factory=list)
    temperature_params: MockTemperatureParams = field(default_factory=MockTemperatureParams)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions

    @property
    def is_overconfident(self) -> bool:
        """Check if overconfident."""
        return hasattr(self, "_overconfident") and self._overconfident

    @property
    def is_underconfident(self) -> bool:
        """Check if underconfident."""
        return hasattr(self, "_underconfident") and self._underconfident


class MockCalibrationTracker:
    """Mock calibration tracker."""

    def __init__(self):
        self._summaries: Dict[str, MockCalibrationSummary] = {}
        self._agents: List[str] = []

    def get_calibration_summary(self, agent: str) -> Optional[MockCalibrationSummary]:
        """Get calibration summary."""
        return self._summaries.get(agent)

    def get_all_agents(self) -> List[str]:
        """Get all tracked agents."""
        return self._agents

    def add_summary(self, summary: MockCalibrationSummary) -> None:
        """Add a summary."""
        self._summaries[summary.agent] = summary
        if summary.agent not in self._agents:
            self._agents.append(summary.agent)


class MockCostTracker:
    """Mock cost tracker."""

    def __init__(self):
        self._workspace_stats: Dict[str, Dict] = {}

    def add_workspace_stats(
        self, workspace_id: str, by_agent: Dict[str, Decimal], api_calls: int
    ) -> None:
        """Add workspace stats."""
        self._workspace_stats[workspace_id] = {
            "by_agent": by_agent,
            "api_calls": api_calls,
        }


class TestAgentCostEfficiency:
    """Tests for AgentCostEfficiency dataclass."""

    def test_defaults(self):
        """Test default values."""
        efficiency = AgentCostEfficiency(
            agent_name="test",
            calibration_score=0.8,
            accuracy_score=0.85,
        )
        assert efficiency.agent_name == "test"
        assert efficiency.efficiency_score == 0.0
        assert efficiency.recommendation == ""

    def test_full_creation(self):
        """Test full creation."""
        efficiency = AgentCostEfficiency(
            agent_name="claude",
            calibration_score=0.9,
            accuracy_score=0.88,
            cost_per_call=Decimal("0.01"),
            efficiency_score=0.85,
            confidence_reliability=0.9,
            is_overconfident=False,
            recommendation="efficient",
        )
        assert efficiency.calibration_score == 0.9
        assert efficiency.recommendation == "efficient"


class TestCalibrationCostBridge:
    """Tests for CalibrationCostBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = CalibrationCostBridge()
        assert bridge.calibration_tracker is None
        assert bridge.cost_tracker is None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = CalibrationCostBridgeConfig(
            min_predictions_for_scoring=10,
            calibration_weight=0.5,
        )
        bridge = CalibrationCostBridge(config=config)
        assert bridge.config.min_predictions_for_scoring == 10
        assert bridge.config.calibration_weight == 0.5

    def test_compute_cost_efficiency_no_tracker(self):
        """Test efficiency with no tracker."""
        bridge = CalibrationCostBridge()
        efficiency = bridge.compute_cost_efficiency("unknown")
        assert efficiency.recommendation == "unknown"

    def test_compute_cost_efficiency_insufficient_data(self):
        """Test efficiency with insufficient data."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(MockCalibrationSummary(agent="claude", total_predictions=5))

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            config=CalibrationCostBridgeConfig(min_predictions_for_scoring=20),
        )

        efficiency = bridge.compute_cost_efficiency("claude")
        assert efficiency.recommendation == "insufficient_data"

    def test_compute_cost_efficiency_well_calibrated(self):
        """Test efficiency for well-calibrated agent."""
        tracker = MockCalibrationTracker()
        summary = MockCalibrationSummary(
            agent="claude",
            total_predictions=100,
            total_correct=88,
            ece=0.05,  # Good calibration
        )
        tracker.add_summary(summary)

        bridge = CalibrationCostBridge(calibration_tracker=tracker)
        efficiency = bridge.compute_cost_efficiency("claude")

        assert efficiency.calibration_score > 0.8
        assert efficiency.efficiency_score > 0.5
        assert efficiency.recommendation in ["efficient", "moderate"]

    def test_compute_cost_efficiency_overconfident(self):
        """Test efficiency for overconfident agent."""
        tracker = MockCalibrationTracker()
        summary = MockCalibrationSummary(
            agent="overconfident",
            total_predictions=100,
            total_correct=60,
            ece=0.2,  # Poor calibration
        )
        summary._overconfident = True
        tracker.add_summary(summary)

        bridge = CalibrationCostBridge(calibration_tracker=tracker)
        efficiency = bridge.compute_cost_efficiency("overconfident")

        assert efficiency.is_overconfident
        assert efficiency.confidence_reliability < 0.9

    def test_estimate_task_cost(self):
        """Test task cost estimation."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="claude",
                total_predictions=100,
                ece=0.05,
            )
        )

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        base_cost = Decimal("0.10")
        estimated = bridge.estimate_task_cost("claude", base_cost, rounds=3)

        # Well-calibrated should get slight discount
        assert estimated <= base_cost * 3

    def test_estimate_task_cost_overconfident(self):
        """Test cost estimation for overconfident agent."""
        tracker = MockCalibrationTracker()
        summary = MockCalibrationSummary(
            agent="overconfident",
            total_predictions=100,
            ece=0.15,
        )
        summary._overconfident = True
        tracker.add_summary(summary)

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            config=CalibrationCostBridgeConfig(overconfident_cost_multiplier=1.3),
        )

        base_cost = Decimal("0.10")
        estimated = bridge.estimate_task_cost("overconfident", base_cost, rounds=3)

        # Overconfident should cost more
        assert estimated > base_cost * 3

    def test_recommend_cost_efficient_agent(self):
        """Test recommending cost-efficient agent."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="efficient",
                total_predictions=100,
                total_correct=90,
                ece=0.05,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="inefficient",
                total_predictions=100,
                total_correct=70,
                ece=0.2,
            )
        )

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        recommended = bridge.recommend_cost_efficient_agent(
            available_agents=["efficient", "inefficient"],
            min_accuracy=0.7,
        )

        assert recommended == "efficient"

    def test_recommend_cost_efficient_agent_none_qualify(self):
        """Test recommendation when none qualify."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="low-accuracy",
                total_predictions=100,
                total_correct=50,  # 50% accuracy
            )
        )

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        recommended = bridge.recommend_cost_efficient_agent(
            available_agents=["low-accuracy"],
            min_accuracy=0.8,  # Require 80%
        )

        assert recommended is None

    def test_rank_agents_by_cost_efficiency(self):
        """Test ranking agents by efficiency."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="best",
                total_predictions=100,
                total_correct=95,
                ece=0.03,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="good",
                total_predictions=100,
                total_correct=85,
                ece=0.08,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="poor",
                total_predictions=100,
                total_correct=60,
                ece=0.2,
            )
        )

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        ranking = bridge.rank_agents_by_cost_efficiency(available_agents=["best", "good", "poor"])

        # Should be sorted by efficiency descending
        assert ranking[0][0] == "best"
        assert ranking[-1][0] == "poor"

    def test_get_budget_aware_selection(self):
        """Test budget-aware agent selection."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="cheap",
                total_predictions=100,
                ece=0.05,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="expensive",
                total_predictions=100,
                ece=0.05,
            )
        )

        cost_tracker = MockCostTracker()
        cost_tracker.add_workspace_stats(
            "ws1",
            {"cheap": Decimal("0.001"), "expensive": Decimal("0.05")},
            100,
        )

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            cost_tracker=cost_tracker,
        )

        agents = bridge.get_budget_aware_selection(
            available_agents=["cheap", "expensive"],
            budget_remaining=Decimal("0.05"),
            estimated_rounds=3,
        )

        # Cheap should be included, expensive might not fit
        assert "cheap" in agents

    def test_get_overconfident_agents(self):
        """Test getting overconfident agents."""
        tracker = MockCalibrationTracker()
        summary1 = MockCalibrationSummary(agent="overconfident", total_predictions=100)
        summary1._overconfident = True
        tracker.add_summary(summary1)

        summary2 = MockCalibrationSummary(agent="well-calibrated", total_predictions=100)
        tracker.add_summary(summary2)

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        overconfident = bridge.get_overconfident_agents()
        assert "overconfident" in overconfident
        assert "well-calibrated" not in overconfident

    def test_get_well_calibrated_agents(self):
        """Test getting well-calibrated agents."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="well-calibrated",
                total_predictions=100,
                ece=0.05,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="poorly-calibrated",
                total_predictions=100,
                ece=0.25,
            )
        )

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            config=CalibrationCostBridgeConfig(well_calibrated_ece_threshold=0.1),
        )

        well_calibrated = bridge.get_well_calibrated_agents()
        assert "well-calibrated" in well_calibrated
        assert "poorly-calibrated" not in well_calibrated

    def test_get_efficiency_cached(self):
        """Test getting cached efficiency."""
        bridge = CalibrationCostBridge()
        bridge._efficiency_cache["test"] = AgentCostEfficiency(
            agent_name="test",
            calibration_score=0.8,
            accuracy_score=0.85,
        )
        bridge._cache_timestamp = __import__("datetime").datetime.now()

        efficiency = bridge.get_efficiency("test")
        assert efficiency is not None
        assert efficiency.agent_name == "test"

    def test_refresh_cache(self):
        """Test refreshing cache."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(MockCalibrationSummary(agent="agent1", total_predictions=100))
        tracker.add_summary(MockCalibrationSummary(agent="agent2", total_predictions=100))

        bridge = CalibrationCostBridge(calibration_tracker=tracker)

        refreshed = bridge.refresh_cache(["agent1", "agent2"])
        assert refreshed == 2

    def test_get_stats(self):
        """Test getting bridge stats."""
        tracker = MockCalibrationTracker()
        cost_tracker = MockCostTracker()

        bridge = CalibrationCostBridge(
            calibration_tracker=tracker,
            cost_tracker=cost_tracker,
        )
        bridge._efficiency_cache["test"] = AgentCostEfficiency(
            agent_name="test",
            calibration_score=0.8,
            accuracy_score=0.85,
        )

        stats = bridge.get_stats()
        assert stats["agents_cached"] == 1
        assert stats["calibration_tracker_attached"]
        assert stats["cost_tracker_attached"]

    def test_factory_function(self):
        """Test factory function."""
        tracker = MockCalibrationTracker()
        bridge = create_calibration_cost_bridge(
            calibration_tracker=tracker,
            min_predictions_for_scoring=15,
            calibration_weight=0.45,
        )
        assert bridge.calibration_tracker is tracker
        assert bridge.config.min_predictions_for_scoring == 15
        assert bridge.config.calibration_weight == 0.45

    def test_cost_score_calculation(self):
        """Test cost score calculation."""
        bridge = CalibrationCostBridge()

        # Cheap call
        cheap_score = bridge._cost_score(Decimal("0.0001"))
        assert cheap_score == 1.0

        # Expensive call
        expensive_score = bridge._cost_score(Decimal("0.10"))
        assert expensive_score == 0.0

        # Mid-range
        mid_score = bridge._cost_score(Decimal("0.05"))
        assert 0.0 < mid_score < 1.0

        # Unknown (zero)
        unknown_score = bridge._cost_score(Decimal("0"))
        assert unknown_score == 0.5
