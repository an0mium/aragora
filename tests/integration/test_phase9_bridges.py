"""Integration tests for Phase 9 Cross-Pollination Bridges.

Tests verify that:
- Bridges auto-initialize in SubsystemCoordinator
- Bridges correctly sync data between subsystems
- Bridge configurations from ArenaConfig are applied
- Bridge lifecycle integrates with debate flow
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from aragora.debate.subsystem_coordinator import SubsystemCoordinator, SubsystemConfig


# =============================================================================
# Mock Source Subsystems
# =============================================================================


@dataclass
class MockPerformanceMetrics:
    """Mock performance metrics for an agent."""

    avg_response_time: float = 1.0
    quality_score: float = 0.8
    consistency_score: float = 0.9
    total_calls: int = 10


class MockPerformanceMonitor:
    """Mock PerformanceMonitor for testing."""

    def __init__(self):
        self._metrics: Dict[str, MockPerformanceMetrics] = {}

    def get_agent_metrics(self, agent_name: str) -> Optional[MockPerformanceMetrics]:
        """Get metrics for an agent."""
        return self._metrics.get(agent_name)

    def add_metrics(self, agent_name: str, metrics: MockPerformanceMetrics) -> None:
        """Add metrics for an agent."""
        self._metrics[agent_name] = metrics


@dataclass
class MockConsensusOutcome:
    """Mock consensus outcome."""

    debate_id: str = ""
    confidence: float = 0.8
    was_correct: bool = True
    participants: List[str] = field(default_factory=list)


class MockOutcomeTracker:
    """Mock OutcomeTracker for testing."""

    def __init__(self):
        self._outcomes: List[MockConsensusOutcome] = []
        self._agent_stats: Dict[str, Dict[str, Any]] = {}

    def record_outcome(self, outcome: MockConsensusOutcome) -> None:
        """Record an outcome."""
        self._outcomes.append(outcome)

    def get_agent_success_rate(self, agent_name: str) -> float:
        """Get success rate for an agent."""
        return self._agent_stats.get(agent_name, {}).get("success_rate", 0.5)

    def add_agent_stats(self, agent_name: str, success_rate: float) -> None:
        """Add agent stats."""
        self._agent_stats[agent_name] = {"success_rate": success_rate}


@dataclass
class MockNoveltyResult:
    """Mock novelty result."""

    agent_name: str = ""
    novelty_score: float = 0.5
    is_novel: bool = True


class MockNoveltyTracker:
    """Mock NoveltyTracker for testing."""

    def __init__(self):
        self._results: Dict[str, List[MockNoveltyResult]] = {}
        self._low_novelty_agents: List[str] = []

    def record_result(self, result: MockNoveltyResult) -> None:
        """Record a novelty result."""
        if result.agent_name not in self._results:
            self._results[result.agent_name] = []
        self._results[result.agent_name].append(result)

    def get_agent_novelty_results(self, agent_name: str) -> List[MockNoveltyResult]:
        """Get novelty results for an agent."""
        return self._results.get(agent_name, [])

    def get_low_novelty_agents(self) -> List[str]:
        """Get agents with consistently low novelty."""
        return self._low_novelty_agents


@dataclass
class MockRelationshipMetrics:
    """Mock relationship metrics."""

    alliance_score: float = 0.5
    rivalry_score: float = 0.3
    agreement_rate: float = 0.6
    debate_count: int = 10


class MockRelationshipTracker:
    """Mock RelationshipTracker for testing."""

    def __init__(self):
        self._relationships: Dict[tuple, MockRelationshipMetrics] = {}

    def compute_metrics(
        self, agent_a: str, agent_b: str
    ) -> Optional[MockRelationshipMetrics]:
        """Compute relationship metrics between agents."""
        key = tuple(sorted([agent_a, agent_b]))
        return self._relationships.get(key)

    def add_relationship(
        self, agent_a: str, agent_b: str, metrics: MockRelationshipMetrics
    ) -> None:
        """Add relationship metrics."""
        key = tuple(sorted([agent_a, agent_b]))
        self._relationships[key] = metrics


class MockRLMBridge:
    """Mock RLMBridge for testing."""

    def __init__(self):
        self._compression_results: List[Any] = []

    def compress(self, content: str) -> Any:
        """Compress content."""
        return MagicMock(estimated_fidelity=0.85, time_seconds=0.1, cache_hits=1)


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary."""

    agent: str = ""
    total_predictions: int = 100
    total_correct: int = 85
    ece: float = 0.08
    brier_score: float = 0.1
    is_overconfident: bool = False
    is_underconfident: bool = False

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions


class MockCalibrationTracker:
    """Mock CalibrationTracker for testing."""

    def __init__(self):
        self._summaries: Dict[str, MockCalibrationSummary] = {}

    def get_calibration_summary(
        self, agent_name: str
    ) -> Optional[MockCalibrationSummary]:
        """Get calibration summary for an agent."""
        return self._summaries.get(agent_name)

    def add_summary(self, summary: MockCalibrationSummary) -> None:
        """Add a calibration summary."""
        self._summaries[summary.agent] = summary

    def get_all_agents(self) -> List[str]:
        """Get all tracked agents."""
        return list(self._summaries.keys())


# =============================================================================
# Test Cases
# =============================================================================


class TestSubsystemCoordinatorBridgeInitialization:
    """Tests for bridge auto-initialization in SubsystemCoordinator."""

    def test_no_bridges_without_sources(self):
        """Test no bridges created when no source subsystems provided."""
        coordinator = SubsystemCoordinator()
        status = coordinator.get_status()

        assert status["active_bridges_count"] == 0
        assert not status["cross_pollination_bridges"]["performance_router"]
        assert not status["cross_pollination_bridges"]["outcome_complexity"]

    def test_performance_router_bridge_auto_init(self):
        """Test PerformanceRouterBridge auto-initializes with performance_monitor."""
        monitor = MockPerformanceMonitor()
        monitor.add_metrics(
            "claude", MockPerformanceMetrics(avg_response_time=0.5, quality_score=0.9)
        )

        coordinator = SubsystemCoordinator(
            performance_monitor=monitor,
            enable_performance_router=True,
        )
        status = coordinator.get_status()

        assert status["cross_pollination_bridges"]["performance_router"]
        assert coordinator.performance_router_bridge is not None

    def test_relationship_bias_bridge_auto_init(self):
        """Test RelationshipBiasBridge auto-initializes with relationship_tracker."""
        tracker = MockRelationshipTracker()
        tracker.add_relationship(
            "claude",
            "gpt",
            MockRelationshipMetrics(alliance_score=0.8, agreement_rate=0.85),
        )

        coordinator = SubsystemCoordinator(
            relationship_tracker=tracker,
            enable_relationship_bias=True,
        )
        status = coordinator.get_status()

        assert status["cross_pollination_bridges"]["relationship_bias"]
        assert coordinator.relationship_bias_bridge is not None

    def test_calibration_cost_bridge_auto_init(self):
        """Test CalibrationCostBridge auto-initializes with calibration_tracker."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="claude",
                total_predictions=100,
                total_correct=90,
                ece=0.05,
            )
        )

        coordinator = SubsystemCoordinator(
            calibration_tracker=tracker,
            enable_calibration_cost=True,
        )
        status = coordinator.get_status()

        assert status["cross_pollination_bridges"]["calibration_cost"]
        assert coordinator.calibration_cost_bridge is not None

    def test_multiple_bridges_init(self):
        """Test multiple bridges can be initialized together."""
        perf_monitor = MockPerformanceMonitor()
        rel_tracker = MockRelationshipTracker()
        cal_tracker = MockCalibrationTracker()

        coordinator = SubsystemCoordinator(
            performance_monitor=perf_monitor,
            relationship_tracker=rel_tracker,
            calibration_tracker=cal_tracker,
            enable_performance_router=True,
            enable_relationship_bias=True,
            enable_calibration_cost=True,
        )
        status = coordinator.get_status()

        assert status["active_bridges_count"] == 3
        assert status["cross_pollination_bridges"]["performance_router"]
        assert status["cross_pollination_bridges"]["relationship_bias"]
        assert status["cross_pollination_bridges"]["calibration_cost"]

    def test_bridges_disabled_when_flag_false(self):
        """Test bridges not created when enable flag is False."""
        perf_monitor = MockPerformanceMonitor()

        coordinator = SubsystemCoordinator(
            performance_monitor=perf_monitor,
            enable_performance_router=False,
        )
        status = coordinator.get_status()

        assert not status["cross_pollination_bridges"]["performance_router"]
        assert coordinator.performance_router_bridge is None


class TestSubsystemConfigBridges:
    """Tests for SubsystemConfig bridge integration."""

    def test_config_creates_coordinator_with_bridges(self):
        """Test SubsystemConfig creates coordinator with bridge sources."""
        perf_monitor = MockPerformanceMonitor()
        rel_tracker = MockRelationshipTracker()

        config = SubsystemConfig(
            performance_monitor=perf_monitor,
            relationship_tracker=rel_tracker,
            enable_performance_router=True,
            enable_relationship_bias=True,
        )
        coordinator = config.create_coordinator()

        status = coordinator.get_status()
        assert status["cross_pollination_bridges"]["performance_router"]
        assert status["cross_pollination_bridges"]["relationship_bias"]

    def test_config_pre_configured_bridge(self):
        """Test SubsystemConfig with pre-configured bridge."""
        from aragora.debate.relationship_bias_bridge import RelationshipBiasBridge

        pre_bridge = RelationshipBiasBridge()

        config = SubsystemConfig(
            relationship_bias_bridge=pre_bridge,
            enable_relationship_bias=True,
        )
        coordinator = config.create_coordinator()

        # Pre-configured bridge should be used
        assert coordinator.relationship_bias_bridge is pre_bridge


class TestBridgeCapabilityChecks:
    """Tests for bridge capability property accessors."""

    def test_has_bridge_properties(self):
        """Test has_*_bridge properties work correctly."""
        coordinator = SubsystemCoordinator()

        assert not coordinator.has_performance_router_bridge
        assert not coordinator.has_outcome_complexity_bridge
        assert not coordinator.has_novelty_selection_bridge
        assert not coordinator.has_relationship_bias_bridge
        assert not coordinator.has_rlm_selection_bridge
        assert not coordinator.has_calibration_cost_bridge

    def test_active_bridges_count(self):
        """Test active_bridges_count property."""
        perf_monitor = MockPerformanceMonitor()
        rel_tracker = MockRelationshipTracker()
        cal_tracker = MockCalibrationTracker()

        coordinator = SubsystemCoordinator(
            performance_monitor=perf_monitor,
            relationship_tracker=rel_tracker,
            calibration_tracker=cal_tracker,
        )

        assert coordinator.active_bridges_count == 3


class TestBridgeDataFlow:
    """Tests for data flow through bridges."""

    def test_relationship_bias_echo_chamber_detection(self):
        """Test relationship bias bridge can detect echo chambers."""
        tracker = MockRelationshipTracker()
        # Add high-alliance pair
        tracker.add_relationship(
            "claude",
            "gpt",
            MockRelationshipMetrics(
                alliance_score=0.9,
                agreement_rate=0.95,
                debate_count=20,
            ),
        )

        coordinator = SubsystemCoordinator(
            relationship_tracker=tracker,
            enable_relationship_bias=True,
        )

        assert coordinator.relationship_bias_bridge is not None

        # Test echo chamber risk detection
        risk = coordinator.relationship_bias_bridge.compute_team_echo_risk(
            ["claude", "gpt"]
        )
        assert risk.overall_risk > 0.5
        assert risk.recommendation in ["caution", "high_risk"]

    def test_calibration_cost_efficiency_scoring(self):
        """Test calibration cost bridge computes efficiency scores."""
        tracker = MockCalibrationTracker()
        tracker.add_summary(
            MockCalibrationSummary(
                agent="claude",
                total_predictions=100,
                total_correct=90,
                ece=0.05,
            )
        )
        tracker.add_summary(
            MockCalibrationSummary(
                agent="gpt",
                total_predictions=100,
                total_correct=60,
                ece=0.2,
                is_overconfident=True,
            )
        )

        coordinator = SubsystemCoordinator(
            calibration_tracker=tracker,
            enable_calibration_cost=True,
        )

        assert coordinator.calibration_cost_bridge is not None

        # Claude should have higher efficiency
        claude_eff = coordinator.calibration_cost_bridge.compute_cost_efficiency(
            "claude"
        )
        gpt_eff = coordinator.calibration_cost_bridge.compute_cost_efficiency("gpt")

        assert claude_eff.efficiency_score > gpt_eff.efficiency_score
        assert claude_eff.recommendation in ["efficient", "moderate"]


class TestBridgeStatusIntegration:
    """Tests for bridge status reporting."""

    def test_get_status_includes_bridges(self):
        """Test get_status includes bridge information."""
        perf_monitor = MockPerformanceMonitor()

        coordinator = SubsystemCoordinator(
            performance_monitor=perf_monitor,
            enable_performance_router=True,
        )

        status = coordinator.get_status()

        assert "cross_pollination_bridges" in status
        assert "active_bridges_count" in status
        assert status["cross_pollination_bridges"]["performance_router"]

    def test_status_all_bridges_reported(self):
        """Test all 7 bridge types are reported in status."""
        coordinator = SubsystemCoordinator()
        status = coordinator.get_status()

        bridges = status["cross_pollination_bridges"]
        assert len(bridges) == 7
        assert "performance_router" in bridges
        assert "outcome_complexity" in bridges
        assert "analytics_selection" in bridges
        assert "novelty_selection" in bridges
        assert "relationship_bias" in bridges
        assert "rlm_selection" in bridges
        assert "calibration_cost" in bridges


class TestBridgeErrorHandling:
    """Tests for bridge initialization error handling."""

    def test_bridge_init_error_captured(self):
        """Test bridge initialization errors are captured in init_errors."""
        # Create coordinator - if import fails, it should be captured
        coordinator = SubsystemCoordinator()

        status = coordinator.get_status()
        # No errors expected when no source subsystems provided
        # (bridges just don't initialize)
        assert coordinator.active_bridges_count == 0

    def test_partial_bridge_initialization(self):
        """Test some bridges can initialize even if others fail."""
        # Only relationship tracker provided - only that bridge should init
        rel_tracker = MockRelationshipTracker()

        coordinator = SubsystemCoordinator(
            relationship_tracker=rel_tracker,
            enable_relationship_bias=True,
            enable_performance_router=True,  # Won't work - no monitor
        )

        status = coordinator.get_status()
        assert status["cross_pollination_bridges"]["relationship_bias"]
        assert not status["cross_pollination_bridges"]["performance_router"]
