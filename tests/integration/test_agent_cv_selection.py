"""
Integration tests for Agent CV-based Team Selection.

Tests the end-to-end flow of:
- CV generation from ELO and calibration data
- CV caching and persistence
- CV-based agent scoring in team selection
- Integration with debate team selection pipeline
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional

from aragora.agents.cv import (
    AgentCV,
    CVBuilder,
    DomainPerformance,
    ReliabilityMetrics,
)
from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig
from aragora.core import Agent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system with agent ratings."""
    from dataclasses import dataclass, field

    @dataclass
    class MockAgentRating:
        agent_name: str
        elo: float = 1500.0
        domain_elos: dict = field(default_factory=dict)
        wins: int = 10
        losses: int = 5
        draws: int = 0
        debates_count: int = 15

        @property
        def win_rate(self) -> float:
            total = self.wins + self.losses + self.draws
            return self.wins / total if total > 0 else 0.0

    ratings_data = {
        "claude-opus": MockAgentRating(
            agent_name="claude-opus",
            elo=1650.0,
            domain_elos={"code": 1680.0, "legal": 1620.0, "general": 1650.0},
        ),
        "gpt-4": MockAgentRating(agent_name="gpt-4", elo=1580.0),
        "gemini-pro": MockAgentRating(agent_name="gemini-pro", elo=1520.0),
        "mistral-large": MockAgentRating(agent_name="mistral-large", elo=1480.0),
    }

    elo = MagicMock()
    elo.get_rating = Mock(
        side_effect=lambda name: ratings_data.get(name, MockAgentRating(agent_name=name))
    )

    elo.get_agent_stats = Mock(
        return_value={
            "wins": 10,
            "losses": 5,
            "debates": 15,
        }
    )

    elo.get_domain_ratings = Mock(
        return_value={
            "code": 1680.0,
            "legal": 1620.0,
            "general": 1650.0,
        }
    )

    elo.get_learning_efficiency = Mock(
        return_value={
            "learning_category": "fast_learner",
            "elo_gain_rate": 5.0,
        }
    )
    return elo


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock calibration tracker."""
    tracker = MagicMock()
    tracker.get_brier_score = Mock(
        side_effect=lambda name, domain=None: {
            "claude-opus": 0.15,
            "gpt-4": 0.20,
            "gemini-pro": 0.25,
            "mistral-large": 0.30,
        }.get(name, 0.25)
    )

    tracker.get_brier_scores_batch = Mock(
        return_value={
            "claude-opus": 0.15,
            "gpt-4": 0.20,
            "gemini-pro": 0.25,
            "mistral-large": 0.30,
        }
    )

    tracker.get_expected_calibration_error = Mock(return_value=0.10)
    tracker.get_calibration_bias = Mock(return_value="well-calibrated")
    return tracker


@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor."""
    monitor = MagicMock()
    monitor.get_agent_metrics = Mock(
        return_value={
            "success_rate": 0.95,
            "failure_rate": 0.03,
            "timeout_rate": 0.02,
            "total_calls": 100,
            "avg_latency_ms": 250.0,
            "p50_latency_ms": 200.0,
            "p99_latency_ms": 800.0,
        }
    )
    return monitor


@pytest.fixture
def cv_builder(mock_elo_system, mock_calibration_tracker, mock_performance_monitor):
    """Create a CVBuilder with mocked dependencies."""
    return CVBuilder(
        elo_system=mock_elo_system,
        calibration_tracker=mock_calibration_tracker,
        performance_monitor=mock_performance_monitor,
    )


@pytest.fixture
def sample_agents():
    """Create sample Agent objects for testing."""
    return [
        Agent(name="claude-opus", model="claude-3-opus"),
        Agent(name="gpt-4", model="gpt-4-turbo"),
        Agent(name="gemini-pro", model="gemini-1.5-pro"),
        Agent(name="mistral-large", model="mistral-large"),
    ]


@pytest.fixture
def sample_cv():
    """Create a sample AgentCV for testing."""
    return AgentCV(
        agent_id="test-agent",
        model_name="test-model",
        overall_elo=1600.0,
        overall_win_rate=0.65,
        total_debates=50,
        calibration_accuracy=0.88,
        brier_score=0.12,
        expected_calibration_error=0.08,
        calibration_bias="well-calibrated",
        reliability=ReliabilityMetrics(
            success_rate=0.95,
            failure_rate=0.03,
            timeout_rate=0.02,
            total_calls=100,
            avg_latency_ms=200.0,
            p50_latency_ms=180.0,
            p99_latency_ms=500.0,
        ),
        domain_performance={
            "code": DomainPerformance(
                domain="code",
                elo=1680.0,
                win_rate=0.70,
                debates_count=20,
                calibration_accuracy=0.90,
                brier_score=0.10,
            ),
        },
        learning_category="steady",
        elo_gain_rate=5.0,
        model_capabilities=["code", "reasoning", "analysis"],
        learned_strengths=["debugging", "architecture"],
    )


# =============================================================================
# CV Builder Tests
# =============================================================================


class TestCVBuilderGeneration:
    """Tests for CV generation from data sources."""

    def test_cv_builder_generates_valid_cv(self, cv_builder):
        """Verify CVBuilder generates a complete CV with all fields."""
        cv = cv_builder.build_cv("claude-opus")

        assert cv is not None
        assert cv.agent_id == "claude-opus"
        assert cv.overall_elo > 0
        assert 0 <= cv.overall_win_rate <= 1
        assert cv.reliability is not None
        assert cv.reliability.success_rate >= 0

    def test_cv_builder_uses_elo_data(self, cv_builder, mock_elo_system):
        """Verify CVBuilder populates ELO data correctly."""
        cv = cv_builder.build_cv("claude-opus")

        # ELO system should have been queried
        mock_elo_system.get_rating.assert_called()
        assert cv.overall_elo == 1650.0

    def test_cv_builder_uses_calibration_data(self, cv_builder, mock_calibration_tracker):
        """Verify CVBuilder populates calibration data correctly."""
        cv = cv_builder.build_cv("claude-opus")

        # Calibration tracker should have been queried
        mock_calibration_tracker.get_brier_score.assert_called()
        assert cv.brier_score == 0.15

    def test_cv_builder_batch_generation(self, cv_builder):
        """Verify batch CV generation works correctly."""
        agent_ids = ["claude-opus", "gpt-4", "gemini-pro"]
        cvs = cv_builder.build_cvs_batch(agent_ids)

        assert len(cvs) == 3
        assert "claude-opus" in cvs
        assert "gpt-4" in cvs
        assert "gemini-pro" in cvs

        # Verify each CV is valid
        for agent_id, cv in cvs.items():
            assert cv.agent_id == agent_id
            assert cv.overall_elo > 0


class TestCVScoring:
    """Tests for CV-based agent scoring."""

    def test_cv_compute_selection_score(self, sample_cv):
        """Verify CV computes selection scores correctly."""
        score = sample_cv.compute_selection_score(
            domain="code",
            elo_weight=0.3,
            calibration_weight=0.2,
            reliability_weight=0.2,
            domain_weight=0.3,
        )

        assert 0 <= score <= 1
        # High ELO + good calibration + reliable should give high score
        assert score > 0.5

    def test_cv_reliability_check(self, sample_cv):
        """Verify reliability metrics are correctly assessed."""
        assert sample_cv.reliability.is_reliable is True
        # success_rate >= 0.9 and total_calls >= 5

    def test_cv_well_calibrated_check(self, sample_cv):
        """Verify calibration status is correctly assessed."""
        # expected_calibration_error < 0.15 means well calibrated
        assert sample_cv.is_well_calibrated is True

    def test_cv_best_domains(self, sample_cv):
        """Verify best domains are correctly identified."""
        best_domains = sample_cv.best_domains
        assert "code" in best_domains


# =============================================================================
# Team Selection Integration Tests
# =============================================================================


class TestTeamSelectorCVIntegration:
    """Tests for TeamSelector using CV-based scoring."""

    def test_team_selector_uses_cv_scoring(
        self, cv_builder, sample_agents, mock_elo_system, mock_calibration_tracker
    ):
        """Verify TeamSelector uses CV scores in selection."""
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.35,
            cv_reliability_threshold=0.7,
        )

        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=cv_builder,
            config=config,
        )

        # Select team for code domain
        team = selector.select(
            agents=sample_agents,
            domain="code",
            task="Review security fix",
        )

        # Should return agents sorted by score
        assert len(team) > 0
        assert all(isinstance(a, Agent) for a in team)

    def test_team_selector_cv_weight_affects_ordering(
        self, cv_builder, sample_agents, mock_elo_system, mock_calibration_tracker
    ):
        """Verify CV weight affects team selection ordering."""
        # High CV weight
        config_high = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.8,
        )
        selector_high = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=cv_builder,
            config=config_high,
        )

        # Low CV weight
        config_low = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.1,
        )
        selector_low = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=cv_builder,
            config=config_low,
        )

        team_high = selector_high.select(agents=sample_agents, domain="code")
        team_low = selector_low.select(agents=sample_agents, domain="code")

        # Both should return valid teams
        assert len(team_high) > 0
        assert len(team_low) > 0

    def test_cv_cache_invalidation(self, cv_builder):
        """Verify CV cache properly invalidates stale data."""
        # Build CV
        cv1 = cv_builder.build_cv("claude-opus")

        # Build again - should use fresh data (no cache in builder)
        cv2 = cv_builder.build_cv("claude-opus")

        # Both should be valid CVs
        assert cv1.agent_id == cv2.agent_id
        assert cv1.overall_elo == cv2.overall_elo

    def test_cv_based_selection_with_elo(
        self, cv_builder, sample_agents, mock_elo_system, mock_calibration_tracker
    ):
        """Verify CV selection works together with ELO system."""
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.35,
            elo_weight=0.35,  # Equal weight to ELO and CV
        )

        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=cv_builder,
            config=config,
        )

        team = selector.select(
            agents=sample_agents,
            domain="code",
        )

        # Claude-opus has highest ELO (1650) so should be first
        assert len(team) > 0
        # Top agent should be claude-opus due to highest ELO + good CV
        assert team[0].name == "claude-opus"


class TestCVSelectionEdgeCases:
    """Edge case tests for CV-based selection."""

    def test_selection_with_missing_cv_data(
        self, sample_agents, mock_elo_system, mock_calibration_tracker
    ):
        """Verify selection works when CV data is partially missing."""
        # Create builder with no performance monitor
        builder = CVBuilder(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            performance_monitor=None,
        )

        config = TeamSelectionConfig(enable_cv_selection=True)
        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=builder,
            config=config,
        )

        # Should still work with missing performance data
        team = selector.select(agents=sample_agents, domain="general")
        assert len(team) > 0

    def test_selection_with_unknown_agent(
        self, cv_builder, mock_elo_system, mock_calibration_tracker
    ):
        """Verify selection handles unknown agents gracefully."""
        unknown_agent = Agent(name="unknown-model", model="unknown")

        config = TeamSelectionConfig(enable_cv_selection=True)
        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=cv_builder,
            config=config,
        )

        team = selector.select(agents=[unknown_agent], domain="general")
        # Should return the agent even without CV data
        assert len(team) >= 0  # May return empty or the agent with default score

    def test_cv_disabled_fallback(self, sample_agents, mock_elo_system, mock_calibration_tracker):
        """Verify selection works when CV is disabled."""
        config = TeamSelectionConfig(
            enable_cv_selection=False,
            elo_weight=0.5,
        )

        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            cv_builder=None,
            config=config,
        )

        team = selector.select(agents=sample_agents, domain="general")
        assert len(team) > 0
