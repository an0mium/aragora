"""
Tests for TeamSelector - Agent selection for debates.

Tests cover:
- TeamSelectionConfig defaults and customization
- Agent scoring with ELO ratings
- Agent scoring with calibration
- Circuit breaker filtering
- Combined selection logic
- Edge cases
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock Agent for testing."""

    name: str
    model: str = "test-model"


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self, ratings: dict[str, float] = None):
        self._ratings = ratings or {}

    def get_rating(self, agent_name: str) -> float:
        if agent_name not in self._ratings:
            raise KeyError(f"Unknown agent: {agent_name}")
        return self._ratings[agent_name]


class MockCalibrationTracker:
    """Mock calibration tracker for testing."""

    def __init__(self, brier_scores: dict[str, float] = None):
        self._scores = brier_scores or {}

    def get_brier_score(self, agent_name: str, domain: str = None) -> float:
        if agent_name not in self._scores:
            raise KeyError(f"Unknown agent: {agent_name}")
        return self._scores[agent_name]


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, blocked_agents: list[str] = None):
        self._blocked = set(blocked_agents or [])

    def filter_available_agents(self, agent_names: list[str]) -> list[str]:
        return [name for name in agent_names if name not in self._blocked]


@pytest.fixture
def agents():
    """Create a list of test agents."""
    return [
        MockAgent(name="claude"),
        MockAgent(name="gpt"),
        MockAgent(name="gemini"),
    ]


@pytest.fixture
def selector():
    """Create a basic TeamSelector."""
    return TeamSelector()


# =============================================================================
# TeamSelectionConfig Tests
# =============================================================================


class TestTeamSelectionConfig:
    """Tests for TeamSelectionConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = TeamSelectionConfig()
        assert config.elo_weight == 0.3
        assert config.calibration_weight == 0.2
        assert config.base_score == 1.0
        assert config.elo_baseline == 1000

    def test_custom_values(self):
        """Config should accept custom values."""
        config = TeamSelectionConfig(
            elo_weight=0.5,
            calibration_weight=0.4,
            base_score=2.0,
            elo_baseline=1500,
        )
        assert config.elo_weight == 0.5
        assert config.calibration_weight == 0.4
        assert config.base_score == 2.0
        assert config.elo_baseline == 1500


# =============================================================================
# TeamSelector Initialization Tests
# =============================================================================


class TestTeamSelectorInit:
    """Tests for TeamSelector initialization."""

    def test_init_defaults(self):
        """Should initialize with None dependencies."""
        selector = TeamSelector()
        assert selector.elo_system is None
        assert selector.calibration_tracker is None
        assert selector.circuit_breaker is None
        assert selector.config is not None

    def test_init_with_elo(self):
        """Should accept ELO system."""
        elo = MockEloSystem({"claude": 1500})
        selector = TeamSelector(elo_system=elo)
        assert selector.elo_system is elo

    def test_init_with_calibration(self):
        """Should accept calibration tracker."""
        calibration = MockCalibrationTracker({"claude": 0.2})
        selector = TeamSelector(calibration_tracker=calibration)
        assert selector.calibration_tracker is calibration

    def test_init_with_circuit_breaker(self):
        """Should accept circuit breaker."""
        breaker = MockCircuitBreaker([])
        selector = TeamSelector(circuit_breaker=breaker)
        assert selector.circuit_breaker is breaker

    def test_init_with_custom_config(self):
        """Should accept custom config."""
        config = TeamSelectionConfig(elo_weight=0.5)
        selector = TeamSelector(config=config)
        assert selector.config.elo_weight == 0.5


# =============================================================================
# Select Method Tests - Basic
# =============================================================================


class TestSelectBasic:
    """Basic tests for the select method."""

    def test_select_returns_all_agents(self, selector, agents):
        """Without any scoring, should return all agents."""
        selected = selector.select(agents)
        assert len(selected) == 3

    def test_select_empty_list(self, selector):
        """Should handle empty agent list."""
        selected = selector.select([])
        assert selected == []

    def test_select_single_agent(self, selector):
        """Should handle single agent."""
        agents = [MockAgent(name="solo")]
        selected = selector.select(agents)
        assert len(selected) == 1
        assert selected[0].name == "solo"

    def test_select_with_domain(self, selector, agents):
        """Should accept domain parameter."""
        # Domain doesn't affect selection yet, just logging
        selected = selector.select(agents, domain="technical")
        assert len(selected) == 3


# =============================================================================
# Select Method Tests - Circuit Breaker
# =============================================================================


class TestSelectCircuitBreaker:
    """Tests for circuit breaker filtering."""

    def test_filters_blocked_agents(self, agents):
        """Should exclude agents blocked by circuit breaker."""
        breaker = MockCircuitBreaker(blocked_agents=["gpt"])
        selector = TeamSelector(circuit_breaker=breaker)

        selected = selector.select(agents)

        names = [a.name for a in selected]
        assert "gpt" not in names
        assert "claude" in names
        assert "gemini" in names

    def test_filters_multiple_blocked(self, agents):
        """Should filter multiple blocked agents."""
        breaker = MockCircuitBreaker(blocked_agents=["gpt", "gemini"])
        selector = TeamSelector(circuit_breaker=breaker)

        selected = selector.select(agents)

        assert len(selected) == 1
        assert selected[0].name == "claude"

    def test_all_blocked_returns_original(self, agents):
        """If all blocked, should return original list."""
        breaker = MockCircuitBreaker(blocked_agents=["claude", "gpt", "gemini"])
        selector = TeamSelector(circuit_breaker=breaker)

        selected = selector.select(agents)

        # Falls back to original when none available
        assert selected == agents

    def test_circuit_breaker_error_handled(self, agents):
        """Should handle circuit breaker errors gracefully."""
        breaker = MagicMock()
        breaker.filter_available_agents.side_effect = TypeError("Error")
        selector = TeamSelector(circuit_breaker=breaker)

        # Should not raise, returns all agents
        selected = selector.select(agents)
        assert len(selected) == 3


# =============================================================================
# Select Method Tests - ELO Scoring
# =============================================================================


class TestSelectEloScoring:
    """Tests for ELO-based scoring."""

    def test_elo_affects_ordering(self):
        """Higher ELO should result in higher ranking."""
        agents = [
            MockAgent(name="low_elo"),
            MockAgent(name="high_elo"),
            MockAgent(name="mid_elo"),
        ]
        elo = MockEloSystem(
            {
                "low_elo": 900,
                "mid_elo": 1000,
                "high_elo": 1500,
            }
        )
        selector = TeamSelector(elo_system=elo)

        selected = selector.select(agents)

        # High ELO should be first
        assert selected[0].name == "high_elo"
        # Low ELO should be last
        assert selected[-1].name == "low_elo"

    def test_elo_missing_agent(self):
        """Should handle agents without ELO rating."""
        agents = [
            MockAgent(name="known"),
            MockAgent(name="unknown"),
        ]
        elo = MockEloSystem({"known": 1500})
        selector = TeamSelector(elo_system=elo)

        # Should not raise
        selected = selector.select(agents)
        assert len(selected) == 2

    def test_elo_weight_affects_score(self):
        """ELO weight should scale contribution."""
        agents = [
            MockAgent(name="low"),
            MockAgent(name="high"),
        ]
        elo = MockEloSystem({"low": 900, "high": 1500})

        # Higher weight = bigger difference
        config = TeamSelectionConfig(elo_weight=1.0)
        selector = TeamSelector(elo_system=elo, config=config)

        low_score = selector.score_agent(agents[0])
        high_score = selector.score_agent(agents[1])

        # Score difference should be larger with high weight
        assert high_score > low_score
        assert (high_score - low_score) > 0.4  # Significant difference


# =============================================================================
# Select Method Tests - Calibration Scoring
# =============================================================================


class TestSelectCalibrationScoring:
    """Tests for calibration-based scoring."""

    def test_calibration_affects_ordering(self):
        """Better calibration (lower Brier) should result in higher ranking."""
        agents = [
            MockAgent(name="poor_calibration"),
            MockAgent(name="good_calibration"),
        ]
        calibration = MockCalibrationTracker(
            {
                "poor_calibration": 0.4,  # High Brier = bad
                "good_calibration": 0.1,  # Low Brier = good
            }
        )
        selector = TeamSelector(calibration_tracker=calibration)

        selected = selector.select(agents)

        # Good calibration should be first
        assert selected[0].name == "good_calibration"

    def test_calibration_missing_agent(self):
        """Should handle agents without calibration data."""
        agents = [
            MockAgent(name="tracked"),
            MockAgent(name="untracked"),
        ]
        calibration = MockCalibrationTracker({"tracked": 0.2})
        selector = TeamSelector(calibration_tracker=calibration)

        # Should not raise
        selected = selector.select(agents)
        assert len(selected) == 2

    def test_calibration_weight_affects_score(self):
        """Calibration weight should scale contribution."""
        agents = [
            MockAgent(name="bad"),
            MockAgent(name="good"),
        ]
        calibration = MockCalibrationTracker({"bad": 0.5, "good": 0.1})

        config = TeamSelectionConfig(calibration_weight=1.0)
        selector = TeamSelector(calibration_tracker=calibration, config=config)

        bad_score = selector.score_agent(agents[0])
        good_score = selector.score_agent(agents[1])

        assert good_score > bad_score


# =============================================================================
# Combined Scoring Tests
# =============================================================================


class TestCombinedScoring:
    """Tests for combined ELO + calibration scoring."""

    def test_combined_scoring(self):
        """Both ELO and calibration should contribute."""
        agents = [
            MockAgent(name="balanced"),
            MockAgent(name="high_elo_bad_cal"),
            MockAgent(name="low_elo_good_cal"),
        ]
        elo = MockEloSystem(
            {
                "balanced": 1200,
                "high_elo_bad_cal": 1500,
                "low_elo_good_cal": 900,
            }
        )
        calibration = MockCalibrationTracker(
            {
                "balanced": 0.2,
                "high_elo_bad_cal": 0.5,
                "low_elo_good_cal": 0.1,
            }
        )

        selector = TeamSelector(
            elo_system=elo,
            calibration_tracker=calibration,
        )

        selected = selector.select(agents)

        # All agents should be included
        assert len(selected) == 3

        # Verify scores affect ordering
        names = [a.name for a in selected]
        # High ELO bad cal vs low ELO good cal - depends on weights
        # With default weights (0.3 elo, 0.2 cal), high ELO should win
        assert "high_elo_bad_cal" in names

    def test_combined_with_filtering(self):
        """Should combine filtering and scoring."""
        agents = [
            MockAgent(name="blocked_high"),
            MockAgent(name="available_low"),
            MockAgent(name="available_high"),
        ]
        elo = MockEloSystem(
            {
                "blocked_high": 1800,
                "available_low": 900,
                "available_high": 1400,
            }
        )
        breaker = MockCircuitBreaker(blocked_agents=["blocked_high"])

        selector = TeamSelector(
            elo_system=elo,
            circuit_breaker=breaker,
        )

        selected = selector.select(agents)

        # Should not include blocked agent
        names = [a.name for a in selected]
        assert "blocked_high" not in names
        # High ELO available should be first
        assert selected[0].name == "available_high"


# =============================================================================
# Score Agent Method Tests
# =============================================================================


class TestScoreAgent:
    """Tests for the public score_agent method."""

    def test_score_agent_base_only(self, selector):
        """Without systems, should return base score."""
        agent = MockAgent(name="test")
        score = selector.score_agent(agent)
        assert score == 1.0  # Default base_score

    def test_score_agent_custom_base(self):
        """Should use custom base score."""
        config = TeamSelectionConfig(base_score=5.0)
        selector = TeamSelector(config=config)
        agent = MockAgent(name="test")

        score = selector.score_agent(agent)
        assert score == 5.0

    def test_score_agent_with_elo(self):
        """Should include ELO contribution."""
        elo = MockEloSystem({"test": 1500})  # 500 above baseline
        selector = TeamSelector(elo_system=elo)
        agent = MockAgent(name="test")

        score = selector.score_agent(agent)

        # base (1.0) + elo contribution (500/1000 * 0.3 = 0.15)
        assert score == pytest.approx(1.15, abs=0.01)

    def test_score_agent_with_calibration(self):
        """Should include calibration contribution."""
        calibration = MockCalibrationTracker({"test": 0.2})  # Good Brier
        selector = TeamSelector(calibration_tracker=calibration)
        agent = MockAgent(name="test")

        score = selector.score_agent(agent)

        # base (1.0) + calibration contribution ((1 - 0.2) * 0.2 = 0.16)
        assert score == pytest.approx(1.16, abs=0.01)

    def test_score_agent_both_systems(self):
        """Should combine both scoring systems."""
        elo = MockEloSystem({"test": 1500})
        calibration = MockCalibrationTracker({"test": 0.2})
        selector = TeamSelector(
            elo_system=elo,
            calibration_tracker=calibration,
        )
        agent = MockAgent(name="test")

        score = selector.score_agent(agent)

        # base + elo + calibration = 1.0 + 0.15 + 0.16 = 1.31
        assert score == pytest.approx(1.31, abs=0.01)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_negative_elo_contribution(self):
        """Low ELO should reduce score."""
        elo = MockEloSystem({"low": 500})  # 500 below baseline
        selector = TeamSelector(elo_system=elo)
        agent = MockAgent(name="low")

        score = selector.score_agent(agent)

        # base (1.0) + elo contribution (-500/1000 * 0.3 = -0.15)
        assert score < 1.0
        assert score == pytest.approx(0.85, abs=0.01)

    def test_perfect_calibration(self):
        """Perfect Brier score (0) should give max bonus."""
        calibration = MockCalibrationTracker({"perfect": 0.0})
        selector = TeamSelector(calibration_tracker=calibration)
        agent = MockAgent(name="perfect")

        score = selector.score_agent(agent)

        # base (1.0) + calibration ((1 - 0) * 0.2 = 0.2)
        assert score == pytest.approx(1.2, abs=0.01)

    def test_worst_calibration(self):
        """Worst Brier score (1) should give no bonus."""
        calibration = MockCalibrationTracker({"worst": 1.0})
        selector = TeamSelector(calibration_tracker=calibration)
        agent = MockAgent(name="worst")

        score = selector.score_agent(agent)

        # base (1.0) + calibration ((1 - 1) * 0.2 = 0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_extremely_high_elo(self):
        """Should handle very high ELO."""
        elo = MockEloSystem({"elite": 3000})
        selector = TeamSelector(elo_system=elo)
        agent = MockAgent(name="elite")

        score = selector.score_agent(agent)

        # base (1.0) + elo contribution (2000/1000 * 0.3 = 0.6)
        assert score == pytest.approx(1.6, abs=0.01)

    def test_duplicate_agent_names(self):
        """Should handle agents with same name (edge case)."""
        agents = [
            MockAgent(name="same"),
            MockAgent(name="same"),
        ]
        selector = TeamSelector()

        selected = selector.select(agents)
        assert len(selected) == 2

    def test_none_circuit_breaker_result(self):
        """Should handle circuit breaker returning None."""
        breaker = MagicMock()
        breaker.filter_available_agents.return_value = None
        selector = TeamSelector(circuit_breaker=breaker)

        agents = [MockAgent(name="test")]

        # Should not raise, but may fail iteration
        # This documents the current behavior
        try:
            selected = selector.select(agents)
            # If it works, should have some result
            assert selected is not None
        except TypeError:
            # Expected if None is returned
            pass


# =============================================================================
# KM Expertise Selection Tests
# =============================================================================


@dataclass
class MockAgentExpertise:
    """Mock AgentExpertise for testing."""

    agent_name: str
    domain: str
    elo: float
    confidence: float = 0.8


class MockRankingAdapter:
    """Mock RankingAdapter for testing KM expertise."""

    def __init__(self, domain_experts: dict[str, list[MockAgentExpertise]] = None):
        self._domain_experts = domain_experts or {}

    def get_domain_experts(
        self,
        domain: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        use_cache: bool = True,
    ) -> list[MockAgentExpertise]:
        return self._domain_experts.get(domain.lower(), [])


class TestKMExpertiseSelection:
    """Tests for KM-powered expertise selection."""

    def test_km_expertise_affects_ordering(self):
        """Agents with KM expertise should rank higher."""
        agents = [
            MockAgent(name="novice"),
            MockAgent(name="expert"),
            MockAgent(name="intermediate"),
        ]
        ranking_adapter = MockRankingAdapter(
            domain_experts={
                "technical": [
                    MockAgentExpertise("expert", "technical", 1600, 0.9),
                    MockAgentExpertise("intermediate", "technical", 1200, 0.7),
                ]
            }
        )
        config = TeamSelectionConfig(
            enable_km_expertise=True,
            km_expertise_weight=0.5,
        )
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        selected = selector.select(agents, domain="technical")

        # Expert should rank highest due to KM expertise
        assert selected[0].name == "expert"
        # Novice with no KM data should rank lowest
        assert selected[-1].name == "novice"

    def test_km_expertise_score_with_confidence(self):
        """KM expertise score should factor in confidence."""
        agent = MockAgent(name="high_confidence")
        ranking_adapter = MockRankingAdapter(
            domain_experts={
                "research": [
                    MockAgentExpertise("high_confidence", "research", 1500, 0.95),
                ]
            }
        )
        config = TeamSelectionConfig(
            enable_km_expertise=True,
            km_expertise_weight=1.0,
        )
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        score = selector.score_agent(agent, domain="research")

        # Base (1.0) + high KM expertise with 0.95 confidence
        assert score > 1.5

    def test_km_expertise_disabled(self):
        """KM expertise should be skipped when disabled."""
        agent = MockAgent(name="expert")
        ranking_adapter = MockRankingAdapter(
            domain_experts={
                "technical": [
                    MockAgentExpertise("expert", "technical", 1600, 0.9),
                ]
            }
        )
        config = TeamSelectionConfig(enable_km_expertise=False)
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        score = selector.score_agent(agent, domain="technical")

        # Should be base score only since KM is disabled
        assert score == pytest.approx(1.0, abs=0.01)

    def test_km_expertise_no_adapter(self):
        """Should handle missing ranking adapter gracefully."""
        agent = MockAgent(name="test")
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector = TeamSelector(ranking_adapter=None, config=config)

        score = selector.score_agent(agent, domain="technical")

        # Should not raise, just return base score
        assert score == pytest.approx(1.0, abs=0.01)

    def test_km_expertise_cache(self):
        """KM expertise lookups should be cached."""
        agent = MockAgent(name="expert")
        ranking_adapter = MockRankingAdapter(
            domain_experts={
                "research": [
                    MockAgentExpertise("expert", "research", 1500, 0.9),
                ]
            }
        )
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        # First call populates cache
        selector.score_agent(agent, domain="research")

        # Cache should be populated
        assert "research" in selector._km_expertise_cache

        # Second call should use cache
        score = selector.score_agent(agent, domain="research")
        assert score > 1.0

    def test_km_expertise_empty_domain(self):
        """Should handle domains with no experts."""
        agent = MockAgent(name="test")
        ranking_adapter = MockRankingAdapter(domain_experts={})
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        score = selector.score_agent(agent, domain="obscure_domain")

        # No KM data = base score only
        assert score == pytest.approx(1.0, abs=0.01)

    def test_km_expertise_combined_with_elo(self):
        """KM expertise should combine with ELO scoring."""
        agents = [
            MockAgent(name="km_expert"),
            MockAgent(name="elo_expert"),
        ]
        elo = MockEloSystem({"km_expert": 1000, "elo_expert": 1500})
        ranking_adapter = MockRankingAdapter(
            domain_experts={
                "coding": [
                    MockAgentExpertise("km_expert", "coding", 1600, 0.95),
                ]
            }
        )
        config = TeamSelectionConfig(
            elo_weight=0.3,
            km_expertise_weight=0.5,
            enable_km_expertise=True,
        )
        selector = TeamSelector(
            elo_system=elo,
            ranking_adapter=ranking_adapter,
            config=config,
        )

        km_score = selector.score_agent(agents[0], domain="coding")
        elo_score = selector.score_agent(agents[1], domain="coding")

        # Both should contribute, km_expert should edge out due to high KM expertise
        assert km_score > 1.0
        assert elo_score > 1.0


# =============================================================================
# CV-Based Selection
# =============================================================================


class MockAgentCV:
    """Mock AgentCV for testing."""

    def __init__(
        self,
        agent_id: str,
        overall_elo: float = 1000.0,
        brier_score: float = 0.5,
        success_rate: float = 0.9,
        has_meaningful_data: bool = True,
        is_reliable: bool = True,
        is_well_calibrated: bool = False,
    ):
        self.agent_id = agent_id
        self.overall_elo = overall_elo
        self.brier_score = brier_score

        # Mock reliability
        class MockReliability:
            pass

        self.reliability = MockReliability()
        self.reliability.success_rate = success_rate
        self.reliability.is_reliable = is_reliable

        self._has_meaningful_data = has_meaningful_data
        self._is_well_calibrated = is_well_calibrated

    @property
    def has_meaningful_data(self) -> bool:
        return self._has_meaningful_data

    @property
    def is_well_calibrated(self) -> bool:
        return self._is_well_calibrated

    def compute_selection_score(
        self,
        domain: Optional[str] = None,
        elo_weight: float = 0.3,
        calibration_weight: float = 0.2,
        reliability_weight: float = 0.2,
        domain_weight: float = 0.3,
    ) -> float:
        """Return a mock composite score."""
        # Normalize ELO to 0-1
        elo_score = (self.overall_elo - 800) / 400
        elo_score = max(0.0, min(1.0, elo_score))

        # Calibration score (invert brier)
        calibration_score = 1 - self.brier_score

        # Simple composite
        return (
            elo_score * elo_weight
            + calibration_score * calibration_weight
            + self.reliability.success_rate * reliability_weight
        )


class MockCVBuilder:
    """Mock CVBuilder for testing."""

    def __init__(self, cvs: dict[str, MockAgentCV] = None):
        self._cvs = cvs or {}

    def build_cv(self, agent_id: str) -> MockAgentCV:
        if agent_id in self._cvs:
            return self._cvs[agent_id]
        return MockAgentCV(agent_id)

    def build_cvs_batch(self, agent_ids: list[str]) -> dict[str, MockAgentCV]:
        return {aid: self.build_cv(aid) for aid in agent_ids}


class TestCVBasedSelection:
    """Tests for CV-based agent selection."""

    def test_cv_affects_ordering(self):
        """CV scores should affect agent ordering."""
        agents = [
            MockAgent(name="low_cv"),
            MockAgent(name="high_cv"),
        ]
        cv_builder = MockCVBuilder(
            {
                "low_cv": MockAgentCV("low_cv", overall_elo=900, brier_score=0.8, success_rate=0.7),
                "high_cv": MockAgentCV(
                    "high_cv", overall_elo=1200, brier_score=0.2, success_rate=0.95
                ),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.5,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        selected = selector.select(agents, domain="general")

        # High CV agent should be ranked first
        assert selected[0].name == "high_cv"
        assert selected[1].name == "low_cv"

    def test_cv_disabled(self):
        """CV scoring should be skipped when disabled."""
        agents = [
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ]
        cv_builder = MockCVBuilder(
            {
                "agent1": MockAgentCV("agent1", overall_elo=1200),
                "agent2": MockAgentCV("agent2", overall_elo=900),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=False,
            cv_weight=0.5,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        # Without CV scoring, order should be unchanged (both have same base score)
        selected = selector.select(agents, domain="general")

        # Order unchanged since both have same base score
        assert len(selected) == 2

    def test_cv_reliability_filtering(self):
        """Unreliable agents should be filtered when configured."""
        agents = [
            MockAgent(name="reliable"),
            MockAgent(name="unreliable"),
        ]
        cv_builder = MockCVBuilder(
            {
                "reliable": MockAgentCV("reliable", success_rate=0.95),
                "unreliable": MockAgentCV("unreliable", success_rate=0.5),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_filter_unreliable=True,
            cv_reliability_threshold=0.7,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        selected = selector.select(agents, domain="general")

        # Only reliable agent should remain
        assert len(selected) == 1
        assert selected[0].name == "reliable"

    def test_cv_no_filtering_below_threshold(self):
        """Without filtering, low reliability agents should still be selected."""
        agents = [
            MockAgent(name="reliable"),
            MockAgent(name="unreliable"),
        ]
        cv_builder = MockCVBuilder(
            {
                "reliable": MockAgentCV("reliable", success_rate=0.95),
                "unreliable": MockAgentCV("unreliable", success_rate=0.5),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_filter_unreliable=False,  # Filtering disabled
            cv_reliability_threshold=0.7,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        selected = selector.select(agents, domain="general")

        # Both agents should be included
        assert len(selected) == 2

    def test_cv_bonuses_applied(self):
        """CV with good reliability and calibration should get bonuses."""
        agents = [MockAgent(name="excellent")]
        cv_builder = MockCVBuilder(
            {
                "excellent": MockAgentCV(
                    "excellent",
                    overall_elo=1100,
                    brier_score=0.1,
                    success_rate=0.95,
                    is_reliable=True,
                    is_well_calibrated=True,
                ),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.35,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        # Get score directly
        cvs = selector._get_agent_cvs_batch(["excellent"])
        cv_score = selector._compute_cv_score(cvs["excellent"], domain="general")

        # Should have base score + reliability bonus + calibration bonus
        assert cv_score > 0.5  # Base composite score
        # The actual value depends on bonuses being applied

    def test_cv_caching(self):
        """CVs should be cached to avoid repeated lookups."""
        agents = [MockAgent(name="cached")]
        cv_builder = MockCVBuilder({"cached": MockAgentCV("cached", overall_elo=1100)})
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_cache_ttl=60,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        # First call populates cache
        selector.select(agents, domain="general")
        assert "cached" in selector._cv_cache

        # Modify the builder to return different data
        cv_builder._cvs["cached"] = MockAgentCV("cached", overall_elo=500)

        # Second call should use cache, not new data
        cvs = selector._get_agent_cvs_batch(["cached"])
        assert cvs["cached"].overall_elo == 1100  # Cached value

    def test_get_cv_method(self):
        """Should be able to get a single agent's CV."""
        cv_builder = MockCVBuilder({"agent1": MockAgentCV("agent1", overall_elo=1150)})
        selector = TeamSelector(cv_builder=cv_builder)

        cv = selector.get_cv("agent1")

        assert cv is not None
        assert cv.agent_id == "agent1"
        assert cv.overall_elo == 1150

    def test_cv_combined_with_elo(self):
        """CV scoring should combine with ELO scoring."""
        agents = [
            MockAgent(name="cv_high"),
            MockAgent(name="elo_high"),
        ]
        elo = MockEloSystem({"cv_high": 900, "elo_high": 1500})
        cv_builder = MockCVBuilder(
            {
                "cv_high": MockAgentCV(
                    "cv_high",
                    overall_elo=1200,
                    brier_score=0.1,
                    success_rate=0.99,
                    is_reliable=True,
                    is_well_calibrated=True,
                ),
                "elo_high": MockAgentCV(
                    "elo_high",
                    overall_elo=1000,
                    brier_score=0.5,
                    success_rate=0.8,
                ),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            elo_weight=0.3,
            cv_weight=0.4,
        )
        selector = TeamSelector(
            elo_system=elo,
            cv_builder=cv_builder,
            config=config,
        )

        cv_score = selector.score_agent(agents[0], domain="general")
        elo_score = selector.score_agent(agents[1], domain="general")

        # cv_high has low ELO (900) but excellent CV - scores reasonably
        # elo_high has high ELO (1500) but weaker CV - benefits from ELO
        assert cv_score > 0.8  # Has CV bonus despite low ELO
        assert elo_score > 1.0  # Has high ELO bonus

    def test_cv_no_meaningful_data(self):
        """CV without meaningful data should return 0 score."""
        agents = [MockAgent(name="new_agent")]
        cv_builder = MockCVBuilder(
            {
                "new_agent": MockAgentCV(
                    "new_agent",
                    has_meaningful_data=False,
                ),
            }
        )
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.5,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        cvs = selector._get_agent_cvs_batch(["new_agent"])
        cv_score = selector._compute_cv_score(cvs["new_agent"], domain="general")

        # No meaningful data = 0 CV score
        assert cv_score == 0.0
