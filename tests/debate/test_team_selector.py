"""
Tests for TeamSelector - agent selection and scoring for debate participation.

Tests cover:
- TeamSelector initialization and configuration
- Agent scoring based on ELO, calibration, delegation, domain
- Domain capability filtering with DOMAIN_CAPABILITY_MAP
- Circuit breaker integration for agent availability
- Pattern-based selection and caching
- KM expertise integration
- CV-based agent selection
- Culture-based recommendations
- Edge cases and error handling
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.team_selector import (
    DOMAIN_CAPABILITY_MAP,
    AgentScorer,
    CalibrationScorer,
    TeamSelectionConfig,
    TeamSelector,
)


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        agent_type: str = "unknown",
        model: str = "",
        capabilities: set | None = None,
        hierarchy_role: str | None = None,
        metadata: dict | None = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.model = model
        self.capabilities = capabilities or set()
        self.hierarchy_role = hierarchy_role
        self.metadata = metadata or {}


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self, ratings: dict[str, float] | None = None):
        self._ratings = ratings or {}

    def get_rating(self, agent_name: str) -> float:
        if agent_name not in self._ratings:
            raise KeyError(f"No rating for {agent_name}")
        return self._ratings.get(agent_name, 1000.0)


class MockEloSystemWithRatingObject:
    """Mock ELO system that returns rating objects."""

    def __init__(self, ratings: dict[str, float] | None = None):
        self._ratings = ratings or {}

    def get_rating(self, agent_name: str):
        """Return an object with .elo attribute."""
        rating = MagicMock()
        rating.elo = self._ratings.get(agent_name, 1000.0)
        return rating


class MockCalibrationTracker:
    """Mock calibration tracker for testing."""

    def __init__(self, scores: dict[str, float] | None = None):
        self._scores = scores or {}

    def get_brier_score(self, agent_name: str, domain: str | None = None) -> float:
        return self._scores.get(agent_name, 0.5)

    def get_brier_scores_batch(
        self, agent_names: list[str], domain: str | None = None
    ) -> dict[str, float]:
        return {name: self._scores.get(name, 0.5) for name in agent_names}


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, unavailable: set[str] | None = None):
        self._unavailable = unavailable or set()

    def filter_available_agents(self, agent_names: list[str]) -> list[str]:
        return [name for name in agent_names if name not in self._unavailable]


class MockDelegationStrategy:
    """Mock delegation strategy for testing."""

    def __init__(self, scores: dict[str, float] | None = None):
        self._scores = scores or {}

    def score_agent(self, agent, task: str, context=None) -> float:
        return self._scores.get(agent.name, 2.5)


class MockPatternMatcher:
    """Mock task pattern matcher for testing."""

    def __init__(
        self,
        pattern: str = "general",
        affinities: dict[str, float] | None = None,
    ):
        self._pattern = pattern
        self._affinities = affinities or {}

    def classify_task(self, task: str) -> str:
        return self._pattern

    def get_agent_affinities(self, pattern: str, critique_store=None) -> dict[str, float]:
        return self._affinities


class MockRankingAdapter:
    """Mock ranking adapter for KM expertise testing."""

    def __init__(self, experts: list | None = None):
        self._experts = experts or []

    def get_domain_experts(
        self,
        domain: str,
        limit: int = 20,
        min_confidence: float = 0.3,
        use_cache: bool = True,
    ) -> list:
        return self._experts


@dataclass
class MockAgentExpertise:
    """Mock expertise data from KM."""

    agent_name: str
    domain: str
    elo: float = 1500.0
    confidence: float = 0.8


@dataclass
class MockAgentCV:
    """Mock Agent CV for testing."""

    agent_id: str
    has_meaningful_data: bool = True
    is_well_calibrated: bool = True
    reliability: MagicMock = field(
        default_factory=lambda: MagicMock(success_rate=0.9, is_reliable=True)
    )

    def compute_selection_score(
        self,
        domain: str | None = None,
        elo_weight: float = 0.25,
        calibration_weight: float = 0.25,
        reliability_weight: float = 0.30,
        domain_weight: float = 0.20,
    ) -> float:
        return 0.75


class MockCVBuilder:
    """Mock CV builder for testing."""

    def __init__(self, cvs: dict[str, MockAgentCV] | None = None):
        self._cvs = cvs or {}

    def build_cv(self, agent_name: str) -> MockAgentCV:
        return self._cvs.get(agent_name, MockAgentCV(agent_id=agent_name))

    def build_cvs_batch(self, agent_names: list[str]) -> dict[str, MockAgentCV]:
        return {name: self.build_cv(name) for name in agent_names}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_agents():
    """Create list of mock agents for testing."""
    return [
        MockAgent("claude-opus", agent_type="claude", model="claude-3-opus"),
        MockAgent("gpt-4", agent_type="gpt", model="gpt-4-turbo"),
        MockAgent("gemini-pro", agent_type="gemini", model="gemini-1.5-pro"),
        MockAgent("codestral", agent_type="codestral", model="codestral"),
        MockAgent("deepseek-r1", agent_type="deepseek", model="deepseek-r1"),
    ]


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system with ratings."""
    return MockEloSystem(
        ratings={
            "claude-opus": 1800.0,
            "gpt-4": 1750.0,
            "gemini-pro": 1650.0,
            "codestral": 1600.0,
            "deepseek-r1": 1700.0,
        }
    )


@pytest.fixture
def mock_calibration_tracker():
    """Create mock calibration tracker."""
    return MockCalibrationTracker(
        scores={
            "claude-opus": 0.15,  # Well calibrated (low Brier = good)
            "gpt-4": 0.20,
            "gemini-pro": 0.25,
            "codestral": 0.30,
            "deepseek-r1": 0.18,
        }
    )


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker."""
    return MockCircuitBreaker()


# ===========================================================================
# Test: TeamSelectionConfig
# ===========================================================================


class TestTeamSelectionConfig:
    """Tests for TeamSelectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TeamSelectionConfig()

        assert config.elo_weight == 0.3
        assert config.calibration_weight == 0.2
        assert config.delegation_weight == 0.2
        assert config.domain_capability_weight == 0.25
        assert config.culture_weight == 0.15
        assert config.km_expertise_weight == 0.25
        assert config.pattern_weight == 0.2
        assert config.base_score == 1.0
        assert config.elo_baseline == 1000
        assert config.enable_domain_filtering is True
        assert config.domain_filter_fallback is True
        assert config.enable_culture_selection is False
        assert config.enable_km_expertise is True
        assert config.enable_pattern_selection is True
        assert config.km_expertise_cache_ttl == 300
        assert config.custom_domain_map == {}

    def test_custom_config(self):
        """Test custom configuration."""
        config = TeamSelectionConfig(
            elo_weight=0.5,
            calibration_weight=0.3,
            enable_domain_filtering=False,
            custom_domain_map={"custom": ["agent1", "agent2"]},
        )

        assert config.elo_weight == 0.5
        assert config.calibration_weight == 0.3
        assert config.enable_domain_filtering is False
        assert config.custom_domain_map == {"custom": ["agent1", "agent2"]}

    def test_cv_selection_config(self):
        """Test CV selection configuration."""
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_weight=0.4,
            cv_reliability_threshold=0.8,
            cv_filter_unreliable=True,
            cv_cache_ttl=120,
        )

        assert config.enable_cv_selection is True
        assert config.cv_weight == 0.4
        assert config.cv_reliability_threshold == 0.8
        assert config.cv_filter_unreliable is True
        assert config.cv_cache_ttl == 120

    def test_hierarchy_config(self):
        """Test hierarchy filtering configuration."""
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            hierarchy_filter_fallback=False,
        )

        assert config.enable_hierarchy_filtering is True
        assert config.hierarchy_filter_fallback is False


# ===========================================================================
# Test: DOMAIN_CAPABILITY_MAP
# ===========================================================================


class TestDomainCapabilityMap:
    """Tests for DOMAIN_CAPABILITY_MAP constant."""

    def test_code_domain_includes_specialists(self):
        """Test code domain includes coding specialists."""
        code_patterns = DOMAIN_CAPABILITY_MAP.get("code", [])

        assert "claude" in code_patterns
        assert "codex" in code_patterns
        assert "codestral" in code_patterns
        assert "deepseek" in code_patterns
        assert "gpt" in code_patterns

    def test_research_domain(self):
        """Test research domain includes research models."""
        research_patterns = DOMAIN_CAPABILITY_MAP.get("research", [])

        assert "claude" in research_patterns
        assert "gemini" in research_patterns
        assert "gpt" in research_patterns
        assert "deepseek-r1" in research_patterns

    def test_creative_domain(self):
        """Test creative domain includes creative models."""
        creative_patterns = DOMAIN_CAPABILITY_MAP.get("creative", [])

        assert "claude" in creative_patterns
        assert "gpt" in creative_patterns
        assert "gemini" in creative_patterns
        assert "llama" in creative_patterns

    def test_reasoning_domain(self):
        """Test reasoning domain includes reasoning models."""
        reasoning_patterns = DOMAIN_CAPABILITY_MAP.get("reasoning", [])

        assert "claude" in reasoning_patterns
        assert "deepseek-r1" in reasoning_patterns
        assert "gpt" in reasoning_patterns

    def test_general_domain_is_empty(self):
        """Test general domain has no filtering patterns."""
        general_patterns = DOMAIN_CAPABILITY_MAP.get("general", [])

        assert general_patterns == []


# ===========================================================================
# Test: TeamSelector Initialization
# ===========================================================================


class TestTeamSelectorInitialization:
    """Tests for TeamSelector initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        selector = TeamSelector()

        assert selector.elo_system is None
        assert selector.calibration_tracker is None
        assert selector.circuit_breaker is None
        assert selector.delegation_strategy is None
        assert selector.config is not None

    def test_init_with_elo_system(self, mock_elo_system):
        """Test initialization with ELO system."""
        selector = TeamSelector(elo_system=mock_elo_system)

        assert selector.elo_system is mock_elo_system

    def test_init_with_calibration_tracker(self, mock_calibration_tracker):
        """Test initialization with calibration tracker."""
        selector = TeamSelector(calibration_tracker=mock_calibration_tracker)

        assert selector.calibration_tracker is mock_calibration_tracker

    def test_init_with_circuit_breaker(self, mock_circuit_breaker):
        """Test initialization with circuit breaker."""
        selector = TeamSelector(circuit_breaker=mock_circuit_breaker)

        assert selector.circuit_breaker is mock_circuit_breaker

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = TeamSelectionConfig(elo_weight=0.5)
        selector = TeamSelector(config=config)

        assert selector.config.elo_weight == 0.5

    def test_init_with_all_components(
        self,
        mock_elo_system,
        mock_calibration_tracker,
        mock_circuit_breaker,
    ):
        """Test initialization with all components."""
        delegation = MockDelegationStrategy()
        pattern_matcher = MockPatternMatcher()

        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            circuit_breaker=mock_circuit_breaker,
            delegation_strategy=delegation,
            pattern_matcher=pattern_matcher,
        )

        assert selector.elo_system is mock_elo_system
        assert selector.calibration_tracker is mock_calibration_tracker
        assert selector.circuit_breaker is mock_circuit_breaker
        assert selector.delegation_strategy is delegation
        assert selector.pattern_matcher is pattern_matcher


# ===========================================================================
# Test: Agent Selection
# ===========================================================================


class TestAgentSelection:
    """Tests for agent selection logic."""

    def test_select_returns_all_agents_with_no_filtering(self, mock_agents):
        """Test selection returns all agents when no filtering configured."""
        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(config=config)

        selected = selector.select(agents=mock_agents, domain="code")

        assert len(selected) == len(mock_agents)

    def test_select_respects_domain_filtering(self, mock_agents):
        """Test selection filters by domain capability."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        selected = selector.select(agents=mock_agents, domain="code")

        # Should include coding specialists
        selected_names = [a.name for a in selected]
        assert "claude-opus" in selected_names
        assert "codestral" in selected_names
        assert "deepseek-r1" in selected_names
        assert "gpt-4" in selected_names

    def test_select_orders_by_score(self, mock_agents, mock_elo_system):
        """Test selection orders agents by score."""
        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(elo_system=mock_elo_system, config=config)

        selected = selector.select(agents=mock_agents, domain="general")

        # First agent should have highest ELO (claude-opus at 1800)
        assert selected[0].name == "claude-opus"

    def test_select_with_circuit_breaker(self, mock_agents):
        """Test selection respects circuit breaker."""
        breaker = MockCircuitBreaker(unavailable={"gpt-4", "gemini-pro"})
        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(circuit_breaker=breaker, config=config)

        selected = selector.select(agents=mock_agents, domain="general")

        # gpt-4 and gemini-pro should be filtered out
        selected_names = [a.name for a in selected]
        assert "gpt-4" not in selected_names
        assert "gemini-pro" not in selected_names
        assert len(selected) == 3

    def test_select_fallback_when_all_filtered(self, mock_agents):
        """Test fallback to all agents when all are filtered."""
        # All agents unavailable via circuit breaker
        breaker = MockCircuitBreaker(
            unavailable={"claude-opus", "gpt-4", "gemini-pro", "codestral", "deepseek-r1"}
        )
        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(circuit_breaker=breaker, config=config)

        selected = selector.select(agents=mock_agents, domain="general")

        # Should fall back to returning all agents
        assert len(selected) == len(mock_agents)

    def test_select_with_empty_agents(self):
        """Test selection with empty agent list."""
        selector = TeamSelector()

        selected = selector.select(agents=[], domain="general")

        assert selected == []


# ===========================================================================
# Test: Agent Scoring
# ===========================================================================


class TestAgentScoring:
    """Tests for agent scoring logic."""

    def test_base_score_applied(self, mock_agents):
        """Test base score is applied to all agents."""
        config = TeamSelectionConfig(
            base_score=2.0,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(config=config)

        score = selector.score_agent(mock_agents[0])

        assert score == 2.0

    def test_elo_contribution(self, mock_agents, mock_elo_system):
        """Test ELO rating contributes to score."""
        config = TeamSelectionConfig(
            elo_weight=0.3,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(elo_system=mock_elo_system, config=config)

        # claude-opus has ELO 1800 (800 above baseline)
        score = selector.score_agent(mock_agents[0])

        # base (1.0) + (800/1000 * 0.3) = 1.0 + 0.24 = 1.24
        assert score == pytest.approx(1.24, rel=0.01)

    def test_elo_contribution_with_rating_object(self, mock_agents):
        """Test ELO contribution when system returns rating objects."""
        elo_system = MockEloSystemWithRatingObject(ratings={"claude-opus": 1800.0})
        config = TeamSelectionConfig(
            elo_weight=0.3,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(elo_system=elo_system, config=config)

        score = selector.score_agent(mock_agents[0])

        assert score == pytest.approx(1.24, rel=0.01)

    def test_calibration_contribution(self, mock_agents, mock_calibration_tracker):
        """Test calibration score contributes to score."""
        config = TeamSelectionConfig(
            calibration_weight=0.2,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(calibration_tracker=mock_calibration_tracker, config=config)

        # claude-opus has Brier score 0.15
        score = selector.score_agent(mock_agents[0], domain="general")

        # base (1.0) + ((1 - 0.15) * 0.2) = 1.0 + 0.17 = 1.17
        assert score == pytest.approx(1.17, rel=0.01)

    def test_delegation_contribution(self, mock_agents):
        """Test delegation strategy contributes to score."""
        delegation = MockDelegationStrategy(scores={"claude-opus": 4.0})
        config = TeamSelectionConfig(
            delegation_weight=0.2,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(delegation_strategy=delegation, config=config)

        score = selector.score_agent(mock_agents[0], task="Write code")

        # base (1.0) + (min(4.0/5.0, 1.0) * 0.2) = 1.0 + 0.16 = 1.16
        assert score == pytest.approx(1.16, rel=0.01)

    def test_domain_score_contribution(self, mock_agents):
        """Test domain capability contributes to score."""
        config = TeamSelectionConfig(
            domain_capability_weight=0.25,
            enable_domain_filtering=True,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(config=config)

        # claude is first in code domain patterns
        score = selector.score_agent(mock_agents[0], domain="code")

        # Should have domain bonus
        assert score > 1.0

    def test_combined_scoring(self, mock_agents, mock_elo_system, mock_calibration_tracker):
        """Test combined scoring from multiple sources."""
        delegation = MockDelegationStrategy(scores={"claude-opus": 4.0})
        config = TeamSelectionConfig(
            elo_weight=0.3,
            calibration_weight=0.2,
            delegation_weight=0.2,
            domain_capability_weight=0.25,
            enable_domain_filtering=True,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            delegation_strategy=delegation,
            config=config,
        )

        score = selector.score_agent(mock_agents[0], domain="code", task="Write code")

        # Should be higher than base score with all contributions
        assert score > 1.5

    def test_score_agent_handles_missing_elo(self, mock_agents):
        """Test scoring handles missing ELO gracefully."""
        elo_system = MockEloSystem(ratings={})  # No ratings
        config = TeamSelectionConfig(
            elo_weight=0.3,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(elo_system=elo_system, config=config)

        # Should not raise, just use base score
        score = selector.score_agent(mock_agents[0])

        assert score == 1.0


# ===========================================================================
# Test: Domain Capability Filtering
# ===========================================================================


class TestDomainCapabilityFiltering:
    """Tests for domain capability filtering."""

    def test_filter_code_domain(self, mock_agents):
        """Test filtering for code domain."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "code")

        # Should include claude, gpt, codestral, deepseek
        filtered_names = [a.name for a in filtered]
        assert "claude-opus" in filtered_names
        assert "gpt-4" in filtered_names
        assert "codestral" in filtered_names
        assert "deepseek-r1" in filtered_names

    def test_filter_creative_domain(self, mock_agents):
        """Test filtering for creative domain."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "creative")

        # Should include claude, gpt, gemini
        filtered_names = [a.name for a in filtered]
        assert "claude-opus" in filtered_names
        assert "gpt-4" in filtered_names
        assert "gemini-pro" in filtered_names

    def test_filter_general_domain_returns_all(self, mock_agents):
        """Test general domain returns all agents (no patterns)."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "general")

        assert len(filtered) == len(mock_agents)

    def test_filter_unknown_domain_returns_all(self, mock_agents):
        """Test unknown domain returns all agents."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "unknown_domain_xyz")

        assert len(filtered) == len(mock_agents)

    def test_filter_disabled_returns_all(self, mock_agents):
        """Test filtering disabled returns all agents."""
        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "code")

        assert len(filtered) == len(mock_agents)

    def test_filter_custom_domain_map(self, mock_agents):
        """Test filtering with custom domain map."""
        config = TeamSelectionConfig(
            enable_domain_filtering=True,
            custom_domain_map={"security": ["claude", "deepseek"]},
        )
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(mock_agents, "security")

        filtered_names = [a.name for a in filtered]
        assert "claude-opus" in filtered_names
        assert "deepseek-r1" in filtered_names
        # Others should not be included
        assert len(filtered) == 2

    def test_filter_fallback_disabled(self, mock_agents):
        """Test fallback disabled returns empty list when no match."""
        # Create agents that won't match any pattern
        agents = [MockAgent("unknown-agent", agent_type="unknown")]
        config = TeamSelectionConfig(
            enable_domain_filtering=True,
            domain_filter_fallback=False,
            custom_domain_map={"strict": ["specific-agent"]},
        )
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_domain_capability(agents, "strict")

        assert filtered == []

    def test_agent_matches_by_name(self):
        """Test agent matching by name."""
        agent = MockAgent("claude-3-opus", agent_type="api")
        selector = TeamSelector()

        matches = selector._agent_matches_capability(agent, ["claude"])

        assert matches is True

    def test_agent_matches_by_type(self):
        """Test agent matching by type."""
        agent = MockAgent("my-agent", agent_type="gpt")
        selector = TeamSelector()

        matches = selector._agent_matches_capability(agent, ["gpt"])

        assert matches is True

    def test_agent_matches_by_model(self):
        """Test agent matching by model."""
        agent = MockAgent("agent", agent_type="api", model="deepseek-coder")
        selector = TeamSelector()

        matches = selector._agent_matches_capability(agent, ["deepseek"])

        assert matches is True


# ===========================================================================
# Test: Domain Score Computation
# ===========================================================================


class TestDomainScoreComputation:
    """Tests for domain score computation."""

    def test_first_in_list_gets_highest_score(self, mock_agents):
        """Test first pattern match gets highest score."""
        selector = TeamSelector()

        # claude is first in code domain patterns
        score = selector._compute_domain_score(mock_agents[0], "code")

        assert score == 1.0

    def test_later_position_lower_score(self, mock_agents):
        """Test later position in pattern list gets lower score."""
        selector = TeamSelector()

        # gpt is not first in code patterns
        score = selector._compute_domain_score(mock_agents[1], "code")

        # Should be less than 1.0 (position-based reduction)
        assert 0.0 < score < 1.0

    def test_no_match_returns_zero(self):
        """Test no pattern match returns zero."""
        agent = MockAgent("unknown-agent", agent_type="unknown")
        selector = TeamSelector()

        score = selector._compute_domain_score(agent, "code")

        assert score == 0.0

    def test_general_domain_returns_zero(self, mock_agents):
        """Test general domain returns zero (no patterns)."""
        selector = TeamSelector()

        score = selector._compute_domain_score(mock_agents[0], "general")

        assert score == 0.0


# ===========================================================================
# Test: Pattern-Based Selection
# ===========================================================================


class TestPatternBasedSelection:
    """Tests for pattern-based selection."""

    def test_pattern_score_contribution(self, mock_agents):
        """Test pattern score contributes to agent score."""
        pattern_matcher = MockPatternMatcher(
            pattern="code_review",
            affinities={"claude": 0.9, "gpt": 0.7},
        )
        config = TeamSelectionConfig(
            pattern_weight=0.2,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(pattern_matcher=pattern_matcher, config=config)

        score = selector._compute_pattern_score(mock_agents[0], "Review this code")

        assert score == 0.9

    def test_pattern_cache(self, mock_agents):
        """Test pattern affinities are cached."""
        pattern_matcher = MockPatternMatcher(
            pattern="code_review",
            affinities={"claude": 0.9},
        )
        config = TeamSelectionConfig(enable_pattern_selection=True)
        selector = TeamSelector(pattern_matcher=pattern_matcher, config=config)

        # First call populates cache
        selector._compute_pattern_score(mock_agents[0], "Review code")

        # Cache should contain the pattern
        assert "code_review" in selector._pattern_affinities_cache

    def test_general_pattern_returns_zero(self, mock_agents):
        """Test general pattern returns zero score."""
        pattern_matcher = MockPatternMatcher(pattern="general")
        selector = TeamSelector(pattern_matcher=pattern_matcher)

        score = selector._compute_pattern_score(mock_agents[0], "General task")

        assert score == 0.0

    def test_no_pattern_matcher_returns_zero(self, mock_agents):
        """Test no pattern matcher returns zero."""
        selector = TeamSelector()

        score = selector._compute_pattern_score(mock_agents[0], "Some task")

        assert score == 0.0

    def test_empty_task_returns_zero(self, mock_agents):
        """Test empty task returns zero."""
        pattern_matcher = MockPatternMatcher()
        selector = TeamSelector(pattern_matcher=pattern_matcher)

        score = selector._compute_pattern_score(mock_agents[0], "")

        assert score == 0.0

    def test_pattern_telemetry(self, mock_agents):
        """Test pattern telemetry is tracked."""
        pattern_matcher = MockPatternMatcher(
            pattern="debugging",
            affinities={"claude": 0.8},
        )
        selector = TeamSelector(pattern_matcher=pattern_matcher)

        # Make multiple classifications
        for _ in range(5):
            selector._compute_pattern_score(mock_agents[0], "Debug this")

        telemetry = selector.get_pattern_telemetry()

        assert "classification_counts" in telemetry
        assert telemetry["classification_counts"].get("debugging", 0) >= 5


# ===========================================================================
# Test: KM Expertise
# ===========================================================================


class TestKMExpertise:
    """Tests for Knowledge Mound expertise integration."""

    def test_km_expertise_score(self, mock_agents):
        """Test KM expertise contributes to score."""
        experts = [
            MockAgentExpertise("claude-opus", "code", elo=1800, confidence=0.9),
            MockAgentExpertise("gpt-4", "code", elo=1700, confidence=0.8),
        ]
        ranking_adapter = MockRankingAdapter(experts=experts)
        config = TeamSelectionConfig(
            km_expertise_weight=0.25,
            enable_km_expertise=True,
        )
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        score = selector._compute_km_expertise_score(mock_agents[0], "code")

        # First expert should get high score
        assert score > 0.5

    def test_km_expertise_cache(self, mock_agents):
        """Test KM expertise is cached."""
        experts = [MockAgentExpertise("claude-opus", "code")]
        ranking_adapter = MockRankingAdapter(experts=experts)
        config = TeamSelectionConfig(
            enable_km_expertise=True,
            km_expertise_cache_ttl=300,
        )
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        # First call
        selector._get_km_domain_experts("code")

        # Should be cached
        assert "code" in selector._km_expertise_cache

    def test_km_expertise_disabled(self, mock_agents):
        """Test KM expertise returns zero when disabled."""
        ranking_adapter = MockRankingAdapter()
        config = TeamSelectionConfig(enable_km_expertise=False)
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        score = selector._compute_km_expertise_score(mock_agents[0], "code")

        assert score == 0.0

    def test_km_expertise_no_adapter(self, mock_agents):
        """Test KM expertise returns zero without adapter."""
        selector = TeamSelector()

        score = selector._compute_km_expertise_score(mock_agents[0], "code")

        assert score == 0.0


# ===========================================================================
# Test: ELO Domain Win Rate Scoring
# ===========================================================================


class TestEloWinRateScoring:
    """Tests for domain-specific win rate scoring from the ELO system."""

    def _make_elo_with_win_rates(self, domain_agents):
        """Create an ELO system mock with get_top_agents_for_domain support."""
        elo = MockEloSystem(ratings={"claude-opus": 1800, "gpt-4": 1700})

        def get_top(domain, limit=20):
            agents = domain_agents.get(domain, [])[:limit]
            results = []
            for name, wr in agents:
                r = MagicMock()
                r.agent_name = name
                r.win_rate = wr
                results.append(r)
            return results

        elo.get_top_agents_for_domain = get_top
        return elo

    def test_high_win_rate_boosts_score(self, mock_agents):
        """Agent with >50% win rate gets positive score."""
        elo = self._make_elo_with_win_rates({
            "security": [("claude-opus", 0.8), ("gpt-4", 0.6)],
        })
        config = TeamSelectionConfig(enable_elo_win_rate=True, elo_win_rate_weight=0.2)
        selector = TeamSelector(elo_system=elo, config=config)

        score = selector._compute_elo_win_rate_score(mock_agents[0], "security")

        assert score == pytest.approx(0.6, abs=0.01)

    def test_low_win_rate_penalizes_score(self, mock_agents):
        """Agent with <50% win rate gets negative score."""
        elo = self._make_elo_with_win_rates({
            "security": [("claude-opus", 0.2)],
        })
        config = TeamSelectionConfig(enable_elo_win_rate=True)
        selector = TeamSelector(elo_system=elo, config=config)

        score = selector._compute_elo_win_rate_score(mock_agents[0], "security")

        assert score == pytest.approx(-0.5, abs=0.01)

    def test_fifty_percent_win_rate_is_neutral(self, mock_agents):
        """Agent at exactly 50% win rate gets zero score."""
        elo = self._make_elo_with_win_rates({
            "code": [("claude-opus", 0.5)],
        })
        selector = TeamSelector(elo_system=elo)

        score = selector._compute_elo_win_rate_score(mock_agents[0], "code")

        assert score == pytest.approx(0.0, abs=0.01)

    def test_disabled_returns_zero(self, mock_agents):
        """Returns 0.0 when enable_elo_win_rate is False."""
        elo = self._make_elo_with_win_rates({"code": [("claude-opus", 0.9)]})
        config = TeamSelectionConfig(enable_elo_win_rate=False)
        selector = TeamSelector(elo_system=elo, config=config)

        assert selector._compute_elo_win_rate_score(mock_agents[0], "code") == 0.0

    def test_no_elo_system_returns_zero(self, mock_agents):
        """Returns 0.0 when no ELO system is provided."""
        selector = TeamSelector()

        assert selector._compute_elo_win_rate_score(mock_agents[0], "code") == 0.0

    def test_agent_not_in_domain_returns_zero(self, mock_agents):
        """Returns 0.0 when agent has no domain win rate data."""
        elo = self._make_elo_with_win_rates({
            "security": [("other-agent", 0.8)],
        })
        selector = TeamSelector(elo_system=elo)

        score = selector._compute_elo_win_rate_score(mock_agents[0], "security")

        assert score == 0.0

    def test_empty_domain_returns_zero(self, mock_agents):
        """Returns 0.0 when domain has no top agents."""
        elo = self._make_elo_with_win_rates({})
        selector = TeamSelector(elo_system=elo)

        assert selector._compute_elo_win_rate_score(mock_agents[0], "unknown") == 0.0

    def test_win_rate_contributes_to_composite_score(self, mock_agents):
        """Win rate score is included in composite _compute_score."""
        elo = self._make_elo_with_win_rates({
            "security": [("claude-opus", 0.9), ("gpt-4", 0.4)],
        })
        config = TeamSelectionConfig(
            elo_weight=0.0,
            calibration_weight=0.0,
            delegation_weight=0.0,
            domain_capability_weight=0.0,
            km_expertise_weight=0.0,
            pattern_weight=0.0,
            enable_cv_selection=False,
            enable_culture_selection=False,
            enable_elo_win_rate=True,
            elo_win_rate_weight=0.2,
        )
        selector = TeamSelector(elo_system=elo, config=config)

        score_claude = selector._compute_score(mock_agents[0], domain="security")
        score_gpt = selector._compute_score(mock_agents[1], domain="security")

        assert score_claude > score_gpt

    def test_elo_without_method_returns_zero(self, mock_agents):
        """Returns 0.0 when ELO system lacks get_top_agents_for_domain."""
        elo = MockEloSystem(ratings={"claude-opus": 1800})
        selector = TeamSelector(elo_system=elo)

        assert selector._compute_elo_win_rate_score(mock_agents[0], "code") == 0.0


# ===========================================================================
# Test: CV-Based Selection
# ===========================================================================


class TestCVBasedSelection:
    """Tests for Agent CV-based selection."""

    def test_cv_score_contribution(self, mock_agents):
        """Test CV score contributes to agent score."""
        cvs = {
            "claude-opus": MockAgentCV(
                agent_id="claude-opus",
                has_meaningful_data=True,
                is_well_calibrated=True,
            )
        }
        cv_builder = MockCVBuilder(cvs=cvs)
        config = TeamSelectionConfig(
            cv_weight=0.35,
            enable_cv_selection=True,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        # Need to get CVs first
        agent_cvs = selector._get_agent_cvs_batch(["claude-opus"])
        score = selector._compute_cv_score(agent_cvs["claude-opus"])

        # Should have meaningful score with bonuses
        assert score > 0.7

    def test_cv_cache(self, mock_agents):
        """Test CV data is cached."""
        cv_builder = MockCVBuilder()
        config = TeamSelectionConfig(enable_cv_selection=True, cv_cache_ttl=60)
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        # First call
        selector._get_agent_cvs_batch(["claude-opus"])

        # Should be cached
        assert "claude-opus" in selector._cv_cache

    def test_cv_filter_unreliable(self, mock_agents):
        """Test unreliable agents are filtered."""
        unreliable_cv = MockAgentCV(
            agent_id="unreliable-agent",
            has_meaningful_data=True,
        )
        unreliable_cv.reliability.success_rate = 0.5
        unreliable_cv.reliability.is_reliable = False

        cvs = {"unreliable-agent": unreliable_cv}
        cv_builder = MockCVBuilder(cvs=cvs)

        agents = [MockAgent("unreliable-agent")]
        config = TeamSelectionConfig(
            enable_cv_selection=True,
            cv_filter_unreliable=True,
            cv_reliability_threshold=0.7,
            enable_domain_filtering=False,
        )
        selector = TeamSelector(cv_builder=cv_builder, config=config)

        selected = selector.select(agents=agents, domain="general")

        # Should fall back to all agents since none pass filter
        assert len(selected) == 1

    def test_cv_no_meaningful_data(self):
        """Test CV without meaningful data returns zero."""
        cv = MockAgentCV(agent_id="new-agent", has_meaningful_data=False)
        selector = TeamSelector()

        score = selector._compute_cv_score(cv)

        assert score == 0.0

    def test_get_cv_single_agent(self, mock_agents):
        """Test getting CV for single agent."""
        cv_builder = MockCVBuilder()
        selector = TeamSelector(cv_builder=cv_builder)

        cv = selector.get_cv("claude-opus")

        assert cv is not None
        assert cv.agent_id == "claude-opus"


# ===========================================================================
# Test: Circuit Breaker Integration
# ===========================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_filter_available_all_available(self, mock_agents):
        """Test all agents available when no breaker."""
        selector = TeamSelector()

        available = selector._filter_available(mock_agents)

        assert len(available) == len(mock_agents)

    def test_filter_available_some_unavailable(self, mock_agents):
        """Test some agents filtered when unavailable."""
        breaker = MockCircuitBreaker(unavailable={"gpt-4", "gemini-pro"})
        selector = TeamSelector(circuit_breaker=breaker)

        available = selector._filter_available(mock_agents)

        assert "gpt-4" not in available
        assert "gemini-pro" not in available
        assert len(available) == 3

    def test_filter_available_breaker_error(self, mock_agents):
        """Test graceful handling of breaker errors."""
        breaker = MagicMock()
        breaker.filter_available_agents = MagicMock(side_effect=AttributeError("error"))
        selector = TeamSelector(circuit_breaker=breaker)

        available = selector._filter_available(mock_agents)

        # Should return all agents on error
        assert len(available) == len(mock_agents)


# ===========================================================================
# Test: Delegation Strategy
# ===========================================================================


class TestDelegationStrategy:
    """Tests for delegation strategy integration."""

    def test_set_delegation_strategy(self):
        """Test setting delegation strategy."""
        selector = TeamSelector()
        strategy = MockDelegationStrategy()

        selector.set_delegation_strategy(strategy)

        assert selector.delegation_strategy is strategy

    def test_delegation_score_applied(self, mock_agents):
        """Test delegation score is applied when task provided."""
        delegation = MockDelegationStrategy(scores={"claude-opus": 5.0})
        config = TeamSelectionConfig(
            delegation_weight=0.2,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(delegation_strategy=delegation, config=config)

        score = selector.score_agent(mock_agents[0], task="Complex task")

        # base (1.0) + (min(5.0/5.0, 1.0) * 0.2) = 1.0 + 0.2 = 1.2
        assert score == pytest.approx(1.2, rel=0.01)

    def test_delegation_score_not_applied_without_task(self, mock_agents):
        """Test delegation score not applied without task."""
        delegation = MockDelegationStrategy(scores={"claude-opus": 5.0})
        config = TeamSelectionConfig(
            delegation_weight=0.2,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(delegation_strategy=delegation, config=config)

        score = selector.score_agent(mock_agents[0], task="")

        # No task = no delegation bonus
        assert score == 1.0


# ===========================================================================
# Test: Hierarchy Role Support
# ===========================================================================


class TestHierarchyRoleSupport:
    """Tests for hierarchy role filtering support."""

    def test_get_hierarchy_role_from_attribute(self):
        """Test getting hierarchy role from agent attribute."""
        agent = MockAgent("agent", hierarchy_role="orchestrator")
        selector = TeamSelector()

        role = selector._get_agent_hierarchy_role(agent)

        assert role == "orchestrator"

    def test_get_hierarchy_role_from_metadata(self):
        """Test getting hierarchy role from agent metadata."""
        agent = MockAgent("agent", metadata={"hierarchy_role": "worker"})
        selector = TeamSelector()

        role = selector._get_agent_hierarchy_role(agent)

        assert role == "worker"

    def test_get_hierarchy_role_none(self):
        """Test getting hierarchy role when not set."""
        agent = MockAgent("agent")
        selector = TeamSelector()

        role = selector._get_agent_hierarchy_role(agent)

        assert role is None

    def test_filter_by_hierarchy_role(self, mock_agents):
        """Test filtering by hierarchy role."""
        # Set up agents with hierarchy roles
        mock_agents[0].hierarchy_role = "orchestrator"
        mock_agents[1].hierarchy_role = "worker"
        mock_agents[2].hierarchy_role = "worker"
        mock_agents[3].hierarchy_role = "monitor"
        mock_agents[4].hierarchy_role = "worker"

        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_hierarchy_role(mock_agents, required_roles={"worker"})

        # Should only include workers
        assert len(filtered) == 3
        for agent in filtered:
            assert agent.hierarchy_role == "worker"

    def test_filter_by_hierarchy_role_disabled(self, mock_agents):
        """Test filtering disabled returns all agents."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=False)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_hierarchy_role(mock_agents, required_roles={"worker"})

        assert len(filtered) == len(mock_agents)

    def test_filter_by_hierarchy_role_no_roles_specified(self, mock_agents):
        """Test filtering with no roles specified returns all."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        filtered = selector._filter_by_hierarchy_role(mock_agents, required_roles=None)

        assert len(filtered) == len(mock_agents)


# ===========================================================================
# Test: Agent Capabilities Inference
# ===========================================================================


class TestAgentCapabilitiesInference:
    """Tests for inferring agent capabilities."""

    def test_get_capabilities_from_attribute(self):
        """Test getting capabilities from agent attribute."""
        agent = MockAgent("agent", capabilities={"coding", "reasoning"})
        selector = TeamSelector()

        caps = selector._get_agent_capabilities(agent)

        assert "coding" in caps
        assert "reasoning" in caps

    def test_infer_claude_capabilities(self):
        """Test inferring Claude capabilities."""
        agent = MockAgent("claude-3-opus")
        selector = TeamSelector()

        caps = selector._get_agent_capabilities(agent)

        assert "reasoning" in caps
        assert "synthesis" in caps
        assert "coordination" in caps
        assert "analysis" in caps
        assert "creativity" in caps

    def test_infer_gpt_capabilities(self):
        """Test inferring GPT capabilities."""
        agent = MockAgent("gpt-4-turbo")
        selector = TeamSelector()

        caps = selector._get_agent_capabilities(agent)

        assert "reasoning" in caps
        assert "synthesis" in caps
        assert "coordination" in caps
        assert "analysis" in caps

    def test_infer_codex_capabilities(self):
        """Test inferring Codex capabilities."""
        agent = MockAgent("codex-v2")
        selector = TeamSelector()

        caps = selector._get_agent_capabilities(agent)

        assert "reasoning" in caps
        assert "coding" in caps
        assert "analysis" in caps

    def test_infer_gemini_capabilities(self):
        """Test inferring Gemini capabilities."""
        agent = MockAgent("gemini-pro")
        selector = TeamSelector()

        caps = selector._get_agent_capabilities(agent)

        assert "reasoning" in caps
        assert "analysis" in caps
        assert "quality_assessment" in caps


# ===========================================================================
# Test: Cache Expiration
# ===========================================================================


class TestCacheExpiration:
    """Tests for cache expiration behavior."""

    def test_km_expertise_cache_expires(self, mock_agents):
        """Test KM expertise cache expires."""
        experts = [MockAgentExpertise("claude-opus", "code")]
        ranking_adapter = MockRankingAdapter(experts=experts)
        config = TeamSelectionConfig(
            enable_km_expertise=True,
            km_expertise_cache_ttl=0,  # Immediate expiration
        )
        selector = TeamSelector(ranking_adapter=ranking_adapter, config=config)

        # First call
        selector._get_km_domain_experts("code")

        # Manually set old timestamp
        if "code" in selector._km_expertise_cache:
            old_time, experts = selector._km_expertise_cache["code"]
            selector._km_expertise_cache["code"] = (old_time - 1000, experts)

        # Second call should refresh
        selector._get_km_domain_experts("code")

        # Cache should have recent timestamp
        assert "code" in selector._km_expertise_cache


# ===========================================================================
# Test: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_select_with_single_agent(self, mock_elo_system):
        """Test selection with single agent."""
        agent = MockAgent("solo-agent")
        selector = TeamSelector(elo_system=mock_elo_system)

        selected = selector.select(agents=[agent], domain="general")

        assert len(selected) == 1
        assert selected[0] == agent

    def test_select_handles_elo_errors(self, mock_agents):
        """Test selection handles ELO lookup errors gracefully."""
        elo_system = MagicMock()
        # Use KeyError which is in the caught exception types
        elo_system.get_rating.side_effect = KeyError("ELO error")

        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(elo_system=elo_system, config=config)

        # Should not raise, should use base score
        selected = selector.select(agents=mock_agents, domain="general")

        assert len(selected) == len(mock_agents)

    def test_select_handles_calibration_errors(self, mock_agents):
        """Test selection handles calibration errors gracefully."""
        tracker = MagicMock()
        # Use KeyError which is in the caught exception types
        tracker.get_brier_score.side_effect = KeyError("Calibration error")
        tracker.get_brier_scores_batch.side_effect = KeyError("Batch error")

        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(calibration_tracker=tracker, config=config)

        # Should not raise
        selected = selector.select(agents=mock_agents, domain="general")

        assert len(selected) == len(mock_agents)

    def test_score_negative_elo(self, mock_agents):
        """Test scoring with below-baseline ELO."""
        elo_system = MockEloSystem(ratings={"claude-opus": 500.0})  # Below 1000 baseline
        config = TeamSelectionConfig(
            elo_weight=0.3,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
        )
        selector = TeamSelector(elo_system=elo_system, config=config)

        score = selector.score_agent(mock_agents[0])

        # base (1.0) + (-500/1000 * 0.3) = 1.0 - 0.15 = 0.85
        assert score == pytest.approx(0.85, rel=0.01)

    def test_batch_calibration_fallback(self, mock_agents):
        """Test fallback to individual calibration lookups."""
        tracker = MagicMock()
        # No batch method
        del tracker.get_brier_scores_batch
        tracker.get_brier_score.return_value = 0.2

        config = TeamSelectionConfig(enable_domain_filtering=False)
        selector = TeamSelector(calibration_tracker=tracker, config=config)

        # Should work with individual lookups
        selected = selector.select(agents=mock_agents, domain="general")

        assert len(selected) == len(mock_agents)

    def test_case_insensitive_domain_matching(self, mock_agents):
        """Test domain matching is case insensitive."""
        config = TeamSelectionConfig(enable_domain_filtering=True)
        selector = TeamSelector(config=config)

        filtered_lower = selector._filter_by_domain_capability(mock_agents, "code")
        filtered_upper = selector._filter_by_domain_capability(mock_agents, "CODE")
        filtered_mixed = selector._filter_by_domain_capability(mock_agents, "Code")

        assert len(filtered_lower) == len(filtered_upper) == len(filtered_mixed)


# ===========================================================================
# Test: Async Culture Score
# ===========================================================================


class TestAsyncCultureScore:
    """Tests for async culture score computation."""

    @pytest.mark.asyncio
    async def test_compute_culture_score_async(self, mock_agents):
        """Test async culture score computation."""
        knowledge_mound = AsyncMock()
        knowledge_mound.recommend_agents.return_value = ["claude-opus", "gpt-4"]

        config = TeamSelectionConfig(enable_culture_selection=True)
        selector = TeamSelector(knowledge_mound=knowledge_mound, config=config)

        score = await selector.compute_culture_score_async(mock_agents[0], "code")

        # claude-opus is first in recommendations
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_compute_culture_score_async_no_recommendations(self, mock_agents):
        """Test async culture score with no recommendations."""
        knowledge_mound = AsyncMock()
        knowledge_mound.recommend_agents.return_value = []

        config = TeamSelectionConfig(enable_culture_selection=True)
        selector = TeamSelector(knowledge_mound=knowledge_mound, config=config)

        score = await selector.compute_culture_score_async(mock_agents[0], "code")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_compute_culture_score_async_disabled(self, mock_agents):
        """Test async culture score when disabled."""
        knowledge_mound = AsyncMock()

        config = TeamSelectionConfig(enable_culture_selection=False)
        selector = TeamSelector(knowledge_mound=knowledge_mound, config=config)

        score = await selector.compute_culture_score_async(mock_agents[0], "code")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_compute_culture_score_async_error_handling(self, mock_agents):
        """Test async culture score handles errors."""
        knowledge_mound = AsyncMock()
        knowledge_mound.recommend_agents.side_effect = Exception("KM error")

        config = TeamSelectionConfig(enable_culture_selection=True)
        selector = TeamSelector(knowledge_mound=knowledge_mound, config=config)

        score = await selector.compute_culture_score_async(mock_agents[0], "code")

        # Should return 0.0 on error
        assert score == 0.0


# ===========================================================================
# Performance Adapter Score Tests
# ===========================================================================


class TestPerformanceAdapterScore:
    """Tests for KM PerformanceAdapter-driven agent recommendation."""

    @pytest.fixture
    def mock_agents(self):
        return [
            MockAgent("claude-3", agent_type="anthropic"),
            MockAgent("gpt-4", agent_type="openai"),
            MockAgent("gemini-pro", agent_type="google"),
            MockAgent("deepseek-r1", agent_type="deepseek"),
        ]

    @pytest.fixture
    def mock_performance_adapter(self):
        """Create a mock PerformanceAdapter with domain experts."""
        adapter = MagicMock()
        # Return experts sorted by ELO (highest first)
        experts = [
            MagicMock(agent_name="claude-3", elo=1800, confidence=0.9, domain="security"),
            MagicMock(agent_name="gpt-4", elo=1650, confidence=0.7, domain="security"),
            MagicMock(agent_name="deepseek-r1", elo=1500, confidence=0.4, domain="security"),
        ]
        adapter.get_domain_experts.return_value = experts
        return adapter

    def test_performance_adapter_boosts_top_expert(self, mock_agents, mock_performance_adapter):
        """Top-ranked agent in performance adapter gets highest score."""
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector = TeamSelector(
            performance_adapter=mock_performance_adapter, config=config
        )

        claude_score = selector._compute_performance_adapter_score(mock_agents[0], "security")
        gpt_score = selector._compute_performance_adapter_score(mock_agents[1], "security")

        assert claude_score > gpt_score
        assert claude_score > 0.0
        assert gpt_score > 0.0

    def test_performance_adapter_unknown_agent_returns_zero(
        self, mock_agents, mock_performance_adapter
    ):
        """Agent not in expert list gets 0.0."""
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector = TeamSelector(
            performance_adapter=mock_performance_adapter, config=config
        )

        # gemini-pro is not in the experts list
        score = selector._compute_performance_adapter_score(mock_agents[2], "security")
        assert score == 0.0

    def test_performance_adapter_none_returns_zero(self, mock_agents):
        """No performance adapter returns 0.0."""
        selector = TeamSelector()
        score = selector._compute_performance_adapter_score(mock_agents[0], "security")
        assert score == 0.0

    def test_performance_adapter_empty_experts_returns_zero(self, mock_agents):
        """Empty expert list returns 0.0."""
        adapter = MagicMock()
        adapter.get_domain_experts.return_value = []
        selector = TeamSelector(performance_adapter=adapter)

        score = selector._compute_performance_adapter_score(mock_agents[0], "security")
        assert score == 0.0

    def test_performance_adapter_error_handling(self, mock_agents):
        """Handles adapter errors gracefully."""
        adapter = MagicMock()
        adapter.get_domain_experts.side_effect = TypeError("bad call")
        selector = TeamSelector(performance_adapter=adapter)

        score = selector._compute_performance_adapter_score(mock_agents[0], "security")
        assert score == 0.0

    def test_performance_adapter_confidence_affects_score(self, mock_agents):
        """Higher confidence yields higher score at same rank."""
        high_conf = MagicMock()
        high_conf.get_domain_experts.return_value = [
            MagicMock(agent_name="claude-3", confidence=0.95),
        ]

        low_conf = MagicMock()
        low_conf.get_domain_experts.return_value = [
            MagicMock(agent_name="claude-3", confidence=0.2),
        ]

        sel_high = TeamSelector(performance_adapter=high_conf)
        sel_low = TeamSelector(performance_adapter=low_conf)

        score_high = sel_high._compute_performance_adapter_score(mock_agents[0], "coding")
        score_low = sel_low._compute_performance_adapter_score(mock_agents[0], "coding")

        assert score_high > score_low

    def test_performance_adapter_wired_in_compute_score(
        self, mock_agents, mock_performance_adapter
    ):
        """Performance adapter score is included in _compute_score total."""
        config = TeamSelectionConfig(enable_km_expertise=True)
        selector_with = TeamSelector(
            performance_adapter=mock_performance_adapter, config=config
        )
        selector_without = TeamSelector(config=config)

        score_with = selector_with._compute_score(mock_agents[0], domain="security")
        score_without = selector_without._compute_score(mock_agents[0], domain="security")

        # The adapter should contribute a positive delta for top-ranked expert
        assert score_with > score_without

    def test_performance_adapter_select_reranks_agents(
        self, mock_agents, mock_performance_adapter
    ):
        """Full select() flow: adapter influences agent ordering."""
        config = TeamSelectionConfig(
            enable_km_expertise=True,
            enable_domain_filtering=False,
            enable_cv_selection=False,
            enable_pattern_selection=False,
        )
        selector = TeamSelector(
            performance_adapter=mock_performance_adapter, config=config
        )

        selected = selector.select(mock_agents, domain="security")

        # claude-3 should be ranked first (top expert with highest confidence)
        assert selected[0].name == "claude-3"
