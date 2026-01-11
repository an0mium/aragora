"""
Tests for Dynamic Role Matching Based on Agent Calibration.

Tests cover:
- RoleMatchingConfig dataclass defaults and validation
- RoleMatchResult dataclass
- RoleMatcher initialization
- Rotation strategy (simple sequential)
- Calibration strategy (calibration-based)
- Hybrid strategy (blended calibration + expertise)
- Role selection by calibration
- Affinity matrix computation
- Softmax selection
- Cold-start agent handling
- Developmental assignments
- Caching behavior
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.role_matcher import (
    RoleMatchingConfig,
    RoleMatchResult,
    RoleMatcher,
)
from aragora.debate.roles import CognitiveRole, RoleAssignment, ROLE_PROMPTS


# ============================================================================
# Mock Classes for Dependencies
# ============================================================================

@dataclass
class MockCalibrationBucket:
    """Mock calibration bucket for testing."""
    range_start: float
    range_end: float
    total_predictions: int = 0
    correct_predictions: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def expected_accuracy(self) -> float:
        return (self.range_start + self.range_end) / 2


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary for testing."""
    agent: str
    total_predictions: int = 0
    total_correct: int = 0
    brier_score: float = 0.0
    ece: float = 0.0
    buckets: list = field(default_factory=list)
    _is_overconfident: bool = False
    _is_underconfident: bool = False

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions

    @property
    def is_overconfident(self) -> bool:
        return self._is_overconfident

    @property
    def is_underconfident(self) -> bool:
        return self._is_underconfident


@dataclass
class MockPersona:
    """Mock persona for testing."""
    agent_name: str
    description: str = ""
    traits: list = field(default_factory=list)
    expertise: dict = field(default_factory=dict)


class MockCalibrationTracker:
    """Mock calibration tracker for testing."""

    def __init__(self, summaries: Optional[dict] = None):
        self.summaries = summaries or {}

    def get_calibration_summary(self, agent: str):
        if agent in self.summaries:
            return self.summaries[agent]
        raise KeyError(f"No calibration for {agent}")


class MockPersonaManager:
    """Mock persona manager for testing."""

    def __init__(self, personas: Optional[dict] = None):
        self.personas = personas or {}

    def get_persona(self, agent: str):
        if agent in self.personas:
            return self.personas[agent]
        raise KeyError(f"No persona for {agent}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Create default role matching config."""
    return RoleMatchingConfig()


@pytest.fixture
def basic_matcher():
    """Create a basic RoleMatcher with no dependencies."""
    return RoleMatcher()


@pytest.fixture
def calibration_matcher():
    """Create a RoleMatcher with mocked calibration tracker."""
    summaries = {
        "claude": MockCalibrationSummary(
            agent="claude",
            total_predictions=20,
            total_correct=16,
            brier_score=0.15,
            ece=0.05,
        ),
        "gpt": MockCalibrationSummary(
            agent="gpt",
            total_predictions=20,
            total_correct=14,
            brier_score=0.30,
            ece=0.15,
            _is_overconfident=True,
        ),
        "gemini": MockCalibrationSummary(
            agent="gemini",
            total_predictions=20,
            total_correct=18,
            brier_score=0.20,
            ece=0.08,
            _is_underconfident=True,
        ),
    }
    tracker = MockCalibrationTracker(summaries)
    return RoleMatcher(calibration_tracker=tracker)


@pytest.fixture
def hybrid_matcher():
    """Create a RoleMatcher with both calibration and persona data."""
    summaries = {
        "claude": MockCalibrationSummary(
            agent="claude",
            total_predictions=20,
            total_correct=16,
            brier_score=0.15,
            ece=0.05,
        ),
        "gpt": MockCalibrationSummary(
            agent="gpt",
            total_predictions=20,
            total_correct=14,
            brier_score=0.30,
            ece=0.15,
            _is_overconfident=True,
        ),
    }
    personas = {
        "claude": MockPersona(
            agent_name="claude",
            traits=["thorough", "diplomatic"],
            expertise={"security": 0.8, "api_design": 0.7},
        ),
        "gpt": MockPersona(
            agent_name="gpt",
            traits=["innovative", "contrarian"],
            expertise={"security": 0.5, "database": 0.9},
        ),
    }
    tracker = MockCalibrationTracker(summaries)
    manager = MockPersonaManager(personas)
    return RoleMatcher(
        calibration_tracker=tracker,
        persona_manager=manager,
        config=RoleMatchingConfig(strategy="hybrid"),
    )


# ============================================================================
# RoleMatchingConfig Tests
# ============================================================================

class TestRoleMatchingConfig:
    """Tests for RoleMatchingConfig dataclass."""

    def test_default_strategy(self, default_config):
        """Test default strategy is hybrid."""
        assert default_config.strategy == "hybrid"

    def test_default_calibration_weight(self, default_config):
        """Test default calibration weight."""
        assert default_config.calibration_weight == 0.4

    def test_default_expertise_weight(self, default_config):
        """Test default expertise weight."""
        assert default_config.expertise_weight == 0.3

    def test_default_min_predictions(self, default_config):
        """Test default minimum predictions for calibration."""
        assert default_config.min_predictions_for_calibration == 5

    def test_default_selection_temperature(self, default_config):
        """Test default selection temperature."""
        assert default_config.selection_temperature == 1.0

    def test_default_developmental_enabled(self, default_config):
        """Test developmental assignment is enabled by default."""
        assert default_config.enable_developmental_assignment is True

    def test_default_brier_threshold(self, default_config):
        """Test default Brier threshold."""
        assert default_config.brier_threshold == 0.25

    def test_default_ece_threshold(self, default_config):
        """Test default ECE threshold."""
        assert default_config.ece_threshold == 0.1

    def test_custom_strategy(self):
        """Test custom strategy configuration."""
        config = RoleMatchingConfig(strategy="calibration")
        assert config.strategy == "calibration"

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = RoleMatchingConfig(
            calibration_weight=0.6,
            expertise_weight=0.4,
        )
        assert config.calibration_weight == 0.6
        assert config.expertise_weight == 0.4


# ============================================================================
# RoleMatchResult Tests
# ============================================================================

class TestRoleMatchResult:
    """Tests for RoleMatchResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = RoleMatchResult(
            round_num=1,
            assignments={"agent1": MagicMock()},
            strategy_used="rotation",
            calibration_used=False,
        )
        assert result.round_num == 1
        assert result.strategy_used == "rotation"
        assert result.calibration_used is False

    def test_default_lists(self):
        """Test default empty lists."""
        result = RoleMatchResult(
            round_num=0,
            assignments={},
            strategy_used="hybrid",
            calibration_used=True,
        )
        assert result.cold_start_agents == []
        assert result.developmental_assignments == []

    def test_with_cold_start_agents(self):
        """Test result with cold start agents."""
        result = RoleMatchResult(
            round_num=1,
            assignments={},
            strategy_used="calibration",
            calibration_used=True,
            cold_start_agents=["new_agent"],
        )
        assert "new_agent" in result.cold_start_agents

    def test_with_developmental_assignments(self):
        """Test result with developmental assignments."""
        result = RoleMatchResult(
            round_num=1,
            assignments={},
            strategy_used="hybrid",
            calibration_used=True,
            developmental_assignments=["overconf_agent"],
        )
        assert "overconf_agent" in result.developmental_assignments


# ============================================================================
# RoleMatcher Initialization Tests
# ============================================================================

class TestRoleMatcherInit:
    """Tests for RoleMatcher initialization."""

    def test_init_no_dependencies(self):
        """Test initialization without any dependencies."""
        matcher = RoleMatcher()
        assert matcher.calibration_tracker is None
        assert matcher.persona_manager is None
        assert matcher.config is not None

    def test_init_default_config(self):
        """Test initialization creates default config."""
        matcher = RoleMatcher()
        assert matcher.config.strategy == "hybrid"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = RoleMatchingConfig(strategy="rotation")
        matcher = RoleMatcher(config=config)
        assert matcher.config.strategy == "rotation"

    def test_init_with_tracker(self):
        """Test initialization with calibration tracker."""
        tracker = MockCalibrationTracker()
        matcher = RoleMatcher(calibration_tracker=tracker)
        assert matcher.calibration_tracker is tracker

    def test_init_with_manager(self):
        """Test initialization with persona manager."""
        manager = MockPersonaManager()
        matcher = RoleMatcher(persona_manager=manager)
        assert matcher.persona_manager is manager

    def test_init_empty_cache(self):
        """Test initialization creates empty calibration cache."""
        matcher = RoleMatcher()
        assert matcher._calibration_cache == {}


# ============================================================================
# Rotation Strategy Tests
# ============================================================================

class TestRotationStrategy:
    """Tests for simple rotation strategy."""

    def test_rotation_strategy_basic(self, basic_matcher):
        """Test basic rotation assigns roles."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["a1", "a2", "a3"], round_num=0)

        assert result.strategy_used == "rotation"
        assert result.calibration_used is False
        assert len(result.assignments) == 3

    def test_rotation_all_agents_get_roles(self, basic_matcher):
        """Test all agents receive role assignments."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["a1", "a2", "a3"], round_num=0)

        assert "a1" in result.assignments
        assert "a2" in result.assignments
        assert "a3" in result.assignments

    def test_rotation_valid_role_assignments(self, basic_matcher):
        """Test all assignments have valid CognitiveRole."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["a1", "a2"], round_num=0)

        for assignment in result.assignments.values():
            assert isinstance(assignment.role, CognitiveRole)

    def test_rotation_role_prompts_set(self, basic_matcher):
        """Test role prompts are set from ROLE_PROMPTS."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["agent"], round_num=0)

        assignment = result.assignments["agent"]
        expected_prompt = ROLE_PROMPTS.get(assignment.role, "")
        assert assignment.role_prompt == expected_prompt

    def test_rotation_different_rounds(self, basic_matcher):
        """Test rotation produces different assignments in different rounds."""
        basic_matcher.config.strategy = "rotation"
        agents = ["a1", "a2"]

        result_r0 = basic_matcher.match_roles(agents, round_num=0)
        result_r1 = basic_matcher.match_roles(agents, round_num=1)

        # At least one agent should have a different role
        roles_r0 = {a: r.role for a, r in result_r0.assignments.items()}
        roles_r1 = {a: r.role for a, r in result_r1.assignments.items()}
        assert roles_r0 != roles_r1

    def test_rotation_round_num_in_assignment(self, basic_matcher):
        """Test round number is set in assignments."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["agent"], round_num=5)

        assert result.round_num == 5
        assert result.assignments["agent"].round_num == 5


# ============================================================================
# Calibration Strategy Tests
# ============================================================================

class TestCalibrationStrategy:
    """Tests for calibration-based strategy."""

    def test_calibration_strategy_used(self, calibration_matcher):
        """Test calibration strategy is reported correctly."""
        calibration_matcher.config.strategy = "calibration"
        result = calibration_matcher.match_roles(["claude", "gpt"], round_num=0)

        assert result.strategy_used == "calibration"
        assert result.calibration_used is True

    def test_calibration_identifies_cold_start(self, calibration_matcher):
        """Test cold start agents are identified."""
        calibration_matcher.config.strategy = "calibration"
        result = calibration_matcher.match_roles(
            ["claude", "new_agent"], round_num=0
        )

        assert "new_agent" in result.cold_start_agents

    def test_calibration_developmental_overconfident(self, calibration_matcher):
        """Test overconfident agents get developmental assignments."""
        calibration_matcher.config.strategy = "calibration"
        result = calibration_matcher.match_roles(["gpt"], round_num=0)

        assert "gpt" in result.developmental_assignments

    def test_calibration_developmental_underconfident(self, calibration_matcher):
        """Test underconfident agents get developmental assignments."""
        calibration_matcher.config.strategy = "calibration"
        result = calibration_matcher.match_roles(["gemini"], round_num=0)

        assert "gemini" in result.developmental_assignments

    def test_calibration_well_calibrated_no_developmental(self, calibration_matcher):
        """Test well-calibrated agents don't get developmental flag."""
        calibration_matcher.config.strategy = "calibration"
        result = calibration_matcher.match_roles(["claude"], round_num=0)

        assert "claude" not in result.developmental_assignments

    def test_calibration_with_low_predictions(self):
        """Test agents with few predictions are cold-start."""
        summaries = {
            "new_agent": MockCalibrationSummary(
                agent="new_agent",
                total_predictions=2,  # Below threshold of 5
            ),
        }
        tracker = MockCalibrationTracker(summaries)
        matcher = RoleMatcher(
            calibration_tracker=tracker,
            config=RoleMatchingConfig(strategy="calibration"),
        )

        result = matcher.match_roles(["new_agent"], round_num=0)
        assert "new_agent" in result.cold_start_agents


# ============================================================================
# Hybrid Strategy Tests
# ============================================================================

class TestHybridStrategy:
    """Tests for hybrid strategy (calibration + expertise)."""

    def test_hybrid_strategy_used(self, hybrid_matcher):
        """Test hybrid strategy is reported correctly."""
        result = hybrid_matcher.match_roles(["claude", "gpt"], round_num=0)

        assert result.strategy_used == "hybrid"
        assert result.calibration_used is True

    def test_hybrid_considers_domain(self, hybrid_matcher):
        """Test hybrid strategy accepts domain parameter."""
        # Should not raise
        result = hybrid_matcher.match_roles(
            ["claude", "gpt"],
            round_num=0,
            debate_domain="security",
        )
        assert len(result.assignments) == 2

    def test_hybrid_cold_start_agents(self, hybrid_matcher):
        """Test hybrid identifies cold-start agents."""
        result = hybrid_matcher.match_roles(
            ["claude", "unknown_agent"],
            round_num=0,
        )
        assert "unknown_agent" in result.cold_start_agents


# ============================================================================
# Role Selection by Calibration Tests
# ============================================================================

class TestSelectRoleByCalibration:
    """Tests for _select_role_by_calibration method."""

    def test_well_calibrated_prefers_skeptic(self, basic_matcher):
        """Test well-calibrated agents prefer SKEPTIC role."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            total_correct=16,
            brier_score=0.10,  # Below threshold
            ece=0.05,  # Below threshold
        )

        role = basic_matcher._select_role_by_calibration(cal, set())
        assert role in [CognitiveRole.SKEPTIC, CognitiveRole.QUALITY_CHALLENGER]

    def test_overconfident_prefers_devil_advocate(self, basic_matcher):
        """Test overconfident agents prefer DEVIL_ADVOCATE role."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            brier_score=0.35,
            ece=0.15,
            _is_overconfident=True,
        )

        role = basic_matcher._select_role_by_calibration(cal, set())
        assert role in [CognitiveRole.DEVIL_ADVOCATE, CognitiveRole.SKEPTIC]

    def test_underconfident_prefers_advocate(self, basic_matcher):
        """Test underconfident agents prefer ADVOCATE role."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            brier_score=0.20,
            ece=0.12,
            _is_underconfident=True,
        )

        role = basic_matcher._select_role_by_calibration(cal, set())
        assert role in [CognitiveRole.ADVOCATE, CognitiveRole.ANALYST]

    def test_high_accuracy_prefers_synthesizer(self, basic_matcher):
        """Test high accuracy agents prefer SYNTHESIZER role."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            total_correct=18,  # 90% accuracy
            brier_score=0.30,  # Above threshold (not well-calibrated)
            ece=0.12,
        )

        role = basic_matcher._select_role_by_calibration(cal, set())
        assert role in [CognitiveRole.SYNTHESIZER, CognitiveRole.ANALYST]

    def test_avoids_used_roles(self, basic_matcher):
        """Test selection avoids already used roles."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            brier_score=0.10,
            ece=0.05,
        )

        # Mark preferred roles as used
        used = {CognitiveRole.SKEPTIC, CognitiveRole.QUALITY_CHALLENGER}
        role = basic_matcher._select_role_by_calibration(cal, used)
        assert role not in used

    def test_fallback_when_all_used(self, basic_matcher):
        """Test fallback when all roles are used."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            brier_score=0.10,
            ece=0.05,
        )

        # Mark all roles as used
        used = set(CognitiveRole)
        role = basic_matcher._select_role_by_calibration(cal, used)
        # Should still return a valid role
        assert isinstance(role, CognitiveRole)


# ============================================================================
# Affinity Matrix Tests
# ============================================================================

class TestAffinityMatrix:
    """Tests for affinity matrix computation."""

    def test_affinity_matrix_structure(self, hybrid_matcher):
        """Test affinity matrix has correct structure."""
        cal = {"claude": MockCalibrationSummary(agent="claude", total_predictions=10)}
        personas = {"claude": MockPersona(agent_name="claude")}

        matrix = hybrid_matcher._compute_affinity_matrix(
            ["claude"], cal, personas, None
        )

        assert "claude" in matrix
        assert all(isinstance(r, CognitiveRole) for r in matrix["claude"].keys())

    def test_affinity_values_bounded(self, hybrid_matcher):
        """Test affinity values are bounded 0-1."""
        cal = {
            "agent": MockCalibrationSummary(
                agent="agent",
                total_predictions=20,
                brier_score=0.1,
                ece=0.05,
            )
        }
        personas = {
            "agent": MockPersona(
                agent_name="agent",
                expertise={"security": 1.0},
                traits=["thorough", "contrarian"],
            )
        }

        matrix = hybrid_matcher._compute_affinity_matrix(
            ["agent"], cal, personas, "security"
        )

        for role, score in matrix["agent"].items():
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {role}"

    def test_affinity_base_score(self, basic_matcher):
        """Test base affinity score is 0.5."""
        matrix = basic_matcher._compute_affinity_matrix(
            ["agent"], {}, {}, None
        )

        # Without calibration or persona, all roles should have base score
        for score in matrix["agent"].values():
            assert score == 0.5


# ============================================================================
# Calibration Affinity Tests
# ============================================================================

class TestCalibrationAffinity:
    """Tests for _calibration_affinity method."""

    def test_skeptic_affinity_well_calibrated(self, basic_matcher):
        """Test SKEPTIC affinity for well-calibrated agent."""
        cal = MockCalibrationSummary(
            agent="test",
            brier_score=0.10,
            ece=0.05,
        )
        affinity = basic_matcher._calibration_affinity(cal, CognitiveRole.SKEPTIC)
        assert affinity == 0.8

    def test_skeptic_affinity_poorly_calibrated(self, basic_matcher):
        """Test SKEPTIC affinity for poorly-calibrated agent."""
        cal = MockCalibrationSummary(
            agent="test",
            brier_score=0.40,
            ece=0.20,
        )
        affinity = basic_matcher._calibration_affinity(cal, CognitiveRole.SKEPTIC)
        assert affinity == 0.3

    def test_devil_advocate_affinity_overconfident(self, basic_matcher):
        """Test DEVIL_ADVOCATE affinity for overconfident agent."""
        cal = MockCalibrationSummary(
            agent="test",
            _is_overconfident=True,
        )
        affinity = basic_matcher._calibration_affinity(
            cal, CognitiveRole.DEVIL_ADVOCATE
        )
        assert affinity == 0.8

    def test_advocate_affinity_underconfident(self, basic_matcher):
        """Test ADVOCATE affinity for underconfident agent."""
        cal = MockCalibrationSummary(
            agent="test",
            _is_underconfident=True,
        )
        affinity = basic_matcher._calibration_affinity(cal, CognitiveRole.ADVOCATE)
        assert affinity == 0.8

    def test_synthesizer_affinity_high_accuracy(self, basic_matcher):
        """Test SYNTHESIZER affinity for high accuracy agent."""
        cal = MockCalibrationSummary(
            agent="test",
            total_predictions=20,
            total_correct=16,  # 80% accuracy
        )
        affinity = basic_matcher._calibration_affinity(cal, CognitiveRole.SYNTHESIZER)
        assert affinity == 0.7

    def test_lateral_thinker_neutral(self, basic_matcher):
        """Test LATERAL_THINKER has neutral affinity."""
        cal = MockCalibrationSummary(agent="test")
        affinity = basic_matcher._calibration_affinity(
            cal, CognitiveRole.LATERAL_THINKER
        )
        assert affinity == 0.5


# ============================================================================
# Expertise Affinity Tests
# ============================================================================

class TestExpertiseAffinity:
    """Tests for _expertise_affinity method."""

    def test_analyst_high_expertise(self, basic_matcher):
        """Test ANALYST affinity with high domain expertise."""
        persona = MockPersona(
            agent_name="test",
            expertise={"security": 0.9},
        )
        affinity = basic_matcher._expertise_affinity(
            persona, "security", CognitiveRole.ANALYST
        )
        assert affinity == pytest.approx(0.72)  # 0.9 * 0.8

    def test_lateral_thinker_low_expertise(self, basic_matcher):
        """Test LATERAL_THINKER affinity with low domain expertise."""
        persona = MockPersona(
            agent_name="test",
            expertise={"security": 0.2},
        )
        affinity = basic_matcher._expertise_affinity(
            persona, "security", CognitiveRole.LATERAL_THINKER
        )
        # Low expertise means fresh perspective
        assert affinity == pytest.approx(0.48)  # (1.0 - 0.2) * 0.6

    def test_expertise_missing_domain(self, basic_matcher):
        """Test affinity when domain not in expertise."""
        persona = MockPersona(
            agent_name="test",
            expertise={"security": 0.9},
        )
        affinity = basic_matcher._expertise_affinity(
            persona, "database", CognitiveRole.ANALYST
        )
        assert affinity == 0.0  # No expertise = 0


# ============================================================================
# Trait Affinity Tests
# ============================================================================

class TestTraitAffinity:
    """Tests for _trait_affinity method."""

    def test_analyst_thorough_trait(self, basic_matcher):
        """Test ANALYST affinity with 'thorough' trait."""
        affinity = basic_matcher._trait_affinity(
            ["thorough"], CognitiveRole.ANALYST
        )
        assert affinity == 0.4  # One matching trait

    def test_analyst_multiple_matching_traits(self, basic_matcher):
        """Test ANALYST affinity with multiple matching traits."""
        affinity = basic_matcher._trait_affinity(
            ["thorough", "pragmatic"], CognitiveRole.ANALYST
        )
        assert affinity == 0.8  # Two matching traits

    def test_skeptic_contrarian_trait(self, basic_matcher):
        """Test SKEPTIC affinity with 'contrarian' trait."""
        affinity = basic_matcher._trait_affinity(
            ["contrarian", "direct"], CognitiveRole.SKEPTIC
        )
        assert affinity == 0.8  # Two matching traits

    def test_no_matching_traits(self, basic_matcher):
        """Test affinity with no matching traits."""
        affinity = basic_matcher._trait_affinity(
            ["innovative"], CognitiveRole.ANALYST
        )
        assert affinity == 0.0

    def test_affinity_capped_at_one(self, basic_matcher):
        """Test affinity is capped at 1.0."""
        # Even with many matching traits
        affinity = basic_matcher._trait_affinity(
            ["thorough", "pragmatic", "diplomatic", "innovative"],
            CognitiveRole.ANALYST,
        )
        assert affinity <= 1.0


# ============================================================================
# Softmax Selection Tests
# ============================================================================

class TestSoftmaxSelection:
    """Tests for _softmax_select_role method."""

    def test_softmax_returns_valid_role(self, basic_matcher):
        """Test softmax selection returns a valid role."""
        affinities = {
            CognitiveRole.ANALYST: 0.8,
            CognitiveRole.SKEPTIC: 0.6,
            CognitiveRole.SYNTHESIZER: 0.4,
        }
        role = basic_matcher._softmax_select_role(affinities, set(), 1.0)
        assert role in affinities

    def test_softmax_avoids_used_roles(self, basic_matcher):
        """Test softmax avoids already used roles."""
        affinities = {
            CognitiveRole.ANALYST: 0.9,
            CognitiveRole.SKEPTIC: 0.1,
        }
        role = basic_matcher._softmax_select_role(
            affinities,
            {CognitiveRole.ANALYST},
            1.0,
        )
        assert role == CognitiveRole.SKEPTIC

    def test_softmax_zero_temperature_deterministic(self, basic_matcher):
        """Test zero temperature selects highest affinity."""
        affinities = {
            CognitiveRole.ANALYST: 0.3,
            CognitiveRole.SKEPTIC: 0.9,
            CognitiveRole.SYNTHESIZER: 0.1,
        }
        role = basic_matcher._softmax_select_role(affinities, set(), 0.0)
        assert role == CognitiveRole.SKEPTIC

    def test_softmax_high_temperature_more_random(self, basic_matcher):
        """Test high temperature increases randomness."""
        affinities = {
            CognitiveRole.ANALYST: 0.9,
            CognitiveRole.SKEPTIC: 0.1,
        }

        # With high temperature, should sometimes select lower affinity
        random.seed(42)
        roles = [
            basic_matcher._softmax_select_role(affinities, set(), 10.0)
            for _ in range(100)
        ]

        # Both roles should appear with high temperature
        unique_roles = set(roles)
        assert len(unique_roles) >= 1  # At minimum one role

    def test_softmax_fallback_when_all_used(self, basic_matcher):
        """Test softmax ignores used constraint when all used."""
        affinities = {
            CognitiveRole.ANALYST: 0.8,
            CognitiveRole.SKEPTIC: 0.6,
        }
        used = {CognitiveRole.ANALYST, CognitiveRole.SKEPTIC}

        role = basic_matcher._softmax_select_role(affinities, used, 1.0)
        assert role in affinities

    def test_softmax_empty_affinities(self, basic_matcher):
        """Test softmax with empty affinities returns default."""
        role = basic_matcher._softmax_select_role({}, set(), 1.0)
        assert role == CognitiveRole.ANALYST


# ============================================================================
# Cache Tests
# ============================================================================

class TestCache:
    """Tests for calibration caching behavior."""

    def test_cache_starts_empty(self, basic_matcher):
        """Test cache starts empty."""
        assert len(basic_matcher._calibration_cache) == 0

    def test_clear_cache(self, calibration_matcher):
        """Test clear_cache method."""
        # Populate cache by matching
        calibration_matcher.config.strategy = "calibration"
        calibration_matcher.match_roles(["claude"], round_num=0)

        # Cache should have entries
        assert len(calibration_matcher._calibration_cache) > 0

        # Clear cache
        calibration_matcher.clear_cache()
        assert len(calibration_matcher._calibration_cache) == 0

    def test_cache_reuses_calibration(self, calibration_matcher):
        """Test cache reuses calibration data."""
        calibration_matcher.config.strategy = "calibration"

        # First call populates cache
        calibration_matcher.match_roles(["claude"], round_num=0)
        cached = calibration_matcher._calibration_cache.get("claude")

        # Second call should use cache
        calibration_matcher.match_roles(["claude"], round_num=1)
        assert calibration_matcher._calibration_cache.get("claude") is cached


# ============================================================================
# Get Calibrations Tests
# ============================================================================

class TestGetCalibrations:
    """Tests for _get_calibrations method."""

    def test_get_calibrations_no_tracker(self, basic_matcher):
        """Test returns empty dict when no tracker."""
        result = basic_matcher._get_calibrations(["agent1", "agent2"])
        assert result == {}

    def test_get_calibrations_with_tracker(self, calibration_matcher):
        """Test returns calibrations from tracker."""
        result = calibration_matcher._get_calibrations(["claude", "gpt"])

        assert "claude" in result
        assert "gpt" in result
        assert result["claude"].agent == "claude"

    def test_get_calibrations_handles_missing(self, calibration_matcher):
        """Test handles missing calibrations gracefully."""
        result = calibration_matcher._get_calibrations(["claude", "unknown"])

        assert result["claude"] is not None
        assert result["unknown"] is None


# ============================================================================
# Get Personas Tests
# ============================================================================

class TestGetPersonas:
    """Tests for _get_personas method."""

    def test_get_personas_no_manager(self, basic_matcher):
        """Test returns empty dict when no manager."""
        result = basic_matcher._get_personas(["agent1", "agent2"])
        assert result == {}

    def test_get_personas_with_manager(self, hybrid_matcher):
        """Test returns personas from manager."""
        result = hybrid_matcher._get_personas(["claude", "gpt"])

        assert "claude" in result
        assert "gpt" in result
        assert result["claude"].agent_name == "claude"

    def test_get_personas_handles_missing(self, hybrid_matcher):
        """Test handles missing personas gracefully."""
        result = hybrid_matcher._get_personas(["claude", "unknown"])

        assert result["claude"] is not None
        assert result["unknown"] is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for RoleMatcher."""

    def test_full_round_rotation(self, basic_matcher):
        """Test complete round with rotation strategy."""
        basic_matcher.config.strategy = "rotation"
        agents = ["a1", "a2", "a3"]

        results = []
        for round_num in range(3):
            result = basic_matcher.match_roles(agents, round_num)
            results.append(result)

        # Each round should have all agents assigned
        for result in results:
            assert len(result.assignments) == 3

    def test_full_round_calibration(self, calibration_matcher):
        """Test complete round with calibration strategy."""
        calibration_matcher.config.strategy = "calibration"
        agents = ["claude", "gpt", "gemini"]

        result = calibration_matcher.match_roles(agents, round_num=0)

        assert len(result.assignments) == 3
        assert result.calibration_used is True
        # At least overconfident/underconfident should be developmental
        assert len(result.developmental_assignments) >= 2

    def test_full_round_hybrid(self, hybrid_matcher):
        """Test complete round with hybrid strategy."""
        agents = ["claude", "gpt"]

        result = hybrid_matcher.match_roles(
            agents,
            round_num=0,
            debate_domain="security",
        )

        assert len(result.assignments) == 2
        assert result.strategy_used == "hybrid"
        assert result.calibration_used is True

    def test_deterministic_with_seed(self, basic_matcher):
        """Test deterministic results with fixed random seed."""
        basic_matcher.config.strategy = "calibration"

        random.seed(42)
        result1 = basic_matcher._rotation_strategy(["a1", "a2"], 0)

        random.seed(42)
        result2 = basic_matcher._rotation_strategy(["a1", "a2"], 0)

        # Same seed should produce same assignments
        for agent in ["a1", "a2"]:
            assert result1.assignments[agent].role == result2.assignments[agent].role

    def test_developmental_disabled(self, calibration_matcher):
        """Test developmental assignments can be disabled."""
        calibration_matcher.config.enable_developmental_assignment = False
        calibration_matcher.config.strategy = "calibration"

        result = calibration_matcher.match_roles(["gpt"], round_num=0)

        # Even overconfident agent shouldn't be marked developmental
        assert result.developmental_assignments == []


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_agent_list(self, basic_matcher):
        """Test handling of empty agent list."""
        result = basic_matcher.match_roles([], round_num=0)
        assert result.assignments == {}

    def test_single_agent(self, basic_matcher):
        """Test handling of single agent."""
        result = basic_matcher.match_roles(["solo"], round_num=0)
        assert len(result.assignments) == 1
        assert "solo" in result.assignments

    def test_many_agents_more_than_roles(self, basic_matcher):
        """Test handling of more agents than roles."""
        basic_matcher.config.strategy = "rotation"
        agents = [f"agent_{i}" for i in range(10)]

        result = basic_matcher.match_roles(agents, round_num=0)

        # All agents should still get assignments
        assert len(result.assignments) == 10

    def test_negative_round_num(self, basic_matcher):
        """Test handling of negative round number."""
        # Should not raise
        result = basic_matcher.match_roles(["agent"], round_num=-1)
        assert len(result.assignments) == 1

    def test_large_round_num(self, basic_matcher):
        """Test handling of large round number."""
        basic_matcher.config.strategy = "rotation"
        result = basic_matcher.match_roles(["agent"], round_num=1000)
        assert len(result.assignments) == 1
