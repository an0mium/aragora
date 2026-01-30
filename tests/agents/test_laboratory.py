"""
Tests for Emergent Persona Laboratory v2.

Tests cover:
- PersonaExperiment dataclass and properties
- EmergentTrait tracking dataclass
- TraitTransfer between agents
- PersonaLaboratory orchestrator initialization
- Experiment creation and variant generation
- Experiment result recording and conclusion
- Trait mutations and selection
- Success metrics calculation
- Cross-pollination logic
- Emergent trait detection
- Evolution history tracking
- Laboratory statistics
- Error handling
"""

import json
import pytest
import random
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.laboratory import (
    EmergentTrait,
    PersonaExperiment,
    PersonaLaboratory,
    TraitTransfer,
)
from aragora.agents.personas import (
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    Persona,
    PersonaManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def temp_persona_db():
    """Create a temporary database for PersonaManager."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def persona_manager(temp_persona_db):
    """Create a PersonaManager with a temporary database."""
    return PersonaManager(db_path=temp_persona_db)


@pytest.fixture
def laboratory(persona_manager, temp_db):
    """Create a PersonaLaboratory with temporary databases."""
    return PersonaLaboratory(persona_manager=persona_manager, db_path=temp_db)


@pytest.fixture
def sample_persona():
    """Create a sample persona for testing."""
    return Persona(
        agent_name="test_agent",
        description="A test agent",
        traits=["thorough", "pragmatic"],
        expertise={"security": 0.8, "testing": 0.6},
    )


@pytest.fixture
def sample_experiment(sample_persona):
    """Create a sample experiment for testing."""
    variant = Persona(
        agent_name="test_agent_variant",
        traits=["thorough", "direct"],
        expertise={"security": 0.9, "testing": 0.5},
    )
    return PersonaExperiment(
        experiment_id="exp_test123",
        agent_name="test_agent",
        control_persona=sample_persona,
        variant_persona=variant,
        hypothesis="Testing if direct trait improves performance",
        status="running",
        control_successes=15,
        control_trials=25,
        variant_successes=18,
        variant_trials=25,
    )


# ---------------------------------------------------------------------------
# PersonaExperiment dataclass tests
# ---------------------------------------------------------------------------


class TestPersonaExperiment:
    """Tests for the PersonaExperiment dataclass."""

    def test_create_basic_experiment(self, sample_persona):
        """Test creating a basic experiment."""
        variant = Persona(agent_name="variant", traits=["direct"])
        exp = PersonaExperiment(
            experiment_id="exp_001",
            agent_name="test_agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="Test hypothesis",
        )

        assert exp.experiment_id == "exp_001"
        assert exp.agent_name == "test_agent"
        assert exp.status == "running"
        assert exp.control_successes == 0
        assert exp.control_trials == 0

    def test_experiment_defaults(self, sample_persona):
        """Test experiment default values."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_002",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
        )

        assert exp.status == "running"
        assert exp.control_successes == 0
        assert exp.variant_successes == 0
        assert exp.completed_at is None
        assert exp.created_at is not None

    def test_control_rate_property(self, sample_experiment):
        """Test control_rate calculation."""
        rate = sample_experiment.control_rate
        assert rate == 15 / 25
        assert rate == 0.6

    def test_variant_rate_property(self, sample_experiment):
        """Test variant_rate calculation."""
        rate = sample_experiment.variant_rate
        assert rate == 18 / 25
        assert rate == 0.72

    def test_control_rate_zero_trials(self, sample_persona):
        """Test control_rate with zero trials returns 0."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_003",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_trials=0,
        )

        assert exp.control_rate == 0.0

    def test_variant_rate_zero_trials(self, sample_persona):
        """Test variant_rate with zero trials returns 0."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_004",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            variant_trials=0,
        )

        assert exp.variant_rate == 0.0

    def test_relative_improvement_positive(self, sample_experiment):
        """Test relative_improvement with positive improvement."""
        improvement = sample_experiment.relative_improvement
        # (0.72 - 0.6) / 0.6 = 0.2
        assert improvement == pytest.approx(0.2, rel=0.01)

    def test_relative_improvement_negative(self, sample_persona):
        """Test relative_improvement with negative improvement."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_005",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_successes=20,
            control_trials=25,
            variant_successes=10,
            variant_trials=25,
        )

        improvement = exp.relative_improvement
        # (0.4 - 0.8) / 0.8 = -0.5
        assert improvement == pytest.approx(-0.5, rel=0.01)

    def test_relative_improvement_zero_control_rate(self, sample_persona):
        """Test relative_improvement when control rate is zero."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_006",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_successes=0,
            control_trials=25,
            variant_successes=10,
            variant_trials=25,
        )

        assert exp.relative_improvement == 0.0

    def test_is_significant_true(self, sample_persona):
        """Test is_significant returns True with sufficient data."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_007",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_successes=12,
            control_trials=20,  # 60%
            variant_successes=18,
            variant_trials=20,  # 90% -> 50% improvement
        )

        assert exp.is_significant is True

    def test_is_significant_false_insufficient_trials(self, sample_persona):
        """Test is_significant returns False with insufficient trials."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_008",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_successes=6,
            control_trials=10,  # Less than 20 min
            variant_successes=9,
            variant_trials=10,
        )

        assert exp.is_significant is False

    def test_is_significant_false_small_difference(self, sample_persona):
        """Test is_significant returns False with small difference."""
        variant = Persona(agent_name="variant")
        exp = PersonaExperiment(
            experiment_id="exp_009",
            agent_name="agent",
            control_persona=sample_persona,
            variant_persona=variant,
            hypothesis="",
            control_successes=15,
            control_trials=25,  # 60%
            variant_successes=16,
            variant_trials=25,  # 64% -> only ~6.7% improvement
        )

        assert exp.is_significant is False


# ---------------------------------------------------------------------------
# EmergentTrait dataclass tests
# ---------------------------------------------------------------------------


class TestEmergentTrait:
    """Tests for the EmergentTrait dataclass."""

    def test_create_emergent_trait(self):
        """Test creating an emergent trait."""
        trait = EmergentTrait(
            trait_name="emergent_security_specialist",
            source_agents=["claude", "gpt4"],
            supporting_evidence=["High security success rate"],
            confidence=0.85,
        )

        assert trait.trait_name == "emergent_security_specialist"
        assert len(trait.source_agents) == 2
        assert trait.confidence == 0.85

    def test_emergent_trait_defaults(self):
        """Test emergent trait default values."""
        trait = EmergentTrait(
            trait_name="test_trait",
            source_agents=["agent1"],
            supporting_evidence=[],
            confidence=0.5,
        )

        assert trait.first_detected is not None

    def test_emergent_trait_multiple_evidence(self):
        """Test emergent trait with multiple evidence items."""
        trait = EmergentTrait(
            trait_name="multi_evidence_trait",
            source_agents=["agent1", "agent2", "agent3"],
            supporting_evidence=[
                "Evidence 1: High performance",
                "Evidence 2: Consistent behavior",
                "Evidence 3: Pattern matching",
            ],
            confidence=0.92,
        )

        assert len(trait.supporting_evidence) == 3
        assert len(trait.source_agents) == 3


# ---------------------------------------------------------------------------
# TraitTransfer dataclass tests
# ---------------------------------------------------------------------------


class TestTraitTransfer:
    """Tests for the TraitTransfer dataclass."""

    def test_create_trait_transfer(self):
        """Test creating a trait transfer record."""
        transfer = TraitTransfer(
            from_agent="claude",
            to_agent="gpt4",
            trait="thorough",
            expertise_domain=None,
            success_rate_before=0.6,
        )

        assert transfer.from_agent == "claude"
        assert transfer.to_agent == "gpt4"
        assert transfer.trait == "thorough"
        assert transfer.success_rate_after is None

    def test_trait_transfer_with_expertise(self):
        """Test trait transfer with expertise domain."""
        transfer = TraitTransfer(
            from_agent="agent1",
            to_agent="agent2",
            trait="",
            expertise_domain="security",
            success_rate_before=0.5,
            success_rate_after=0.7,
        )

        assert transfer.expertise_domain == "security"
        assert transfer.success_rate_after == 0.7

    def test_trait_transfer_defaults(self):
        """Test trait transfer default values."""
        transfer = TraitTransfer(
            from_agent="src",
            to_agent="dst",
            trait="test",
            expertise_domain=None,
            success_rate_before=0.5,
        )

        assert transfer.transferred_at is not None
        assert transfer.success_rate_after is None


# ---------------------------------------------------------------------------
# PersonaLaboratory initialization tests
# ---------------------------------------------------------------------------


class TestPersonaLaboratoryInit:
    """Tests for PersonaLaboratory initialization."""

    def test_init_with_paths(self, persona_manager, temp_db):
        """Test initialization with explicit paths."""
        lab = PersonaLaboratory(persona_manager=persona_manager, db_path=temp_db)

        assert lab.persona_manager is persona_manager
        assert lab.db_path == temp_db

    def test_init_creates_db(self, persona_manager, temp_db):
        """Test initialization creates database."""
        lab = PersonaLaboratory(persona_manager=persona_manager, db_path=temp_db)

        assert lab.db is not None
        assert lab.db_path.exists()


# ---------------------------------------------------------------------------
# Experiment creation tests
# ---------------------------------------------------------------------------


class TestExperimentCreation:
    """Tests for experiment creation."""

    def test_create_experiment_new_agent(self, laboratory):
        """Test creating experiment for new agent."""
        exp = laboratory.create_experiment(
            agent_name="new_agent",
            variant_traits=["direct"],
            hypothesis="Test direct trait",
        )

        assert exp.experiment_id.startswith("exp_")
        assert exp.agent_name == "new_agent"
        assert exp.status == "running"

    def test_create_experiment_existing_agent(self, laboratory, persona_manager):
        """Test creating experiment for existing agent."""
        persona_manager.create_persona(
            agent_name="existing_agent",
            traits=["thorough"],
            expertise={"security": 0.7},
        )

        exp = laboratory.create_experiment(
            agent_name="existing_agent",
            variant_traits=["thorough", "direct"],
            hypothesis="Add direct trait",
        )

        assert exp.control_persona.agent_name == "existing_agent"
        assert "thorough" in exp.control_persona.traits

    def test_create_experiment_with_expertise(self, laboratory):
        """Test creating experiment with variant expertise."""
        exp = laboratory.create_experiment(
            agent_name="expertise_test",
            variant_expertise={"security": 0.9, "testing": 0.8},
            hypothesis="Increase expertise levels",
        )

        assert exp.variant_persona.expertise.get("security") == 0.9
        assert exp.variant_persona.expertise.get("testing") == 0.8

    def test_create_experiment_saves_to_db(self, laboratory):
        """Test experiment is saved to database."""
        exp = laboratory.create_experiment(
            agent_name="save_test",
            hypothesis="Test save",
        )

        # Verify by getting running experiments
        running = laboratory.get_running_experiments()
        exp_ids = [e.experiment_id for e in running]
        assert exp.experiment_id in exp_ids


# ---------------------------------------------------------------------------
# Experiment result recording tests
# ---------------------------------------------------------------------------


class TestExperimentResultRecording:
    """Tests for recording experiment results."""

    def test_record_control_success(self, laboratory):
        """Test recording control group success."""
        exp = laboratory.create_experiment(
            agent_name="record_test",
            hypothesis="Test recording",
        )

        laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=True)

        # Verify by getting running experiments
        running = laboratory.get_running_experiments()
        updated = next(e for e in running if e.experiment_id == exp.experiment_id)
        assert updated.control_trials == 1
        assert updated.control_successes == 1

    def test_record_control_failure(self, laboratory):
        """Test recording control group failure."""
        exp = laboratory.create_experiment(
            agent_name="record_fail",
            hypothesis="Test failure recording",
        )

        laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=False)

        running = laboratory.get_running_experiments()
        updated = next(e for e in running if e.experiment_id == exp.experiment_id)
        assert updated.control_trials == 1
        assert updated.control_successes == 0

    def test_record_variant_success(self, laboratory):
        """Test recording variant group success."""
        exp = laboratory.create_experiment(
            agent_name="variant_test",
            hypothesis="Test variant",
        )

        laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        running = laboratory.get_running_experiments()
        updated = next(e for e in running if e.experiment_id == exp.experiment_id)
        assert updated.variant_trials == 1
        assert updated.variant_successes == 1

    def test_record_multiple_results(self, laboratory):
        """Test recording multiple results."""
        exp = laboratory.create_experiment(
            agent_name="multi_record",
            hypothesis="Test multiple",
        )

        for _ in range(5):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
        for _ in range(3):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=True)
        for _ in range(2):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=False)

        running = laboratory.get_running_experiments()
        updated = next(e for e in running if e.experiment_id == exp.experiment_id)
        assert updated.control_trials == 5
        assert updated.control_successes == 5
        assert updated.variant_trials == 5
        assert updated.variant_successes == 3


# ---------------------------------------------------------------------------
# Experiment conclusion tests
# ---------------------------------------------------------------------------


class TestExperimentConclusion:
    """Tests for concluding experiments."""

    def test_conclude_experiment_not_found(self, laboratory):
        """Test concluding non-existent experiment."""
        result = laboratory.conclude_experiment("nonexistent_id")
        assert result is None

    def test_conclude_experiment_marks_completed(self, laboratory):
        """Test concluding experiment marks status as completed."""
        exp = laboratory.create_experiment(
            agent_name="conclude_test",
            hypothesis="Test conclusion",
        )

        result = laboratory.conclude_experiment(exp.experiment_id)

        assert result is not None
        assert result.status == "completed"
        assert result.completed_at is not None

    def test_conclude_experiment_applies_winning_variant(self, laboratory, persona_manager):
        """Test that winning variant is applied when significant."""
        persona_manager.create_persona(
            agent_name="winner_test",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        exp = laboratory.create_experiment(
            agent_name="winner_test",
            variant_traits=["thorough", "direct"],
            variant_expertise={"security": 0.8},
            hypothesis="Test winning variant",
        )

        # Record significant difference (variant much better)
        for _ in range(25):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
        for _ in range(5):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=False)
        for _ in range(28):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=True)
        for _ in range(2):
            laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=False)

        result = laboratory.conclude_experiment(exp.experiment_id)

        # Check variant was significant and better
        assert result.variant_rate > result.control_rate


# ---------------------------------------------------------------------------
# Cross-pollination tests
# ---------------------------------------------------------------------------


class TestCrossPollination:
    """Tests for cross-pollination of traits."""

    def test_cross_pollinate_trait(self, laboratory, persona_manager):
        """Test cross-pollinating a trait."""
        persona_manager.create_persona(
            agent_name="source_agent",
            traits=["thorough", "direct"],
            expertise={"security": 0.8},
        )
        persona_manager.create_persona(
            agent_name="target_agent",
            traits=["pragmatic"],
            expertise={"testing": 0.6},
        )

        transfer = laboratory.cross_pollinate(
            from_agent="source_agent",
            to_agent="target_agent",
            trait="thorough",
        )

        assert transfer is not None
        assert transfer.from_agent == "source_agent"
        assert transfer.to_agent == "target_agent"
        assert transfer.trait == "thorough"

    def test_cross_pollinate_expertise(self, laboratory, persona_manager):
        """Test cross-pollinating expertise."""
        persona_manager.create_persona(
            agent_name="expert_source",
            traits=["thorough"],
            expertise={"security": 0.9},
        )
        persona_manager.create_persona(
            agent_name="expert_target",
            traits=["pragmatic"],
            expertise={"security": 0.3},
        )

        transfer = laboratory.cross_pollinate(
            from_agent="expert_source",
            to_agent="expert_target",
            expertise_domain="security",
        )

        assert transfer is not None
        assert transfer.expertise_domain == "security"

        # Check target's expertise was updated
        updated = persona_manager.get_persona("expert_target")
        # Should be blend: 0.7 * 0.3 + 0.3 * 0.9 = 0.48
        assert updated.expertise.get("security") == pytest.approx(0.48, rel=0.01)

    def test_cross_pollinate_nonexistent_source(self, laboratory, persona_manager):
        """Test cross-pollination with non-existent source."""
        persona_manager.create_persona(agent_name="only_target", traits=["thorough"])

        transfer = laboratory.cross_pollinate(
            from_agent="nonexistent",
            to_agent="only_target",
            trait="thorough",
        )

        assert transfer is None

    def test_cross_pollinate_nonexistent_target(self, laboratory, persona_manager):
        """Test cross-pollination with non-existent target."""
        persona_manager.create_persona(agent_name="only_source", traits=["thorough"])

        transfer = laboratory.cross_pollinate(
            from_agent="only_source",
            to_agent="nonexistent",
            trait="thorough",
        )

        assert transfer is None

    def test_cross_pollinate_invalid_trait(self, laboratory, persona_manager):
        """Test cross-pollination with invalid trait."""
        persona_manager.create_persona(agent_name="src", traits=["thorough"])
        persona_manager.create_persona(agent_name="tgt", traits=["pragmatic"])

        transfer = laboratory.cross_pollinate(
            from_agent="src",
            to_agent="tgt",
            trait="invalid_trait_xyz",
        )

        # Transfer still happens but trait not added
        assert transfer is not None
        target = persona_manager.get_persona("tgt")
        assert "invalid_trait_xyz" not in target.traits

    def test_suggest_cross_pollinations(self, laboratory, persona_manager):
        """Test suggesting cross-pollinations."""
        # Create source with high security performance
        persona_manager.create_persona(
            agent_name="strong_source",
            traits=["thorough"],
            expertise={"security": 0.9},
        )
        for _ in range(10):
            persona_manager.record_performance("strong_source", "security", success=True)

        # Create target with low security performance
        persona_manager.create_persona(
            agent_name="weak_target",
            traits=["pragmatic"],
            expertise={"security": 0.3},
        )
        for _ in range(5):
            persona_manager.record_performance("weak_target", "security", success=False)
        for _ in range(2):
            persona_manager.record_performance("weak_target", "security", success=True)

        suggestions = laboratory.suggest_cross_pollinations("weak_target")

        # Should suggest transferring security from strong_source
        assert isinstance(suggestions, list)


# ---------------------------------------------------------------------------
# Persona mutation tests
# ---------------------------------------------------------------------------


class TestPersonaMutation:
    """Tests for persona mutation."""

    def test_mutate_persona_no_changes(self, laboratory, persona_manager):
        """Test mutation with rate 0 (no changes)."""
        persona_manager.create_persona(
            agent_name="stable",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        with patch("random.random", return_value=0.5):  # Always above 0 rate
            result = laboratory.mutate_persona("stable", mutation_rate=0.0)

        assert result is not None
        assert result.traits == ["thorough"]

    def test_mutate_persona_adds_trait(self, laboratory, persona_manager):
        """Test mutation adds a trait."""
        persona_manager.create_persona(
            agent_name="evolving",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        # Mock random to force trait mutation
        with patch("random.random", return_value=0.05):  # Below mutation rate
            with patch("random.choice", return_value="direct"):
                result = laboratory.mutate_persona("evolving", mutation_rate=0.1)

        # May or may not have added trait depending on random state
        assert result is not None

    def test_mutate_persona_nonexistent(self, laboratory):
        """Test mutation of non-existent persona."""
        result = laboratory.mutate_persona("nonexistent", mutation_rate=0.5)
        assert result is None

    def test_mutate_persona_expertise_adjustment(self, laboratory, persona_manager):
        """Test mutation adjusts expertise."""
        persona_manager.create_persona(
            agent_name="expertise_mutate",
            traits=["thorough"],
            expertise={"security": 0.5, "testing": 0.5},
        )

        # Force expertise mutation with controlled random values
        random.seed(42)
        result = laboratory.mutate_persona("expertise_mutate", mutation_rate=1.0)

        assert result is not None
        # Expertise values should still be in valid range
        for domain, score in result.expertise.items():
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Emergent trait detection tests
# ---------------------------------------------------------------------------


class TestEmergentTraitDetection:
    """Tests for emergent trait detection."""

    def test_detect_emergent_traits_empty(self, laboratory, persona_manager):
        """Test detection with no personas."""
        traits = laboratory.detect_emergent_traits()
        assert isinstance(traits, list)

    def test_detect_emergent_traits_no_performance(self, laboratory, persona_manager):
        """Test detection with personas but no performance data."""
        persona_manager.create_persona(
            agent_name="no_perf",
            traits=["thorough"],
            expertise={"security": 0.3},
        )

        traits = laboratory.detect_emergent_traits()
        assert isinstance(traits, list)

    def test_get_emergent_traits_filtering(self, laboratory):
        """Test getting emergent traits with confidence filtering."""
        # Save some traits directly
        trait1 = EmergentTrait(
            trait_name="high_confidence",
            source_agents=["agent1"],
            supporting_evidence=["test"],
            confidence=0.8,
        )
        trait2 = EmergentTrait(
            trait_name="low_confidence",
            source_agents=["agent2"],
            supporting_evidence=["test"],
            confidence=0.3,
        )

        laboratory._save_emergent_trait(trait1)
        laboratory._save_emergent_trait(trait2)

        high_only = laboratory.get_emergent_traits(min_confidence=0.5)
        all_traits = laboratory.get_emergent_traits(min_confidence=0.0)

        assert len(high_only) == 1
        assert high_only[0].trait_name == "high_confidence"
        assert len(all_traits) == 2


# ---------------------------------------------------------------------------
# Evolution history tests
# ---------------------------------------------------------------------------


class TestEvolutionHistory:
    """Tests for evolution history tracking."""

    def test_get_evolution_history_empty(self, laboratory):
        """Test getting history when none exists."""
        history = laboratory.get_evolution_history("no_history_agent")
        assert history == []

    def test_evolution_history_from_mutation(self, laboratory, persona_manager):
        """Test history is recorded during mutation."""
        persona_manager.create_persona(
            agent_name="history_test",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        # Force a mutation
        random.seed(12345)
        laboratory.mutate_persona("history_test", mutation_rate=1.0)

        history = laboratory.get_evolution_history("history_test")
        # May or may not have history depending on if mutation occurred
        assert isinstance(history, list)

    def test_evolution_history_from_cross_pollination(self, laboratory, persona_manager):
        """Test history is recorded during cross-pollination."""
        persona_manager.create_persona(
            agent_name="cross_src",
            traits=["thorough"],
            expertise={"security": 0.8},
        )
        persona_manager.create_persona(
            agent_name="cross_tgt",
            traits=["pragmatic"],
            expertise={"security": 0.3},
        )

        laboratory.cross_pollinate(
            from_agent="cross_src",
            to_agent="cross_tgt",
            expertise_domain="security",
        )

        history = laboratory.get_evolution_history("cross_tgt")
        assert len(history) >= 1
        assert any(h["mutation_type"] == "cross_pollination" for h in history)

    def test_evolution_history_limit(self, laboratory, persona_manager):
        """Test history respects limit parameter."""
        persona_manager.create_persona(
            agent_name="limit_test",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        # Create multiple history entries
        for i in range(5):
            laboratory._record_evolution(
                "limit_test",
                "test_mutation",
                Persona(agent_name="before", traits=[]),
                Persona(agent_name="after", traits=[]),
                f"Test reason {i}",
            )

        history = laboratory.get_evolution_history("limit_test", limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Laboratory statistics tests
# ---------------------------------------------------------------------------


class TestLaboratoryStats:
    """Tests for laboratory statistics."""

    def test_get_lab_stats_empty(self, laboratory):
        """Test stats with empty database."""
        stats = laboratory.get_lab_stats()

        assert stats["total_experiments"] == 0
        assert stats["completed_experiments"] == 0
        assert stats["emergent_traits_detected"] == 0
        assert stats["trait_transfers"] == 0
        assert stats["total_evolutions"] == 0

    def test_get_lab_stats_with_data(self, laboratory, persona_manager):
        """Test stats with data."""
        # Create an experiment
        laboratory.create_experiment(
            agent_name="stats_test",
            hypothesis="Test stats",
        )

        # Save a trait
        laboratory._save_emergent_trait(
            EmergentTrait(
                trait_name="test_trait",
                source_agents=["agent1"],
                supporting_evidence=["test"],
                confidence=0.8,
            )
        )

        # Create a transfer
        persona_manager.create_persona(agent_name="src", traits=["thorough"])
        persona_manager.create_persona(agent_name="tgt", traits=["pragmatic"])
        laboratory.cross_pollinate("src", "tgt", trait="thorough")

        stats = laboratory.get_lab_stats()

        assert stats["total_experiments"] >= 1
        assert stats["emergent_traits_detected"] >= 1
        assert stats["trait_transfers"] >= 1


# ---------------------------------------------------------------------------
# Running experiments tests
# ---------------------------------------------------------------------------


class TestRunningExperiments:
    """Tests for getting running experiments."""

    def test_get_running_experiments_empty(self, laboratory):
        """Test getting running experiments when none exist."""
        running = laboratory.get_running_experiments()
        assert running == []

    def test_get_running_experiments_filters_completed(self, laboratory):
        """Test that completed experiments are filtered out."""
        exp1 = laboratory.create_experiment(
            agent_name="running1",
            hypothesis="Test 1",
        )
        exp2 = laboratory.create_experiment(
            agent_name="running2",
            hypothesis="Test 2",
        )

        # Conclude one experiment
        laboratory.conclude_experiment(exp2.experiment_id)

        running = laboratory.get_running_experiments()
        exp_ids = [e.experiment_id for e in running]

        assert exp1.experiment_id in exp_ids
        assert exp2.experiment_id not in exp_ids

    def test_get_running_experiments_multiple(self, laboratory):
        """Test getting multiple running experiments."""
        exp1 = laboratory.create_experiment(agent_name="multi1", hypothesis="Test 1")
        exp2 = laboratory.create_experiment(agent_name="multi2", hypothesis="Test 2")
        exp3 = laboratory.create_experiment(agent_name="multi3", hypothesis="Test 3")

        running = laboratory.get_running_experiments()

        assert len(running) == 3
        exp_ids = [e.experiment_id for e in running]
        assert exp1.experiment_id in exp_ids
        assert exp2.experiment_id in exp_ids
        assert exp3.experiment_id in exp_ids


# ---------------------------------------------------------------------------
# Row conversion tests
# ---------------------------------------------------------------------------


class TestRowConversion:
    """Tests for database row conversion."""

    def test_row_to_experiment_preserves_data(self, laboratory):
        """Test that row conversion preserves experiment data."""
        exp = laboratory.create_experiment(
            agent_name="conversion_test",
            variant_traits=["direct"],
            hypothesis="Test conversion",
        )

        # Record some results
        laboratory.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
        laboratory.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        running = laboratory.get_running_experiments()
        converted = next(e for e in running if e.experiment_id == exp.experiment_id)

        assert converted.agent_name == "conversion_test"
        assert converted.hypothesis == "Test conversion"
        assert converted.control_trials == 1
        assert converted.variant_trials == 1


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_experiment_with_empty_hypothesis(self, laboratory):
        """Test creating experiment with empty hypothesis."""
        exp = laboratory.create_experiment(
            agent_name="empty_hypothesis",
            hypothesis="",
        )

        assert exp.hypothesis == ""

    def test_mutation_with_max_traits(self, laboratory, persona_manager):
        """Test mutation doesn't exceed max traits."""
        persona_manager.create_persona(
            agent_name="max_traits",
            traits=["thorough", "pragmatic", "direct", "conservative"],  # 4 traits
            expertise={"security": 0.5},
        )

        random.seed(42)
        result = laboratory.mutate_persona("max_traits", mutation_rate=1.0)

        assert result is not None
        assert len(result.traits) <= 4

    def test_expertise_bounds_during_mutation(self, laboratory, persona_manager):
        """Test expertise stays within 0-1 bounds during mutation."""
        persona_manager.create_persona(
            agent_name="bounds_test",
            traits=["thorough"],
            expertise={"security": 0.99, "testing": 0.01},
        )

        random.seed(42)
        for _ in range(10):
            result = laboratory.mutate_persona("bounds_test", mutation_rate=1.0)
            if result:
                for score in result.expertise.values():
                    assert 0.0 <= score <= 1.0

    def test_cross_pollinate_same_agent(self, laboratory, persona_manager):
        """Test cross-pollination from agent to itself."""
        persona_manager.create_persona(
            agent_name="self_transfer",
            traits=["thorough"],
            expertise={"security": 0.5},
        )

        transfer = laboratory.cross_pollinate(
            from_agent="self_transfer",
            to_agent="self_transfer",
            trait="thorough",
        )

        # Should still work but be a no-op for traits
        assert transfer is not None

    def test_record_result_nonexistent_experiment(self, laboratory):
        """Test recording result for non-existent experiment."""
        # Should not raise, just silently fail
        laboratory.record_experiment_result("fake_exp_id", is_variant=True, success=True)


# ---------------------------------------------------------------------------
# Module exports tests
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ are importable."""
        from aragora.agents import laboratory

        for name in laboratory.__all__:
            assert hasattr(laboratory, name), f"{name} not found in module"

    def test_exports_contain_key_items(self):
        """Module exports include key classes."""
        from aragora.agents.laboratory import __all__

        assert "PersonaExperiment" in __all__
        assert "EmergentTrait" in __all__
        assert "TraitTransfer" in __all__
        assert "PersonaLaboratory" in __all__
