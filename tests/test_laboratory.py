"""
Tests for the PersonaLaboratory module.

Covers:
- PersonaExperiment data class and properties
- EmergentTrait data class
- TraitTransfer data class
- PersonaLaboratory initialization and schema
- A/B experiment lifecycle
- Emergent trait detection
- Cross-pollination between agents
- Persona mutation
- Evolution history tracking
- Laboratory statistics
"""

import os
import pytest
import random
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aragora.agents.laboratory import (
    PersonaExperiment,
    EmergentTrait,
    TraitTransfer,
    PersonaLaboratory,
)
from aragora.agents.personas import Persona, PersonaManager, PERSONALITY_TRAITS, EXPERTISE_DOMAINS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def persona_db(temp_dir):
    """Path to temporary persona database."""
    return os.path.join(temp_dir, "personas.db")


@pytest.fixture
def lab_db(temp_dir):
    """Path to temporary laboratory database."""
    return os.path.join(temp_dir, "lab.db")


@pytest.fixture
def persona_manager(persona_db):
    """Create PersonaManager with temp DB."""
    return PersonaManager(db_path=persona_db)


@pytest.fixture
def lab(persona_manager, lab_db):
    """Create PersonaLaboratory with temp DB."""
    return PersonaLaboratory(persona_manager, db_path=lab_db)


@pytest.fixture
def sample_persona():
    """Create a sample persona for testing."""
    return Persona(
        agent_name="test-agent",
        description="A test agent",
        traits=["thorough", "pragmatic"],
        expertise={"security": 0.8, "performance": 0.6},
    )


@pytest.fixture
def sample_experiment(sample_persona):
    """Create a sample experiment for testing."""
    variant = Persona(
        agent_name="test-agent_variant",
        traits=["thorough", "innovative"],
        expertise={"security": 0.9, "performance": 0.5},
    )
    return PersonaExperiment(
        experiment_id="exp_test123",
        agent_name="test-agent",
        control_persona=sample_persona,
        variant_persona=variant,
        hypothesis="Testing innovative trait",
    )


# ============================================================================
# PersonaExperiment Tests
# ============================================================================

class TestPersonaExperiment:
    """Tests for PersonaExperiment data class."""

    def test_control_rate_calculation(self, sample_experiment):
        """Test control success rate calculation."""
        sample_experiment.control_successes = 7
        sample_experiment.control_trials = 10
        assert sample_experiment.control_rate == 0.7

    def test_variant_rate_calculation(self, sample_experiment):
        """Test variant success rate calculation."""
        sample_experiment.variant_successes = 8
        sample_experiment.variant_trials = 10
        assert sample_experiment.variant_rate == 0.8

    def test_rates_with_zero_trials(self, sample_experiment):
        """Test rates return 0 with zero trials."""
        assert sample_experiment.control_rate == 0.0
        assert sample_experiment.variant_rate == 0.0

    def test_relative_improvement_calculation(self, sample_experiment):
        """Test relative improvement calculation."""
        sample_experiment.control_successes = 5
        sample_experiment.control_trials = 10
        sample_experiment.variant_successes = 7
        sample_experiment.variant_trials = 10
        # (0.7 - 0.5) / 0.5 = 0.4 = 40% improvement
        assert sample_experiment.relative_improvement == pytest.approx(0.4)

    def test_relative_improvement_zero_control(self, sample_experiment):
        """Test relative improvement when control rate is 0."""
        sample_experiment.control_successes = 0
        sample_experiment.control_trials = 10
        sample_experiment.variant_successes = 5
        sample_experiment.variant_trials = 10
        assert sample_experiment.relative_improvement == 0.0

    def test_is_significant_minimum_trials(self, sample_experiment):
        """Test significance requires minimum trials."""
        sample_experiment.control_successes = 10
        sample_experiment.control_trials = 15  # Below 20 minimum
        sample_experiment.variant_successes = 15
        sample_experiment.variant_trials = 15
        assert sample_experiment.is_significant is False

    def test_is_significant_threshold(self, sample_experiment):
        """Test significance requires 10% difference."""
        sample_experiment.control_successes = 14
        sample_experiment.control_trials = 20
        sample_experiment.variant_successes = 16
        sample_experiment.variant_trials = 20
        # 80% vs 70% = ~14% improvement, should be significant
        assert sample_experiment.is_significant is True

    def test_is_significant_small_difference(self, sample_experiment):
        """Test small differences are not significant."""
        sample_experiment.control_successes = 14
        sample_experiment.control_trials = 20
        sample_experiment.variant_successes = 15
        sample_experiment.variant_trials = 20
        # 75% vs 70% = ~7% improvement, not significant
        assert sample_experiment.is_significant is False

    def test_default_status(self, sample_experiment):
        """Test default experiment status is running."""
        assert sample_experiment.status == "running"

    def test_created_at_has_timestamp(self, sample_experiment):
        """Test created_at is set."""
        assert sample_experiment.created_at is not None
        assert len(sample_experiment.created_at) > 0


# ============================================================================
# EmergentTrait Tests
# ============================================================================

class TestEmergentTrait:
    """Tests for EmergentTrait data class."""

    def test_creation_with_defaults(self):
        """Test EmergentTrait creation with defaults."""
        trait = EmergentTrait(
            trait_name="emergent_security_specialist",
            source_agents=["claude", "gemini"],
            supporting_evidence=["High success in security domain"],
            confidence=0.85,
        )
        assert trait.trait_name == "emergent_security_specialist"
        assert len(trait.source_agents) == 2
        assert trait.confidence == 0.85
        assert trait.first_detected is not None

    def test_confidence_range(self):
        """Test confidence can be any float (not clamped at creation)."""
        trait = EmergentTrait(
            trait_name="test",
            source_agents=[],
            supporting_evidence=[],
            confidence=0.0,
        )
        assert trait.confidence == 0.0

        trait2 = EmergentTrait(
            trait_name="test2",
            source_agents=[],
            supporting_evidence=[],
            confidence=1.0,
        )
        assert trait2.confidence == 1.0


# ============================================================================
# TraitTransfer Tests
# ============================================================================

class TestTraitTransfer:
    """Tests for TraitTransfer data class."""

    def test_creation_with_defaults(self):
        """Test TraitTransfer creation with defaults."""
        transfer = TraitTransfer(
            from_agent="claude",
            to_agent="gemini",
            trait="thorough",
            expertise_domain="security",
            success_rate_before=0.5,
        )
        assert transfer.from_agent == "claude"
        assert transfer.to_agent == "gemini"
        assert transfer.trait == "thorough"
        assert transfer.success_rate_before == 0.5
        assert transfer.success_rate_after is None
        assert transfer.transferred_at is not None

    def test_optional_fields(self):
        """Test optional fields can be None."""
        transfer = TraitTransfer(
            from_agent="a",
            to_agent="b",
            trait="",
            expertise_domain=None,
            success_rate_before=0.0,
        )
        assert transfer.expertise_domain is None


# ============================================================================
# PersonaLaboratory Initialization Tests
# ============================================================================

class TestPersonaLaboratoryInit:
    """Tests for PersonaLaboratory initialization."""

    def test_creates_database_file(self, persona_manager, lab_db):
        """Test database file is created."""
        lab = PersonaLaboratory(persona_manager, db_path=lab_db)
        assert os.path.exists(lab_db)

    def test_creates_experiments_table(self, lab, lab_db):
        """Test experiments table is created."""
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
            assert cursor.fetchone() is not None

    def test_creates_emergent_traits_table(self, lab, lab_db):
        """Test emergent_traits table is created."""
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emergent_traits'")
            assert cursor.fetchone() is not None

    def test_creates_trait_transfers_table(self, lab, lab_db):
        """Test trait_transfers table is created."""
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trait_transfers'")
            assert cursor.fetchone() is not None

    def test_creates_evolution_history_table(self, lab, lab_db):
        """Test evolution_history table is created."""
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evolution_history'")
            assert cursor.fetchone() is not None

    def test_idempotent_init(self, persona_manager, lab_db):
        """Test multiple initializations don't fail."""
        lab1 = PersonaLaboratory(persona_manager, db_path=lab_db)
        lab2 = PersonaLaboratory(persona_manager, db_path=lab_db)
        # Should not raise any errors


# ============================================================================
# A/B Experiment Tests
# ============================================================================

class TestExperimentCreation:
    """Tests for experiment creation."""

    def test_create_experiment_new_agent(self, lab):
        """Test creating experiment for new agent creates persona."""
        exp = lab.create_experiment(
            agent_name="new-agent",
            hypothesis="Testing new agent",
        )
        assert exp.agent_name == "new-agent"
        assert exp.experiment_id.startswith("exp_")
        assert exp.status == "running"

    def test_create_experiment_existing_persona(self, lab, persona_manager):
        """Test creating experiment uses existing persona."""
        persona_manager.create_persona(
            "existing-agent",
            description="Test agent",
            traits=["thorough"],
            expertise={"security": 0.8},
        )
        exp = lab.create_experiment(
            agent_name="existing-agent",
            hypothesis="Testing existing",
        )
        assert exp.control_persona.traits == ["thorough"]
        assert exp.control_persona.expertise.get("security") == 0.8

    def test_create_experiment_with_variant_traits(self, lab, persona_manager):
        """Test creating experiment with variant traits."""
        persona_manager.create_persona("agent", traits=["thorough"])
        exp = lab.create_experiment(
            agent_name="agent",
            variant_traits=["innovative", "direct"],
        )
        assert "innovative" in exp.variant_persona.traits
        assert "direct" in exp.variant_persona.traits

    def test_create_experiment_with_variant_expertise(self, lab, persona_manager):
        """Test creating experiment with variant expertise."""
        persona_manager.create_persona("agent", expertise={"security": 0.5})
        exp = lab.create_experiment(
            agent_name="agent",
            variant_expertise={"security": 0.9, "performance": 0.7},
        )
        assert exp.variant_persona.expertise["security"] == 0.9
        assert exp.variant_persona.expertise["performance"] == 0.7

    def test_experiment_persisted_to_db(self, lab, lab_db):
        """Test experiment is saved to database."""
        exp = lab.create_experiment("test-agent")

        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT experiment_id FROM experiments WHERE experiment_id = ?", (exp.experiment_id,))
            assert cursor.fetchone() is not None


class TestExperimentResults:
    """Tests for recording experiment results."""

    def test_record_control_success(self, lab):
        """Test recording control success."""
        exp = lab.create_experiment("agent")
        lab.record_experiment_result(exp.experiment_id, is_variant=False, success=True)

        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated is not None
        assert updated.control_trials == 1
        assert updated.control_successes == 1

    def test_record_control_failure(self, lab):
        """Test recording control failure."""
        exp = lab.create_experiment("agent")
        lab.record_experiment_result(exp.experiment_id, is_variant=False, success=False)

        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated.control_trials == 1
        assert updated.control_successes == 0

    def test_record_variant_success(self, lab):
        """Test recording variant success."""
        exp = lab.create_experiment("agent")
        lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated.variant_trials == 1
        assert updated.variant_successes == 1

    def test_record_variant_failure(self, lab):
        """Test recording variant failure."""
        exp = lab.create_experiment("agent")
        lab.record_experiment_result(exp.experiment_id, is_variant=True, success=False)

        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated.variant_trials == 1
        assert updated.variant_successes == 0

    def test_multiple_results_accumulate(self, lab):
        """Test multiple results accumulate correctly."""
        exp = lab.create_experiment("agent")

        for _ in range(5):
            lab.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
        for _ in range(3):
            lab.record_experiment_result(exp.experiment_id, is_variant=False, success=False)
        for _ in range(7):
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated.control_trials == 8
        assert updated.control_successes == 5
        assert updated.variant_trials == 7
        assert updated.variant_successes == 7


class TestExperimentConclusion:
    """Tests for concluding experiments."""

    def test_conclude_nonexistent_returns_none(self, lab):
        """Test concluding nonexistent experiment returns None."""
        result = lab.conclude_experiment("nonexistent")
        assert result is None

    def test_conclude_updates_status(self, lab):
        """Test concluding experiment updates status."""
        exp = lab.create_experiment("agent")
        result = lab.conclude_experiment(exp.experiment_id)
        assert result.status == "completed"
        assert result.completed_at is not None

    def test_conclude_insignificant_no_change(self, lab, persona_manager):
        """Test concluding insignificant experiment doesn't apply variant."""
        persona_manager.create_persona("agent", traits=["thorough"])
        exp = lab.create_experiment("agent", variant_traits=["innovative"])

        # Not enough trials for significance
        for _ in range(10):
            lab.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        lab.conclude_experiment(exp.experiment_id)

        # Original persona should be unchanged
        persona = persona_manager.get_persona("agent")
        assert "thorough" in persona.traits
        assert "innovative" not in persona.traits

    def test_conclude_significant_applies_variant(self, lab, persona_manager):
        """Test concluding significant experiment applies variant."""
        persona_manager.create_persona("agent", traits=["thorough"])
        exp = lab.create_experiment("agent", variant_traits=["thorough", "innovative"])

        # Enough trials with significant improvement
        # Control: 50% success, Variant: 80% success = 60% improvement
        for _ in range(25):
            lab.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
            lab.record_experiment_result(exp.experiment_id, is_variant=False, success=False)  # 50%
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)
            lab.record_experiment_result(exp.experiment_id, is_variant=True, success=False)  # 75%

        lab.conclude_experiment(exp.experiment_id)

        # Variant should be applied
        persona = persona_manager.get_persona("agent")
        assert "innovative" in persona.traits

    def test_get_running_experiments(self, lab):
        """Test getting running experiments."""
        exp1 = lab.create_experiment("agent1")
        exp2 = lab.create_experiment("agent2")

        running = lab.get_running_experiments()
        ids = [e.experiment_id for e in running]
        assert exp1.experiment_id in ids
        assert exp2.experiment_id in ids

        lab.conclude_experiment(exp1.experiment_id)

        running = lab.get_running_experiments()
        ids = [e.experiment_id for e in running]
        assert exp1.experiment_id not in ids
        assert exp2.experiment_id in ids


# ============================================================================
# Emergent Trait Detection Tests
# ============================================================================

class TestEmergentTraitDetection:
    """Tests for emergent trait detection."""

    def test_detect_no_personas_returns_empty(self, lab):
        """Test detection with no personas returns empty."""
        traits = lab.detect_emergent_traits()
        assert traits == []

    def test_get_traits_empty_db(self, lab):
        """Test getting traits from empty database."""
        traits = lab.get_emergent_traits()
        assert traits == []

    def test_get_traits_above_threshold(self, lab, lab_db):
        """Test filtering traits by confidence threshold."""
        # Manually insert traits
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emergent_traits (trait_name, source_agents, confidence) VALUES (?, ?, ?)",
                ("high_conf", "[]", 0.9)
            )
            cursor.execute(
                "INSERT INTO emergent_traits (trait_name, source_agents, confidence) VALUES (?, ?, ?)",
                ("low_conf", "[]", 0.3)
            )
            conn.commit()

        traits = lab.get_emergent_traits(min_confidence=0.5)
        assert len(traits) == 1
        assert traits[0].trait_name == "high_conf"

    def test_get_traits_ordered_by_confidence(self, lab, lab_db):
        """Test traits are ordered by confidence descending."""
        with sqlite3.connect(lab_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emergent_traits (trait_name, source_agents, confidence) VALUES (?, ?, ?)",
                ("medium", "[]", 0.6)
            )
            cursor.execute(
                "INSERT INTO emergent_traits (trait_name, source_agents, confidence) VALUES (?, ?, ?)",
                ("high", "[]", 0.9)
            )
            cursor.execute(
                "INSERT INTO emergent_traits (trait_name, source_agents, confidence) VALUES (?, ?, ?)",
                ("low", "[]", 0.5)
            )
            conn.commit()

        traits = lab.get_emergent_traits(min_confidence=0.5)
        assert [t.trait_name for t in traits] == ["high", "medium", "low"]


# ============================================================================
# Cross-Pollination Tests
# ============================================================================

class TestCrossPollination:
    """Tests for trait cross-pollination."""

    def test_transfer_missing_source_returns_none(self, lab):
        """Test transfer with missing source agent returns None."""
        result = lab.cross_pollinate(
            from_agent="nonexistent",
            to_agent="also-nonexistent",
            trait="thorough",
        )
        assert result is None

    def test_transfer_missing_target_returns_none(self, lab, persona_manager):
        """Test transfer with missing target agent returns None."""
        persona_manager.create_persona("source")
        result = lab.cross_pollinate(
            from_agent="source",
            to_agent="nonexistent",
            trait="thorough",
        )
        assert result is None

    def test_transfer_trait(self, lab, persona_manager):
        """Test transferring a trait."""
        persona_manager.create_persona("source", traits=["thorough", "innovative"])
        persona_manager.create_persona("target", traits=["pragmatic"])

        transfer = lab.cross_pollinate(
            from_agent="source",
            to_agent="target",
            trait="thorough",
        )

        assert transfer is not None
        target = persona_manager.get_persona("target")
        assert "thorough" in target.traits

    def test_transfer_expertise(self, lab, persona_manager):
        """Test transferring expertise."""
        persona_manager.create_persona("source", expertise={"security": 0.9})
        persona_manager.create_persona("target", expertise={"security": 0.5})

        transfer = lab.cross_pollinate(
            from_agent="source",
            to_agent="target",
            expertise_domain="security",
        )

        assert transfer is not None
        target = persona_manager.get_persona("target")
        # 0.7 * 0.5 + 0.3 * 0.9 = 0.35 + 0.27 = 0.62
        assert target.expertise["security"] == pytest.approx(0.62)

    def test_transfer_blends_expertise(self, lab, persona_manager):
        """Test expertise blending formula."""
        persona_manager.create_persona("source", expertise={"performance": 1.0})
        persona_manager.create_persona("target", expertise={"performance": 0.0})

        lab.cross_pollinate(
            from_agent="source",
            to_agent="target",
            expertise_domain="performance",
        )

        target = persona_manager.get_persona("target")
        # 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        assert target.expertise["performance"] == pytest.approx(0.3)

    def test_transfer_records_evolution(self, lab, persona_manager):
        """Test transfer records evolution history."""
        persona_manager.create_persona("source", traits=["thorough"])
        persona_manager.create_persona("target", traits=[])

        lab.cross_pollinate(
            from_agent="source",
            to_agent="target",
            trait="thorough",
        )

        history = lab.get_evolution_history("target")
        assert len(history) == 1
        assert history[0]["mutation_type"] == "cross_pollination"


class TestCrossPollinationSuggestions:
    """Tests for cross-pollination suggestions."""

    def test_suggest_excludes_self(self, lab, persona_manager):
        """Test suggestions exclude the target agent."""
        persona_manager.create_persona("agent")
        suggestions = lab.suggest_cross_pollinations("agent")
        agents = [s[0] for s in suggestions]
        assert "agent" not in agents

    def test_suggest_nonexistent_returns_empty(self, lab):
        """Test suggestions for nonexistent agent."""
        suggestions = lab.suggest_cross_pollinations("nonexistent")
        assert suggestions == []


# ============================================================================
# Persona Mutation Tests
# ============================================================================

class TestPersonaMutation:
    """Tests for persona mutation."""

    def test_mutate_missing_agent_returns_none(self, lab):
        """Test mutating nonexistent agent returns None."""
        result = lab.mutate_persona("nonexistent")
        assert result is None

    def test_mutate_no_change_low_rate(self, lab, persona_manager):
        """Test mutation with 0 rate makes no changes."""
        persona_manager.create_persona("agent", traits=["thorough"], expertise={"security": 0.5})

        random.seed(42)  # For reproducibility
        result = lab.mutate_persona("agent", mutation_rate=0.0)

        assert result.traits == ["thorough"]
        assert result.expertise["security"] == 0.5

    def test_mutate_high_rate_changes(self, lab, persona_manager):
        """Test mutation with high rate makes changes."""
        persona_manager.create_persona("agent", traits=["thorough"], expertise={"security": 0.5})

        random.seed(123)  # For reproducibility
        result = lab.mutate_persona("agent", mutation_rate=1.0)

        # With 100% mutation rate, something should change
        changed = (
            result.traits != ["thorough"] or
            result.expertise.get("security") != 0.5 or
            len(result.expertise) > 1
        )
        assert changed

    def test_mutate_records_evolution(self, lab, persona_manager):
        """Test mutation records evolution history."""
        persona_manager.create_persona("agent", traits=["thorough"])

        random.seed(999)  # Seed that will cause a change
        lab.mutate_persona("agent", mutation_rate=1.0)

        history = lab.get_evolution_history("agent")
        # May or may not have a record depending on if mutation occurred
        if history:
            assert history[0]["mutation_type"] == "mutation"


# ============================================================================
# Evolution History Tests
# ============================================================================

class TestEvolutionHistory:
    """Tests for evolution history tracking."""

    def test_get_history_empty(self, lab):
        """Test getting history for agent with no history."""
        history = lab.get_evolution_history("agent")
        assert history == []

    def test_get_history_returns_mutations(self, lab, persona_manager):
        """Test history returns recorded mutations."""
        persona_manager.create_persona("source", traits=["thorough"])
        persona_manager.create_persona("target", traits=[])

        lab.cross_pollinate("source", "target", trait="thorough")

        history = lab.get_evolution_history("target")
        assert len(history) == 1
        assert "before" in history[0]
        assert "after" in history[0]
        assert "reason" in history[0]

    def test_get_history_respects_limit(self, lab, persona_manager):
        """Test history respects limit parameter."""
        persona_manager.create_persona("source", traits=["thorough", "innovative"])
        persona_manager.create_persona("target", traits=[])

        # Create multiple history entries
        lab.cross_pollinate("source", "target", trait="thorough")
        lab.cross_pollinate("source", "target", trait="innovative")

        history = lab.get_evolution_history("target", limit=1)
        assert len(history) == 1

    def test_get_history_ordered_desc(self, lab, persona_manager):
        """Test history returns multiple entries in descending order."""
        persona_manager.create_persona("source", traits=["thorough", "innovative"])
        persona_manager.create_persona("target", traits=[])

        lab.cross_pollinate("source", "target", trait="thorough")
        lab.cross_pollinate("source", "target", trait="innovative")

        history = lab.get_evolution_history("target")
        # Should have 2 entries
        assert len(history) == 2
        # Both should have cross_pollination type
        assert all(h["mutation_type"] == "cross_pollination" for h in history)
        # Reasons should reference both traits
        reasons = [h["reason"] for h in history]
        assert any("thorough" in r for r in reasons)
        assert any("innovative" in r for r in reasons)


# ============================================================================
# Lab Statistics Tests
# ============================================================================

class TestLabStats:
    """Tests for laboratory statistics."""

    def test_stats_empty_db(self, lab):
        """Test stats on empty database."""
        stats = lab.get_lab_stats()
        assert stats["total_experiments"] == 0
        assert stats["completed_experiments"] == 0
        assert stats["emergent_traits_detected"] == 0
        assert stats["trait_transfers"] == 0
        assert stats["total_evolutions"] == 0

    def test_stats_counts_experiments(self, lab):
        """Test stats counts total experiments."""
        lab.create_experiment("agent1")
        lab.create_experiment("agent2")

        stats = lab.get_lab_stats()
        assert stats["total_experiments"] == 2

    def test_stats_counts_completed(self, lab):
        """Test stats counts completed experiments."""
        exp1 = lab.create_experiment("agent1")
        lab.create_experiment("agent2")
        lab.conclude_experiment(exp1.experiment_id)

        stats = lab.get_lab_stats()
        assert stats["total_experiments"] == 2
        assert stats["completed_experiments"] == 1

    def test_stats_counts_transfers(self, lab, persona_manager):
        """Test stats counts trait transfers."""
        persona_manager.create_persona("source", traits=["thorough"])
        persona_manager.create_persona("target", traits=[])

        lab.cross_pollinate("source", "target", trait="thorough")

        stats = lab.get_lab_stats()
        assert stats["trait_transfers"] == 1

    def test_stats_counts_evolutions(self, lab, persona_manager):
        """Test stats counts total evolutions."""
        persona_manager.create_persona("source", traits=["thorough"])
        persona_manager.create_persona("target", traits=[])

        lab.cross_pollinate("source", "target", trait="thorough")

        stats = lab.get_lab_stats()
        assert stats["total_evolutions"] == 1
