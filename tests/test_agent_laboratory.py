"""
Tests for aragora.agents.laboratory module.

Tests PersonaExperiment, EmergentTrait, TraitTransfer, and PersonaLaboratory
for A/B testing and emergent persona evolution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.laboratory import (
    PersonaExperiment,
    EmergentTrait,
    TraitTransfer,
    PersonaLaboratory,
)
from aragora.agents.personas import Persona


class TestPersonaExperiment:
    """Tests for PersonaExperiment dataclass."""

    @pytest.fixture
    def sample_personas(self):
        """Create sample control and variant personas."""
        control = Persona(
            agent_name="claude",
            traits=["thorough", "conservative"],
            expertise={"ethics": 0.8},
        )
        variant = Persona(
            agent_name="claude",
            traits=["thorough", "innovative"],
            expertise={"ethics": 0.8},
        )
        return control, variant

    def test_experiment_creation(self, sample_personas):
        """Test creating an experiment."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Bold trait improves debate performance",
        )

        assert exp.experiment_id == "exp-001"
        assert exp.status == "running"
        assert exp.control_trials == 0
        assert exp.variant_trials == 0

    def test_control_rate(self, sample_personas):
        """Test control success rate calculation."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            control_successes=8,
            control_trials=10,
        )
        assert exp.control_rate == 0.8

    def test_control_rate_zero_trials(self, sample_personas):
        """Test control rate with zero trials."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
        )
        assert exp.control_rate == 0.0

    def test_variant_rate(self, sample_personas):
        """Test variant success rate calculation."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            variant_successes=9,
            variant_trials=10,
        )
        assert exp.variant_rate == 0.9

    def test_relative_improvement(self, sample_personas):
        """Test relative improvement calculation."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            control_successes=8,
            control_trials=10,  # 80%
            variant_successes=9,
            variant_trials=10,  # 90%
        )
        # (0.9 - 0.8) / 0.8 = 0.125
        assert exp.relative_improvement == pytest.approx(0.125)

    def test_relative_improvement_zero_control(self, sample_personas):
        """Test relative improvement when control rate is zero."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            control_successes=0,
            control_trials=10,
        )
        assert exp.relative_improvement == 0.0

    def test_is_significant_true(self, sample_personas):
        """Test significance detection with enough trials."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            control_successes=16,
            control_trials=25,  # 64%
            variant_successes=20,
            variant_trials=25,  # 80%
        )
        # 25% improvement > 10% threshold
        assert exp.is_significant is True

    def test_is_significant_insufficient_trials(self, sample_personas):
        """Test significance with insufficient trials."""
        control, variant = sample_personas
        exp = PersonaExperiment(
            experiment_id="exp-001",
            agent_name="claude",
            control_persona=control,
            variant_persona=variant,
            hypothesis="Test",
            control_successes=5,
            control_trials=10,  # < 20 min trials
            variant_successes=8,
            variant_trials=10,
        )
        assert exp.is_significant is False


class TestEmergentTrait:
    """Tests for EmergentTrait dataclass."""

    def test_trait_creation(self):
        """Test creating an emergent trait."""
        trait = EmergentTrait(
            trait_name="systematic_skeptic",
            source_agents=["claude", "gemini"],
            supporting_evidence=[
                "High critique acceptance rate",
                "Consistent position reversals on new evidence",
            ],
            confidence=0.85,
        )

        assert trait.trait_name == "systematic_skeptic"
        assert len(trait.source_agents) == 2
        assert trait.confidence == 0.85

    def test_trait_timestamps(self):
        """Test that trait has automatic timestamp."""
        trait = EmergentTrait(
            trait_name="test_trait",
            source_agents=["claude"],
            supporting_evidence=["evidence"],
            confidence=0.5,
        )
        assert trait.first_detected is not None


class TestTraitTransfer:
    """Tests for TraitTransfer dataclass."""

    def test_transfer_creation(self):
        """Test creating a trait transfer record."""
        transfer = TraitTransfer(
            from_agent="claude",
            to_agent="gemini",
            trait="thorough",
            expertise_domain="security",
            success_rate_before=0.6,
            success_rate_after=0.75,
        )

        assert transfer.from_agent == "claude"
        assert transfer.to_agent == "gemini"
        assert transfer.trait == "thorough"
        assert transfer.expertise_domain == "security"


class TestPersonaLaboratory:
    """Tests for PersonaLaboratory class.

    These tests verify the PersonaLaboratory API for A/B testing personas.
    """

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_persona_db(self):
        """Create a temporary persona database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def persona_manager(self, temp_persona_db):
        """Create a PersonaManager with temp database."""
        from aragora.agents.personas import PersonaManager
        return PersonaManager(temp_persona_db)

    @pytest.fixture
    def lab(self, persona_manager, temp_db):
        """Create a PersonaLaboratory with temp database."""
        return PersonaLaboratory(persona_manager, temp_db)

    def test_lab_initialization(self, lab):
        """Test lab initializes correctly."""
        assert lab is not None
        assert lab.persona_manager is not None

    def test_create_experiment_creates_new_persona(self, lab):
        """Test creating an experiment for a new agent."""
        exp = lab.create_experiment(
            agent_name="new-agent",
            variant_traits=["innovative", "direct"],
            hypothesis="Testing innovation trait",
        )

        assert exp is not None
        assert exp.agent_name == "new-agent"
        assert exp.status == "running"
        assert "innovative" in exp.variant_persona.traits

    def test_get_running_experiments(self, lab):
        """Test getting running experiments."""
        # Create experiment
        exp = lab.create_experiment(
            agent_name="test-agent",
            hypothesis="Test hypothesis",
        )

        running = lab.get_running_experiments()
        assert len(running) >= 1
        assert any(e.experiment_id == exp.experiment_id for e in running)

    def test_record_experiment_result(self, lab):
        """Test recording experiment trial results."""
        exp = lab.create_experiment(
            agent_name="trial-agent",
            hypothesis="Testing trials",
        )

        # Record control and variant results
        lab.record_experiment_result(exp.experiment_id, is_variant=False, success=True)
        lab.record_experiment_result(exp.experiment_id, is_variant=True, success=True)

        # Verify by getting running experiments
        running = lab.get_running_experiments()
        updated = next((e for e in running if e.experiment_id == exp.experiment_id), None)
        assert updated is not None
        assert updated.control_trials == 1
        assert updated.variant_trials == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
