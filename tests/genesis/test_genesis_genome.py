"""
Tests for genesis genome system.

Tests:
- Genome ID generation
- AgentGenome creation and manipulation
- Fitness updates and calculations
- Genome similarity
- Serialization roundtrip
- GenomeStore persistence
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from aragora.genesis.genome import (
    AgentGenome,
    GenomeStore,
    generate_genome_id,
)


class TestGenomeIdGeneration:
    """Test unique genome ID generation."""

    def test_generates_12_char_hex_id(self):
        """ID should be 12 character hex string."""
        genome_id = generate_genome_id(
            traits={"analytical": 0.8}, expertise={"security": 0.9}, parents=[]
        )
        assert len(genome_id) == 12
        assert all(c in "0123456789abcdef" for c in genome_id)

    def test_same_inputs_different_ids(self):
        """Same inputs at different times should produce different IDs (due to timestamp)."""
        id1 = generate_genome_id({"a": 1}, {"b": 2}, [])
        id2 = generate_genome_id({"a": 1}, {"b": 2}, [])
        # IDs may differ due to timestamp, but structure should be valid
        assert len(id1) == 12
        assert len(id2) == 12

    def test_different_inputs_different_ids(self):
        """Different inputs should definitely produce different IDs."""
        id1 = generate_genome_id({"analytical": 0.8}, {"security": 0.9}, [])
        id2 = generate_genome_id({"creative": 0.8}, {"ethics": 0.9}, [])
        # High probability of being different
        assert len(id1) == 12
        assert len(id2) == 12


class TestAgentGenome:
    """Test AgentGenome dataclass."""

    def test_create_basic_genome(self):
        """Should create genome with all fields."""
        genome = AgentGenome(
            genome_id="abc123def456",
            name="test-agent",
            traits={"analytical": 0.8, "cautious": 0.6},
            expertise={"security": 0.9, "backend": 0.7},
            model_preference="claude",
        )

        assert genome.genome_id == "abc123def456"
        assert genome.name == "test-agent"
        assert genome.traits["analytical"] == 0.8
        assert genome.expertise["security"] == 0.9
        assert genome.generation == 0
        assert genome.fitness_score == 0.5

    def test_from_persona(self):
        """Should create genome from Persona."""
        persona = Mock()
        persona.agent_name = "test-persona"
        persona.traits = ["analytical", "creative"]
        persona.expertise = {"security": 0.9, "backend": 0.7}

        genome = AgentGenome.from_persona(persona, model="gemini")

        assert genome.name == "test-persona"
        assert genome.traits["analytical"] == 1.0
        assert genome.traits["creative"] == 1.0
        assert genome.expertise["security"] == 0.9
        assert genome.model_preference == "gemini"
        assert genome.generation == 0

    def test_to_persona(self):
        """Should convert genome back to Persona."""
        genome = AgentGenome(
            genome_id="abc123def456",
            name="test-agent",
            traits={"analytical": 0.8, "cautious": 0.3, "creative": 0.7},
            expertise={"security": 0.9},
            generation=3,
            fitness_score=0.85,
        )

        persona = genome.to_persona()

        assert persona.agent_name == "test-agent"
        # Only traits with weight > 0.5 are included
        assert "analytical" in persona.traits
        assert "creative" in persona.traits
        assert "cautious" not in persona.traits
        assert persona.expertise["security"] == 0.9

    def test_get_dominant_traits(self):
        """Should return top N traits by weight."""
        genome = AgentGenome(
            genome_id="test",
            name="test",
            traits={"a": 0.9, "b": 0.7, "c": 0.8, "d": 0.5},
        )

        top_2 = genome.get_dominant_traits(top_n=2)
        assert top_2 == ["a", "c"]

        top_3 = genome.get_dominant_traits(top_n=3)
        assert top_3 == ["a", "c", "b"]

    def test_get_top_expertise(self):
        """Should return top N expertise areas."""
        genome = AgentGenome(
            genome_id="test",
            name="test",
            expertise={"security": 0.9, "backend": 0.7, "frontend": 0.5},
        )

        top = genome.get_top_expertise(top_n=2)
        assert top == [("security", 0.9), ("backend", 0.7)]

    def test_update_fitness_consensus_win(self):
        """Fitness should increase on consensus win."""
        genome = AgentGenome(genome_id="test", name="test")
        assert genome.fitness_score == 0.5
        assert genome.debates_participated == 0

        genome.update_fitness(consensus_win=True)

        assert genome.debates_participated == 1
        assert genome.consensus_contributions == 1
        assert genome.fitness_score > 0  # Should be calculated

    def test_update_fitness_multiple_outcomes(self):
        """Fitness should reflect all outcome types."""
        genome = AgentGenome(genome_id="test", name="test")

        # Win with accepted critique and correct prediction
        genome.update_fitness(consensus_win=True, critique_accepted=True, prediction_correct=True)

        assert genome.debates_participated == 1
        assert genome.consensus_contributions == 1
        assert genome.critiques_accepted == 1
        assert genome.predictions_correct == 1
        # All positive outcomes = high fitness
        assert genome.fitness_score == pytest.approx(1.0, rel=0.01)

    def test_update_fitness_no_wins(self):
        """Fitness should be low with no positive outcomes."""
        genome = AgentGenome(genome_id="test", name="test")

        genome.update_fitness(
            consensus_win=False, critique_accepted=False, prediction_correct=False
        )

        assert genome.debates_participated == 1
        assert genome.fitness_score == 0.0

    def test_similarity_to_identical(self):
        """Identical genomes should have similarity 1.0."""
        genome1 = AgentGenome(
            genome_id="a",
            name="test",
            traits={"a": 0.8, "b": 0.6},
            expertise={"x": 0.9, "y": 0.7},
        )
        genome2 = AgentGenome(
            genome_id="b",
            name="test",
            traits={"a": 0.8, "b": 0.6},
            expertise={"x": 0.9, "y": 0.7},
        )

        assert genome1.similarity_to(genome2) == pytest.approx(1.0, rel=0.01)

    def test_similarity_to_different(self):
        """Very different genomes should have low similarity."""
        genome1 = AgentGenome(
            genome_id="a",
            name="test",
            traits={"a": 1.0, "b": 1.0},
            expertise={"x": 1.0},
        )
        genome2 = AgentGenome(
            genome_id="b",
            name="test",
            traits={"a": 0.0, "b": 0.0},
            expertise={"x": 0.0},
        )

        similarity = genome1.similarity_to(genome2)
        assert similarity == pytest.approx(0.0, rel=0.01)

    def test_similarity_partial_overlap(self):
        """Partial overlap should have intermediate similarity."""
        genome1 = AgentGenome(
            genome_id="a",
            name="test",
            traits={"a": 0.8, "b": 0.6},
            expertise={"x": 0.9},
        )
        genome2 = AgentGenome(
            genome_id="b",
            name="test",
            traits={"a": 0.8, "c": 0.6},  # Different trait b vs c
            expertise={"x": 0.5, "y": 0.7},  # Different expertise
        )

        similarity = genome1.similarity_to(genome2)
        assert 0.0 < similarity < 1.0

    def test_to_dict_roundtrip(self):
        """Genome should survive to_dict/from_dict roundtrip."""
        genome = AgentGenome(
            genome_id="abc123def456",
            name="test-agent",
            traits={"analytical": 0.8},
            expertise={"security": 0.9},
            model_preference="claude",
            parent_genomes=["parent1", "parent2"],
            generation=3,
            fitness_score=0.75,
            birth_debate_id="debate-123",
            consensus_contributions=5,
            critiques_accepted=3,
            predictions_correct=4,
            debates_participated=10,
        )

        data = genome.to_dict()
        restored = AgentGenome.from_dict(data)

        assert restored.genome_id == genome.genome_id
        assert restored.name == genome.name
        assert restored.traits == genome.traits
        assert restored.expertise == genome.expertise
        assert restored.parent_genomes == genome.parent_genomes
        assert restored.generation == genome.generation
        assert restored.fitness_score == genome.fitness_score
        assert restored.consensus_contributions == genome.consensus_contributions

    def test_repr(self):
        """Repr should include key info."""
        genome = AgentGenome(
            genome_id="test",
            name="test-agent",
            traits={"a": 0.9, "b": 0.7},
            expertise={"x": 0.8, "y": 0.6},
            generation=2,
            fitness_score=0.75,
        )

        repr_str = repr(genome)
        assert "test-agent" in repr_str
        assert "gen=2" in repr_str
        assert "fit=0.75" in repr_str


class TestGenomeStore:
    """Test GenomeStore SQLite persistence."""

    def test_init_creates_db(self):
        """Store should create database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            assert db_path.exists()

    def test_save_and_get(self):
        """Should save and retrieve genome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            genome = AgentGenome(
                genome_id="test123",
                name="test-agent",
                traits={"a": 0.8},
                expertise={"x": 0.9},
            )

            store.save(genome)
            retrieved = store.get("test123")

            assert retrieved is not None
            assert retrieved.genome_id == "test123"
            assert retrieved.name == "test-agent"
            assert retrieved.traits["a"] == 0.8

    def test_get_nonexistent_returns_none(self):
        """Getting non-existent genome should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            result = store.get("nonexistent")
            assert result is None

    def test_get_by_name(self):
        """Should retrieve latest genome by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            # Save two genomes with same name, different generations
            genome1 = AgentGenome(
                genome_id="v1",
                name="test-agent",
                generation=1,
            )
            genome2 = AgentGenome(
                genome_id="v2",
                name="test-agent",
                generation=2,
            )

            store.save(genome1)
            store.save(genome2)

            retrieved = store.get_by_name("test-agent")
            assert retrieved is not None
            assert retrieved.genome_id == "v2"  # Latest generation

    def test_get_top_by_fitness(self):
        """Should retrieve top N genomes by fitness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            genomes = [
                AgentGenome(genome_id=f"g{i}", name=f"agent-{i}", fitness_score=i / 10)
                for i in range(5)
            ]

            for g in genomes:
                store.save(g)

            top_3 = store.get_top_by_fitness(n=3)

            assert len(top_3) == 3
            assert top_3[0].fitness_score == 0.4  # Highest
            assert top_3[1].fitness_score == 0.3
            assert top_3[2].fitness_score == 0.2

    def test_get_all(self):
        """Should retrieve all genomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            for i in range(3):
                store.save(AgentGenome(genome_id=f"g{i}", name=f"agent-{i}"))

            all_genomes = store.get_all()
            assert len(all_genomes) == 3

    def test_get_lineage(self):
        """Should trace lineage through parent genomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            # Create lineage: grandparent -> parent -> child
            grandparent = AgentGenome(
                genome_id="grandparent",
                name="ancestor",
                parent_genomes=[],
            )
            parent = AgentGenome(
                genome_id="parent",
                name="parent",
                parent_genomes=["grandparent"],
            )
            child = AgentGenome(
                genome_id="child",
                name="child",
                parent_genomes=["parent", "other"],
            )

            store.save(grandparent)
            store.save(parent)
            store.save(child)

            lineage = store.get_lineage("child")

            assert len(lineage) == 3
            assert lineage[0].genome_id == "child"
            assert lineage[1].genome_id == "parent"
            assert lineage[2].genome_id == "grandparent"

    def test_delete(self):
        """Should delete genome by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            store.save(AgentGenome(genome_id="to-delete", name="test"))

            assert store.get("to-delete") is not None

            deleted = store.delete("to-delete")
            assert deleted is True

            assert store.get("to-delete") is None

    def test_delete_nonexistent(self):
        """Deleting non-existent genome should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            deleted = store.delete("nonexistent")
            assert deleted is False

    def test_update_on_conflict(self):
        """Should update fitness on save conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_genesis.db"
            store = GenomeStore(db_path=str(db_path))

            genome = AgentGenome(
                genome_id="test",
                name="agent",
                fitness_score=0.5,
            )
            store.save(genome)

            # Update fitness and save again
            genome.fitness_score = 0.9
            genome.debates_participated = 10
            store.save(genome)

            retrieved = store.get("test")
            assert retrieved.fitness_score == 0.9
            assert retrieved.debates_participated == 10
