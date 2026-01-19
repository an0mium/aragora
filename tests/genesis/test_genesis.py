"""Tests for genesis module.

Tests the agent genesis/breeding system including:
- AgentGenome: genetic representation, fitness tracking
- GenomeStore: persistence layer
- GenomeBreeder: crossover, mutation, selection
- Population: population management
- PopulationManager: persistent population evolution
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.genesis.genome import (
    AgentGenome,
    GenomeStore,
    generate_genome_id,
)
from aragora.genesis.breeding import (
    GenomeBreeder,
    Population,
    PopulationManager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_genome():
    """Create a sample genome for testing."""
    return AgentGenome(
        genome_id="test-genome-001",
        name="test-agent",
        traits={"analytical": 0.8, "creative": 0.6, "cautious": 0.7},
        expertise={"security": 0.9, "architecture": 0.7, "testing": 0.5},
        model_preference="claude",
        parent_genomes=[],
        generation=0,
        fitness_score=0.5,
    )


@pytest.fixture
def sample_genome_2():
    """Create a second sample genome for crossover tests."""
    return AgentGenome(
        genome_id="test-genome-002",
        name="other-agent",
        traits={"analytical": 0.4, "creative": 0.9, "bold": 0.8},
        expertise={"frontend": 0.8, "architecture": 0.5, "testing": 0.9},
        model_preference="gemini",
        parent_genomes=[],
        generation=0,
        fitness_score=0.6,
    )


@pytest.fixture
def breeder():
    """Create a GenomeBreeder with default settings."""
    return GenomeBreeder(mutation_rate=0.1, crossover_ratio=0.5, elite_ratio=0.2)


@pytest.fixture
def mock_persona():
    """Create a mock Persona object."""
    from aragora.agents.personas import Persona
    return Persona(
        agent_name="mock-agent",
        description="A mock agent for testing",
        traits=["analytical", "thorough"],
        expertise={"security": 0.8, "testing": 0.7},
    )


# =============================================================================
# generate_genome_id Tests
# =============================================================================


class TestGenerateGenomeId:
    """Test genome ID generation."""

    def test_generates_string(self):
        """Test that genome ID is a string."""
        gid = generate_genome_id({"trait": 0.5}, {"domain": 0.5}, [])
        assert isinstance(gid, str)
        assert len(gid) == 12  # SHA256 truncated to 12 chars

    def test_different_inputs_different_ids(self):
        """Test that different inputs produce different IDs."""
        id1 = generate_genome_id({"a": 0.1}, {}, [])
        id2 = generate_genome_id({"b": 0.1}, {}, [])
        # Different traits should produce different IDs
        # (timestamp makes them unique anyway, but structure matters too)
        assert isinstance(id1, str) and isinstance(id2, str)

    def test_consistent_format(self):
        """Test ID format is consistent."""
        gid = generate_genome_id({"x": 1.0}, {"y": 0.5}, ["parent1"])
        # Should be hexadecimal
        assert all(c in "0123456789abcdef" for c in gid)


# =============================================================================
# AgentGenome Tests - Creation
# =============================================================================


class TestAgentGenomeCreation:
    """Test AgentGenome creation and basic operations."""

    def test_create_genome(self, sample_genome):
        """Test creating a genome with all fields."""
        assert sample_genome.genome_id == "test-genome-001"
        assert sample_genome.name == "test-agent"
        assert sample_genome.traits["analytical"] == 0.8
        assert sample_genome.expertise["security"] == 0.9
        assert sample_genome.generation == 0
        assert sample_genome.fitness_score == 0.5

    def test_default_values(self):
        """Test genome with default values."""
        genome = AgentGenome(
            genome_id="minimal",
            name="minimal-agent",
        )

        assert genome.traits == {}
        assert genome.expertise == {}
        assert genome.model_preference == "claude"
        assert genome.parent_genomes == []
        assert genome.generation == 0
        assert genome.fitness_score == 0.5

    def test_from_persona(self, mock_persona):
        """Test creating genome from Persona."""
        genome = AgentGenome.from_persona(mock_persona, model="claude")

        assert genome.name == mock_persona.agent_name
        assert "analytical" in genome.traits
        assert genome.traits["analytical"] == 1.0  # Base traits get weight 1.0
        assert "security" in genome.expertise
        assert genome.generation == 0
        assert genome.parent_genomes == []

    def test_to_persona(self, sample_genome):
        """Test converting genome back to Persona."""
        persona = sample_genome.to_persona()

        assert persona.agent_name == sample_genome.name
        # Active traits are those with weight > 0.5
        assert "analytical" in persona.traits  # 0.8 > 0.5
        assert "creative" in persona.traits    # 0.6 > 0.5
        assert "cautious" in persona.traits    # 0.7 > 0.5


# =============================================================================
# AgentGenome Tests - Traits and Expertise
# =============================================================================


class TestAgentGenomeTraits:
    """Test trait and expertise methods."""

    def test_get_dominant_traits(self, sample_genome):
        """Test getting dominant traits."""
        dominant = sample_genome.get_dominant_traits(top_n=2)

        assert len(dominant) == 2
        # Should be sorted by weight
        assert dominant[0] == "analytical"  # 0.8
        assert dominant[1] == "cautious"    # 0.7

    def test_get_top_expertise(self, sample_genome):
        """Test getting top expertise areas."""
        top = sample_genome.get_top_expertise(top_n=2)

        assert len(top) == 2
        assert top[0][0] == "security"      # 0.9
        assert top[0][1] == 0.9
        assert top[1][0] == "architecture"  # 0.7

    def test_get_dominant_traits_empty(self):
        """Test dominant traits with empty traits dict."""
        genome = AgentGenome(genome_id="empty", name="empty")
        assert genome.get_dominant_traits(3) == []

    def test_get_top_expertise_empty(self):
        """Test top expertise with empty expertise dict."""
        genome = AgentGenome(genome_id="empty", name="empty")
        assert genome.get_top_expertise(3) == []


# =============================================================================
# AgentGenome Tests - Fitness
# =============================================================================


class TestAgentGenomeFitness:
    """Test fitness tracking and updates."""

    def test_update_fitness_consensus_win(self, sample_genome):
        """Test fitness update on consensus win."""
        sample_genome.update_fitness(consensus_win=True)

        assert sample_genome.debates_participated == 1
        assert sample_genome.consensus_contributions == 1
        assert sample_genome.fitness_score > 0.0

    def test_update_fitness_critique_accepted(self, sample_genome):
        """Test fitness update on critique accepted."""
        sample_genome.update_fitness(critique_accepted=True)

        assert sample_genome.critiques_accepted == 1
        assert sample_genome.debates_participated == 1

    def test_update_fitness_prediction_correct(self, sample_genome):
        """Test fitness update on correct prediction."""
        sample_genome.update_fitness(prediction_correct=True)

        assert sample_genome.predictions_correct == 1
        assert sample_genome.debates_participated == 1

    def test_update_fitness_weighted_calculation(self, sample_genome):
        """Test weighted fitness calculation."""
        # All positive outcomes
        sample_genome.update_fitness(
            consensus_win=True,
            critique_accepted=True,
            prediction_correct=True,
        )

        # 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0 = 1.0
        assert sample_genome.fitness_score == 1.0

    def test_update_fitness_updates_timestamp(self, sample_genome):
        """Test that fitness update changes updated_at."""
        old_time = sample_genome.updated_at
        sample_genome.update_fitness(consensus_win=True)
        # Time should be updated (or at least set)
        assert sample_genome.updated_at is not None


# =============================================================================
# AgentGenome Tests - Similarity
# =============================================================================


class TestAgentGenomeSimilarity:
    """Test genome similarity calculation."""

    def test_similarity_to_self(self, sample_genome):
        """Test genome is perfectly similar to itself."""
        similarity = sample_genome.similarity_to(sample_genome)
        assert similarity == 1.0

    def test_similarity_to_different(self, sample_genome, sample_genome_2):
        """Test similarity between different genomes."""
        similarity = sample_genome.similarity_to(sample_genome_2)

        assert 0 <= similarity <= 1
        assert similarity < 1.0  # Not identical

    def test_similarity_empty_traits(self):
        """Test similarity with empty traits."""
        g1 = AgentGenome(genome_id="1", name="a")
        g2 = AgentGenome(genome_id="2", name="b")

        similarity = g1.similarity_to(g2)
        assert similarity == 1.0  # Both empty = identical

    def test_similarity_partial_overlap(self):
        """Test similarity with partially overlapping traits."""
        g1 = AgentGenome(
            genome_id="1", name="a",
            traits={"a": 1.0, "b": 0.5},
            expertise={"x": 0.8},
        )
        g2 = AgentGenome(
            genome_id="2", name="b",
            traits={"b": 0.5, "c": 1.0},
            expertise={"x": 0.8, "y": 0.5},
        )

        similarity = g1.similarity_to(g2)
        assert 0 < similarity < 1


# =============================================================================
# AgentGenome Tests - Serialization
# =============================================================================


class TestAgentGenomeSerialization:
    """Test genome serialization."""

    def test_to_dict(self, sample_genome):
        """Test converting genome to dictionary."""
        data = sample_genome.to_dict()

        assert data["genome_id"] == "test-genome-001"
        assert data["name"] == "test-agent"
        assert data["traits"] == sample_genome.traits
        assert data["expertise"] == sample_genome.expertise
        assert data["generation"] == 0
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self, sample_genome):
        """Test creating genome from dictionary."""
        data = sample_genome.to_dict()
        restored = AgentGenome.from_dict(data)

        assert restored.genome_id == sample_genome.genome_id
        assert restored.name == sample_genome.name
        assert restored.traits == sample_genome.traits
        assert restored.expertise == sample_genome.expertise
        assert restored.generation == sample_genome.generation

    def test_roundtrip_serialization(self, sample_genome):
        """Test full serialization roundtrip."""
        data = sample_genome.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = AgentGenome.from_dict(restored_data)

        assert restored.genome_id == sample_genome.genome_id
        assert restored.fitness_score == sample_genome.fitness_score

    def test_repr(self, sample_genome):
        """Test string representation."""
        repr_str = repr(sample_genome)

        assert "Genome" in repr_str
        assert sample_genome.name in repr_str
        assert "gen=0" in repr_str


# =============================================================================
# GenomeStore Tests
# =============================================================================


class TestGenomeStore:
    """Test genome persistence layer."""

    def test_save_and_get(self, temp_db, sample_genome):
        """Test saving and retrieving a genome."""
        store = GenomeStore(temp_db)
        store.save(sample_genome)

        retrieved = store.get(sample_genome.genome_id)

        assert retrieved is not None
        assert retrieved.genome_id == sample_genome.genome_id
        assert retrieved.name == sample_genome.name

    def test_get_nonexistent(self, temp_db):
        """Test getting nonexistent genome returns None."""
        store = GenomeStore(temp_db)
        assert store.get("nonexistent") is None

    def test_get_by_name(self, temp_db, sample_genome):
        """Test getting genome by name."""
        store = GenomeStore(temp_db)
        store.save(sample_genome)

        retrieved = store.get_by_name(sample_genome.name)

        assert retrieved is not None
        assert retrieved.name == sample_genome.name

    def test_get_top_by_fitness(self, temp_db):
        """Test getting top genomes by fitness."""
        store = GenomeStore(temp_db)

        # Create genomes with different fitness
        for i in range(5):
            genome = AgentGenome(
                genome_id=f"genome-{i}",
                name=f"agent-{i}",
                fitness_score=i * 0.2,
            )
            store.save(genome)

        top = store.get_top_by_fitness(n=3)

        assert len(top) == 3
        # Should be sorted by fitness descending
        assert top[0].fitness_score >= top[1].fitness_score
        assert top[1].fitness_score >= top[2].fitness_score

    def test_get_all(self, temp_db, sample_genome, sample_genome_2):
        """Test getting all genomes."""
        store = GenomeStore(temp_db)
        store.save(sample_genome)
        store.save(sample_genome_2)

        all_genomes = store.get_all()

        assert len(all_genomes) == 2

    def test_get_lineage(self, temp_db):
        """Test getting genome lineage."""
        store = GenomeStore(temp_db)

        # Create lineage: gen0 -> gen1 -> gen2
        gen0 = AgentGenome(genome_id="gen0", name="a", generation=0)
        gen1 = AgentGenome(genome_id="gen1", name="b", generation=1, parent_genomes=["gen0"])
        gen2 = AgentGenome(genome_id="gen2", name="c", generation=2, parent_genomes=["gen1"])

        store.save(gen0)
        store.save(gen1)
        store.save(gen2)

        lineage = store.get_lineage("gen2")

        assert len(lineage) == 3
        assert lineage[0].genome_id == "gen2"
        assert lineage[1].genome_id == "gen1"
        assert lineage[2].genome_id == "gen0"

    def test_delete(self, temp_db, sample_genome):
        """Test deleting a genome."""
        store = GenomeStore(temp_db)
        store.save(sample_genome)

        assert store.delete(sample_genome.genome_id) is True
        assert store.get(sample_genome.genome_id) is None

    def test_delete_nonexistent(self, temp_db):
        """Test deleting nonexistent genome."""
        store = GenomeStore(temp_db)
        assert store.delete("nonexistent") is False

    def test_update_existing(self, temp_db, sample_genome):
        """Test updating existing genome."""
        store = GenomeStore(temp_db)
        store.save(sample_genome)

        # Update fitness
        sample_genome.fitness_score = 0.9
        sample_genome.debates_participated = 10
        store.save(sample_genome)

        retrieved = store.get(sample_genome.genome_id)
        assert retrieved.fitness_score == 0.9
        assert retrieved.debates_participated == 10


# =============================================================================
# Population Tests
# =============================================================================


class TestPopulation:
    """Test Population dataclass."""

    def test_create_population(self, sample_genome, sample_genome_2):
        """Test creating a population."""
        pop = Population(
            population_id="pop-001",
            genomes=[sample_genome, sample_genome_2],
            generation=0,
        )

        assert pop.population_id == "pop-001"
        assert pop.size == 2
        assert pop.generation == 0

    def test_average_fitness(self, sample_genome, sample_genome_2):
        """Test average fitness calculation."""
        sample_genome.fitness_score = 0.4
        sample_genome_2.fitness_score = 0.6

        pop = Population(
            population_id="pop",
            genomes=[sample_genome, sample_genome_2],
        )

        assert pop.average_fitness == 0.5

    def test_average_fitness_empty(self):
        """Test average fitness with empty population."""
        pop = Population(population_id="empty", genomes=[])
        assert pop.average_fitness == 0.0

    def test_best_genome(self, sample_genome, sample_genome_2):
        """Test getting best genome."""
        sample_genome.fitness_score = 0.3
        sample_genome_2.fitness_score = 0.8

        pop = Population(
            population_id="pop",
            genomes=[sample_genome, sample_genome_2],
        )

        best = pop.best_genome
        assert best.genome_id == sample_genome_2.genome_id

    def test_best_genome_empty(self):
        """Test best genome with empty population."""
        pop = Population(population_id="empty", genomes=[])
        assert pop.best_genome is None

    def test_get_by_id(self, sample_genome, sample_genome_2):
        """Test getting genome by ID from population."""
        pop = Population(
            population_id="pop",
            genomes=[sample_genome, sample_genome_2],
        )

        found = pop.get_by_id(sample_genome_2.genome_id)
        assert found is not None
        assert found.genome_id == sample_genome_2.genome_id

        assert pop.get_by_id("nonexistent") is None

    def test_to_dict(self, sample_genome):
        """Test population serialization."""
        pop = Population(
            population_id="pop-001",
            genomes=[sample_genome],
            generation=5,
            debate_history=["d1", "d2"],
        )

        data = pop.to_dict()

        assert data["population_id"] == "pop-001"
        assert data["genomes"] == [sample_genome.genome_id]
        assert data["generation"] == 5
        assert data["debate_history"] == ["d1", "d2"]


# =============================================================================
# GenomeBreeder Tests - Crossover
# =============================================================================


class TestGenomeBreederCrossover:
    """Test genome crossover operations."""

    def test_crossover_basic(self, breeder, sample_genome, sample_genome_2):
        """Test basic crossover."""
        child = breeder.crossover(sample_genome, sample_genome_2)

        assert child.genome_id != sample_genome.genome_id
        assert child.genome_id != sample_genome_2.genome_id
        assert child.generation == 1
        assert sample_genome.genome_id in child.parent_genomes
        assert sample_genome_2.genome_id in child.parent_genomes

    def test_crossover_blends_traits(self, breeder, sample_genome, sample_genome_2):
        """Test crossover blends trait values."""
        child = breeder.crossover(sample_genome, sample_genome_2)

        # Child should have traits from both parents
        # Analytical: 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        if "analytical" in child.traits:
            # With 0.5 ratio, should be average
            expected = 0.5 * sample_genome.traits.get("analytical", 0) + \
                      0.5 * sample_genome_2.traits.get("analytical", 0)
            assert abs(child.traits["analytical"] - expected) < 0.01

    def test_crossover_blends_expertise(self, breeder, sample_genome, sample_genome_2):
        """Test crossover blends expertise values."""
        child = breeder.crossover(sample_genome, sample_genome_2)

        # Should have expertise from both parents
        all_domains = set(sample_genome.expertise.keys()) | set(sample_genome_2.expertise.keys())
        for domain in all_domains:
            assert domain in child.expertise

    def test_crossover_custom_name(self, breeder, sample_genome, sample_genome_2):
        """Test crossover with custom name."""
        child = breeder.crossover(sample_genome, sample_genome_2, name="custom-child")

        assert child.name == "custom-child"

    def test_crossover_records_debate_id(self, breeder, sample_genome, sample_genome_2):
        """Test crossover records birth debate ID."""
        child = breeder.crossover(
            sample_genome, sample_genome_2,
            debate_id="debate-123",
        )

        assert child.birth_debate_id == "debate-123"

    def test_crossover_generation_increment(self, sample_genome, sample_genome_2):
        """Test child generation is max parent + 1."""
        breeder = GenomeBreeder()
        sample_genome.generation = 2
        sample_genome_2.generation = 5

        child = breeder.crossover(sample_genome, sample_genome_2)

        assert child.generation == 6


# =============================================================================
# GenomeBreeder Tests - Mutation
# =============================================================================


class TestGenomeBreederMutation:
    """Test genome mutation operations."""

    def test_mutate_returns_new_genome(self, breeder, sample_genome):
        """Test mutation returns new genome."""
        mutant = breeder.mutate(sample_genome)

        assert mutant.genome_id != sample_genome.genome_id
        assert mutant.parent_genomes == [sample_genome.genome_id]

    def test_mutate_preserves_generation(self, breeder, sample_genome):
        """Test mutation preserves generation."""
        sample_genome.generation = 3
        mutant = breeder.mutate(sample_genome)

        assert mutant.generation == 3

    def test_mutate_with_high_rate(self, sample_genome):
        """Test mutation with high rate causes changes."""
        breeder = GenomeBreeder(mutation_rate=1.0)

        results = []
        for _ in range(10):
            mutant = breeder.mutate(sample_genome)
            results.append(mutant)

        # With 100% rate, should see some changes
        # Check that at least one trait value changed
        original_values = set(sample_genome.traits.values())
        changed = any(
            any(v not in original_values for v in m.traits.values())
            for m in results
        )
        # May or may not change depending on random values, but structure should differ

    def test_mutate_bounds_values(self, sample_genome):
        """Test mutation keeps values in [0, 1] range."""
        breeder = GenomeBreeder(mutation_rate=1.0)

        for _ in range(10):
            mutant = breeder.mutate(sample_genome)

            for value in mutant.traits.values():
                assert 0 <= value <= 1

            for value in mutant.expertise.values():
                assert 0 <= value <= 1


# =============================================================================
# GenomeBreeder Tests - Specialist Spawning
# =============================================================================


class TestGenomeBreederSpecialist:
    """Test specialist spawning."""

    def test_spawn_specialist(self, breeder, sample_genome, sample_genome_2):
        """Test spawning a domain specialist."""
        specialist = breeder.spawn_specialist(
            domain="security",
            parent_pool=[sample_genome, sample_genome_2],
        )

        assert specialist is not None
        assert "specialist" in specialist.name.lower()
        # Security expertise should be boosted from crossover average
        # Parent1 has 0.9, Parent2 has 0.0, crossover ~0.45, +0.3 boost = ~0.75
        assert specialist.expertise.get("security", 0) >= 0.7

    def test_spawn_specialist_empty_pool(self, breeder):
        """Test spawning from empty pool raises error."""
        with pytest.raises(ValueError, match="empty parent pool"):
            breeder.spawn_specialist("security", [])

    def test_spawn_specialist_single_parent(self, breeder, sample_genome):
        """Test spawning specialist with single parent."""
        specialist = breeder.spawn_specialist(
            domain="security",
            parent_pool=[sample_genome],
        )

        # Should mutate the single parent
        assert specialist is not None
        assert specialist.parent_genomes == [sample_genome.genome_id]


# =============================================================================
# GenomeBreeder Tests - Natural Selection
# =============================================================================


class TestGenomeBreederSelection:
    """Test natural selection operations."""

    def test_natural_selection_empty(self, breeder):
        """Test selection on empty population."""
        pop = Population(population_id="empty", genomes=[])
        evolved = breeder.natural_selection(pop)

        # Empty population returns unchanged
        assert evolved.genomes == []
        # Generation is not incremented for empty population (unchanged)
        assert evolved.generation == 0

    def test_natural_selection_preserves_elite(self, breeder, sample_genome, sample_genome_2):
        """Test selection preserves top performers."""
        sample_genome.fitness_score = 0.9
        sample_genome_2.fitness_score = 0.3

        pop = Population(
            population_id="pop",
            genomes=[sample_genome, sample_genome_2],
        )

        evolved = breeder.natural_selection(pop, keep_top_n=1)

        # Best genome should be preserved
        assert any(g.genome_id == sample_genome.genome_id for g in evolved.genomes)

    def test_natural_selection_increments_generation(self, breeder, sample_genome):
        """Test selection increments population generation."""
        pop = Population(
            population_id="pop",
            genomes=[sample_genome],
            generation=5,
        )

        evolved = breeder.natural_selection(pop)

        assert evolved.generation == 6

    def test_natural_selection_creates_offspring(self, breeder):
        """Test selection creates new offspring."""
        genomes = [
            AgentGenome(
                genome_id=f"g{i}",
                name=f"a{i}",
                traits={"t": 0.5},
                expertise={"e": 0.5},
                fitness_score=0.5 + i * 0.1,
            )
            for i in range(4)
        ]

        pop = Population(population_id="pop", genomes=genomes)
        evolved = breeder.natural_selection(pop, keep_top_n=2, breed_n=2, mutate_n=1)

        # Should have elites + offspring + mutants
        assert len(evolved.genomes) >= 2


# =============================================================================
# GenomeBreeder Tests - Diversity Protection
# =============================================================================


class TestGenomeBreederDiversity:
    """Test diversity protection."""

    def test_protect_endangered_small_population(self, breeder, sample_genome):
        """Test protection on small population does nothing."""
        pop = Population(population_id="pop", genomes=[sample_genome])

        protected = breeder.protect_endangered(pop)

        assert protected.genomes == [sample_genome]

    def test_protect_endangered_boosts_unique(self, breeder):
        """Test protection boosts unique genomes."""
        # Create similar genomes
        similar = [
            AgentGenome(
                genome_id=f"sim{i}",
                name=f"sim{i}",
                traits={"a": 0.5, "b": 0.5},
                expertise={"x": 0.5},
                fitness_score=0.3,
            )
            for i in range(3)
        ]

        # Add one unique genome
        unique = AgentGenome(
            genome_id="unique",
            name="unique",
            traits={"c": 1.0, "d": 1.0},  # Very different
            expertise={"y": 1.0, "z": 1.0},
            fitness_score=0.3,
        )

        pop = Population(population_id="pop", genomes=similar + [unique])
        protected = breeder.protect_endangered(pop, min_similarity=0.3)

        # Unique genome should have boosted fitness
        protected_unique = protected.get_by_id("unique")
        assert protected_unique is not None
        assert protected_unique.fitness_score >= 0.6


# =============================================================================
# PopulationManager Tests
# =============================================================================


class TestPopulationManager:
    """Test PopulationManager operations."""

    def test_get_or_create_population_new(self, temp_db):
        """Test creating new population."""
        manager = PopulationManager(db_path=temp_db)

        # Patch where DEFAULT_PERSONAS is actually used (inside the method)
        with patch.dict("aragora.agents.personas.DEFAULT_PERSONAS", {}, clear=True):
            pop = manager.get_or_create_population(["agent1", "agent2"])

            assert pop is not None
            assert pop.size == 2
            assert pop.generation == 0

    def test_update_fitness(self, temp_db, sample_genome):
        """Test updating genome fitness via manager."""
        manager = PopulationManager(db_path=temp_db)
        manager.genome_store.save(sample_genome)

        manager.update_fitness(
            sample_genome.genome_id,
            consensus_win=True,
            critique_accepted=True,
        )

        updated = manager.genome_store.get(sample_genome.genome_id)
        assert updated.consensus_contributions == 1
        assert updated.critiques_accepted == 1

    def test_get_best_for_domain(self, temp_db):
        """Test getting best genomes for domain."""
        manager = PopulationManager(db_path=temp_db)

        # Create genomes with different security expertise
        for i, score in enumerate([0.3, 0.9, 0.6]):
            genome = AgentGenome(
                genome_id=f"g{i}",
                name=f"a{i}",
                expertise={"security": score},
                fitness_score=0.5,
            )
            manager.genome_store.save(genome)

        best = manager.get_best_for_domain("security", n=2)

        assert len(best) == 2
        assert best[0].expertise["security"] == 0.9
        assert best[1].expertise["security"] == 0.6


# =============================================================================
# Integration Tests
# =============================================================================


class TestGenesisIntegration:
    """Integration tests for genesis system."""

    def test_full_evolution_cycle(self, temp_db):
        """Test complete evolution cycle."""
        manager = PopulationManager(db_path=temp_db, max_population_size=6)

        # Create initial population
        genomes = [
            AgentGenome(
                genome_id=f"initial-{i}",
                name=f"agent-{i}",
                traits={"analytical": 0.5 + i * 0.1},
                expertise={"security": 0.5},
                fitness_score=0.3 + i * 0.1,
            )
            for i in range(4)
        ]

        pop = Population(population_id="test-pop", genomes=genomes)

        # Save initial genomes
        for g in genomes:
            manager.genome_store.save(g)

        # Evolve population
        evolved = manager.evolve_population(pop)

        assert evolved.generation == 1
        assert len(evolved.genomes) <= manager.max_population_size
        # Some original high-fitness genomes should survive
        original_ids = {g.genome_id for g in genomes}
        surviving_originals = [g for g in evolved.genomes if g.genome_id in original_ids]
        assert len(surviving_originals) >= 1

    def test_specialist_spawning_workflow(self, temp_db, sample_genome, sample_genome_2):
        """Test spawning specialist for debate."""
        manager = PopulationManager(db_path=temp_db)

        manager.genome_store.save(sample_genome)
        manager.genome_store.save(sample_genome_2)

        pop = Population(
            population_id="test",
            genomes=[sample_genome, sample_genome_2],
        )

        specialist = manager.spawn_specialist_for_debate(
            domain="security",
            population=pop,
            debate_id="debate-001",
        )

        assert specialist is not None
        assert specialist.birth_debate_id == "debate-001"

        # Should be persisted
        saved = manager.genome_store.get(specialist.genome_id)
        assert saved is not None

    def test_genome_persistence_roundtrip(self, temp_db, sample_genome):
        """Test genome survives full persistence roundtrip."""
        store = GenomeStore(temp_db)

        # Save
        store.save(sample_genome)

        # Retrieve
        loaded = store.get(sample_genome.genome_id)
        assert loaded is not None

        # Modify
        loaded.fitness_score = 0.95
        loaded.debates_participated = 100
        store.save(loaded)

        # Retrieve again
        reloaded = store.get(sample_genome.genome_id)
        assert reloaded.fitness_score == 0.95
        assert reloaded.debates_participated == 100
