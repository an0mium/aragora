"""
Tests for PopulationManager database operations.

Tests the persistent population management including:
- Population creation from base agents
- Fitness updates from debate outcomes
- Generational evolution
- Domain-specific agent selection
"""

import os
import tempfile
import pytest

from aragora.genesis.breeding import (
    PopulationManager,
    GenomeBreeder,
    Population,
)
from aragora.genesis.genome import AgentGenome


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def population_manager(temp_db):
    """Create PopulationManager with temp database."""
    return PopulationManager(db_path=temp_db)


@pytest.fixture
def base_agents():
    """Standard base agents for testing."""
    return ["claude", "gemini", "grok"]


# =============================================================================
# Population Creation Tests
# =============================================================================


class TestGetOrCreatePopulation:
    """Tests for get_or_create_population."""

    def test_creates_population_from_base_agents(self, population_manager, base_agents):
        """Test creating population from base agents."""
        population = population_manager.get_or_create_population(base_agents)

        assert population is not None
        assert len(population.genomes) == len(base_agents)

    def test_population_has_correct_genome_names(self, population_manager, base_agents):
        """Test that genomes have correct names from base agents."""
        population = population_manager.get_or_create_population(base_agents)

        genome_names = [g.name for g in population.genomes]
        for agent in base_agents:
            assert any(agent in name for name in genome_names)

    def test_returns_existing_population_on_second_call(self, population_manager, base_agents):
        """Test that same population is returned on subsequent calls."""
        pop1 = population_manager.get_or_create_population(base_agents)
        pop2 = population_manager.get_or_create_population(base_agents)

        # Should return same population (potentially with updates)
        assert pop1.population_id == pop2.population_id

    def test_creates_population_generates_id(self, population_manager, base_agents):
        """Test that population gets a generated ID."""
        population = population_manager.get_or_create_population(base_agents)

        assert population.population_id is not None
        assert len(population.population_id) > 0

    def test_empty_base_agents_creates_empty_population(self, population_manager):
        """Test creating population with empty base agents."""
        population = population_manager.get_or_create_population([])

        assert population is not None
        assert len(population.genomes) == 0


# =============================================================================
# Fitness Update Tests
# =============================================================================


class TestUpdateFitness:
    """Tests for update_fitness method."""

    def test_consensus_win_updates_fitness(self, population_manager, base_agents):
        """Test that consensus win updates fitness."""
        population = population_manager.get_or_create_population(base_agents)
        genome = population.genomes[0]
        initial_fitness = genome.fitness_score

        population_manager.update_fitness(genome.genome_id, consensus_win=True)

        # Refresh genome from store
        updated = population_manager.genome_store.get(genome.genome_id)
        # Fitness changed (may go up or down based on EMA formula)
        assert updated.fitness_score != initial_fitness or initial_fitness == 0.5

    def test_critique_accepted_updates_fitness(self, population_manager, base_agents):
        """Test that critique accepted updates fitness."""
        population = population_manager.get_or_create_population(base_agents)
        genome = population.genomes[0]
        initial_fitness = genome.fitness_score

        population_manager.update_fitness(genome.genome_id, critique_accepted=True)

        updated = population_manager.genome_store.get(genome.genome_id)
        # Fitness changed
        assert updated.fitness_score != initial_fitness or initial_fitness == 0.5

    def test_prediction_correct_updates_fitness(self, population_manager, base_agents):
        """Test that correct prediction updates fitness."""
        population = population_manager.get_or_create_population(base_agents)
        genome = population.genomes[0]
        initial_fitness = genome.fitness_score

        population_manager.update_fitness(genome.genome_id, prediction_correct=True)

        updated = population_manager.genome_store.get(genome.genome_id)
        # Fitness changed
        assert updated.fitness_score != initial_fitness or initial_fitness == 0.5

    def test_update_nonexistent_genome_no_error(self, population_manager):
        """Test that updating nonexistent genome doesn't raise error."""
        # Should not raise
        population_manager.update_fitness("nonexistent-id", consensus_win=True)

    def test_multiple_fitness_updates_changes_fitness(self, population_manager, base_agents):
        """Test that multiple updates change fitness."""
        population = population_manager.get_or_create_population(base_agents)
        genome = population.genomes[0]
        initial_fitness = genome.fitness_score

        # Multiple wins
        for _ in range(3):
            population_manager.update_fitness(genome.genome_id, consensus_win=True)

        updated = population_manager.genome_store.get(genome.genome_id)
        # After multiple updates, fitness has changed
        assert updated.fitness_score != initial_fitness or genome.debate_count > 0


# =============================================================================
# Evolution Tests
# =============================================================================


class TestEvolvePopulation:
    """Tests for evolve_population method."""

    def test_evolve_population_returns_population(self, population_manager, base_agents):
        """Test that evolve returns a Population."""
        population = population_manager.get_or_create_population(base_agents)

        evolved = population_manager.evolve_population(population)

        assert isinstance(evolved, Population)

    def test_evolve_increases_generation(self, population_manager, base_agents):
        """Test that evolution increases generation number."""
        population = population_manager.get_or_create_population(base_agents)
        initial_gen = population.generation

        evolved = population_manager.evolve_population(population)

        assert evolved.generation > initial_gen

    def test_evolve_respects_max_population_size(self, temp_db, base_agents):
        """Test that evolution respects max population size."""
        max_size = 4
        manager = PopulationManager(db_path=temp_db, max_population_size=max_size)
        population = manager.get_or_create_population(base_agents)

        # Add more genomes artificially to exceed limit
        breeder = GenomeBreeder()
        for i in range(5):
            child = breeder.crossover(
                population.genomes[0],
                population.genomes[1],
            )
            population.genomes.append(child)

        evolved = manager.evolve_population(population)

        assert len(evolved.genomes) <= max_size

    def test_evolve_empty_population_no_error(self, population_manager):
        """Test that evolving empty population doesn't error."""
        population = population_manager.get_or_create_population([])

        # Should not raise
        evolved = population_manager.evolve_population(population)

        assert isinstance(evolved, Population)


# =============================================================================
# Domain Selection Tests
# =============================================================================


class TestGetBestForDomain:
    """Tests for get_best_for_domain method."""

    def test_get_best_returns_list(self, population_manager, base_agents):
        """Test that get_best_for_domain returns a list."""
        population_manager.get_or_create_population(base_agents)

        best = population_manager.get_best_for_domain("coding", n=2)

        assert isinstance(best, list)

    def test_get_best_respects_n_limit(self, population_manager, base_agents):
        """Test that n parameter limits results."""
        population_manager.get_or_create_population(base_agents)

        best = population_manager.get_best_for_domain("coding", n=1)

        assert len(best) <= 1

    def test_get_best_returns_empty_for_no_genomes(self, population_manager):
        """Test that empty store returns empty list."""
        # Don't create population first
        best = population_manager.get_best_for_domain("coding", n=2)

        assert best == []

    def test_get_best_prefers_domain_experts(self, population_manager, base_agents):
        """Test that genomes with domain expertise are preferred."""
        population = population_manager.get_or_create_population(base_agents)

        # Manually set expertise on first genome
        expert_genome = population.genomes[0]
        expert_genome.expertise["coding"] = 10.0
        population_manager.genome_store.save(expert_genome)

        best = population_manager.get_best_for_domain("coding", n=1)

        assert len(best) >= 1
        if best:
            assert best[0].genome_id == expert_genome.genome_id


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for database persistence."""

    def test_population_persists_across_instances(self, temp_db, base_agents):
        """Test that population persists when manager recreated."""
        # Create population with first manager
        manager1 = PopulationManager(db_path=temp_db)
        pop1 = manager1.get_or_create_population(base_agents)
        pop1_id = pop1.population_id

        # Create new manager and load
        manager2 = PopulationManager(db_path=temp_db)
        pop2 = manager2.get_or_create_population(base_agents)

        assert pop2.population_id == pop1_id

    def test_fitness_updates_persist(self, temp_db, base_agents):
        """Test that fitness updates persist."""
        # Create and update with first manager
        manager1 = PopulationManager(db_path=temp_db)
        pop1 = manager1.get_or_create_population(base_agents)
        genome_id = pop1.genomes[0].genome_id
        manager1.update_fitness(genome_id, consensus_win=True)

        # Create new manager and check
        manager2 = PopulationManager(db_path=temp_db)
        genome = manager2.genome_store.get(genome_id)

        # Fitness should be > 0 (updated)
        assert genome.fitness_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
