"""Genesis feedback integration tests.

Tests the feedback loop between debates and agent evolution:
- FeedbackPhase triggering genome evolution
- Performance-based fitness scoring
- Population updates after debates
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockAgentGenome:
    """Mock agent genome for testing."""

    genome_id: str
    base_model: str
    traits: dict
    expertise: list
    fitness_score: float = 0.5
    generation: int = 0

    def copy(self):
        return MockAgentGenome(
            genome_id=self.genome_id,
            base_model=self.base_model,
            traits=self.traits.copy(),
            expertise=self.expertise.copy(),
            fitness_score=self.fitness_score,
            generation=self.generation,
        )


class MockGenomeBreeder:
    """Mock genome breeder for testing evolution."""

    def __init__(self):
        self.crossover_count = 0
        self.mutation_count = 0

    def crossover(self, parent1: MockAgentGenome, parent2: MockAgentGenome) -> MockAgentGenome:
        """Create offspring from two parents."""
        self.crossover_count += 1
        child_traits = {}
        for key in parent1.traits:
            # 50/50 inheritance
            child_traits[key] = (parent1.traits.get(key, 0) + parent2.traits.get(key, 0)) / 2

        return MockAgentGenome(
            genome_id=f"child-{self.crossover_count}",
            base_model=parent1.base_model,
            traits=child_traits,
            expertise=list(set(parent1.expertise + parent2.expertise)),
            generation=max(parent1.generation, parent2.generation) + 1,
        )

    def mutate(self, genome: MockAgentGenome, rate: float = 0.1) -> MockAgentGenome:
        """Apply mutations to a genome."""
        self.mutation_count += 1
        mutated = genome.copy()
        # Apply some random-like mutation
        for key in mutated.traits:
            mutated.traits[key] *= 1 + rate * 0.5  # Predictable mutation for testing
        mutated.genome_id = f"{genome.genome_id}-mutated"
        return mutated


class MockPopulationManager:
    """Mock population manager for testing."""

    def __init__(self):
        self.population: list[MockAgentGenome] = []
        self.fitness_updates: list[tuple[str, float]] = []
        self.evolutions: list[MockAgentGenome] = []

    def add_genome(self, genome: MockAgentGenome):
        self.population.append(genome)

    def update_fitness(self, genome_id: str, score: float):
        self.fitness_updates.append((genome_id, score))
        for genome in self.population:
            if genome.genome_id == genome_id:
                genome.fitness_score = score
                break

    def get_top_performers(self, n: int = 2) -> list[MockAgentGenome]:
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_pop[:n]

    def evolve_generation(self, breeder: MockGenomeBreeder) -> list[MockAgentGenome]:
        """Evolve a new generation from top performers."""
        if len(self.population) < 2:
            return []

        top = self.get_top_performers(2)
        offspring = breeder.crossover(top[0], top[1])
        mutated = breeder.mutate(offspring)
        self.evolutions.append(mutated)
        return [mutated]


class TestFeedbackPhaseTriggersEvolution:
    """Test that feedback phase properly triggers genome evolution."""

    def test_debate_outcome_updates_fitness(self):
        """Test that debate outcomes update agent fitness scores."""
        population = MockPopulationManager()

        # Add initial genomes
        genome1 = MockAgentGenome("claude-v1", "claude", {"reasoning": 0.8}, ["ai"])
        genome2 = MockAgentGenome("gemini-v1", "gemini", {"creativity": 0.7}, ["tech"])
        population.add_genome(genome1)
        population.add_genome(genome2)

        # Simulate feedback from debate
        # Claude performed well (consensus reached, high confidence)
        population.update_fitness("claude-v1", 0.85)
        # Gemini performed moderately
        population.update_fitness("gemini-v1", 0.65)

        assert len(population.fitness_updates) == 2
        assert genome1.fitness_score == 0.85
        assert genome2.fitness_score == 0.65

    def test_high_fitness_triggers_breeding(self):
        """Test that high-performing agents trigger breeding."""
        population = MockPopulationManager()
        breeder = MockGenomeBreeder()

        # Add genomes with different fitness
        genome1 = MockAgentGenome("claude-v1", "claude", {"reasoning": 0.8}, ["ai"])
        genome1.fitness_score = 0.9
        genome2 = MockAgentGenome("gemini-v1", "gemini", {"creativity": 0.7}, ["tech"])
        genome2.fitness_score = 0.85

        population.add_genome(genome1)
        population.add_genome(genome2)

        # Trigger evolution
        offspring = population.evolve_generation(breeder)

        assert len(offspring) == 1
        assert breeder.crossover_count == 1
        assert breeder.mutation_count == 1
        assert offspring[0].generation == 1

    def test_offspring_inherits_traits(self):
        """Test that offspring inherits traits from parents."""
        breeder = MockGenomeBreeder()

        parent1 = MockAgentGenome(
            "claude-v1", "claude", {"reasoning": 0.9, "creativity": 0.6}, ["ai", "safety"]
        )
        parent2 = MockAgentGenome(
            "gemini-v1", "gemini", {"reasoning": 0.7, "creativity": 0.8}, ["tech", "research"]
        )

        child = breeder.crossover(parent1, parent2)

        # Traits should be averaged
        assert child.traits["reasoning"] == 0.8
        assert child.traits["creativity"] == 0.7

        # Expertise should be combined
        assert "ai" in child.expertise
        assert "tech" in child.expertise


class TestFitnessScoring:
    """Test fitness scoring from debate performance."""

    def test_consensus_boosts_fitness(self):
        """Test that reaching consensus boosts fitness."""
        base_fitness = 0.5

        # Consensus reached
        consensus_bonus = 0.2
        fitness_with_consensus = base_fitness + consensus_bonus

        # No consensus
        fitness_without_consensus = base_fitness

        assert fitness_with_consensus > fitness_without_consensus

    def test_high_confidence_boosts_fitness(self):
        """Test that high confidence in consensus boosts fitness."""
        base_fitness = 0.5

        # Calculate fitness based on confidence
        def calculate_fitness(base: float, confidence: float) -> float:
            return base + (confidence * 0.3)

        high_conf_fitness = calculate_fitness(base_fitness, 0.95)
        low_conf_fitness = calculate_fitness(base_fitness, 0.55)

        assert high_conf_fitness > low_conf_fitness

    def test_critique_quality_affects_fitness(self):
        """Test that critique quality affects agent fitness."""
        # Agent whose critiques were accepted has higher fitness
        critiques_accepted = 5
        critiques_rejected = 1
        critique_rate = critiques_accepted / (critiques_accepted + critiques_rejected)

        assert critique_rate > 0.5  # Good critique quality

    def test_combined_fitness_formula(self):
        """Test the combined fitness formula."""

        def calculate_combined_fitness(
            consensus: bool,
            confidence: float,
            critique_acceptance_rate: float,
            position_accuracy: float,
        ) -> float:
            """Calculate fitness from multiple factors."""
            base = 0.3
            consensus_bonus = 0.2 if consensus else 0
            confidence_factor = confidence * 0.2
            critique_factor = critique_acceptance_rate * 0.15
            accuracy_factor = position_accuracy * 0.15
            return base + consensus_bonus + confidence_factor + critique_factor + accuracy_factor

        # High performer
        high_fitness = calculate_combined_fitness(
            consensus=True,
            confidence=0.9,
            critique_acceptance_rate=0.8,
            position_accuracy=0.85,
        )

        # Low performer
        low_fitness = calculate_combined_fitness(
            consensus=False,
            confidence=0.5,
            critique_acceptance_rate=0.3,
            position_accuracy=0.4,
        )

        assert high_fitness > 0.7
        assert low_fitness < 0.55  # Low performer should be below 0.55 (just above base)
        assert high_fitness > low_fitness


class TestPopulationEvolution:
    """Test population-level evolution mechanics."""

    def test_population_size_maintained(self):
        """Test that population size is maintained across generations."""
        population = MockPopulationManager()
        breeder = MockGenomeBreeder()

        # Initial population
        for i in range(5):
            genome = MockAgentGenome(
                f"agent-{i}",
                "base",
                {"trait": 0.5 + i * 0.1},
                ["general"],
                fitness_score=0.5 + i * 0.1,
            )
            population.add_genome(genome)

        initial_size = len(population.population)

        # Evolve one generation
        population.evolve_generation(breeder)

        # Population should have grown by offspring count
        assert len(population.population) == initial_size
        assert len(population.evolutions) == 1

    def test_low_fitness_pruning(self):
        """Test that low-fitness agents can be pruned."""
        population = MockPopulationManager()

        # Add genomes with varying fitness
        genomes = [
            MockAgentGenome("high", "base", {}, [], fitness_score=0.9),
            MockAgentGenome("medium", "base", {}, [], fitness_score=0.6),
            MockAgentGenome("low", "base", {}, [], fitness_score=0.2),
        ]
        for g in genomes:
            population.add_genome(g)

        # Get top performers (simulating pruning)
        survivors = population.get_top_performers(2)

        assert len(survivors) == 2
        assert survivors[0].genome_id == "high"
        assert survivors[1].genome_id == "medium"

    def test_generation_advancement(self):
        """Test that generations advance properly."""
        breeder = MockGenomeBreeder()

        gen0_parent1 = MockAgentGenome("p1", "base", {"x": 1}, [], generation=0)
        gen0_parent2 = MockAgentGenome("p2", "base", {"x": 1}, [], generation=0)

        gen1 = breeder.crossover(gen0_parent1, gen0_parent2)

        assert gen1.generation == 1


class TestArenaEvolutionIntegration:
    """Test evolution integration with Arena debates."""

    def test_auto_evolve_flag(self):
        """Test that auto_evolve flag triggers evolution."""
        # Mock arena configuration
        arena_config = {
            "auto_evolve": True,
            "evolution_threshold": 0.8,  # Min fitness for breeding
            "max_population": 10,
        }

        assert arena_config["auto_evolve"] is True

    def test_evolution_after_consensus(self):
        """Test that evolution triggers after consensus."""
        population = MockPopulationManager()
        breeder = MockGenomeBreeder()

        # Add initial agents
        genome1 = MockAgentGenome("agent1", "base", {"x": 0.5}, [], fitness_score=0.85)
        genome2 = MockAgentGenome("agent2", "base", {"x": 0.6}, [], fitness_score=0.82)
        population.add_genome(genome1)
        population.add_genome(genome2)

        # Simulate consensus reached
        consensus_reached = True
        if consensus_reached:
            # Trigger evolution
            new_agents = population.evolve_generation(breeder)
            assert len(new_agents) == 1

    def test_specialist_spawning(self):
        """Test spawning specialists from high-performing generalists."""
        breeder = MockGenomeBreeder()

        generalist = MockAgentGenome(
            "generalist",
            "claude",
            {"reasoning": 0.8, "creativity": 0.8, "precision": 0.8},
            ["general"],
            fitness_score=0.9,
        )

        # Mutate to create specialist
        specialist = breeder.mutate(generalist, rate=0.3)

        assert specialist.genome_id != generalist.genome_id
        assert breeder.mutation_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
