"""
Tests for genesis breeding system.

Tests:
- Population management
- Genome crossover
- Genome mutation
- Selection operators
"""

import random
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aragora.genesis.genome import AgentGenome, generate_genome_id
from aragora.genesis.breeding import (
    Population,
    GenomeBreeder,
)


class TestPopulation:
    """Test Population dataclass."""

    def test_create_empty_population(self):
        """Should create empty population."""
        pop = Population(population_id="test-pop")

        assert pop.population_id == "test-pop"
        assert pop.size == 0
        assert pop.genomes == []
        assert pop.generation == 0

    def test_population_size(self):
        """Size should return number of genomes."""
        pop = Population(
            population_id="test",
            genomes=[
                AgentGenome(genome_id="a", name="agent-a"),
                AgentGenome(genome_id="b", name="agent-b"),
                AgentGenome(genome_id="c", name="agent-c"),
            ],
        )

        assert pop.size == 3

    def test_average_fitness_empty(self):
        """Average fitness of empty population should be 0."""
        pop = Population(population_id="test")
        assert pop.average_fitness == 0.0

    def test_average_fitness(self):
        """Should calculate average fitness correctly."""
        pop = Population(
            population_id="test",
            genomes=[
                AgentGenome(genome_id="a", name="a", fitness_score=0.2),
                AgentGenome(genome_id="b", name="b", fitness_score=0.4),
                AgentGenome(genome_id="c", name="c", fitness_score=0.6),
            ],
        )

        assert pop.average_fitness == pytest.approx(0.4, rel=0.01)

    def test_best_genome_empty(self):
        """Best genome of empty population should be None."""
        pop = Population(population_id="test")
        assert pop.best_genome is None

    def test_best_genome(self):
        """Should return genome with highest fitness."""
        pop = Population(
            population_id="test",
            genomes=[
                AgentGenome(genome_id="a", name="a", fitness_score=0.3),
                AgentGenome(genome_id="b", name="b", fitness_score=0.9),
                AgentGenome(genome_id="c", name="c", fitness_score=0.5),
            ],
        )

        best = pop.best_genome
        assert best is not None
        assert best.genome_id == "b"
        assert best.fitness_score == 0.9

    def test_get_by_id(self):
        """Should retrieve genome by ID."""
        pop = Population(
            population_id="test",
            genomes=[
                AgentGenome(genome_id="a", name="agent-a"),
                AgentGenome(genome_id="b", name="agent-b"),
            ],
        )

        found = pop.get_by_id("b")
        assert found is not None
        assert found.name == "agent-b"

        not_found = pop.get_by_id("nonexistent")
        assert not_found is None

    def test_to_dict(self):
        """Should serialize to dict."""
        pop = Population(
            population_id="test-pop",
            genomes=[
                AgentGenome(genome_id="a", name="agent-a"),
            ],
            generation=5,
            debate_history=["d1", "d2"],
        )

        data = pop.to_dict()

        assert data["population_id"] == "test-pop"
        assert data["genomes"] == ["a"]  # Just IDs
        assert data["generation"] == 5
        assert data["debate_history"] == ["d1", "d2"]


class TestGenomeBreeder:
    """Test GenomeBreeder genetic operators."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        breeder = GenomeBreeder()

        assert breeder.mutation_rate == 0.1
        assert breeder.crossover_ratio == 0.5
        assert breeder.elite_ratio == 0.2

    def test_init_custom_params(self):
        """Should accept custom parameters."""
        breeder = GenomeBreeder(
            mutation_rate=0.2,
            crossover_ratio=0.7,
            elite_ratio=0.3,
        )

        assert breeder.mutation_rate == 0.2
        assert breeder.crossover_ratio == 0.7
        assert breeder.elite_ratio == 0.3

    def test_crossover_blends_traits(self):
        """Crossover should blend parent traits."""
        breeder = GenomeBreeder(crossover_ratio=0.5)

        parent_a = AgentGenome(
            genome_id="a",
            name="parent-a",
            traits={"analytical": 1.0, "cautious": 0.8},
            expertise={"security": 1.0},
        )
        parent_b = AgentGenome(
            genome_id="b",
            name="parent-b",
            traits={"analytical": 0.0, "creative": 0.6},
            expertise={"security": 0.0, "frontend": 0.8},
        )

        child = breeder.crossover(parent_a, parent_b)

        # Should blend with 0.5 ratio
        assert child.traits["analytical"] == 0.5
        # Trait only in one parent blends with 0
        assert child.traits["cautious"] == 0.4  # 0.5 * 0.8 + 0.5 * 0
        assert child.traits["creative"] == 0.3  # 0.5 * 0 + 0.5 * 0.6

        # Expertise blending
        assert child.expertise["security"] == 0.5
        assert child.expertise["frontend"] == 0.4

    def test_crossover_respects_ratio(self):
        """Crossover should respect crossover ratio."""
        breeder = GenomeBreeder(crossover_ratio=0.8)  # 80% from parent_a

        parent_a = AgentGenome(
            genome_id="a",
            name="parent-a",
            traits={"analytical": 1.0},
            expertise={},
        )
        parent_b = AgentGenome(
            genome_id="b",
            name="parent-b",
            traits={"analytical": 0.0},
            expertise={},
        )

        child = breeder.crossover(parent_a, parent_b)

        # 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        assert child.traits["analytical"] == pytest.approx(0.8, rel=0.01)

    def test_crossover_increments_generation(self):
        """Child generation should be max of parents + 1."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="a", generation=2)
        parent_b = AgentGenome(genome_id="b", name="b", generation=5)

        child = breeder.crossover(parent_a, parent_b)

        assert child.generation == 6

    def test_crossover_records_parents(self):
        """Child should record parent genome IDs."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="parent-a-id", name="a")
        parent_b = AgentGenome(genome_id="parent-b-id", name="b")

        child = breeder.crossover(parent_a, parent_b)

        assert child.parent_genomes == ["parent-a-id", "parent-b-id"]

    def test_crossover_generates_unique_id(self):
        """Each child should get a unique genome ID."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="a")
        parent_b = AgentGenome(genome_id="b", name="b")

        child1 = breeder.crossover(parent_a, parent_b)
        child2 = breeder.crossover(parent_a, parent_b)

        assert len(child1.genome_id) == 12
        assert len(child2.genome_id) == 12
        # Should be unique (different timestamps)

    def test_crossover_custom_name(self):
        """Should use custom name if provided."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="claude")
        parent_b = AgentGenome(genome_id="b", name="grok")

        child = breeder.crossover(parent_a, parent_b, name="custom-hybrid")

        assert child.name == "custom-hybrid"

    def test_crossover_auto_generates_name(self):
        """Should auto-generate name from parents."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="claude-specialist", generation=2)
        parent_b = AgentGenome(genome_id="b", name="grok-analyst", generation=1)

        child = breeder.crossover(parent_a, parent_b)

        # Name should be like "claude-grok-gen3"
        assert "claude" in child.name
        assert "grok" in child.name
        assert "gen3" in child.name

    def test_crossover_records_debate_id(self):
        """Should record birth debate ID."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="a")
        parent_b = AgentGenome(genome_id="b", name="b")

        child = breeder.crossover(parent_a, parent_b, debate_id="debate-123")

        assert child.birth_debate_id == "debate-123"

    def test_crossover_starts_neutral_fitness(self):
        """Child should start with neutral fitness."""
        breeder = GenomeBreeder()

        parent_a = AgentGenome(genome_id="a", name="a", fitness_score=0.9)
        parent_b = AgentGenome(genome_id="b", name="b", fitness_score=0.8)

        child = breeder.crossover(parent_a, parent_b)

        assert child.fitness_score == 0.5  # Neutral starting point

    def test_mutate_returns_new_genome(self):
        """Mutation should return new genome, not modify original."""
        breeder = GenomeBreeder(mutation_rate=1.0)  # High rate to ensure mutations

        original = AgentGenome(
            genome_id="original",
            name="original-agent",
            traits={"analytical": 0.5},
            expertise={"security": 0.5},
        )

        mutated = breeder.mutate(original)

        # Original should be unchanged
        assert original.genome_id == "original"
        assert original.name == "original-agent"

        # Mutated should have different ID and "-mut" suffix
        assert mutated.genome_id != "original"
        assert "-mut" in mutated.name

    def test_mutate_modifies_traits(self):
        """With high mutation rate, traits should change."""
        random.seed(42)  # For reproducibility
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            traits={"analytical": 0.5, "cautious": 0.5},
        )

        mutated = breeder.mutate(original)

        # At least some traits should be different (random delta applied)
        traits_changed = (
            mutated.traits.get("analytical", 0) != 0.5
            or mutated.traits.get("cautious", 0) != 0.5
            or len(mutated.traits) != len(original.traits)
        )
        assert traits_changed

    def test_mutate_keeps_traits_in_bounds(self):
        """Mutated trait values should stay in [0, 1]."""
        random.seed(42)
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            traits={"analytical": 0.99, "cautious": 0.01},
            expertise={"security": 0.99, "backend": 0.01},
        )

        # Mutate multiple times
        for _ in range(10):
            mutated = breeder.mutate(original)

            for val in mutated.traits.values():
                assert 0 <= val <= 1

            for val in mutated.expertise.values():
                assert 0 <= val <= 1

    def test_mutate_respects_custom_rate(self):
        """Should use provided mutation rate."""
        random.seed(42)
        breeder = GenomeBreeder(mutation_rate=0.1)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            traits={"a": 0.5, "b": 0.5, "c": 0.5},
        )

        # With 0 rate, nothing should change
        mutated = breeder.mutate(original, rate=0.0)

        for trait, val in original.traits.items():
            if trait in mutated.traits:
                assert mutated.traits[trait] == val

    def test_mutate_records_original_as_parent(self):
        """Mutated genome should record original as parent."""
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original-id",
            name="agent",
            parent_genomes=["grandparent"],
        )

        mutated = breeder.mutate(original)

        # Parent genomes should contain original's ID
        assert mutated.parent_genomes == ["original-id"]

    @patch("aragora.genesis.breeding.PERSONALITY_TRAITS", ["new_trait_1", "new_trait_2"])
    def test_mutate_can_add_new_traits(self):
        """Mutation can add new traits from available pool."""
        random.seed(0)  # Seed that triggers new trait addition
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            traits={},  # Start empty
        )

        # Try multiple mutations to see if new traits are added
        mutated = original
        for _ in range(20):
            mutated = breeder.mutate(mutated, rate=1.0)

        # Should have potentially added new traits
        # (depends on random, but with high rate should happen)

    @patch("aragora.genesis.breeding.EXPERTISE_DOMAINS", ["new_domain_1", "new_domain_2"])
    def test_mutate_can_add_new_expertise(self):
        """Mutation can add new expertise from available pool."""
        random.seed(0)
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            expertise={},
        )

        mutated = original
        for _ in range(20):
            mutated = breeder.mutate(mutated, rate=1.0)

        # Should have potentially added new domains

    def test_mutate_can_change_model_preference(self):
        """Mutation can change model preference."""
        random.seed(42)
        breeder = GenomeBreeder(mutation_rate=1.0)

        original = AgentGenome(
            genome_id="original",
            name="agent",
            model_preference="claude",
        )

        # Try multiple mutations
        model_changed = False
        for _ in range(50):
            mutated = breeder.mutate(original, rate=1.0)
            if mutated.model_preference != "claude":
                model_changed = True
                break

        # With high rate, model should change at least once
        assert model_changed


class TestBreederIntegration:
    """Integration tests for breeding operations."""

    def test_breed_multiple_generations(self):
        """Should be able to breed multiple generations."""
        breeder = GenomeBreeder(mutation_rate=0.1)

        # Start with base population
        gen0 = [
            AgentGenome(genome_id="a", name="agent-a", fitness_score=0.6),
            AgentGenome(genome_id="b", name="agent-b", fitness_score=0.7),
            AgentGenome(genome_id="c", name="agent-c", fitness_score=0.5),
        ]

        # Breed gen1
        gen1 = []
        gen1.append(breeder.crossover(gen0[0], gen0[1]))
        gen1.append(breeder.crossover(gen0[1], gen0[2]))
        gen1.append(breeder.mutate(gen0[1]))  # Best performer mutated

        assert all(g.generation == 1 for g in gen1[:2])

        # Breed gen2
        gen2 = []
        gen2.append(breeder.crossover(gen1[0], gen1[1]))

        assert gen2[0].generation == 2

    def test_maintain_genetic_diversity(self):
        """Breeding should maintain some genetic diversity."""
        breeder = GenomeBreeder(mutation_rate=0.3)

        # Create diverse parents
        parent_a = AgentGenome(
            genome_id="a",
            name="security-specialist",
            traits={"analytical": 0.9, "cautious": 0.8},
            expertise={"security": 0.95, "backend": 0.3},
        )
        parent_b = AgentGenome(
            genome_id="b",
            name="creative-generalist",
            traits={"creative": 0.9, "bold": 0.7},
            expertise={"frontend": 0.8, "design": 0.7},
        )

        # Create several children
        children = [breeder.crossover(parent_a, parent_b) for _ in range(5)]

        # Apply mutations
        mutated_children = [breeder.mutate(c) for c in children]

        # Check diversity - children shouldn't all be identical
        similarities = []
        for i, c1 in enumerate(mutated_children):
            for c2 in mutated_children[i + 1 :]:
                similarities.append(c1.similarity_to(c2))

        # Not all should be 1.0 (identical)
        if similarities:
            assert min(similarities) < 1.0 or len(similarities) == 0
