#!/usr/bin/env python3
"""
Test script for Aragora Genesis - Fractal debates with agent evolution.

This script tests the genesis module components without requiring API calls.
"""

import asyncio
import sys
from pathlib import Path

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.genesis import (
    AgentGenome,
    GenomeStore,
    GenomeBreeder,
    PopulationManager,
    Population,
    GenesisLedger,
    create_logging_hooks,
)


def test_genome_creation():
    """Test creating and manipulating genomes."""
    print("\n=== Testing Genome Creation ===")

    # Create a genome from scratch
    genome = AgentGenome(
        genome_id="test-001",
        name="claude-test",
        traits={"thorough": 0.8, "pragmatic": 0.6},
        expertise={"security": 0.9, "performance": 0.7},
        model_preference="claude",
        generation=0,
    )

    print(f"Created: {genome}")
    print(f"  Top traits: {genome.get_dominant_traits()}")
    print(f"  Top expertise: {genome.get_top_expertise()}")

    # Update fitness
    genome.update_fitness(consensus_win=True, critique_accepted=True)
    print(f"  Fitness after win: {genome.fitness_score:.2f}")

    return genome


def test_breeding():
    """Test genome breeding operations."""
    print("\n=== Testing Genome Breeding ===")

    breeder = GenomeBreeder(mutation_rate=0.2)

    # Create parent genomes
    parent_a = AgentGenome(
        genome_id="parent-a",
        name="claude",
        traits={"thorough": 0.9, "conservative": 0.7},
        expertise={"security": 0.9, "testing": 0.6},
        model_preference="claude",
    )

    parent_b = AgentGenome(
        genome_id="parent-b",
        name="grok",
        traits={"contrarian": 0.8, "innovative": 0.7},
        expertise={"performance": 0.8, "architecture": 0.7},
        model_preference="grok",
    )

    print(f"Parent A: {parent_a}")
    print(f"Parent B: {parent_b}")

    # Crossover
    child = breeder.crossover(parent_a, parent_b)
    print(f"\nChild (crossover): {child}")
    print(f"  Parents: {child.parent_genomes}")
    print(f"  Generation: {child.generation}")

    # Mutation
    mutant = breeder.mutate(child)
    print(f"\nMutant: {mutant}")

    # Specialist
    specialist = breeder.spawn_specialist(
        domain="security",
        parent_pool=[parent_a, parent_b],
    )
    print(f"\nSpecialist: {specialist}")
    print(f"  Security expertise: {specialist.expertise.get('security', 0):.2f}")

    return child


def test_population():
    """Test population management."""
    print("\n=== Testing Population Management ===")

    # Create population manager with temp database
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = PopulationManager(db_path=db_path, max_population_size=6)

    # Create population from base agents
    population = manager.get_or_create_population(["claude", "gemini", "grok"])

    print(f"Created population: {population.population_id}")
    print(f"  Size: {population.size}")
    print(f"  Generation: {population.generation}")
    print(f"  Average fitness: {population.average_fitness:.2f}")

    for genome in population.genomes:
        print(f"    - {genome.name}: fitness={genome.fitness_score:.2f}")

    # Evolve population
    evolved = manager.evolve_population(population)
    print(f"\nEvolved population:")
    print(f"  Size: {evolved.size}")
    print(f"  Generation: {evolved.generation}")

    for genome in evolved.genomes:
        print(f"    - {genome.name}: fitness={genome.fitness_score:.2f}, gen={genome.generation}")

    # Clean up
    Path(db_path).unlink(missing_ok=True)

    return evolved


def test_ledger():
    """Test genesis ledger."""
    print("\n=== Testing Genesis Ledger ===")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    ledger = GenesisLedger(db_path=db_path)

    # Record some events
    ledger.record_debate_start(
        debate_id="debate-001",
        task="Design a caching system",
        agents=["claude", "gemini", "grok"],
    )

    genome = AgentGenome(
        genome_id="specialist-001",
        name="cache-specialist",
        traits={"pragmatic": 0.8},
        expertise={"performance": 0.9},
        model_preference="claude",
        generation=1,
    )

    ledger.record_agent_birth(
        genome=genome,
        parents=["claude", "grok"],
        birth_type="specialist",
    )

    ledger.record_debate_spawn(
        parent_id="debate-001",
        child_id="debate-001-sub-1",
        trigger="unresolved_tension",
        tension_description="Cache invalidation strategy unclear",
    )

    # Export
    print("\nLedger export (JSON preview):")
    json_export = ledger.export(format="json")
    print(json_export[:500] + "...")

    print("\nLedger Merkle root:", ledger.get_merkle_root()[:32] + "...")

    # Clean up
    Path(db_path).unlink(missing_ok=True)

    return ledger


def test_logging_hooks():
    """Test logging hooks."""
    print("\n=== Testing Logging Hooks ===")

    hooks = create_logging_hooks()

    # Simulate events
    genome = AgentGenome(
        genome_id="test-genome",
        name="test-agent",
        traits={"thorough": 0.8},
        expertise={"security": 0.7},
        model_preference="claude",
    )

    print("Simulating hook calls:")
    hooks["on_fractal_start"]("debate-001", "Test task", 0, None)
    hooks["on_agent_birth"](genome, ["parent-a", "parent-b"], "crossover")
    hooks["on_fractal_spawn"]("sub-001", "debate-001", "Test tension", 1)
    hooks["on_fractal_complete"]("debate-001", 1, 1, True)


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARAGORA GENESIS TEST SUITE")
    print("=" * 60)

    try:
        test_genome_creation()
        test_breeding()
        test_population()
        test_ledger()
        test_logging_hooks()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
