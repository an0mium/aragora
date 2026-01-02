#!/usr/bin/env python3
"""
Run a real fractal debate with actual AI agents.

This demonstrates the full genesis system:
- Fractal sub-debates when tensions are detected
- Agent evolution with genetic operators
- Provenance tracking in the genesis ledger
"""

import asyncio
import sys
from pathlib import Path

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.genesis import (
    FractalOrchestrator,
    PopulationManager,
    GenesisLedger,
    create_logging_hooks,
)
from aragora.agents.cli_agents import ClaudeAgent, CodexAgent
from aragora.agents.api_agents import GeminiAgent


async def run_fractal_debate():
    """Run a fractal debate on a technical topic."""

    print("=" * 70)
    print("ARAGORA GENESIS - FRACTAL DEBATE")
    print("=" * 70)

    # Initialize genesis components
    db_path = ".nomic/genesis.db"
    ledger = GenesisLedger(db_path)
    population_manager = PopulationManager(db_path, max_population_size=8)

    # Create logging hooks
    hooks = create_logging_hooks(lambda msg: print(f"  [genesis] {msg}"))

    # Create agents
    print("\nInitializing agents...")
    agents = []

    try:
        claude = ClaudeAgent(name="claude-visionary", model="claude", role="proposer")
        agents.append(claude)
        print("  ✓ Claude agent ready")
    except Exception as e:
        print(f"  ✗ Claude agent failed: {e}")

    try:
        gemini = GeminiAgent(name="gemini-pragmatic", role="critic")
        agents.append(gemini)
        print("  ✓ Gemini agent ready")
    except Exception as e:
        print(f"  ✗ Gemini agent failed: {e}")

    try:
        codex = CodexAgent(name="codex-engineer", model="codex", role="synthesizer")
        agents.append(codex)
        print("  ✓ Codex agent ready")
    except Exception as e:
        print(f"  ✗ Codex agent failed: {e}")

    if len(agents) < 2:
        print("\nNot enough agents available. Need at least 2.")
        return None

    print(f"\n{len(agents)} agents ready for debate")

    # Create fractal orchestrator
    orchestrator = FractalOrchestrator(
        max_depth=2,
        tension_threshold=0.6,
        timeout_inheritance=0.5,
        evolve_agents=True,
        population_manager=population_manager,
        event_hooks=hooks,
    )

    # Define the debate topic
    task = """Design a distributed rate limiter for a multi-region API gateway.

Requirements:
1. Must handle 1M+ requests/second globally
2. Must be consistent across regions within 100ms
3. Must gracefully degrade under partition
4. Must support per-user, per-API, and global limits

Consider:
- Token bucket vs sliding window algorithms
- Consensus protocols (Raft, Paxos, CRDTs)
- Regional vs global state synchronization
- Failure modes and recovery strategies

Propose a specific architecture with trade-offs clearly stated."""

    print("\n" + "=" * 70)
    print("DEBATE TOPIC:")
    print("=" * 70)
    print(task[:500] + "..." if len(task) > 500 else task)
    print("=" * 70)

    # Get or create population
    agent_names = [a.name.split("-")[0] for a in agents]
    population = population_manager.get_or_create_population(agent_names)

    print(f"\nPopulation: {population.size} genomes, generation {population.generation}")
    for genome in population.genomes:
        print(f"  - {genome.name}: fitness={genome.fitness_score:.2f}")

    # Run the fractal debate
    print("\n" + "=" * 70)
    print("STARTING FRACTAL DEBATE")
    print("=" * 70 + "\n")

    try:
        result = await orchestrator.run(
            task=task,
            agents=agents,
            population=population,
        )

        print("\n" + "=" * 70)
        print("DEBATE COMPLETE")
        print("=" * 70)

        print(f"\nTotal depth: {result.total_depth}")
        print(f"Sub-debates: {len(result.sub_debates)}")
        print(f"Tensions resolved: {result.tensions_resolved}")
        print(f"Tensions unresolved: {result.tensions_unresolved}")
        print(f"Evolved genomes: {len(result.evolved_genomes)}")

        if result.evolved_genomes:
            print("\nEvolved specialists:")
            for genome in result.evolved_genomes:
                print(f"  - {genome.name} (gen {genome.generation})")
                print(f"    Expertise: {list(genome.expertise.keys())[:3]}")

        print("\n" + "-" * 70)
        print("MAIN RESULT:")
        print("-" * 70)
        if result.main_result.final_answer:
            # Truncate for display
            answer = result.main_result.final_answer
            if len(answer) > 2000:
                print(answer[:2000] + "\n\n[... truncated ...]")
            else:
                print(answer)
        else:
            print("No final answer produced")

        # Show debate tree
        if result.sub_debates:
            print("\n" + "-" * 70)
            print("DEBATE TREE:")
            print("-" * 70)
            tree = result.debate_tree
            print_tree(tree)

        # Export ledger
        print("\n" + "-" * 70)
        print("LEDGER SUMMARY:")
        print("-" * 70)
        print(f"Merkle root: {ledger.get_merkle_root()[:32]}...")

        # Save HTML export
        html_export = ledger.export(format="html")
        export_path = Path(".nomic/genesis_debate.html")
        export_path.write_text(html_export)
        print(f"Full ledger exported to: {export_path}")

        return result

    except Exception as e:
        print(f"\nDEBATE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_tree(tree: dict, indent: int = 0):
    """Print a debate tree recursively."""
    prefix = "  " * indent
    debate_id = tree.get("debate_id", "unknown")
    tension = tree.get("tension", "")
    success = tree.get("success", False)

    status = "✓" if success else "○"
    tension_str = f" - {tension[:50]}..." if tension else ""
    print(f"{prefix}{status} {debate_id}{tension_str}")

    for child in tree.get("children", []):
        print_tree(child, indent + 1)


if __name__ == "__main__":
    result = asyncio.run(run_fractal_debate())

    if result:
        print("\n" + "=" * 70)
        print("FRACTAL DEBATE COMPLETED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FRACTAL DEBATE FAILED")
        print("=" * 70)
        sys.exit(1)
