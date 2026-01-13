"""
Aragora Genesis - Evolutionary Fractal Debates with Provenance.

This module synthesizes three powerful concepts:
1. **Fractal Resolution** - Recursive sub-debates for deep problem-solving
2. **Cambrian Explosion** - Agent evolution and spawning mid-debate
3. **Argonaut Ledger** - Cryptographic provenance and replay

Core Components:
- `FractalOrchestrator`: Runs debates with recursive sub-debate spawning
- `AgentGenome`: Genetic representation of agent traits/expertise
- `GenomeBreeder`: Crossover, mutation, and natural selection
- `PopulationManager`: Persistent population across debates
- `GenesisLedger`: Unified provenance tracking

Example Usage:
    ```python
    from aragora.genesis import FractalOrchestrator, GenesisLedger

    # Run a fractal debate with agent evolution
    orchestrator = FractalOrchestrator(max_depth=2, evolve_agents=True)
    result = await orchestrator.run(
        task="Design a distributed cache with strong consistency",
        agents=[claude, gemini, grok, codex]
    )

    # Access the fractal tree
    print(result.debate_tree)
    print(result.evolved_genomes)

    # Query the ledger
    ledger = GenesisLedger()
    lineage = ledger.get_lineage("claude-grok-specialist-v1")
    ```
"""

from aragora.genesis import breeding  # Expose submodule for patching
from aragora.genesis.breeding import (
    GenomeBreeder,
    Population,
    PopulationManager,
)
from aragora.genesis.events import (
    GenesisStreamEventType,
    create_genesis_hooks,
    create_logging_hooks,
)
from aragora.genesis.fractal import (
    FractalOrchestrator,
    FractalResult,
    SubDebateResult,
)
from aragora.genesis.genome import (
    AgentGenome,
    GenomeStore,
    generate_genome_id,
)
from aragora.genesis.ledger import (
    FractalTree,
    GenesisEvent,
    GenesisEventType,
    GenesisLedger,
)

__all__ = [
    # Submodules
    "breeding",
    # Genome
    "AgentGenome",
    "GenomeStore",
    "generate_genome_id",
    # Breeding
    "Population",
    "GenomeBreeder",
    "PopulationManager",
    # Fractal
    "FractalOrchestrator",
    "FractalResult",
    "SubDebateResult",
    # Ledger
    "GenesisLedger",
    "GenesisEvent",
    "GenesisEventType",
    "FractalTree",
    # Events
    "GenesisStreamEventType",
    "create_genesis_hooks",
    "create_logging_hooks",
]

__version__ = "0.1.0"
