---
title: Genesis - Evolutionary Agent System
description: Genesis - Evolutionary Agent System
---

# Genesis - Evolutionary Agent System

Genesis is Aragora's evolutionary system for creating, evolving, and managing AI agent genomes. It combines genetic algorithms with debate performance to produce specialized agents that excel at specific tasks.

## Overview

Genesis synthesizes three concepts:

1. **Fractal Resolution** - Recursive sub-debates for deep problem-solving
2. **Cambrian Explosion** - Agent evolution and spawning mid-debate
3. **Argonaut Ledger** - Cryptographic provenance and replay

## Core Components

### AgentGenome

A genetic representation of an agent's traits, expertise, and behavior:

```python
from aragora.genesis import AgentGenome

genome = AgentGenome(
    genome_id="claude-systems-specialist-v1",
    traits={
        "reasoning_depth": 0.85,
        "creativity": 0.7,
        "precision": 0.9,
        "collaboration": 0.75
    },
    expertise=["distributed_systems", "databases", "api_design"],
    base_model="claude-opus-4-5-20251101",
    prompt_template="You are a systems architecture specialist...",
    fitness_score=0.82
)
```

### PopulationManager

Manages a persistent population of agent genomes across debates:

```python
from aragora.genesis import PopulationManager

population = PopulationManager(db_path=".nomic/genomes.db")

# Get top performers
top_agents = population.get_top_genomes(limit=10)

# Select agents for breeding
parents = population.select_parents(
    selection_strategy="tournament",
    count=4
)
```

### GenomeBreeder

Handles crossover, mutation, and natural selection:

```python
from aragora.genesis import GenomeBreeder

breeder = GenomeBreeder(
    mutation_rate=0.1,
    crossover_points=2
)

# Create offspring from two parents
child = breeder.crossover(parent_a, parent_b)

# Apply mutations
mutated = breeder.mutate(child, mutation_strength=0.05)
```

### FractalOrchestrator

Runs debates with recursive sub-debate spawning and agent evolution:

```python
from aragora.genesis import FractalOrchestrator

orchestrator = FractalOrchestrator(
    max_depth=2,        # Maximum sub-debate nesting
    evolve_agents=True  # Enable mid-debate evolution
)

result = await orchestrator.run(
    task="Design a distributed cache with strong consistency",
    agents=["anthropic-api", "gemini", "grok", "codex"]
)

# Access the debate tree
print(result.debate_tree)
print(result.evolved_genomes)
```

### GenesisLedger

Tracks all evolutionary events with cryptographic provenance:

```python
from aragora.genesis import GenesisLedger

ledger = GenesisLedger(db_path=".nomic/genesis.db")

# Query lineage
lineage = ledger.get_lineage("claude-grok-specialist-v1")

# Get events by type
mutations = ledger.get_events(event_type="mutation", limit=100)
```

## API Endpoints

### GET /api/genesis/stats

Get overall genesis statistics.

**Response:**
```json
{
  "total_genomes": 247,
  "total_events": 1523,
  "event_counts": {
    "spawn": 247,
    "mutation": 892,
    "crossover": 312,
    "selection": 72
  },
  "avg_fitness": 0.68,
  "top_genome_id": "claude-systems-v3"
}
```

### GET /api/genesis/genomes

List all genomes with pagination.

**Query Parameters:**
- `limit` (int, default: 50, max: 200)
- `offset` (int, default: 0)

**Response:**
```json
{
  "genomes": [
    {
      "genome_id": "claude-systems-v3",
      "base_model": "claude-opus-4-5-20251101",
      "fitness_score": 0.92,
      "traits": {"reasoning_depth": 0.95, ...},
      "expertise": ["distributed_systems", "consensus"],
      "generation": 3,
      "created_at": "2026-01-10T10:00:00Z"
    }
  ],
  "total": 247,
  "offset": 0,
  "limit": 50
}
```

### GET /api/genesis/genomes/top

Get top-performing genomes by fitness.

**Query Parameters:**
- `limit` (int, default: 10, max: 50)

### GET /api/genesis/genomes/\{genome_id\}

Get details for a specific genome.

### GET /api/genesis/lineage/\{genome_id\}

Get the full evolutionary lineage (ancestry) of a genome.

**Query Parameters:**
- `max_depth` (int, default: 10, max: 50) - Maximum ancestor depth to trace

**Response:**
```json
{
  "genome_id": "claude-systems-v3",
  "lineage": [
    {
      "genome_id": "claude-systems-v3",
      "generation": 3,
      "fitness_score": 0.92,
      "parent_ids": ["claude-systems-v2"],
      "event_type": "mutation",
      "created_at": "2026-01-10T10:00:00Z"
    },
    {
      "genome_id": "claude-systems-v2",
      "generation": 2,
      "fitness_score": 0.85,
      "parent_ids": ["claude-systems-v1", "grok-perf-v1"],
      "event_type": "crossover",
      "created_at": "2026-01-09T15:30:00Z"
    },
    {
      "genome_id": "claude-systems-v1",
      "generation": 1,
      "fitness_score": 0.78,
      "parent_ids": ["claude-base-v1"],
      "event_type": "mutation",
      "created_at": "2026-01-08T09:00:00Z"
    },
    {
      "genome_id": "claude-base-v1",
      "generation": 0,
      "fitness_score": 0.65,
      "parent_ids": [],
      "event_type": "spawn",
      "created_at": "2026-01-07T12:00:00Z"
    }
  ],
  "generations": 4
}
```

### GET /api/genesis/descendants/\{genome_id\}

Get all descendants of a genome (forward-looking family tree).

**Query Parameters:**
- `max_depth` (int, default: 5, max: 20) - Maximum descendant depth to search

**Response:**
```json
{
  "genome_id": "claude-base-v1",
  "descendants": [
    {
      "genome_id": "claude-systems-v1",
      "name": "Systems Specialist",
      "generation": 1,
      "fitness_score": 0.78,
      "parent_ids": ["claude-base-v1"],
      "depth": 1
    },
    {
      "genome_id": "claude-systems-v2",
      "name": "Systems Specialist v2",
      "generation": 2,
      "fitness_score": 0.85,
      "parent_ids": ["claude-systems-v1", "grok-perf-v1"],
      "depth": 2
    },
    {
      "genome_id": "claude-systems-v3",
      "name": "Systems Specialist v3",
      "generation": 3,
      "fitness_score": 0.92,
      "parent_ids": ["claude-systems-v2"],
      "depth": 3
    }
  ],
  "total_descendants": 3,
  "max_depth_reached": 3
}
```

### GET /api/genesis/events

Get recent genesis events.

**Query Parameters:**
- `limit` (int, default: 20, max: 100)
- `event_type` (string, optional: "spawn", "mutation", "crossover", "selection")

### GET /api/genesis/tree/\{debate_id\}

Get the fractal debate tree structure.

### GET /api/genesis/population

Get current population statistics.

## Evolution Mechanics

### Fitness Calculation

Genome fitness is calculated from debate performance:

```python
fitness = (
    win_rate * 0.4 +
    evidence_quality * 0.2 +
    novelty_score * 0.2 +
    consensus_contribution * 0.2
)
```

### Selection Strategies

**Tournament Selection:**
```python
# Select best from random sample
candidates = random.sample(population, tournament_size=4)
winner = max(candidates, key=lambda g: g.fitness_score)
```

**Roulette Selection:**
```python
# Probability proportional to fitness
total_fitness = sum(g.fitness_score for g in population)
probabilities = [g.fitness_score / total_fitness for g in population]
selected = random.choices(population, weights=probabilities)
```

**Elite Selection:**
```python
# Always keep top performers
elite = sorted(population, key=lambda g: g.fitness_score)[-elite_size:]
```

### Crossover Operations

**Single-point crossover:**
```python
# Split traits at one point
child_traits = {
    **parent_a.traits[:split_point],
    **parent_b.traits[split_point:]
}
```

**Uniform crossover:**
```python
# Randomly select each trait from either parent
child_traits = {
    key: random.choice([parent_a.traits[key], parent_b.traits[key]])
    for key in trait_keys
}
```

### Mutation Types

**Trait mutation:**
```python
# Small random adjustments to trait values
trait_value += random.gauss(0, mutation_strength)
trait_value = max(0, min(1, trait_value))  # Clamp to [0, 1]
```

**Expertise mutation:**
```python
# Add or remove expertise areas
if random.random() < add_expertise_prob:
    genome.expertise.append(random_expertise())
```

**Prompt mutation:**
```python
# Modify prompt template with LLM assistance
new_prompt = await llm.generate(
    f"Improve this agent prompt for {genome.expertise}: {genome.prompt_template}"
)
```

## Auto-Evolution in Debates

Enable automatic evolution during the feedback phase:

```python
from aragora import Arena, Environment, DebateProtocol

protocol = DebateProtocol(
    rounds=5,
    enable_auto_evolve=True  # Enable mid-debate evolution
)

# During debate, top performers may spawn evolved variants
```

### Evolution Triggers

Auto-evolution can be triggered by:

1. **High Performance** - Agent wins multiple rounds
2. **Novelty** - Agent introduces unique perspectives
3. **Synergy** - Two agents complement each other well
4. **Stagnation** - Population needs diversity injection

## Fractal Debates

Fractal debates spawn sub-debates for complex topics:

```
Main Debate: "Design a distributed cache"
├── Sub-debate 1: "What consistency model?"
│   ├── Sub-sub-debate: "How to handle network partitions?"
│   └── Sub-sub-debate: "Consistency vs availability trade-offs?"
├── Sub-debate 2: "Eviction strategy?"
└── Sub-debate 3: "Replication approach?"
```

### Depth Limiting

Control sub-debate depth to prevent runaway recursion:

```python
orchestrator = FractalOrchestrator(
    max_depth=2,          # Maximum nesting level
    min_complexity=0.6,   # Minimum complexity to spawn sub-debate
    spawn_threshold=0.8   # Disagreement threshold for spawning
)
```

## Event Types

The Genesis ledger tracks these event types:

| Event Type | Description |
|------------|-------------|
| `spawn` | New genome created from scratch |
| `mutation` | Genome modified by mutation |
| `crossover` | New genome from two parents |
| `selection` | Genome selected for next generation |
| `elimination` | Genome removed from population |
| `fitness_update` | Fitness score recalculated |
| `fractal_spawn` | Sub-debate spawned |
| `fractal_merge` | Sub-debate results merged |

## Streaming Events

Genesis emits real-time events during debates:

```python
from aragora.genesis import create_genesis_hooks

hooks = create_genesis_hooks(event_emitter)

# Events emitted:
# - genesis:spawn
# - genesis:mutation
# - genesis:crossover
# - genesis:fitness_update
# - genesis:generation_complete
```

## Example: Full Evolution Cycle

```python
from aragora.genesis import (
    FractalOrchestrator,
    PopulationManager,
    GenomeBreeder,
    GenesisLedger
)

# Initialize components
population = PopulationManager()
breeder = GenomeBreeder(mutation_rate=0.1)
ledger = GenesisLedger()

# Run debates with evolution
orchestrator = FractalOrchestrator(
    population=population,
    breeder=breeder,
    ledger=ledger,
    max_depth=2,
    evolve_agents=True
)

# Run multiple generations
for generation in range(10):
    # Select parents from top performers
    parents = population.select_parents(count=4)

    # Create offspring
    offspring = []
    for i in range(0, len(parents), 2):
        child = breeder.crossover(parents[i], parents[i+1])
        child = breeder.mutate(child)
        offspring.append(child)

    # Run debate with new population
    agents = population.get_agents() + offspring
    result = await orchestrator.run(
        task="Design a high-performance message queue",
        agents=agents
    )

    # Update fitness scores
    for agent, score in result.performance_scores.items():
        population.update_fitness(agent.genome_id, score)

    # Natural selection
    population.cull_population(keep_top=20)

    print(f"Generation \{generation\}: Top fitness = {population.top_fitness}")

# Query evolution history
top_genome = population.get_top_genomes(1)[0]
lineage = ledger.get_lineage(top_genome.genome_id)
print(f"Evolution path: {' -> '.join(g['genome_id'] for g in lineage)}")
```

## Related Features

- **ELO Rankings** - Agent performance tracking
- **A/B Testing** - Compare genome variants systematically
- **Trickster** - Quality enforcement during evolution
- **Memory** - Store successful genome configurations
