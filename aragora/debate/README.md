# Debate Module

Core debate orchestration engine for multi-agent consensus building.

## Quick Start

```python
from aragora.debate import Arena, DebateProtocol, ArenaBuilder
from aragora import Environment

# Simple usage
env = Environment(task="Design a rate limiter")
protocol = DebateProtocol(rounds=9, consensus="hybrid")
arena = Arena(env, agents, protocol)
result = await arena.run()

# Builder pattern (recommended)
arena = (ArenaBuilder()
    .with_environment(env)
    .with_agents(agents)
    .with_protocol(protocol)
    .build())
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `Arena` | Core orchestration engine; manages propose → critique → revise → consensus loop |
| `DebateProtocol` | Configuration for rounds, timeouts, and consensus strategy |
| `ArenaConfig` | Type-safe configuration dataclass |
| `ArenaBuilder` | Fluent builder pattern for construction |
| `ConsensusProof` | Auditable artifacts with claims, evidence, dissent tracking |
| `ConvergenceDetector` | Semantic similarity detection for debate convergence |
| `JudgeSelector` | Selects qualified judges and panels for evaluation |

## Architecture

```
debate/
├── orchestrator.py      # Arena class - main entry point
├── consensus.py         # ConsensusProof, ConsensusBuilder
├── convergence.py       # ConvergenceDetector, AdvancedConvergenceAnalyzer
├── protocol.py          # DebateProtocol, RoundPhase
├── arena_config.py      # ArenaConfig dataclass
├── arena_builder.py     # ArenaBuilder fluent API
├── judge_selector.py    # JudgeSelector, JudgePanel
├── phases/              # Extracted phase implementations
├── prompts/             # Domain-specific prompt templates
├── protocol_messages/   # Message handlers and verification
├── cache/               # Embedding cache for optimization
└── similarity/          # 3-tier fallback similarity backends
```

## Advanced Features

- **Counterfactuals**: `CounterfactualOrchestrator` for exploring branches
- **Graph Debates**: `DebateGraph` for non-linear debate structures
- **Checkpointing**: `CheckpointManager` for pause/resume
- **Multilingual**: `TranslationService` for cross-language debates
- **Resilience**: `CircuitBreaker` for agent failure handling

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/STATUS.md](../../docs/STATUS.md) - Feature status
