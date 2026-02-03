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
    .with_knowledge(enable_knowledge_retrieval=True)
    .with_audit_trail(enable_receipt_generation=True)
    .build())

# Config objects
from aragora.debate.arena_config import DebateConfig, MemoryConfig

arena = Arena(
    environment=env,
    agents=agents,
    debate_config=DebateConfig(rounds=5, consensus_threshold=0.8),
    memory_config=MemoryConfig(enable_knowledge_retrieval=True),
)

# Context manager (recommended for cleanup)
async with arena:
    result = await arena.run()
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `Arena` | `orchestrator.py` | Core orchestration engine; manages propose -> critique -> revise -> consensus loop |
| `DebateProtocol` | `protocol.py` | Configuration for rounds, timeouts, consensus strategy, topology |
| `ArenaConfig` | `arena_config.py` | Type-safe configuration with sub-group dataclasses |
| `ArenaBuilder` | `arena_builder.py` | Fluent builder pattern for construction |
| `DebateContext` | `debate_state.py` | Shared mutable state container for all phases |
| `ConsensusProof` | `consensus.py` | Auditable artifacts with claims, evidence, dissent tracking |
| `ConvergenceDetector` | `convergence.py` | Semantic similarity detection (3-tier fallback) |
| `JudgeSelector` | `judge_selector.py` | Judge and panel selection with multiple strategies |
| `TeamSelector` | `team_selector.py` | Agent selection by ELO, calibration, domain expertise |
| `PromptBuilder` | `prompt_builder.py` | Dynamic prompt construction with evidence and memory injection |
| `MemoryManager` | `memory_manager.py` | Multi-tier memory management (fast/medium/slow/glacial) |
| `PhaseExecutor` | `phase_executor.py` | Sequential phase coordination |

## Architecture

```
debate/
├── orchestrator.py          # Arena class - main entry point
├── orchestrator_*.py        # 15 extracted modules (agents, config, memory, etc.)
├── arena_config.py          # Type-safe configuration dataclass
├── arena_builder.py         # Fluent builder API
├── factory.py               # Dependency injection factory
│
├── protocol.py              # DebateProtocol, RoundPhase definitions
├── debate_state.py          # DebateContext shared state
├── consensus.py             # ConsensusProof, Claims, Dissent tracking
├── convergence.py           # ConvergenceDetector, similarity caching
│
├── phases/                  # Phase implementations (40+ files)
│   ├── context_init.py      # Phase 0: Context gathering & evidence
│   ├── proposal_phase.py    # Phase 1: Initial proposals
│   ├── debate_rounds.py     # Phases 2-7: Core critique/revise loop
│   ├── voting.py            # Phase 8a: Vote collection
│   ├── consensus_phase.py   # Phase 8b: Final consensus
│   ├── analytics_phase.py   # Post-debate analytics
│   └── feedback_phase.py    # Post-debate feedback (ELO, personas)
│
├── team_selector.py         # Agent team selection (ELO + calibration + domain)
├── judge_selector.py        # Judge selection and multi-judge panels
├── voting_engine.py         # Vote counting and weighted aggregation
├── prompt_builder.py        # Dynamic prompt generation
├── memory_manager.py        # Memory coordination across tiers
├── roles.py                 # Cognitive roles (Analyst, Skeptic, Synthesizer, etc.)
│
├── similarity/              # 3-tier fallback similarity backends
│   ├── backends.py          # SentenceTransformer, TF-IDF, Jaccard
│   └── factory.py           # Backend factory pattern
│
├── cache/                   # Embedding and similarity caches
├── checkpoint.py            # Debate checkpointing for pause/resume
├── event_bus.py             # Event emission (debate_start, vote, consensus, etc.)
└── lifecycle_manager.py     # Async context manager and cleanup
```

## Execution Flow

```
Arena.run()
  │
  ├─ Phase 0: Context Initialization
  │    ├─ Gather background evidence
  │    ├─ Select agent team (TeamSelector)
  │    ├─ Assign cognitive roles
  │    └─ Initialize memory and convergence caches
  │
  ├─ Phase 1: Proposals
  │    └─ All agents generate initial proposals (parallel)
  │
  ├─ Phases 2-7: Debate Rounds
  │    ├─ Critics challenge proposals (parallel, configurable concurrency)
  │    ├─ Proposers revise based on critique
  │    ├─ Convergence check (early termination if threshold met)
  │    └─ Role rotation per round
  │
  ├─ Phase 8a: Voting
  │    ├─ Agents vote on best proposals
  │    └─ Votes weighted by ELO, calibration, consistency
  │
  ├─ Phase 8b: Consensus
  │    ├─ Judge or mechanism selects winner
  │    ├─ Generate ConsensusProof with dissent tracking
  │    └─ Verify consensus integrity
  │
  ├─ Analytics: Extract insights, belief changes, relationships
  │
  └─ Feedback: Update ELO ratings, persona profiles, calibration scores
```

## Debate Rounds

The default 9-round protocol assigns cognitive modes per round:

| Round | Name | Cognitive Mode |
|-------|------|----------------|
| 0 | Context Gathering | (parallel with Round 1) |
| 1 | Initial Analysis | Analyst |
| 2 | Skeptical Review | Skeptic |
| 3 | Lateral Exploration | Lateral Thinker |
| 4 | Devil's Advocacy | Devil's Advocate |
| 5 | Integration | Synthesizer |
| 6 | Cross-Examination | Quality Challenger |
| 7 | Final Synthesis | Synthesizer |
| 8 | Final Adjudication | Judge |

## Configuration

### DebateConfig
- `rounds: int = 9` - Number of debate rounds
- `consensus_threshold: float = 0.66` - Majority threshold
- `enable_early_termination: bool = True` - Allow early exit on convergence
- `convergence_threshold: float = 0.85` - Semantic similarity cutoff

### AgentConfig
- `use_airlock: bool = False` - Wrap agents with resilience proxy
- `team_size: int | None = None` - Limit agent pool
- `enable_hierarchy: bool = False` - Use agent hierarchy roles

### MemoryConfig
- `enable_knowledge_retrieval: bool = False` - Retrieve past decisions
- `enable_knowledge_ingestion: bool = False` - Store outcomes
- `enable_continuum_memory: bool = True` - Use multi-tier memory

### ObservabilityConfig
- `enable_telemetry: bool = True` - Metrics collection
- `enable_tracing: bool = True` - Distributed tracing
- `enable_perf_monitoring: bool = True` - Performance tracking

## Consensus Mechanisms

| Mechanism | Description |
|-----------|-------------|
| `majority` | Simple majority vote |
| `judge` | Single judge renders verdict |
| `weighted` | Votes weighted by agent reputation |
| `hybrid` | Combines voting with judge adjudication |
| `supermajority` | Requires 2/3+ agreement |
| `unanimous` | Requires full agreement |

## Judge Selection Strategies

| Strategy | Description |
|----------|-------------|
| `last` | Use synthesizer or last agent |
| `random` | Random selection |
| `voted` | Agents vote for judge |
| `elo_ranked` | Highest ELO agent judges |
| `calibrated` | Best composite score (70% ELO + 30% calibration) |
| `crux_aware` | Prefer historical dissenters on similar topics |

Multi-judge panels support `MAJORITY`, `SUPERMAJORITY`, `UNANIMOUS`, and `WEIGHTED` strategies.

## Similarity Detection

3-tier fallback for convergence detection:

1. **SentenceTransformer** - Best accuracy, requires `sentence-transformers`
2. **TF-IDF** - Good accuracy, requires `scikit-learn`
3. **Jaccard** - Always available, zero dependencies

Performance is optimized via LRU caches (256 pairs per backend) and session-scoped similarity caches.

## Advanced Features

- **Counterfactuals**: `CounterfactualOrchestrator` for exploring alternative debate branches
- **Graph Debates**: `DebateGraph` / `GraphOrchestrator` for DAG-based non-linear debates
- **Checkpointing**: `CheckpointManager` for pause/resume across sessions
- **Multilingual**: `TranslationService` for cross-language debates
- **Resilience**: Circuit breakers for agent failure handling
- **Byzantine Resistance**: Detection and mitigation of Byzantine agents
- **Distributed Debates**: Coordination across multiple processes/machines
- **Trickster Detection**: Detect and challenge hollow consensus
- **Bias Mitigation**: Identify and mitigate systematic agent biases
- **Knowledge Mound**: Bidirectional integration with 25 KM adapters
- **ML Delegation**: ML-based complexity-aware agent delegation

## WebSocket Events

Subscribe to real-time debate events:

| Event | When |
|-------|------|
| `debate_start` | Debate begins |
| `round_start` | New round starts |
| `agent_message` | Agent submits proposal/critique |
| `critique` | Critique is generated |
| `vote` | Agent casts a vote |
| `consensus` | Consensus is reached |
| `debate_end` | Debate completes |

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/STATUS.md](../../docs/STATUS.md) - Feature status
- [docs/API_REFERENCE.md](../../docs/API_REFERENCE.md) - REST API documentation
