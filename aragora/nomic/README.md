# Nomic - Autonomous Self-Improvement System

Event-driven state machine for the Nomic Loop - autonomous cycle where agents debate improvements, design solutions, implement code, and verify changes.

## Quick Start

```python
from aragora.nomic import create_nomic_state_machine, NomicStateMachine

# Create state machine (recommended approach)
sm = create_nomic_state_machine(checkpoint_dir="/tmp/nomic")

# Run improvement cycle
result = await sm.run(goal="Improve test coverage in debate module")

# Or use autonomous orchestrator for complex goals
from aragora.nomic import AutonomousOrchestrator, Track

orchestrator = await AutonomousOrchestrator.create()
result = await orchestrator.execute(
    goal="Enhance SME dashboard experience",
    tracks=[Track.SME, Track.DEVELOPER],
    require_approval=True
)
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `NomicStateMachine` | `state_machine.py` | Event-driven orchestration |
| `AutonomousOrchestrator` | `autonomous_orchestrator.py` | Multi-track goal execution |
| `MetaPlanner` | `meta_planner.py` | Debate-driven goal prioritization |
| `TaskDecomposer` | `task_decomposer.py` | Break goals into subtasks |
| `BranchCoordinator` | `branch_coordinator.py` | Parallel branch management |
| `BeadStore` | `beads.py` | Atomic work units (JSONL + git) |
| `ConvoyManager` | `convoys.py` | Grouped work batches |
| `RecoveryManager` | `recovery.py` | Circuit breaker + recovery |

## Architecture

```
nomic/
├── state_machine.py         # Event-driven state machine
├── states.py                # NomicState enum, StateContext
├── events.py                # Event types with OpenTelemetry tracing
├── handlers.py              # Phase handlers for state machine
├── phases/                  # Phase implementations
│   ├── context.py           # Phase 0: Gather codebase context
│   ├── debate.py            # Phase 1: Multi-agent debate
│   ├── design.py            # Phase 2: Architecture planning
│   ├── implement.py         # Phase 3: Code generation
│   ├── verify.py            # Phase 4: Tests and validation
│   ├── commit.py            # Phase 5: Git commit
│   └── scope_limiter.py     # Protected file enforcement
├── autonomous_orchestrator.py  # Goal orchestration
├── meta_planner.py          # Goal prioritization
├── task_decomposer.py       # Task breakdown
├── branch_coordinator.py    # Parallel branch management
├── Gastown Patterns/
│   ├── beads.py             # Atomic work units
│   ├── convoys.py           # Grouped work orders
│   ├── hook_queue.py        # Per-agent work queues
│   ├── agent_roles.py       # Hierarchical agent coordination
│   └── molecules.py         # Durable multi-step workflows
├── recovery.py              # Circuit breaker, backoff
├── checkpoints.py           # State persistence
├── cycle_record.py          # Cross-cycle learning
├── cycle_store.py           # Cycle persistence
├── metrics.py               # Prometheus metrics
└── loop.py                  # Legacy NomicLoop API
```

## Nomic Loop Phases

```
START → CONTEXT → DEBATE → DESIGN → IMPLEMENT → VERIFY → COMMIT → END
         (0)       (1)      (2)       (3)        (4)      (5)
```

| Phase | Purpose | Gate |
|-------|---------|------|
| Context | Gather codebase understanding | Auto |
| Debate | Multi-agent debate on improvements | Consensus required |
| Design | Architecture planning | Human approval (optional) |
| Implement | Code generation (Claude/Codex) | Design approved |
| Verify | Run tests and validation | Tests must pass |
| Commit | Git commit changes | Verification passed |

## Development Tracks

```python
from aragora.nomic import Track, TrackConfig

# 5 development tracks with different constraints
Track.SME          # Dashboard, handlers (2 concurrent)
Track.DEVELOPER    # SDK, docs, tests (2 concurrent)
Track.SELF_HOSTED  # Scripts, docker (1 concurrent)
Track.QA           # Tests, E2E, CI/CD (3 concurrent)
Track.CORE         # Debate engine, agents (1 concurrent, Claude only)
```

## Task Decomposition

```python
from aragora.nomic import TaskDecomposer, analyze_task

# Quick heuristic decomposition (fast)
subtasks = analyze_task("Refactor dashboard.tsx and api.py")

# Debate-based decomposition (thorough)
decomposer = TaskDecomposer(use_debate=True)
result = await decomposer.decompose("Maximize utility for SME businesses")
```

## Gastown Patterns

### Beads (Atomic Work Units)
```python
from aragora.nomic import BeadStore, Bead
from aragora.nomic.beads import create_bead_store

store = await create_bead_store()
bead = Bead(
    id="impl-abc12",
    type="implement",
    description="Add form validation",
    depends_on=["spec-xyz99"]
)
await store.create(bead)
# Status: PENDING → ASSIGNED → RUNNING → DONE/FAILED
```

### Convoys (Work Batches)
```python
from aragora.nomic import ConvoyManager

manager = ConvoyManager(bead_store=store)
convoy = await manager.create_convoy(
    title="Example convoy",
    bead_ids=[bead1.id, bead2.id, bead3.id]
)
# Status: PENDING → ACTIVE → COMPLETED/FAILED/PARTIAL
```

#### Convoy Executor Storage

The Gastown-style convoy executor uses the canonical store root by default.
To opt out, set:

```bash
export NOMIC_CONVOY_CANONICAL_STORE=0
```

When enabled and no explicit `bead_dir` is supplied, the executor resolves
its bead/convoy storage under `<workspace_root>/.aragora_beads`.

## Recovery & Resilience

```python
from aragora.nomic import RecoveryManager, CircuitBreaker

# Per-agent circuit breaker
# Default: 3 failures → open, 300s reset
breaker = CircuitBreaker(agent_id="claude", threshold=3)

# Recovery strategies
# RETRY, SKIP, ROLLBACK, RESTART, PAUSE, FAIL
manager = RecoveryManager()
decision = await manager.decide(error, context)
```

## Cross-Cycle Learning

```python
from aragora.nomic import get_cycle_store, NomicCycleRecord

store = get_cycle_store()

# Records agent performance, patterns, surprise events
record = NomicCycleRecord(
    cycle_id="cycle-123",
    agent_contributions=[...],
    surprise_events=[...],
    pattern_reinforcements=[...]
)
await store.save(record)
```

## Safety Features

- **Checkpoint after every transition** - Resume on failure
- **Protected file enforcement** - ScopeLimiter prevents modification
- **Human approval gates** - Optional checkpoints for dangerous changes
- **Circuit breaker per agent** - Isolate failing agents
- **Exponential backoff** - Prevent cascade failures
- **Full event sourcing** - Audit trail with OpenTelemetry tracing

## CLI Tools

```bash
# Quick preview (dry run)
python scripts/self_develop.py --goal "Improve test coverage" --dry-run

# Full autonomous run
python scripts/self_develop.py --goal "Enhance SDK" --tracks developer qa

# Staged execution
python scripts/nomic_staged.py debate    # Run debate phase only
python scripts/nomic_staged.py all       # Run all phases
```

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Workspace](../workspace/README.md) - Rig/Bead/Convoy containers
- [Debate](../debate/README.md) - Debate orchestration
- [Fabric](../fabric/README.md) - Agent orchestration substrate
