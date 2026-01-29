# Workspace - Gastown Developer Orchestration

Per-repo project containers with work batch tracking and merge queue capabilities for multi-agent parallel development.

## Quick Start

```python
from aragora.workspace import WorkspaceManager

ws = WorkspaceManager(workspace_root="/path/to/projects")

# Create project container
rig = await ws.create_rig("my-repo", repo_url="https://github.com/...")

# Create work batch
convoy = await ws.create_convoy(rig.rig_id, bead_specs=[
    {"type": "implement", "description": "Add feature X"},
    {"type": "test", "description": "Write tests", "depends_on": ["bead-1"]}
])

# Execute
await ws.start_convoy(convoy.convoy_id)
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `Rig` | `rig.py` | Per-repo project container |
| `Bead` | `bead.py` | Atomic work unit (task) |
| `Convoy` | `convoy.py` | Batch of related work items |
| `WorkspaceManager` | `manager.py` | Top-level orchestration |
| `Refinery` | `refinery.py` | Merge queue for integration |

## Architecture

```
workspace/
├── __init__.py    # Public API exports
├── rig.py         # Project containers
├── bead.py        # Atomic work units
├── convoy.py      # Work batch tracking
├── manager.py     # Orchestration facade
└── refinery.py    # Merge queue
```

## Concepts

### Rig (Project Container)
Per-repo container managing agents and tasks.

```python
class RigConfig:
    repo_url: str
    branch: str = "main"
    max_agents: int = 5
    max_concurrent_tasks: int = 10

# Status: INITIALIZING → READY → ACTIVE → DRAINING → STOPPED
```

### Bead (Work Unit)
Atomic task with crash-resilient JSONL persistence.

```python
# Status: PENDING → ASSIGNED → RUNNING → DONE/FAILED/SKIPPED
bead = Bead(
    id="impl-abc12",
    type="implement",
    description="Add validation",
    depends_on=["spec-xyz99"]
)
```

### Convoy (Work Batch)
Coordinated batch of beads executed as a unit.

```python
# Status: CREATED → ASSIGNING → EXECUTING → MERGING → DONE
convoy = Convoy(
    id="convoy-123",
    rig_id="rig-abc",
    beads=[bead1, bead2, bead3]
)
```

### Refinery (Merge Queue)
Orderly integration of completed work with conflict handling.

```python
from aragora.workspace import Refinery, MergeRequest

refinery = Refinery(repo_path="/path/to/repo")

# Queue merge request
request = MergeRequest(
    branch="feature-xyz",
    target="main",
    convoy_id="convoy-123"
)
await refinery.queue(request)

# Process queue
await refinery.process_next()
```

## Status Flows

### Rig Lifecycle
```
INITIALIZING → READY → ACTIVE → DRAINING → STOPPED
                 ↓                           ↓
               ERROR ←─────────────────────────
```

### Bead Lifecycle
```
PENDING → ASSIGNED → RUNNING → DONE
                        ↓      ↓
                     FAILED  SKIPPED
```

### Convoy Lifecycle
```
CREATED → ASSIGNING → EXECUTING → MERGING → DONE
              ↓           ↓          ↓
          CANCELLED    FAILED    FAILED
```

### Merge Request Lifecycle
```
QUEUED → VALIDATING → REBASING → MERGING → MERGED
             ↓            ↓          ↓
          FAILED     CONFLICT   ROLLED_BACK
```

## Features

- **Crash Resilience**: JSONL-backed state persistence
- **Dependency Tracking**: Beads can depend on other beads
- **Priority Queue**: Merge requests processed by priority
- **Conflict Handling**: Configurable resolution strategies
- **Agent Pools**: Rigs manage crew of assigned agents
- **Progress Tracking**: Real-time convoy status and statistics

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Fabric](../fabric/README.md) - Agent orchestration substrate
- [Nomic](../nomic/README.md) - Self-improvement loop (uses Bead/Convoy)
