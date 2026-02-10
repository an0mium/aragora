# Workspace Guide

Gastown-style developer orchestration with rigs, convoys, and beads.

## Overview

The `aragora.workspace` module provides project management capabilities:

- **Rigs**: Per-repo project containers with isolated agent pools
- **Convoys**: Batches of related work items tracked as a unit
- **Beads**: Atomic units of work with JSONL-backed state tracking
- **Refinery**: Merge queue for coordinating parallel agent work

## Storage & persistence

By default, workspace bead/convoy adapters use the canonical Nomic stores and
persist under the canonical store directory.

To control persistence or location:
- Set `ARAGORA_STORE_DIR=/path/to/.aragora_beads` for an explicit location
- Set `ARAGORA_CANONICAL_STORE_PERSIST=0` to force an ephemeral temp directory

Gastown workspace metadata (workspaces/rigs) is persisted to
`state.json` under the canonical store directory when using the
Gastown extension workspace manager.

Gastown extension persistence files:
- `state.json` (workspaces + rigs)
- `hooks.json` (hook definitions + trigger stats)
- `ledger.json` (Mayor/Coordinator issue ledger, when using the coordinator)

You can migrate any legacy Gastown state into the canonical store:

```
aragora gt migrate
aragora gt migrate --apply
```

## Quick Start

```python
from aragora.workspace import WorkspaceManager, RigConfig

# Create workspace manager
ws = WorkspaceManager(workspace_root="/path/to/projects")

# Create a rig (per-repo container)
rig = await ws.create_rig(
    name="my-project",
    config=RigConfig(repo_url="https://github.com/org/repo"),
)

# Create a convoy with work items
convoy = await ws.create_convoy(
    rig_id=rig.rig_id,
    name="Feature Implementation",
    bead_specs=[
        {"title": "Write tests", "description": "Add unit tests"},
        {"title": "Implement feature", "depends_on": ["bead-1"]},
        {"title": "Update docs"},
    ],
)

# Start execution
await ws.start_convoy(convoy.convoy_id)

# Check status
status = await ws.get_convoy_status(convoy.convoy_id)
print(f"Progress: {status['progress_percent']}%")
```

## Core Concepts

### Rigs

A rig is a per-repo project container with its own agent pool and configuration:

```python
from aragora.workspace import Rig, RigConfig, RigStatus

# Configure a rig
config = RigConfig(
    repo_url="https://github.com/org/repo",
    branch="main",
    max_agents=5,
    auto_merge=True,
)

# Create the rig
rig = await ws.create_rig(name="backend", config=config)

# Assign agents
await ws.assign_agent_to_rig(rig.rig_id, "agent-123")
await ws.assign_agent_to_rig(rig.rig_id, "agent-456")

# List rigs by status
active_rigs = await ws.list_rigs(status=RigStatus.READY)
```

### Convoys

A convoy is a batch of related work items (beads) tracked as a unit:

```python
from aragora.workspace import Convoy, ConvoyStatus

# Create convoy with dependencies
convoy = await ws.create_convoy(
    rig_id=rig.rig_id,
    name="Sprint 42",
    description="Implement new feature set",
    bead_specs=[
        {
            "title": "Design API",
            "description": "Design REST endpoints",
            "payload": {"type": "design"},
        },
        {
            "title": "Implement API",
            "depends_on": ["bead-1"],  # Waits for design
            "payload": {"type": "code"},
        },
        {
            "title": "Write tests",
            "depends_on": ["bead-2"],  # Waits for implementation
            "payload": {"type": "test"},
        },
    ],
)

# Start execution
await ws.start_convoy(convoy.convoy_id)

# List active convoys
active = await ws.list_convoys(
    rig_id=rig.rig_id,
    status=ConvoyStatus.EXECUTING,
)
```

### Beads

A bead is an atomic unit of work with JSONL-backed state:

```python
from aragora.workspace import Bead, BeadStatus

# Get ready beads (dependencies satisfied)
ready_beads = await ws.get_ready_beads(convoy.convoy_id)

for bead in ready_beads:
    # Assign to an agent
    await ws.assign_bead(bead.bead_id, agent_id="agent-123")

    # ... agent does work ...

    # Complete the bead
    await ws.complete_bead(
        bead.bead_id,
        result={"files_changed": ["api.py"], "tests_passed": True},
    )

# Handle failures
await ws.fail_bead(bead.bead_id, error="Build failed")
```

### Refinery

The refinery manages the orderly merging of completed convoy work:

```python
from aragora.workspace import Refinery, RefineryConfig, MergeStatus

# Configure refinery
config = RefineryConfig(
    max_concurrent_merges=1,
    auto_rebase=True,
    require_tests=True,
    require_review=False,
    retry_on_conflict=3,
)

refinery = Refinery(config=config)

# Queue convoy for merge
request = await refinery.queue_for_merge(
    convoy_id=convoy.convoy_id,
    rig_id=rig.rig_id,
    source_branch="feature/sprint-42",
    target_branch="main",
    priority=1,  # Higher = more urgent
)

# Process merge queue
processed = await refinery.process_queue()

for req in processed:
    if req.status == MergeStatus.MERGED:
        print(f"Merged: {req.merge_commit}")
    elif req.status == MergeStatus.CONFLICT:
        print(f"Conflicts in: {req.conflict_files}")
```

## API Reference

### WorkspaceManager

Main entry point for workspace management.

| Method | Description |
|--------|-------------|
| `create_rig(name, config, rig_id)` | Create a new rig |
| `get_rig(rig_id)` | Get a rig by ID |
| `list_rigs(status)` | List rigs with optional filter |
| `assign_agent_to_rig(rig_id, agent_id)` | Assign agent to rig |
| `remove_agent_from_rig(rig_id, agent_id)` | Remove agent from rig |
| `stop_rig(rig_id)` | Stop and drain a rig |
| `delete_rig(rig_id)` | Delete a rig |
| `create_convoy(rig_id, name, ...)` | Create a convoy with beads |
| `get_convoy(convoy_id)` | Get a convoy by ID |
| `get_convoy_status(convoy_id)` | Get detailed convoy status |
| `start_convoy(convoy_id)` | Start executing a convoy |
| `complete_convoy(convoy_id, merge_result)` | Mark convoy complete |
| `list_convoys(rig_id, status)` | List convoys with filters |
| `get_bead(bead_id)` | Get a bead by ID |
| `assign_bead(bead_id, agent_id)` | Assign bead to agent |
| `complete_bead(bead_id, result)` | Mark bead complete |
| `fail_bead(bead_id, error)` | Mark bead failed |
| `get_ready_beads(convoy_id)` | Get beads ready for execution |
| `get_stats()` | Get workspace statistics |

### RigConfig

Configuration for a rig.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_url` | str | "" | Repository URL |
| `branch` | str | "main" | Default branch |
| `max_agents` | int | 10 | Max concurrent agents |
| `auto_merge` | bool | False | Auto-merge completed work |

### ConvoyStatus

Convoy lifecycle states.

| Status | Description |
|--------|-------------|
| `PENDING` | Created but not started |
| `EXECUTING` | Work in progress |
| `MERGING` | Being merged via refinery |
| `COMPLETED` | Successfully finished |
| `FAILED` | Failed during execution |

### BeadStatus

Bead lifecycle states.

| Status | Description |
|--------|-------------|
| `PENDING` | Waiting for dependencies |
| `ASSIGNED` | Assigned to an agent |
| `RUNNING` | Work in progress |
| `DONE` | Successfully completed |
| `FAILED` | Failed during execution |

### MergeStatus

Merge request states.

| Status | Description |
|--------|-------------|
| `QUEUED` | Waiting in queue |
| `VALIDATING` | Running tests/reviews |
| `REBASING` | Rebasing onto target |
| `MERGING` | Performing merge |
| `MERGED` | Successfully merged |
| `CONFLICT` | Has merge conflicts |
| `FAILED` | Failed to merge |
| `ROLLED_BACK` | Reverted after merge |

## Integration

### With Agent Fabric

```python
from aragora.fabric import AgentFabric
from aragora.workspace import WorkspaceManager

fabric = AgentFabric()
ws = WorkspaceManager()

# Create pool for rig
pool = await fabric.create_pool(
    pool_id="pool-backend",
    name="Backend Pool",
    model="claude-3-opus",
    min_agents=2,
    max_agents=10,
)

# Assign pool agents to rig
rig = await ws.create_rig(name="backend")
for agent_id in pool.current_agents:
    await ws.assign_agent_to_rig(rig.rig_id, agent_id)
```

### With Gateway

```python
from aragora.gateway import LocalGateway, InboxMessage
from aragora.workspace import WorkspaceManager

gateway = LocalGateway()
ws = WorkspaceManager()

# Route incoming message to create bead
async def handle_message(message: InboxMessage):
    # Create convoy from message
    convoy = await ws.create_convoy(
        rig_id="rig-support",
        name=f"Task: {message.content[:50]}",
        bead_specs=[{
            "title": message.content,
            "payload": {"source": message.channel},
        }],
    )
    await ws.start_convoy(convoy.convoy_id)
```

## Examples

### Sprint Planning

```python
# Create sprint convoy
convoy = await ws.create_convoy(
    rig_id=rig.rig_id,
    name="Sprint 43",
    bead_specs=[
        # Independent tasks (can run in parallel)
        {"title": "JIRA-101: Fix login bug"},
        {"title": "JIRA-102: Add logging"},
        {"title": "JIRA-103: Update docs"},
        # Dependent task
        {
            "title": "JIRA-104: Deploy",
            "depends_on": ["bead-1", "bead-2", "bead-3"],
        },
    ],
)
```

### Multi-Agent Coordination

```python
# Get ready beads and distribute to agents
ready = await ws.get_ready_beads(convoy.convoy_id)

# Assign to available agents
for i, bead in enumerate(ready):
    agent_id = available_agents[i % len(available_agents)]
    await ws.assign_bead(bead.bead_id, agent_id)
```

### Merge Queue

```python
# Queue multiple convoys for merge with priorities
await refinery.queue_for_merge(
    convoy_id="cv-hotfix",
    rig_id=rig.rig_id,
    source_branch="hotfix/critical",
    priority=10,  # High priority
)

await refinery.queue_for_merge(
    convoy_id="cv-feature",
    rig_id=rig.rig_id,
    source_branch="feature/new",
    priority=1,  # Normal priority
)

# Process in priority order
await refinery.process_queue()
```

---

*Part of Aragora control plane for multi-agent robust decisionmaking*
