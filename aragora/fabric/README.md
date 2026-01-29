# Fabric - Agent Orchestration Substrate

High-scale agent orchestration substrate for managing 50+ concurrent AI agents with scheduling, isolation, policy enforcement, and budget tracking.

## Quick Start

```python
from aragora.fabric import AgentFabric, Task, AgentConfig

async with AgentFabric() as fabric:
    # Create agent pool
    pool = await fabric.create_pool("workers", "claude-3-opus", min_agents=5)

    # Schedule task
    task = Task(id="task-1", type="debate", payload={"question": "..."})
    handle = await fabric.schedule_to_pool(task, pool.id)

    # Wait for result
    result = await fabric.wait_for_task(handle.task_id)
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `AgentFabric` | `fabric.py` | Main facade for orchestration |
| `AgentScheduler` | `scheduler.py` | Priority-based task scheduling |
| `LifecycleManager` | `lifecycle.py` | Agent spawn, heartbeat, termination |
| `PolicyEngine` | `policy.py` | Access control and approval workflows |
| `BudgetManager` | `budget.py` | Cost tracking and enforcement |
| `HookManager` | `hooks.py` | GUPP persistence (crash-safe work queues) |
| `Nudge` | `nudge.py` | Inter-agent messaging |

## Architecture

```
fabric/
├── __init__.py       # Public API exports
├── models.py         # Data structures (Task, AgentConfig, Policy, etc.)
├── fabric.py         # AgentFabric facade + AgentPool
├── scheduler.py      # Priority-based fair queuing
├── lifecycle.py      # Agent spawn/termination
├── policy.py         # Access control rules
├── budget.py         # Token/cost tracking
├── hooks.py          # GUPP work persistence
└── nudge.py          # Agent messaging
```

## Features

### Task Scheduling
- 4 priority levels: CRITICAL, HIGH, NORMAL, LOW
- Per-agent queues with configurable depth
- Dependency tracking and resolution
- Timeout and cancellation support

### Agent Lifecycle
- Agent pooling for fast reuse
- Heartbeat monitoring (30s interval)
- Health status: HEALTHY, DEGRADED, UNHEALTHY
- Graceful shutdown with task draining

### Policy Engine
- Glob pattern matching for actions (e.g., `tool.shell.*`)
- Context-aware evaluation (agent, user, tenant)
- Human-in-loop approval workflows
- Priority-based rule evaluation

### Budget Management
- Multi-dimension limits: tokens, compute, USD
- Per-agent, per-user, per-tenant budgets
- Soft (warn) and hard (block) limits
- Usage alerts at configurable thresholds

### GUPP Hooks (Work Persistence)
Core principle: *"If there is work on your Hook, YOU MUST RUN IT"*
- Git worktree-backed work queues
- Crash-safe: work survives agent crashes
- Automatic stale detection and re-assignment

### Nudge (Inter-agent Messaging)
- Priority queues (LOW, NORMAL, HIGH, URGENT)
- Delivery tracking and acknowledgment
- TTL support with expiration
- Broadcast and direct messaging

## Usage Examples

### Create and Scale Pool

```python
async with AgentFabric() as fabric:
    pool = await fabric.create_pool(
        name="analysts",
        model="claude-3-opus",
        min_agents=3,
        max_agents=10
    )

    # Auto-scale based on load
    await fabric.scale_pool(pool.id, target=8)
```

### Policy Enforcement

```python
from aragora.fabric import Policy, PolicyRule, PolicyEffect

policy = Policy(
    id="shell-approval",
    rules=[
        PolicyRule(
            pattern="tool.shell.*",
            effect=PolicyEffect.REQUIRE_APPROVAL
        )
    ]
)
await fabric.add_policy(policy)

# Check before action
decision = await fabric.check_policy("tool.shell.execute", context)
if decision.effect == PolicyEffect.REQUIRE_APPROVAL:
    request = await fabric.request_approval(action, context)
```

### Budget Tracking

```python
from aragora.fabric import BudgetConfig

await fabric.set_budget(
    entity_id="agent-1",
    config=BudgetConfig(
        tokens_per_day=100_000,
        compute_seconds=3600,
        usd_per_day=50.0
    )
)

# Check before expensive operation
allowed, status = await fabric.check_budget("agent-1", estimated_tokens=5000)
```

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Control Plane](../control_plane/README.md) - Enterprise orchestration
- [Workflow Engine](../workflow/README.md) - DAG-based automation
