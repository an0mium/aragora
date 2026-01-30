# Fabric Guide

High-scale agent orchestration with pools, scheduling, and policies.

## Overview

The `aragora.fabric` module provides agent orchestration infrastructure:

- **AgentFabric**: Central facade for agent lifecycle and scheduling
- **AgentScheduler**: Fair task distribution with priority queues
- **LifecycleManager**: Agent spawn, heartbeat, and termination
- **PolicyEngine**: Tool access control and approval workflows
- **BudgetManager**: Cost tracking and enforcement
- **HookManager**: GUPP-compliant work persistence (Gastown parity)

## Quick Start

```python
from aragora.fabric import AgentFabric, FabricConfig, AgentConfig

# Create fabric with configuration
config = FabricConfig(
    max_concurrent_agents=100,
    max_concurrent_tasks_per_agent=5,
    default_timeout_seconds=300.0,
    heartbeat_interval_seconds=30.0,
)

fabric = AgentFabric(config=config)

# Spawn an agent
agent = await fabric.spawn(AgentConfig(
    model="claude-3-opus",
    name="code-review-agent",
))

# Schedule a task
task = await fabric.schedule(
    agent_id=agent.id,
    task_type="code_review",
    payload={"pr_url": "https://github.com/org/repo/pull/123"},
)

# Wait for completion
result = await fabric.wait_for_task(task.id)
print(f"Result: {result}")

# Terminate agent
await fabric.terminate(agent.id)
```

## Core Concepts

### AgentFabric

The central orchestration facade:

```python
from aragora.fabric import AgentFabric, FabricConfig

# Configure fabric
config = FabricConfig(
    max_queue_depth=1000,
    default_timeout_seconds=300.0,
    heartbeat_interval_seconds=30.0,
    max_concurrent_agents=100,
    max_concurrent_tasks_per_agent=5,
)

fabric = AgentFabric(config=config)

# Get fabric statistics
stats = await fabric.get_fabric_stats()
print(f"Active agents: {stats.active_agents}")
print(f"Pending tasks: {stats.pending_tasks}")
print(f"Total tasks completed: {stats.completed_tasks}")
```

### Agent Pools

Manage groups of agents with shared configuration:

```python
from aragora.fabric import AgentFabric, AgentPool

# Create a pool
pool = await fabric.create_pool(
    pool_id="pool-frontend",
    name="Frontend Agents",
    model="claude-3-opus",
    min_agents=2,
    max_agents=10,
)

# Scale pool
await fabric.scale_pool("pool-frontend", target_size=5)

# Schedule to specific pool
task = await fabric.schedule_to_pool(
    pool_id="pool-frontend",
    task_type="ui_review",
    payload={"component": "Dashboard"},
)

# Delete pool (drains agents first)
await fabric.delete_pool("pool-frontend")
```

### Scheduling

Fair task distribution with priorities:

```python
from aragora.fabric import Priority, Task

# Schedule high-priority task
task = await fabric.schedule(
    agent_id=agent.id,
    task_type="critical_fix",
    payload={"issue": "memory_leak"},
    priority=Priority.HIGH,
)

# Schedule with deadline
task = await fabric.schedule(
    agent_id=agent.id,
    task_type="report",
    payload={"type": "weekly"},
    priority=Priority.NORMAL,
    deadline=time.time() + 3600,  # 1 hour
)
```

### Lifecycle Management

Agent spawn, heartbeat, and termination:

```python
from aragora.fabric import LifecycleManager, AgentConfig

lifecycle = LifecycleManager()

# Spawn agent
agent = await lifecycle.spawn(AgentConfig(
    model="claude-3-opus",
    name="worker-1",
    capabilities=["code", "test", "review"],
))

# Record heartbeat
await lifecycle.heartbeat(agent.id)

# Check health
info = await lifecycle.get_agent_info(agent.id)
print(f"Last heartbeat: {info.last_heartbeat}")
print(f"Status: {info.status}")

# Terminate
await lifecycle.terminate(agent.id)
```

### Policy Engine

Tool access control and approvals:

```python
from aragora.fabric import PolicyEngine, Policy, PolicyRule

engine = PolicyEngine()

# Define policy
policy = Policy(
    name="code-execution",
    rules=[
        PolicyRule(
            tool="execute_code",
            action="require_approval",
            conditions={"language": "python"},
        ),
        PolicyRule(
            tool="file_write",
            action="deny",
            conditions={"path": "/etc/*"},
        ),
    ],
)

await engine.add_policy(policy)

# Check access
context = PolicyContext(
    agent_id=agent.id,
    tool="execute_code",
    params={"language": "python", "code": "..."},
)

decision = await engine.evaluate(context)
if decision.action == "require_approval":
    # Wait for human approval
    approved = await wait_for_approval(decision.approval_id)
```

### Budget Management

Cost tracking and enforcement:

```python
from aragora.fabric import BudgetManager, BudgetConfig

budget = BudgetManager()

# Set budget for agent
await budget.set_budget(
    agent_id=agent.id,
    config=BudgetConfig(
        max_tokens_per_hour=100000,
        max_cost_per_day=10.0,
        warn_at_percent=80,
    ),
)

# Record usage
await budget.record_usage(
    agent_id=agent.id,
    tokens=1500,
    cost=0.05,
)

# Check status
status = await budget.get_status(agent.id)
print(f"Tokens used: {status.tokens_used}/{status.tokens_limit}")
print(f"Cost: ${status.cost_used:.2f}/${status.cost_limit:.2f}")
```

### Hook Manager (GUPP)

Git-backed work persistence:

```python
from aragora.fabric import HookManager, HookManagerConfig

hooks = HookManager(HookManagerConfig(
    storage_path="/path/to/hooks",
    use_git_worktree=True,
))

# Create hook (persisted to git worktree)
hook = await hooks.create_hook(
    agent_id=agent.id,
    work_item={"task": "refactor", "file": "api.py"},
    branch="agent/refactor-123",
)

# Check for pending hooks (GUPP patrol)
pending = await hooks.check_pending_hooks()
for h in pending:
    print(f"Pending: {h.hook_id} - {h.work_item}")

# Complete hook
await hooks.complete_hook(
    hook.hook_id,
    result={"files_changed": ["api.py"], "tests_passed": True},
)
```

## API Reference

### FabricConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_queue_depth` | int | 1000 | Max pending tasks |
| `default_timeout_seconds` | float | 300.0 | Task timeout |
| `heartbeat_interval_seconds` | float | 30.0 | Heartbeat interval |
| `max_concurrent_agents` | int | 100 | Max active agents |
| `max_concurrent_tasks_per_agent` | int | 5 | Max tasks per agent |

### AgentConfig

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model identifier (claude-3-opus, etc.) |
| `name` | str | Agent display name |
| `capabilities` | list | List of capabilities |
| `isolation` | IsolationConfig | Isolation settings |
| `budget` | BudgetConfig | Budget limits |

### Priority

Task priority levels.

| Level | Value | Description |
|-------|-------|-------------|
| `CRITICAL` | 0 | Emergency tasks |
| `HIGH` | 1 | Important tasks |
| `NORMAL` | 2 | Standard tasks |
| `LOW` | 3 | Background tasks |

### TaskStatus

| Status | Description |
|--------|-------------|
| `PENDING` | Waiting in queue |
| `SCHEDULED` | Assigned to agent |
| `RUNNING` | In progress |
| `COMPLETED` | Successfully finished |
| `FAILED` | Failed during execution |
| `TIMEOUT` | Exceeded time limit |
| `CANCELLED` | Manually cancelled |

### HookStatus

| Status | Description |
|--------|-------------|
| `PENDING` | Not yet started |
| `ACTIVE` | Work in progress |
| `COMPLETED` | Successfully finished |
| `FAILED` | Failed with error |
| `ABANDONED` | Agent crashed, needs reassignment |

## Integration

### With Workspace

```python
from aragora.fabric import AgentFabric
from aragora.workspace import WorkspaceManager

fabric = AgentFabric()
ws = WorkspaceManager()

# Assign fabric agents to rig
rig = await ws.create_rig(name="backend")

pool = await fabric.create_pool(
    pool_id="pool-backend",
    model="claude-3-opus",
    min_agents=2,
)

for agent_id in pool.current_agents:
    await ws.assign_agent_to_rig(rig.rig_id, agent_id)
```

Note: Workspace convoys/beads persist only when `ARAGORA_CANONICAL_STORE_PERSIST=1`
or `ARAGORA_STORE_DIR` is set. Otherwise, local runs use an ephemeral temp store.

### With Gateway

```python
from aragora.fabric import AgentFabric
from aragora.gateway import LocalGateway

fabric = AgentFabric()
gateway = LocalGateway()

# Route message to fabric for handling
async def handle_message(message):
    agent = await fabric.get_available_agent(pool_id="pool-support")
    task = await fabric.schedule(
        agent_id=agent.id,
        task_type="handle_message",
        payload=message.to_dict(),
    )
    return task.id
```

### With Computer Use

```python
from aragora.fabric import AgentFabric
from aragora.computer_use import ComputerUseOrchestrator

fabric = AgentFabric()
computer_use = ComputerUseOrchestrator()

# Schedule computer-use task through fabric
task = await fabric.schedule(
    agent_id=agent.id,
    task_type="computer_use",
    payload={
        "goal": "Navigate to settings page",
        "max_steps": 10,
    },
)
```

## Examples

### Multi-Pool Setup

```python
# Create pools for different workloads
review_pool = await fabric.create_pool(
    pool_id="pool-review",
    model="claude-3-opus",
    min_agents=2,
    max_agents=5,
)

code_pool = await fabric.create_pool(
    pool_id="pool-code",
    model="claude-3-opus",
    min_agents=3,
    max_agents=10,
)

test_pool = await fabric.create_pool(
    pool_id="pool-test",
    model="claude-3-haiku",  # Faster for tests
    min_agents=2,
    max_agents=8,
)

# Route tasks to appropriate pools
await fabric.schedule_to_pool("pool-review", "code_review", {...})
await fabric.schedule_to_pool("pool-code", "implement", {...})
await fabric.schedule_to_pool("pool-test", "run_tests", {...})
```

### Budget Monitoring

```python
# Set budgets for all agents
for agent in agents:
    await budget.set_budget(
        agent_id=agent.id,
        config=BudgetConfig(
            max_tokens_per_hour=50000,
            max_cost_per_day=5.0,
        ),
    )

# Monitor usage
while True:
    for agent in agents:
        status = await budget.get_status(agent.id)
        if status.percent_used > 80:
            print(f"Warning: {agent.name} at {status.percent_used}% budget")
    await asyncio.sleep(60)
```

### Crash Recovery

```python
# Check for abandoned hooks (agent crashed)
pending = await hooks.check_pending_hooks()

for hook in pending:
    if hook.status == HookStatus.ABANDONED:
        # Reassign to new agent
        new_agent = await fabric.get_available_agent()
        await hooks.reassign_hook(hook.hook_id, new_agent.id)
```

---

*Part of Aragora control plane for multi-agent robust decisionmaking*
