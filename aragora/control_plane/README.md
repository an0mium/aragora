# Control Plane - Enterprise Agent Orchestration

Centralized orchestration for heterogeneous AI agents with distributed coordination, policy enforcement, health monitoring, and multi-region support.

## Quick Start

```python
from aragora.control_plane import ControlPlaneCoordinator, TaskPriority

# Create coordinator
coordinator = await ControlPlaneCoordinator.create()

# Register agent
await coordinator.register_agent(
    agent_id="claude-3-opus",
    capabilities=["debate", "code", "analysis"],
    model="claude-3-opus",
    provider="anthropic"
)

# Submit task
task_id = await coordinator.submit_task(
    task_type="debate",
    payload={"question": "Should we use microservices?"},
    priority=TaskPriority.HIGH
)

# Wait for result
result = await coordinator.wait_for_result(task_id)
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `ControlPlaneCoordinator` | `coordinator.py` | Main orchestration facade |
| `AgentRegistry` | `registry.py` | Service discovery with heartbeats |
| `TaskScheduler` | `scheduler.py` | Priority-based task distribution |
| `HealthMonitor` | `health.py` | Liveness probes and circuit breakers |
| `PolicyManager` | `policy/manager.py` | Policy enforcement |
| `RegionRouter` | `region_router.py` | Multi-region task routing |
| `LeaderElection` | `leader.py` | Distributed leader election |

## Architecture

```
control_plane/
├── coordinator.py        # Main entry point
├── registry.py           # Agent registration/discovery
├── scheduler.py          # Task scheduling
├── health.py             # Health monitoring
├── leader.py             # Leader election
├── region_router.py      # Regional routing
├── regional_sync.py      # Cross-region sync
├── multi_tenancy.py      # Tenant isolation
├── notifications.py      # Event notifications
├── channels.py           # Slack/Teams/Email delivery
├── watchdog.py           # Three-tier watchdog
├── auto_scaling.py       # Dynamic scaling
├── policy/
│   ├── manager.py        # Policy evaluation
│   ├── types.py          # Policy domain types
│   ├── conflicts.py      # Conflict detection
│   ├── cache.py          # Redis policy cache
│   └── sync.py           # Policy synchronization
├── deliberation.py       # Debate execution
├── deliberation_chain.py # Multi-step pipelines
└── arena_bridge.py       # Arena integration
```

## Features

### Agent Management
```python
# Register with capabilities
await coordinator.register_agent(
    agent_id="claude-opus",
    capabilities=["debate", "code", "analysis"],
    region_id="us-west-2",
    available_regions=["us-west-2", "us-east-1"]
)

# Check health
health = coordinator.get_agent_health("claude-opus")
is_available = coordinator.is_agent_available("claude-opus")
```

### Task Scheduling
```python
from aragora.control_plane import TaskPriority, RegionRoutingMode

task_id = await coordinator.submit_task(
    task_type="debate",
    payload={"question": "..."},
    required_capabilities=["debate"],
    priority=TaskPriority.HIGH,
    timeout_seconds=120,
    target_region="us-west-2",
    region_routing_mode=RegionRoutingMode.PREFERRED
)
```

### Multi-Step Deliberations
```python
from aragora.control_plane.deliberation_chain import (
    DeliberationChain, DeliberationStage, ChainExecutor
)

chain = DeliberationChain(
    name="Code Review Pipeline",
    stages=[
        DeliberationStage(
            id="review",
            topic_template="Review: {context.code}",
            agents=["claude", "gpt-4"],
            required_consensus=0.7,
            next_on_success="security"
        ),
        DeliberationStage(
            id="security",
            topic_template="Audit: {previous.output}"
        )
    ]
)

executor = ChainExecutor(coordinator)
result = await executor.execute(chain)
```

### Policy Enforcement
```python
from aragora.control_plane.policy import PolicyManager, EnforcementLevel

# HARD enforcement blocks violations
# WARN enforcement logs and allows
decision = await policy_manager.evaluate(
    action="execute_shell",
    context={"agent_id": "claude", "region": "us-west-2"}
)
```

### Three-Tier Watchdog (Gastown Pattern)
```python
from aragora.control_plane.watchdog import ThreeTierWatchdog

watchdog = ThreeTierWatchdog(coordinator)
await watchdog.start()

# Tier 1: Mechanical (heartbeat, memory, circuits)
# Tier 2: Boot Agent (quality, latency, semantics)
# Tier 3: Deacon (SLA, coordination, global policy)
```

## Status Enums

### Agent Status
```
STARTING → READY → BUSY → DRAINING → OFFLINE
                     ↓
                  FAILED
```

### Task Status
```
PENDING → ASSIGNED → RUNNING → COMPLETED
                        ↓
               FAILED/CANCELLED/TIMEOUT
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `HEARTBEAT_INTERVAL` | `10` | Agent heartbeat (seconds) |
| `HEARTBEAT_TIMEOUT` | `30` | Offline threshold |
| `TASK_TIMEOUT` | `300` | Default task timeout |
| `MAX_TASK_RETRIES` | `3` | Retry limit |
| `CP_ENABLE_KM` | `true` | Knowledge Mound integration |
| `CP_ENABLE_WATCHDOG` | `true` | Three-tier watchdog |

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Fabric](../fabric/README.md) - Agent orchestration substrate
- [RBAC](../rbac/README.md) - Role-based access control
- [Workflow](../workflow/README.md) - DAG-based automation
