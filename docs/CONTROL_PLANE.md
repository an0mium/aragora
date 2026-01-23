# Aragora Control Plane

**Enterprise orchestration for multi-agent deliberation**

---

## Overview

The Aragora Control Plane is the central orchestration layer that manages multi-agent deliberation across your organization's knowledge and communication channels. It provides:

- **Agent Orchestration**: Coordinate 15+ AI models with role assignment and capability matching
- **Task Scheduling**: Priority-based task distribution with Redis-backed queues
- **Service Discovery**: Agent registry with heartbeat-based liveness tracking
- **Multi-Region Support**: Regional routing for compliance and latency optimization
- **Governance**: Policy enforcement, RBAC, and audit logging

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARAGORA CONTROL PLANE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Coordinator │  │   Registry   │  │  Scheduler   │          │
│  │              │  │              │  │              │          │
│  │ - Unified    │  │ - Service    │  │ - Priority   │          │
│  │   API        │  │   discovery  │  │   queues     │          │
│  │ - State      │  │ - Heartbeat  │  │ - Capability │          │
│  │   management │  │   tracking   │  │   matching   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Leader     │  │  Shared      │  │  Multi-      │          │
│  │   Election   │  │  State       │  │  Tenancy     │          │
│  │              │  │              │  │              │          │
│  │ - Distributed│  │ - Agent/task │  │ - Workspace  │          │
│  │   consensus  │  │   visibility │  │   isolation  │          │
│  │ - Failover   │  │ - Dashboard  │  │ - Quota mgmt │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT POOL                                │
├─────────────────────────────────────────────────────────────────┤
│  Claude │ GPT │ Gemini │ Grok │ Mistral │ DeepSeek │ Qwen │ ... │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. ControlPlaneCoordinator

The unified API for agent and task coordination.

**Location:** `aragora/control_plane/coordinator.py`

```python
from aragora.control_plane import ControlPlaneCoordinator

coordinator = ControlPlaneCoordinator()

# Register an agent
await coordinator.register_agent(
    agent_id="claude-1",
    capabilities=["reasoning", "code_review", "synthesis"],
    region="us-east-1"
)

# Submit a task
task_id = await coordinator.submit_task(
    task_type="deliberation",
    payload={"topic": "API design review"},
    required_capabilities=["reasoning", "code_review"]
)

# Get task status
status = await coordinator.get_task_status(task_id)
```

### 2. AgentRegistry

Service discovery with heartbeat-based liveness tracking.

**Location:** `aragora/control_plane/registry.py`

Agent states:
- `STARTING` - Agent is initializing
- `READY` - Agent is available for tasks
- `BUSY` - Agent is processing a task
- `DRAINING` - Agent is finishing current work before shutdown
- `OFFLINE` - Agent is not responding
- `FAILED` - Agent encountered an error

### 3. TaskScheduler

Redis-backed priority task distribution with capability matching.

**Location:** `aragora/control_plane/scheduler.py`

Features:
- Priority queues (critical, high, normal, low)
- Capability-based routing
- Regional affinity
- Retry with exponential backoff
- Dead letter queues

### 4. LeaderElection

Distributed leader election for multi-node deployments.

**Location:** `aragora/control_plane/leader.py`

Uses Redis-based locking for leader election with automatic failover.

---

## Deployment Modes

### Single-Node (Development)

All control plane components run in a single process.

```bash
python -m aragora.server.unified_server --port 8080
```

### Multi-Instance (Production)

Multiple server instances with shared Redis state.

```bash
export ARAGORA_MULTI_INSTANCE=true
export REDIS_URL=redis://redis:6379

# Start multiple instances
python -m aragora.server.unified_server --port 8080
python -m aragora.server.unified_server --port 8081
```

### Multi-Region (Enterprise)

Regional deployments with cross-region coordination.

```yaml
# deploy/multi-region/helm/aragora/values.yaml
regions:
  - name: us-east-1
    primary: true
  - name: eu-west-1
    primary: false
```

---

## REST API

The control plane API is versioned under `/api/v1/control-plane` with legacy
aliases under `/api/control-plane` for backward compatibility.

### Agent Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/control-plane/agents` | Register an agent |
| `GET` | `/api/control-plane/agents` | List all agents |
| `GET` | `/api/control-plane/agents/{id}` | Get agent details |
| `DELETE` | `/api/control-plane/agents/{id}` | Deregister an agent |
| `POST` | `/api/control-plane/agents/{id}/heartbeat` | Send heartbeat |

### Task Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/control-plane/tasks` | Submit a task |
| `GET` | `/api/control-plane/tasks/{id}` | Get task status |
| `POST` | `/api/control-plane/tasks/{id}/complete` | Complete a task |
| `POST` | `/api/control-plane/tasks/{id}/fail` | Fail a task |
| `POST` | `/api/control-plane/tasks/{id}/cancel` | Cancel a task |
| `POST` | `/api/control-plane/tasks/claim` | Claim the next task |

### Deliberation Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/control-plane/deliberations` | Run or queue a deliberation |
| `GET` | `/api/control-plane/deliberations/{id}` | Get deliberation result |
| `GET` | `/api/control-plane/deliberations/{id}/status` | Get deliberation status |

### Queue Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/control-plane/queue` | Pending and running tasks |

### Health & Metrics Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/control-plane/health` | System health |
| `GET` | `/api/control-plane/health/{agent_id}` | Agent health |
| `GET` | `/api/control-plane/stats` | Scheduler and registry stats |
| `GET` | `/api/control-plane/metrics` | Dashboard metrics |

---

## Integration with Knowledge Mound

The control plane integrates with the Knowledge Mound for organizational knowledge accumulation.

```python
from aragora.control_plane import ControlPlaneCoordinator
from aragora.knowledge.mound import KnowledgeMound

coordinator = ControlPlaneCoordinator()
mound = KnowledgeMound()

# Submit a task that uses Knowledge Mound context
task_id = await coordinator.submit_task(
    task_type="deliberation",
    payload={
        "topic": "Security architecture review",
        "knowledge_context": await mound.retrieve_context("security policies")
    }
)
```

---

## Observability

### Metrics

The control plane exports Prometheus metrics:

- `aragora_agents_total` - Total registered agents
- `aragora_agents_active` - Currently active agents
- `aragora_tasks_submitted_total` - Total tasks submitted
- `aragora_tasks_completed_total` - Completed tasks
- `aragora_task_latency_seconds` - Task processing latency

### Dashboard

Access the control plane dashboard at `/admin/control-plane`.

Features:
- Real-time agent status
- Task queue visualization
- Performance metrics
- Audit log viewer

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_MULTI_INSTANCE` | `false` | Enable multi-instance mode |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `CONTROL_PLANE_HEARTBEAT_INTERVAL` | `30` | Heartbeat interval in seconds |
| `CONTROL_PLANE_AGENT_TIMEOUT` | `90` | Agent timeout in seconds |
| `CONTROL_PLANE_MAX_RETRIES` | `3` | Maximum task retries |

### YAML Configuration

```yaml
# config/control_plane.yaml
control_plane:
  heartbeat_interval: 30
  agent_timeout: 90
  max_retries: 3
  queues:
    critical:
      max_size: 100
      workers: 4
    normal:
      max_size: 1000
      workers: 2
```

---

## Security

### Authentication

The control plane requires API token authentication for all endpoints.

```bash
export ARAGORA_API_TOKEN=your-secret-token
```

### RBAC

Control plane operations require specific permissions:

| Operation | Permission |
|-----------|------------|
| Register agent | `control_plane:agents:write` |
| View agents | `control_plane:agents:read` |
| Submit task | `control_plane:tasks:write` |
| Cancel task | `control_plane:tasks:delete` |

See [GOVERNANCE.md](./GOVERNANCE.md) for full RBAC documentation.

---

## Related Documentation

- [GOVERNANCE.md](./GOVERNANCE.md) - Policies, RBAC, and audit logging
- [CHANNELS.md](./CHANNELS.md) - Channel integrations and bidirectional communication
- [CONNECTORS.md](./CONNECTORS.md) - Data connectors for organizational knowledge
