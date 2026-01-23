# Control Plane Guide

> **Deprecated:** This guide is superseded by `docs/CONTROL_PLANE_GUIDE.md`.
> Use the main guide for current endpoints and permissions.

The Enterprise Control Plane provides centralized management for distributed Aragora deployments, including agent orchestration, task scheduling, and health monitoring.

## Overview

The Control Plane coordinates:
- **Agent Registry**: Registration, discovery, and status tracking
- **Task Scheduler**: Priority-based task distribution and execution
- **Health Monitor**: Agent health checks and automatic recovery
- **Metrics Dashboard**: Real-time operational metrics

## API Endpoints

### Agent Management

#### List Agents
```
GET /api/control-plane/agents
```

Query parameters:
- `status` - Filter by status: `available`, `busy`, `offline`
- `type` - Filter by agent type: `cli`, `api`, `local`
- `limit` - Maximum results (default: 100)

Response:
```json
{
  "agents": [
    {
      "id": "claude-opus",
      "name": "Claude Opus",
      "type": "api",
      "status": "available",
      "capabilities": ["coding", "analysis", "creative"],
      "registered_at": "2026-01-18T10:00:00Z",
      "last_heartbeat": "2026-01-18T17:30:00Z"
    }
  ],
  "total": 8
}
```

#### Register Agent
```
POST /api/control-plane/agents
```

Request body:
```json
{
  "id": "custom-agent-1",
  "name": "Custom Analysis Agent",
  "type": "api",
  "capabilities": ["analysis", "research"],
  "endpoint": "https://api.example.com/agent",
  "api_key_ref": "CUSTOM_AGENT_KEY"
}
```

#### Get Agent Status
```
GET /api/control-plane/agents/{agent_id}
```

#### Deregister Agent
```
DELETE /api/control-plane/agents/{agent_id}
```

### Task Management

#### Submit Task
```
POST /api/control-plane/tasks
```

Request body:
```json
{
  "task_type": "debate",
  "payload": {
    "topic": "Design a rate limiter",
    "agents": ["claude", "gpt-4", "codex"],
    "rounds": 3
  },
  "priority": "high",
  "metadata": {
    "name": "Rate Limiter Design",
    "requester": "user@example.com"
  }
}
```

Priority levels: `critical`, `high`, `normal`, `low`, `background`

Response:
```json
{
  "task_id": "task_abc123",
  "status": "pending",
  "created_at": "2026-01-18T17:35:00Z",
  "estimated_wait": "30s"
}
```

#### Get Task Status
```
GET /api/control-plane/tasks/{task_id}
```

Response:
```json
{
  "task_id": "task_abc123",
  "status": "running",
  "progress": 0.6,
  "assigned_agent": "claude-opus",
  "started_at": "2026-01-18T17:35:30Z",
  "metadata": {
    "current_round": 2,
    "total_rounds": 3
  }
}
```

#### Cancel Task
```
DELETE /api/control-plane/tasks/{task_id}
```

### Queue Management (New)

#### Get Job Queue
```
GET /api/control-plane/queue
```

Query parameters:
- `limit` - Maximum jobs to return (default: 50)

Returns pending and running tasks formatted as jobs:

```json
{
  "jobs": [
    {
      "id": "task_abc123",
      "type": "debate",
      "name": "Rate Limiter Design",
      "status": "running",
      "progress": 0.6,
      "started_at": "2026-01-18T17:35:30Z",
      "created_at": "2026-01-18T17:35:00Z",
      "document_count": 0,
      "agents_assigned": ["claude-opus"],
      "priority": "high"
    },
    {
      "id": "task_def456",
      "type": "document_processing",
      "name": "Contract Analysis",
      "status": "pending",
      "progress": 0.0,
      "started_at": null,
      "created_at": "2026-01-18T17:36:00Z",
      "document_count": 5,
      "agents_assigned": [],
      "priority": "normal"
    }
  ],
  "total": 2
}
```

### Metrics Dashboard (New)

#### Get Dashboard Metrics
```
GET /api/control-plane/metrics
```

Returns aggregated metrics for dashboard display:

```json
{
  "active_jobs": 3,
  "queued_jobs": 12,
  "completed_jobs": 156,
  "agents_available": 5,
  "agents_busy": 3,
  "total_agents": 8,
  "documents_processed_today": 47,
  "audits_completed_today": 8,
  "tokens_used_today": 125000
}
```

### Health Monitoring

#### System Health
```
GET /api/control-plane/health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "scheduler": "healthy",
    "registry": "healthy",
    "database": "healthy"
  },
  "uptime_seconds": 86400,
  "version": "2.0.0"
}
```

#### Agent Health
```
GET /api/control-plane/health/{agent_id}
```

Response:
```json
{
  "agent_id": "claude-opus",
  "status": "healthy",
  "last_heartbeat": "2026-01-18T17:30:00Z",
  "latency_ms": 150,
  "success_rate": 0.98,
  "tasks_completed": 45
}
```

### Statistics

#### Control Plane Stats
```
GET /api/control-plane/stats
```

Response:
```json
{
  "scheduler": {
    "total_tasks": 1250,
    "by_status": {
      "pending": 12,
      "running": 3,
      "completed": 1200,
      "failed": 35
    },
    "by_type": {
      "debate": 800,
      "document_processing": 300,
      "audit": 150
    },
    "avg_wait_time_ms": 2500,
    "avg_execution_time_ms": 45000
  },
  "registry": {
    "total_agents": 8,
    "available_agents": 5,
    "by_status": {
      "available": 5,
      "busy": 3,
      "offline": 0
    },
    "by_type": {
      "api": 6,
      "cli": 2
    }
  }
}
```

## Integration Examples

### Python SDK

```python
from aragora.control_plane import ControlPlaneClient

# Initialize client
client = ControlPlaneClient(base_url="http://localhost:8080")

# Submit a task
task = await client.submit_task(
    task_type="debate",
    payload={
        "topic": "Design a caching strategy",
        "agents": ["claude", "gpt-4"],
    },
    priority="high",
)

# Monitor progress
while True:
    status = await client.get_task_status(task.id)
    print(f"Progress: {status.progress * 100:.0f}%")
    if status.status in ("completed", "failed"):
        break
    await asyncio.sleep(5)

# Get result
result = await client.get_task_result(task.id)
```

### REST API (curl)

```bash
# List available agents
curl http://localhost:8080/api/control-plane/agents?status=available

# Submit a task
curl -X POST http://localhost:8080/api/control-plane/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "debate",
    "payload": {"topic": "API design best practices"},
    "priority": "normal"
  }'

# Get queue status
curl http://localhost:8080/api/control-plane/queue

# Get dashboard metrics
curl http://localhost:8080/api/control-plane/metrics
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Control Plane                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Agent     │  │    Task      │  │   Health     │      │
│  │   Registry   │  │  Scheduler   │  │   Monitor    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │ Coordinator │                         │
│                    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
      │  Agent 1  │   │  Agent 2  │   │  Agent N  │
      │ (Claude)  │   │  (GPT-4)  │   │  (Codex)  │
      └───────────┘   └───────────┘   └───────────┘
```

## Configuration

Environment variables:

```bash
# Control plane settings
CONTROL_PLANE_ENABLED=true
CONTROL_PLANE_HEARTBEAT_INTERVAL=30  # seconds
CONTROL_PLANE_TASK_TIMEOUT=300       # seconds
CONTROL_PLANE_MAX_RETRIES=3

# Agent registration
AGENT_REGISTRATION_TTL=300           # seconds before re-registration needed
```

## Monitoring

The Control Plane exposes Prometheus metrics at `/metrics`:

- `control_plane_tasks_total{status,type}` - Total tasks by status and type
- `control_plane_task_duration_seconds` - Task execution duration histogram
- `control_plane_agents_total{status,type}` - Registered agents by status
- `control_plane_queue_depth` - Current queue depth

Grafana dashboards are available in `monitoring/grafana/dashboards/control-plane.json`.

## Troubleshooting

### Agent Not Appearing

1. Check agent registration endpoint is reachable
2. Verify API key is configured correctly
3. Check control plane logs for registration errors

### Tasks Stuck in Pending

1. Verify agents are available: `GET /api/control-plane/agents?status=available`
2. Check queue depth: `GET /api/control-plane/queue`
3. Review scheduler stats: `GET /api/control-plane/stats`

### High Latency

1. Check agent health: `GET /api/control-plane/health/{agent_id}`
2. Review metrics for bottlenecks
3. Consider increasing agent pool size
