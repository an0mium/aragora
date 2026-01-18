# Control Plane Setup Guide

This guide covers setting up and configuring the Enterprise Multi-Agent Control Plane for coordinating distributed agent workloads.

## Overview

The Control Plane provides centralized coordination for multi-agent systems:
- **Agent Registry**: Dynamic agent registration and discovery
- **Task Scheduler**: Work distribution across agents
- **Health Monitor**: Liveness and readiness monitoring
- **Metrics Collection**: Real-time performance tracking

## Quick Start

### Basic Setup

```python
from aragora.control_plane import (
    ControlPlaneCoordinator,
    ControlPlaneConfig,
)

# Create configuration
config = ControlPlaneConfig(
    redis_url="redis://localhost:6379",
    heartbeat_timeout=30.0,
    task_timeout=300.0,
    max_task_retries=3,
    cleanup_interval=60.0,
)

# Initialize coordinator
coordinator = ControlPlaneCoordinator(config)
await coordinator.connect()

# Coordinator is now ready
```

### Registering Agents

```python
from aragora.control_plane import AgentRegistry, AgentInfo

registry = AgentRegistry(coordinator)

# Register an agent
agent_info = AgentInfo(
    agent_id="claude-agent-001",
    capabilities=["reasoning", "code_review"],
    endpoint="http://agent1:8080",
    metadata={"model": "claude-3-opus"},
)
await registry.register(agent_info)

# List available agents
agents = await registry.list_agents(capabilities=["reasoning"])
```

### Scheduling Tasks

```python
from aragora.control_plane import TaskScheduler, TaskDefinition

scheduler = TaskScheduler(coordinator)

# Define a task
task = TaskDefinition(
    task_id="task-001",
    task_type="debate",
    payload={"topic": "API design patterns"},
    required_capabilities=["reasoning"],
    priority=1,
)

# Schedule the task
await scheduler.schedule(task)

# Check task status
status = await scheduler.get_status("task-001")
```

## Configuration Reference

### ControlPlaneConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | str | `"redis://localhost:6379"` | Redis connection URL |
| `heartbeat_timeout` | float | `30.0` | Seconds before agent considered unhealthy |
| `task_timeout` | float | `300.0` | Default task timeout in seconds |
| `max_task_retries` | int | `3` | Maximum retry attempts for failed tasks |
| `cleanup_interval` | float | `60.0` | Interval for cleanup operations |
| `enable_metrics` | bool | `True` | Enable Prometheus metrics |

### Environment Variables

```bash
# Redis Configuration
ARAGORA_REDIS_URL=redis://localhost:6379
ARAGORA_REDIS_PASSWORD=your-password

# Control Plane Settings
ARAGORA_CP_HEARTBEAT_TIMEOUT=30
ARAGORA_CP_TASK_TIMEOUT=300
ARAGORA_CP_MAX_RETRIES=3
```

## Health Monitoring

### Health Probes

```python
from aragora.control_plane import HealthMonitor

monitor = HealthMonitor(coordinator)

# Register health probe
async def custom_probe():
    # Return True if healthy
    return check_my_service()

monitor.register_probe("my-service", custom_probe)

# Check system health
health = await monitor.check_health()
# {'status': 'healthy', 'probes': {...}, 'agents': {...}}
```

### Agent Health

Agents report health via heartbeats:

```python
# Agent-side heartbeat
await coordinator.send_heartbeat(
    agent_id="claude-agent-001",
    status="healthy",
    metrics={"tasks_processed": 42, "avg_latency_ms": 150},
)
```

## Task Distribution

### Distribution Strategies

The scheduler supports multiple distribution strategies:

```python
# Round-robin (default)
scheduler = TaskScheduler(coordinator, strategy="round_robin")

# Capability-based routing
scheduler = TaskScheduler(coordinator, strategy="capability_match")

# Load-balanced
scheduler = TaskScheduler(coordinator, strategy="least_loaded")
```

### Task Priorities

```python
# High priority (processed first)
task = TaskDefinition(..., priority=0)

# Normal priority
task = TaskDefinition(..., priority=1)

# Low priority (processed last)
task = TaskDefinition(..., priority=2)
```

## Multi-Tenant Setup

For organizations with multiple tenants:

```python
config = ControlPlaneConfig(
    redis_url="redis://localhost:6379",
    tenant_isolation=True,
    tenant_id="org-123",
)

coordinator = ControlPlaneCoordinator(config)
# All operations scoped to tenant
```

## Metrics & Observability

### Prometheus Metrics

The control plane exposes metrics at `/metrics`:

- `aragora_cp_tasks_total` - Total tasks processed
- `aragora_cp_tasks_active` - Currently active tasks
- `aragora_cp_agents_total` - Registered agents
- `aragora_cp_agents_healthy` - Healthy agents
- `aragora_cp_task_duration_seconds` - Task execution duration

### Logging

```python
import logging

# Enable debug logging
logging.getLogger("aragora.control_plane").setLevel(logging.DEBUG)
```

## Error Handling

### Task Failures

```python
# Configure retry behavior
task = TaskDefinition(
    ...,
    max_retries=5,
    retry_delay=10.0,  # Seconds between retries
    retry_backoff=2.0,  # Exponential backoff multiplier
)

# Handle task failure callback
async def on_failure(task_id: str, error: str):
    logger.error(f"Task {task_id} failed: {error}")

scheduler.on_task_failure(on_failure)
```

### Agent Failures

```python
# Configure agent failure handling
async def on_agent_unhealthy(agent_id: str):
    # Reschedule tasks from failed agent
    await scheduler.reassign_agent_tasks(agent_id)

monitor.on_agent_unhealthy(on_agent_unhealthy)
```

## Production Deployment

### High Availability

For production, use Redis Cluster:

```python
config = ControlPlaneConfig(
    redis_url="redis://node1:6379,node2:6379,node3:6379",
    redis_cluster=True,
)
```

### Kubernetes Integration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora-control-plane
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aragora-control-plane
  template:
    spec:
      containers:
      - name: control-plane
        image: aragora/control-plane:latest
        env:
        - name: ARAGORA_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

## API Reference

See `aragora/control_plane/` for complete API documentation:
- `coordinator.py` - Main coordinator class
- `agent_registry.py` - Agent management
- `task_scheduler.py` - Task distribution
- `health_monitor.py` - Health checking
