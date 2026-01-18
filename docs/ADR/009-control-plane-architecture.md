# ADR-009: Control Plane Architecture

## Status
Accepted

## Context
As Aragora scaled to enterprise deployments, the system needed:
- Multi-tenant isolation for workspace separation
- Centralized agent management across distributed instances
- Task scheduling with priority queuing and retry logic
- Health monitoring with circuit breaker patterns
- Policy enforcement for access control and resource limits

The original design tightly coupled agents to specific debates, limiting scalability.

## Decision
We implemented a **Unified Control Plane** with Redis-backed coordination for distributed agent management.

### Core Components

**ControlPlaneCoordinator** (`aragora/control_plane/coordinator.py`):
```python
class ControlPlaneCoordinator:
    """
    Unified coordinator providing high-level operations that coordinate:
    - AgentRegistry: Service discovery and agent management
    - TaskScheduler: Task distribution and lifecycle
    - HealthMonitor: Health tracking and circuit breakers
    """

    async def register_agent(
        agent_id: str,
        capabilities: List[str],
        model: str,
    ) -> None

    async def submit_task(
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: List[str],
    ) -> str

    async def wait_for_result(task_id: str, timeout: float) -> Any
```

**ControlPlaneConfig**:
```python
@dataclass
class ControlPlaneConfig:
    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "aragora:cp:"
    heartbeat_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    task_timeout: float = 300.0
    max_task_retries: int = 3
```

### Subsystems

**AgentRegistry** (`aragora/control_plane/registry.py`):
- Agent capability declaration (debate, code, search)
- Status tracking (IDLE, BUSY, OFFLINE)
- Heartbeat-based health detection
- Capability-based task routing

**TaskScheduler** (`aragora/control_plane/scheduler.py`):
- Priority queuing (CRITICAL, HIGH, NORMAL, LOW)
- Task lifecycle management (PENDING, RUNNING, COMPLETED, FAILED)
- Automatic retry with exponential backoff
- Result caching and cleanup

**HealthMonitor** (`aragora/control_plane/health.py`):
- Liveness and readiness probes
- Circuit breaker pattern for failing agents
- Health aggregation across instances
- Alerting integration points

### Multi-Tenancy

**TenancyMiddleware** (`aragora/server/middleware/tenancy.py`):
- Workspace isolation via headers/tokens
- Per-tenant resource quotas
- Audit logging for compliance

**PolicyEngine** (`aragora/policy/engine.py`):
- Rule-based access control
- Rate limiting per workspace
- Feature flag evaluation
- Cost tracking and budgets

### HTTP Handlers

**Control Plane API** (`aragora/server/handlers/control_plane.py`):
```
POST /api/control-plane/agents/register
GET  /api/control-plane/agents
POST /api/control-plane/tasks/submit
GET  /api/control-plane/tasks/{task_id}
GET  /api/control-plane/health
```

## Consequences

**Positive:**
- Horizontal scaling via Redis coordination
- True multi-tenancy with workspace isolation
- Unified agent management across instances
- Circuit breakers prevent cascade failures
- Policy-based governance for enterprise compliance

**Negative:**
- Redis dependency for distributed state
- Added latency for cross-instance coordination
- Complexity in debugging distributed flows
- Heartbeat overhead for large agent pools

**Mitigations:**
- Local caching with Redis as source of truth
- Connection pooling for Redis
- Structured logging with correlation IDs
- Configurable heartbeat intervals

## Security Considerations

- Workspace tokens validated per request
- Agent registration requires authentication
- Task payloads can be encrypted at rest
- Audit trail for all control plane operations

## References
- `aragora/control_plane/coordinator.py` - Main coordinator
- `aragora/control_plane/registry.py` - Agent registry
- `aragora/control_plane/scheduler.py` - Task scheduler
- `aragora/control_plane/health.py` - Health monitoring
- `aragora/policy/engine.py` - Policy enforcement
- `aragora/server/middleware/tenancy.py` - Multi-tenant middleware
- `aragora/server/handlers/control_plane.py` - HTTP API
