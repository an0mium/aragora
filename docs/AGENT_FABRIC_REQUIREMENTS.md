# Agent Fabric Requirements

The Agent Fabric is the shared orchestration substrate that enables high-scale
multi-agent execution with isolation, policy enforcement, and observability.

## Overview

The Agent Fabric provides:
1. **Scheduling** - Fair task distribution across agents
2. **Isolation** - Per-agent resource boundaries
3. **Policy** - Tool access control and approvals
4. **Budget** - Cost tracking and enforcement
5. **Lifecycle** - Agent spawn, heartbeat, termination
6. **Telemetry** - Metrics, traces, and logs

## Functional Requirements

### 1. Agent Scheduler

**FR-SCHED-1**: Support scheduling tasks to 100+ concurrent agents
**FR-SCHED-2**: Implement fair scheduling with priority levels (critical, high, normal, low)
**FR-SCHED-3**: Support task queuing with configurable max queue depth per agent
**FR-SCHED-4**: Allow task cancellation and timeout handling
**FR-SCHED-5**: Support task dependencies (task B waits for task A)
**FR-SCHED-6**: Provide task status queries (pending, running, completed, failed)

Interface:
```python
class AgentScheduler:
    async def schedule(
        self,
        task: Task,
        agent_id: str,
        priority: Priority = Priority.NORMAL,
        depends_on: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TaskHandle

    async def cancel(self, task_id: str) -> bool
    async def get_status(self, task_id: str) -> TaskStatus
    async def list_pending(self, agent_id: str) -> list[Task]
```

### 2. Agent Isolation

**FR-ISO-1**: Isolate agent execution environments (process or container level)
**FR-ISO-2**: Enforce memory limits per agent (configurable, default 512MB)
**FR-ISO-3**: Enforce CPU limits per agent (configurable, default 1 core)
**FR-ISO-4**: Isolate file system access (per-agent working directory)
**FR-ISO-5**: Isolate network access (egress allowlists)
**FR-ISO-6**: Support isolation levels: none, process, container

Interface:
```python
class IsolationManager:
    async def create_sandbox(
        self,
        agent_id: str,
        config: IsolationConfig,
    ) -> Sandbox

    async def destroy_sandbox(self, agent_id: str) -> None
    async def get_resource_usage(self, agent_id: str) -> ResourceUsage
```

### 3. Policy Engine

**FR-POL-1**: Define tool access policies per agent/role/tenant
**FR-POL-2**: Support explicit approval gates for sensitive actions
**FR-POL-3**: Allow/deny lists for file paths, URLs, shell commands
**FR-POL-4**: Support policy inheritance (tenant -> workspace -> agent)
**FR-POL-5**: Log all policy decisions for audit
**FR-POL-6**: Support real-time policy updates without restart

Interface:
```python
class PolicyEngine:
    async def check(
        self,
        action: Action,
        context: PolicyContext,
    ) -> PolicyDecision

    async def require_approval(
        self,
        action: Action,
        context: PolicyContext,
        approvers: list[str],
    ) -> ApprovalResult

    async def update_policy(
        self,
        policy_id: str,
        policy: Policy,
    ) -> None
```

### 4. Budget Enforcement

**FR-BUD-1**: Track token usage per agent/user/tenant
**FR-BUD-2**: Track compute time per agent/user/tenant
**FR-BUD-3**: Enforce budget limits with soft/hard thresholds
**FR-BUD-4**: Support budget alerts and notifications
**FR-BUD-5**: Allow budget carryover or reset policies
**FR-BUD-6**: Provide usage reports and projections

Interface:
```python
class BudgetManager:
    async def track(
        self,
        agent_id: str,
        usage: Usage,
    ) -> BudgetStatus

    async def check_budget(
        self,
        agent_id: str,
        estimated_cost: Cost,
    ) -> bool

    async def get_usage(
        self,
        entity_id: str,
        period: TimePeriod,
    ) -> UsageReport
```

### 5. Agent Lifecycle

**FR-LIFE-1**: Spawn agents with configuration (model, tools, limits)
**FR-LIFE-2**: Maintain agent heartbeat and health status
**FR-LIFE-3**: Handle agent crashes with automatic cleanup
**FR-LIFE-4**: Support graceful shutdown with task draining
**FR-LIFE-5**: Track agent session history
**FR-LIFE-6**: Support agent pooling for fast spawn

Interface:
```python
class LifecycleManager:
    async def spawn(
        self,
        config: AgentConfig,
    ) -> AgentHandle

    async def terminate(
        self,
        agent_id: str,
        graceful: bool = True,
        drain_timeout: float = 30.0,
    ) -> None

    async def heartbeat(self, agent_id: str) -> None
    async def get_health(self, agent_id: str) -> HealthStatus
    async def list_agents(self, filters: AgentFilters) -> list[AgentInfo]
```

### 6. Telemetry

**FR-TEL-1**: Emit Prometheus metrics for all operations
**FR-TEL-2**: Support OpenTelemetry tracing
**FR-TEL-3**: Structured logging with correlation IDs
**FR-TEL-4**: Agent-level metrics (tasks, tokens, latency, errors)
**FR-TEL-5**: Real-time dashboards via existing observability stack
**FR-TEL-6**: Alert on anomalies (high error rate, budget exceeded)

Metrics:
- `agent_fabric_tasks_total` (counter, labels: agent_id, status)
- `agent_fabric_task_duration_seconds` (histogram)
- `agent_fabric_tokens_total` (counter, labels: agent_id, model)
- `agent_fabric_budget_usage_ratio` (gauge, labels: entity_id)
- `agent_fabric_agents_active` (gauge)
- `agent_fabric_policy_decisions_total` (counter, labels: decision)

## Non-Functional Requirements

### Performance
- **NFR-PERF-1**: Task scheduling latency < 10ms P99
- **NFR-PERF-2**: Policy check latency < 5ms P99
- **NFR-PERF-3**: Support 1000 tasks/second throughput
- **NFR-PERF-4**: Agent spawn time < 1 second

### Reliability
- **NFR-REL-1**: No task loss on scheduler crash (persistent queue)
- **NFR-REL-2**: Automatic recovery from agent crashes
- **NFR-REL-3**: Graceful degradation under load

### Security
- **NFR-SEC-1**: All inter-component communication authenticated
- **NFR-SEC-2**: Secrets never logged or exposed
- **NFR-SEC-3**: Audit trail for all administrative actions

## Data Models

```python
@dataclass
class Task:
    id: str
    type: str
    payload: dict[str, Any]
    created_at: datetime
    timeout_seconds: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

@dataclass
class AgentConfig:
    id: str
    model: str
    tools: list[str]
    isolation: IsolationConfig
    budget: BudgetConfig
    policies: list[str]

@dataclass
class IsolationConfig:
    level: Literal["none", "process", "container"]
    memory_mb: int = 512
    cpu_cores: float = 1.0
    filesystem_root: str | None = None
    network_egress: list[str] = field(default_factory=list)

@dataclass
class BudgetConfig:
    max_tokens_per_day: int | None = None
    max_compute_seconds_per_day: int | None = None
    max_cost_per_day_usd: float | None = None
    alert_threshold_percent: float = 80.0

@dataclass
class Policy:
    id: str
    rules: list[PolicyRule]
    priority: int = 0

@dataclass
class PolicyRule:
    action_pattern: str  # glob pattern
    effect: Literal["allow", "deny", "require_approval"]
    conditions: dict[str, Any] = field(default_factory=dict)
```

## Integration Points

- **Aragora Core**: Uses fabric for debate agent orchestration
- **Gastown Extension**: Uses fabric for workspace agents and convoys
- **Moltbot Extension**: Uses fabric for device-initiated agents
- **Computer Use**: Uses fabric policy for action approvals

## Implementation Plan

### Phase 1: Foundation (Week 1)
- Core data models and interfaces
- In-memory scheduler implementation
- Basic lifecycle manager

### Phase 2: Policy & Budget (Week 2)
- Policy engine with rule evaluation
- Budget tracking and enforcement
- Approval workflow

### Phase 3: Isolation & Telemetry (Week 3)
- Process-level isolation
- Prometheus metrics integration
- Structured logging

### Phase 4: Hardening (Week 4)
- Persistent task queue
- Container isolation (optional)
- Performance testing and optimization
