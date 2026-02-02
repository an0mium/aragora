"""
Agent Fabric data models.

Defines the core data structures used across the Agent Fabric components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from aragora.resilience.health import HealthStatus


class Priority(Enum):
    """Task priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PolicyEffect(Enum):
    """Policy rule effect."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class Task:
    """A unit of work to be executed by an agent."""

    id: str
    type: str
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TaskHandle:
    """Handle to a scheduled task."""

    task_id: str
    agent_id: str
    status: TaskStatus
    scheduled_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None


@dataclass
class IsolationConfig:
    """Agent isolation configuration."""

    level: Literal["none", "process", "container"] = "process"
    memory_mb: int = 512
    cpu_cores: float = 1.0
    filesystem_root: str | None = None
    network_egress: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Agent budget configuration."""

    max_tokens_per_day: int | None = None
    max_compute_seconds_per_day: int | None = None
    max_cost_per_day_usd: float | None = None
    alert_threshold_percent: float = 80.0
    hard_limit: bool = True  # If True, block on limit; if False, warn only


@dataclass
class AgentConfig:
    """Configuration for spawning an agent."""

    id: str
    model: str
    tools: list[str] = field(default_factory=list)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    policies: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    # Pool settings
    pool_id: str | None = None
    max_concurrent_tasks: int = 1


@dataclass
class AgentHandle:
    """Handle to a running agent."""

    agent_id: str
    config: AgentConfig
    spawned_at: datetime
    status: HealthStatus = HealthStatus.HEALTHY
    last_heartbeat: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class AgentInfo:
    """Summary information about an agent."""

    agent_id: str
    model: str
    status: HealthStatus
    spawned_at: datetime
    last_heartbeat: datetime | None
    tasks_pending: int
    tasks_running: int
    tasks_completed: int
    tasks_failed: int
    budget_usage_percent: float


@dataclass
class PolicyRule:
    """A single policy rule."""

    action_pattern: str  # glob pattern for matching actions
    effect: PolicyEffect
    conditions: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Policy:
    """A policy containing multiple rules."""

    id: str
    name: str
    rules: list[PolicyRule] = field(default_factory=list)
    priority: int = 0  # Higher priority policies are evaluated first
    enabled: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyContext:
    """Context for policy evaluation."""

    agent_id: str
    user_id: str | None = None
    tenant_id: str | None = None
    workspace_id: str | None = None
    action: str = ""
    resource: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""

    allowed: bool
    effect: PolicyEffect
    matching_rule: PolicyRule | None = None
    matching_policy: str | None = None
    reason: str = ""
    requires_approval: bool = False
    approvers: list[str] = field(default_factory=list)


@dataclass
class Usage:
    """Resource usage record."""

    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens_input: int = 0
    tokens_output: int = 0
    compute_seconds: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    task_id: str | None = None


@dataclass
class BudgetStatus:
    """Current budget status for an entity."""

    entity_id: str
    entity_type: Literal["agent", "user", "tenant"]
    period_start: datetime
    period_end: datetime
    tokens_used: int = 0
    tokens_limit: int | None = None
    compute_seconds_used: float = 0.0
    compute_seconds_limit: float | None = None
    cost_used_usd: float = 0.0
    cost_limit_usd: float | None = None
    usage_percent: float = 0.0
    over_limit: bool = False
    alert_triggered: bool = False


@dataclass
class UsageReport:
    """Usage report for a time period."""

    entity_id: str
    period_start: datetime
    period_end: datetime
    total_tokens: int = 0
    total_compute_seconds: float = 0.0
    total_cost_usd: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_day: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceUsage:
    """Current resource usage for an agent."""

    agent_id: str
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class ApprovalRequest:
    """Request for action approval."""

    id: str
    action: str
    context: PolicyContext
    requested_at: datetime
    requested_by: str
    approvers: list[str]
    status: Literal["pending", "approved", "denied", "expired"] = "pending"
    approved_by: str | None = None
    approved_at: datetime | None = None
    expires_at: datetime | None = None
    reason: str = ""


@dataclass
class ApprovalResult:
    """Result of an approval request."""

    approved: bool
    request: ApprovalRequest
    waited_seconds: float = 0.0
