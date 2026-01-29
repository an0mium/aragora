"""
Control Plane Policy Types.

Enums, exceptions, and dataclasses for the control plane policy system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PolicyScope(Enum):
    """Scope of a control plane policy."""

    GLOBAL = "global"  # Applies to all tasks
    TASK_TYPE = "task_type"  # Applies to specific task types
    CAPABILITY = "capability"  # Applies to specific capabilities
    REGION = "region"  # Applies to specific regions
    WORKSPACE = "workspace"  # Applies to specific workspaces


class EnforcementLevel(Enum):
    """How strictly a policy is enforced."""

    WARN = "warn"  # Log warning but allow
    SOFT = "soft"  # Deny but allow override
    HARD = "hard"  # Deny with no override


class PolicyDecision(Enum):
    """Result of a policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    ESCALATE = "escalate"


class PolicyViolationError(Exception):
    """Raised when task violates HARD enforcement policy."""

    def __init__(
        self,
        result: "PolicyEvaluationResult",
        task_type: str = None,
        agent_id: str = None,
        region: str = None,
    ):
        self.result = result
        self.task_type = task_type
        self.agent_id = agent_id
        self.region = region
        super().__init__(f"Policy violation ({result.enforcement_level.value}): {result.reason}")


@dataclass
class SLARequirements:
    """SLA requirements for a policy.

    Attributes:
        max_execution_seconds: Maximum allowed execution time
        max_queue_seconds: Maximum time a task can wait in queue
        min_agents_available: Minimum agents that must be available
        max_concurrent_tasks: Maximum concurrent tasks per agent
        response_time_p99_ms: Target P99 response time
    """

    max_execution_seconds: float = 300.0
    max_queue_seconds: float = 60.0
    min_agents_available: int = 1
    max_concurrent_tasks: int = 5
    response_time_p99_ms: float = 5000.0

    def is_execution_time_compliant(self, duration_seconds: float) -> bool:
        """Check if execution time is within SLA."""
        return duration_seconds <= self.max_execution_seconds

    def is_queue_time_compliant(self, wait_seconds: float) -> bool:
        """Check if queue time is within SLA."""
        return wait_seconds <= self.max_queue_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "max_execution_seconds": self.max_execution_seconds,
            "max_queue_seconds": self.max_queue_seconds,
            "min_agents_available": self.min_agents_available,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "response_time_p99_ms": self.response_time_p99_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SLARequirements":
        """Deserialize from dict."""
        return cls(
            max_execution_seconds=data.get("max_execution_seconds", 300.0),
            max_queue_seconds=data.get("max_queue_seconds", 60.0),
            min_agents_available=data.get("min_agents_available", 1),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 5),
            response_time_p99_ms=data.get("response_time_p99_ms", 5000.0),
        )


@dataclass
class RegionConstraint:
    """Constraints on which regions can execute tasks.

    Attributes:
        allowed_regions: Explicit list of allowed regions (empty = all)
        blocked_regions: Explicit list of blocked regions
        require_data_residency: Region must match data residency rules
        allow_cross_region: Whether cross-region execution is allowed
    """

    allowed_regions: list[str] = field(default_factory=list)
    blocked_regions: list[str] = field(default_factory=list)
    require_data_residency: bool = False
    allow_cross_region: bool = True

    def is_region_allowed(self, region: str, data_region: str | None = None) -> bool:
        """Check if a region is allowed for execution."""
        # Check explicit block list
        if region in self.blocked_regions:
            return False

        # Check explicit allow list (if specified)
        if self.allowed_regions and region not in self.allowed_regions:
            return False

        # Check data residency
        if self.require_data_residency and data_region and region != data_region:
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "allowed_regions": self.allowed_regions,
            "blocked_regions": self.blocked_regions,
            "require_data_residency": self.require_data_residency,
            "allow_cross_region": self.allow_cross_region,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionConstraint":
        """Deserialize from dict."""
        return cls(
            allowed_regions=data.get("allowed_regions", []),
            blocked_regions=data.get("blocked_regions", []),
            require_data_residency=data.get("require_data_residency", False),
            allow_cross_region=data.get("allow_cross_region", True),
        )


@dataclass
class ControlPlanePolicy:
    """A control plane policy defining agent/region/SLA constraints.

    Policies can restrict:
    - Which agents can execute tasks (allowlist/blocklist)
    - Which regions can execute tasks
    - SLA requirements (execution time, queue time)
    - Which task types/capabilities the policy applies to
    """

    name: str
    description: str = ""

    # Scope - what this policy applies to
    scope: PolicyScope = PolicyScope.GLOBAL
    task_types: list[str] = field(default_factory=list)  # Empty = all
    capabilities: list[str] = field(default_factory=list)  # Empty = all
    workspaces: list[str] = field(default_factory=list)  # Empty = all

    # Agent restrictions
    agent_allowlist: list[str] = field(default_factory=list)  # Empty = all
    agent_blocklist: list[str] = field(default_factory=list)

    # Region constraints
    region_constraint: RegionConstraint | None = None

    # SLA requirements
    sla: SLARequirements | None = None

    # Enforcement
    enforcement_level: EnforcementLevel = EnforcementLevel.HARD
    enabled: bool = True
    priority: int = 0  # Higher = evaluated first

    # Versioning
    version: int = 1
    updated_at: datetime | None = None
    updated_by: str | None = None
    previous_version_id: str | None = None  # Link to previous version

    # Metadata
    id: str = field(default_factory=lambda: f"policy_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        task_type: str | None = None,
        capability: str | None = None,
        workspace: str | None = None,
    ) -> bool:
        """Check if this policy applies to the given context."""
        if not self.enabled:
            return False

        # Check task type match
        if self.task_types and task_type and task_type not in self.task_types:
            return False

        # Check capability match
        if self.capabilities and capability and capability not in self.capabilities:
            return False

        # Check workspace match
        if self.workspaces and workspace and workspace not in self.workspaces:
            return False

        return True

    def is_agent_allowed(self, agent_id: str) -> bool:
        """Check if an agent is allowed by this policy."""
        # Check blocklist first
        if agent_id in self.agent_blocklist:
            return False

        # Check allowlist (if specified)
        if self.agent_allowlist and agent_id not in self.agent_allowlist:
            return False

        return True

    def is_region_allowed(self, region: str, data_region: str | None = None) -> bool:
        """Check if a region is allowed by this policy."""
        if not self.region_constraint:
            return True
        return self.region_constraint.is_region_allowed(region, data_region)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope.value,
            "task_types": self.task_types,
            "capabilities": self.capabilities,
            "workspaces": self.workspaces,
            "agent_allowlist": self.agent_allowlist,
            "agent_blocklist": self.agent_blocklist,
            "region_constraint": (
                self.region_constraint.to_dict() if self.region_constraint else None
            ),
            "sla": self.sla.to_dict() if self.sla else None,
            "enforcement_level": self.enforcement_level.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "updated_by": self.updated_by,
            "previous_version_id": self.previous_version_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlPlanePolicy":
        """Deserialize from dict."""
        region_constraint = None
        if data.get("region_constraint"):
            region_constraint = RegionConstraint.from_dict(data["region_constraint"])

        sla = None
        if data.get("sla"):
            sla = SLARequirements.from_dict(data["sla"])

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", f"policy_{uuid.uuid4().hex[:12]}"),
            name=data["name"],
            description=data.get("description", ""),
            scope=PolicyScope(data.get("scope", "global")),
            task_types=data.get("task_types", []),
            capabilities=data.get("capabilities", []),
            workspaces=data.get("workspaces", []),
            agent_allowlist=data.get("agent_allowlist", []),
            agent_blocklist=data.get("agent_blocklist", []),
            region_constraint=region_constraint,
            sla=sla,
            enforcement_level=EnforcementLevel(data.get("enforcement_level", "hard")),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
            created_at=created_at,
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating a policy."""

    decision: PolicyDecision
    allowed: bool
    policy_id: str
    policy_name: str
    reason: str
    enforcement_level: EnforcementLevel
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context
    task_type: str | None = None
    agent_id: str | None = None
    region: str | None = None
    sla_violation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "decision": self.decision.value,
            "allowed": self.allowed,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "enforcement_level": self.enforcement_level.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "task_type": self.task_type,
            "agent_id": self.agent_id,
            "region": self.region,
            "sla_violation": self.sla_violation,
        }


@dataclass
class PolicyViolation:
    """A policy violation record."""

    id: str
    policy_id: str
    policy_name: str
    violation_type: str  # "agent", "region", "sla"
    description: str
    task_id: str | None = None
    task_type: str | None = None
    agent_id: str | None = None
    region: str | None = None
    workspace_id: str | None = None
    enforcement_level: EnforcementLevel = EnforcementLevel.HARD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "violation_type": self.violation_type,
            "description": self.description,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "agent_id": self.agent_id,
            "region": self.region,
            "workspace_id": self.workspace_id,
            "enforcement_level": self.enforcement_level.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
