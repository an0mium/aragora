"""
Control Plane Policy Integration.

Provides policy enforcement for the control plane, integrating the core
PolicyEngine with control plane-specific policies for:
- Agent restrictions (allow/deny lists per task type or capability)
- Region constraints (which regions can execute which tasks)
- SLA enforcement (time limits, response time requirements)

Usage:
    policy_manager = ControlPlanePolicyManager()

    # Add agent restriction
    policy_manager.add_policy(ControlPlanePolicy(
        name="restrict-production-agents",
        agent_allowlist=["claude-3-opus", "gpt-4"],
        task_types=["production-deployment"],
    ))

    # Evaluate before dispatch
    result = policy_manager.evaluate_task_dispatch(
        task_type="production-deployment",
        agent_id="gpt-3.5-turbo",
        region="us-east-1",
        capabilities=["deploy"],
    )

    if not result.allowed:
        raise PolicyViolation(result)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from aragora.observability import get_logger

logger = get_logger(__name__)

# Redis availability check (optional - for distributed cache)
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore

# Audit logging (optional)
try:
    from aragora.control_plane.audit import log_policy_decision

    HAS_AUDIT = True
except ImportError:
    HAS_AUDIT = False
    log_policy_decision = None  # type: ignore

# Prometheus metrics (optional)
try:
    from aragora.server.prometheus_control_plane import (
        record_policy_decision as prometheus_record_policy,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    prometheus_record_policy = None  # type: ignore


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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "max_execution_seconds": self.max_execution_seconds,
            "max_queue_seconds": self.max_queue_seconds,
            "min_agents_available": self.min_agents_available,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "response_time_p99_ms": self.response_time_p99_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLARequirements":
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

    allowed_regions: List[str] = field(default_factory=list)
    blocked_regions: List[str] = field(default_factory=list)
    require_data_residency: bool = False
    allow_cross_region: bool = True

    def is_region_allowed(self, region: str, data_region: Optional[str] = None) -> bool:
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "allowed_regions": self.allowed_regions,
            "blocked_regions": self.blocked_regions,
            "require_data_residency": self.require_data_residency,
            "allow_cross_region": self.allow_cross_region,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionConstraint":
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
    task_types: List[str] = field(default_factory=list)  # Empty = all
    capabilities: List[str] = field(default_factory=list)  # Empty = all
    workspaces: List[str] = field(default_factory=list)  # Empty = all

    # Agent restrictions
    agent_allowlist: List[str] = field(default_factory=list)  # Empty = all
    agent_blocklist: List[str] = field(default_factory=list)

    # Region constraints
    region_constraint: Optional[RegionConstraint] = None

    # SLA requirements
    sla: Optional[SLARequirements] = None

    # Enforcement
    enforcement_level: EnforcementLevel = EnforcementLevel.HARD
    enabled: bool = True
    priority: int = 0  # Higher = evaluated first

    # Versioning
    version: int = 1
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    previous_version_id: Optional[str] = None  # Link to previous version

    # Metadata
    id: str = field(default_factory=lambda: f"policy_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        task_type: Optional[str] = None,
        capability: Optional[str] = None,
        workspace: Optional[str] = None,
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

    def is_region_allowed(self, region: str, data_region: Optional[str] = None) -> bool:
        """Check if a region is allowed by this policy."""
        if not self.region_constraint:
            return True
        return self.region_constraint.is_region_allowed(region, data_region)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPlanePolicy":
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
    task_type: Optional[str] = None
    agent_id: Optional[str] = None
    region: Optional[str] = None
    sla_violation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    task_id: Optional[str] = None
    task_type: Optional[str] = None
    agent_id: Optional[str] = None
    region: Optional[str] = None
    workspace_id: Optional[str] = None
    enforcement_level: EnforcementLevel = EnforcementLevel.HARD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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


class ControlPlanePolicyManager:
    """
    Manages control plane policies and evaluates them for task dispatch.

    Provides:
    - Policy CRUD operations
    - Policy evaluation for task dispatch
    - Violation tracking
    - Metrics and reporting
    """

    def __init__(
        self,
        violation_callback: Optional[Callable[[PolicyViolation], None]] = None,
    ):
        """
        Initialize the policy manager.

        Args:
            violation_callback: Optional callback for violations (e.g., audit logging)
        """
        self._policies: Dict[str, ControlPlanePolicy] = {}
        self._violations: List[PolicyViolation] = []
        self._violation_callback = violation_callback

        # Metrics
        self._metrics: Dict[str, int] = {
            "evaluations": 0,
            "allowed": 0,
            "denied": 0,
            "warned": 0,
            "violations": 0,
        }

    def add_policy(self, policy: ControlPlanePolicy) -> None:
        """Add a policy to the manager."""
        self._policies[policy.id] = policy
        logger.info(
            "policy_added",
            policy_id=policy.id,
            policy_name=policy.name,
            scope=policy.scope.value,
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info("policy_removed", policy_id=policy_id)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[ControlPlanePolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        enabled_only: bool = True,
        task_type: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> List[ControlPlanePolicy]:
        """List policies with optional filters."""
        policies = list(self._policies.values())

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        if task_type:
            policies = [p for p in policies if not p.task_types or task_type in p.task_types]

        if workspace:
            policies = [p for p in policies if not p.workspaces or workspace in p.workspaces]

        # Sort by priority (descending)
        return sorted(policies, key=lambda p: -p.priority)

    def _extract_control_plane_policy(self, policy: Any) -> Optional[ControlPlanePolicy]:
        """Extract a control plane policy from a compliance policy record."""
        metadata = getattr(policy, "metadata", {}) or {}
        payload = metadata.get("control_plane_policy") or metadata.get("control_plane")
        if not payload:
            return None

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("policy_payload_parse_failed", policy_id=getattr(policy, "id", ""))
                return None

        if not isinstance(payload, dict):
            return None

        data = dict(payload)
        data.setdefault("id", getattr(policy, "id", f"policy_{uuid.uuid4().hex[:12]}"))
        data.setdefault("name", getattr(policy, "name", "Unnamed policy"))
        data.setdefault("description", getattr(policy, "description", ""))
        data.setdefault("enabled", getattr(policy, "enabled", True))
        if getattr(policy, "workspace_id", None) and not data.get("workspaces"):
            data["workspaces"] = [policy.workspace_id]
        data.setdefault("created_by", getattr(policy, "created_by", None))
        if getattr(policy, "updated_at", None):
            data.setdefault("created_at", policy.updated_at)

        meta = dict(data.get("metadata") or {})
        meta.setdefault("compliance_policy_id", getattr(policy, "id", None))
        meta.setdefault("framework_id", getattr(policy, "framework_id", None))
        meta.setdefault("vertical_id", getattr(policy, "vertical_id", None))
        meta.setdefault("source", "compliance_policy_store")
        data["metadata"] = meta

        try:
            return ControlPlanePolicy.from_dict(data)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                "policy_convert_failed",
                policy_id=getattr(policy, "id", ""),
                error=str(e),
            )
            return None

    def sync_from_compliance_store(
        self,
        store: Optional[Any] = None,
        replace: bool = True,
        workspace_id: Optional[str] = None,
    ) -> int:
        """
        Load control plane policies from the compliance policy store.

        Policies are expected to include a `control_plane_policy` payload in
        the compliance policy metadata.

        Args:
            store: Optional compliance policy store instance
            replace: If True, replace existing policies before loading
            workspace_id: Optional workspace filter for policy loading

        Returns:
            Number of policies loaded
        """
        try:
            if store is None:
                from aragora.compliance.policy_store import get_policy_store

                store = get_policy_store()
        except ImportError as e:
            logger.warning("policy_store_unavailable", error=str(e))
            return 0

        if replace:
            self._policies = {}

        loaded = 0
        offset = 0
        limit = 200

        while True:
            try:
                policies = store.list_policies(
                    workspace_id=workspace_id,
                    enabled_only=True,
                    limit=limit,
                    offset=offset,
                )
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning("policy_store_list_failed", error=str(e))
                break

            if not policies:
                break

            for policy in policies:
                cp_policy = self._extract_control_plane_policy(policy)
                if not cp_policy:
                    continue
                self.add_policy(cp_policy)
                loaded += 1

            if len(policies) < limit:
                break
            offset += len(policies)

        logger.info("policy_sync_completed", loaded=loaded)
        return loaded

    def evaluate_task_dispatch(
        self,
        task_type: str,
        agent_id: str,
        region: str,
        capabilities: Optional[List[str]] = None,
        workspace: Optional[str] = None,
        data_region: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate policies for a task dispatch.

        Returns the result of the first matching policy that denies/warns,
        or an allow result if all policies pass.

        Args:
            task_type: Type of task being dispatched
            agent_id: Agent being considered for the task
            region: Region where task would execute
            capabilities: Task required capabilities
            workspace: Workspace context
            data_region: Region where data resides (for residency checks)
            task_id: Optional task ID for tracking

        Returns:
            PolicyEvaluationResult with decision
        """
        self._metrics["evaluations"] += 1
        capabilities = capabilities or []

        # Get applicable policies (sorted by priority)
        policies = self.list_policies(enabled_only=True, workspace=workspace)

        for policy in policies:
            # Check if policy matches this context
            matches_task = policy.matches(
                task_type=task_type,
                workspace=workspace,
            )

            # Also check capability-level matches
            matches_capability = (
                any(policy.matches(capability=cap, workspace=workspace) for cap in capabilities)
                if capabilities
                else policy.matches(workspace=workspace)
            )

            if not (matches_task or matches_capability):
                continue

            # Check agent restriction
            if not policy.is_agent_allowed(agent_id):
                result = self._create_deny_result(
                    policy=policy,
                    reason=f"Agent '{agent_id}' not allowed by policy '{policy.name}'",
                    task_type=task_type,
                    agent_id=agent_id,
                    region=region,
                )
                self._record_violation(
                    policy=policy,
                    violation_type="agent",
                    description=result.reason,
                    task_id=task_id,
                    task_type=task_type,
                    agent_id=agent_id,
                    region=region,
                    workspace_id=workspace,
                )
                # Log to audit trail
                self._fire_audit_log(
                    policy_id=policy.id,
                    decision="deny",
                    task_type=task_type,
                    reason=result.reason,
                    workspace_id=workspace,
                    task_id=task_id,
                    agent_id=agent_id,
                    violations=[f"agent:{agent_id}"],
                )
                # Record Prometheus metric
                self._record_policy_metric(decision="deny", policy_type="agent_restriction")
                return result

            # Check region restriction
            if not policy.is_region_allowed(region, data_region):
                result = self._create_deny_result(
                    policy=policy,
                    reason=f"Region '{region}' not allowed by policy '{policy.name}'",
                    task_type=task_type,
                    agent_id=agent_id,
                    region=region,
                )
                self._record_violation(
                    policy=policy,
                    violation_type="region",
                    description=result.reason,
                    task_id=task_id,
                    task_type=task_type,
                    agent_id=agent_id,
                    region=region,
                    workspace_id=workspace,
                )
                # Log to audit trail
                self._fire_audit_log(
                    policy_id=policy.id,
                    decision="deny",
                    task_type=task_type,
                    reason=result.reason,
                    workspace_id=workspace,
                    task_id=task_id,
                    agent_id=agent_id,
                    violations=[f"region:{region}"],
                )
                # Record Prometheus metric
                self._record_policy_metric(decision="deny", policy_type="region_restriction")
                return result

        # All policies passed
        self._metrics["allowed"] += 1
        result = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            policy_id="",
            policy_name="",
            reason="All policies passed",
            enforcement_level=EnforcementLevel.HARD,
            task_type=task_type,
            agent_id=agent_id,
            region=region,
        )

        # Log to audit trail (fire and forget)
        self._fire_audit_log(
            policy_id="all",
            decision="allow",
            task_type=task_type,
            reason="All policies passed",
            workspace_id=workspace,
            task_id=task_id,
            agent_id=agent_id,
        )
        # Record Prometheus metric
        self._record_policy_metric(decision="allow", policy_type="all")

        return result

    def evaluate_sla_compliance(
        self,
        policy_id: str,
        execution_seconds: Optional[float] = None,
        queue_seconds: Optional[float] = None,
        available_agents: Optional[int] = None,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate SLA compliance for a running or completed task.

        Args:
            policy_id: Policy with SLA to check
            execution_seconds: Task execution duration
            queue_seconds: Time task waited in queue
            available_agents: Number of available agents
            task_id: Task being evaluated
            task_type: Type of task
            agent_id: Agent executing task
            workspace: Workspace context

        Returns:
            PolicyEvaluationResult with SLA compliance status
        """
        policy = self.get_policy(policy_id)
        if not policy or not policy.sla:
            return PolicyEvaluationResult(
                decision=PolicyDecision.ALLOW,
                allowed=True,
                policy_id=policy_id,
                policy_name=policy.name if policy else "unknown",
                reason="No SLA requirements",
                enforcement_level=EnforcementLevel.HARD,
            )

        sla = policy.sla
        violations = []

        # Check execution time
        if execution_seconds is not None:
            if not sla.is_execution_time_compliant(execution_seconds):
                violations.append(
                    f"Execution time {execution_seconds:.1f}s exceeds limit "
                    f"{sla.max_execution_seconds:.1f}s"
                )

        # Check queue time
        if queue_seconds is not None:
            if not sla.is_queue_time_compliant(queue_seconds):
                violations.append(
                    f"Queue time {queue_seconds:.1f}s exceeds limit {sla.max_queue_seconds:.1f}s"
                )

        # Check available agents
        if available_agents is not None:
            if available_agents < sla.min_agents_available:
                violations.append(
                    f"Available agents {available_agents} below minimum {sla.min_agents_available}"
                )

        if violations:
            violation_str = "; ".join(violations)
            result = self._create_deny_result(
                policy=policy,
                reason=f"SLA violation: {violation_str}",
                task_type=task_type,
                agent_id=agent_id,
                region=None,
                sla_violation=violation_str,
            )
            self._record_violation(
                policy=policy,
                violation_type="sla",
                description=violation_str,
                task_id=task_id,
                task_type=task_type,
                agent_id=agent_id,
                workspace_id=workspace,
            )
            return result

        return PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            policy_id=policy.id,
            policy_name=policy.name,
            reason="SLA requirements met",
            enforcement_level=policy.enforcement_level,
            task_type=task_type,
            agent_id=agent_id,
        )

    def _create_deny_result(
        self,
        policy: ControlPlanePolicy,
        reason: str,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        region: Optional[str] = None,
        sla_violation: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        """Create a deny/warn result based on enforcement level."""
        if policy.enforcement_level == EnforcementLevel.WARN:
            self._metrics["warned"] += 1
            decision = PolicyDecision.WARN
            allowed = True
        else:
            self._metrics["denied"] += 1
            decision = PolicyDecision.DENY
            allowed = False

        return PolicyEvaluationResult(
            decision=decision,
            allowed=allowed,
            policy_id=policy.id,
            policy_name=policy.name,
            reason=reason,
            enforcement_level=policy.enforcement_level,
            task_type=task_type,
            agent_id=agent_id,
            region=region,
            sla_violation=sla_violation,
        )

    def _record_violation(
        self,
        policy: ControlPlanePolicy,
        violation_type: str,
        description: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        region: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> PolicyViolation:
        """Record a policy violation."""
        self._metrics["violations"] += 1

        violation = PolicyViolation(
            id=f"violation_{uuid.uuid4().hex[:12]}",
            policy_id=policy.id,
            policy_name=policy.name,
            violation_type=violation_type,
            description=description,
            task_id=task_id,
            task_type=task_type,
            agent_id=agent_id,
            region=region,
            workspace_id=workspace_id,
            enforcement_level=policy.enforcement_level,
        )

        self._violations.append(violation)

        # Notify callback if configured
        if self._violation_callback:
            try:
                self._violation_callback(violation)
            except Exception as e:  # User callback - any exception possible
                logger.warning(
                    "violation_callback_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )

        logger.warning(
            "policy_violation",
            policy_id=policy.id,
            policy_name=policy.name,
            violation_type=violation_type,
            description=description,
            enforcement_level=policy.enforcement_level.value,
        )

        return violation

    def _fire_audit_log(
        self,
        policy_id: str,
        decision: str,
        task_type: str,
        reason: str,
        workspace_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        violations: Optional[List[str]] = None,
    ) -> None:
        """Fire audit log entry asynchronously (fire and forget)."""
        if not HAS_AUDIT or log_policy_decision is None:
            return

        import asyncio

        try:
            # Try to get running loop
            asyncio.get_running_loop()  # Check if loop exists
            asyncio.ensure_future(
                log_policy_decision(
                    policy_id=policy_id,
                    decision=decision,
                    task_type=task_type,
                    reason=reason,
                    workspace_id=workspace_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    violations=violations,
                )
            )
        except RuntimeError:
            # No running loop - log synchronously in debug mode
            logger.debug(
                "audit_log_skipped",
                policy_id=policy_id,
                decision=decision,
                reason="No event loop running",
            )

    def _record_policy_metric(
        self,
        decision: str,
        policy_type: str,
    ) -> None:
        """Record policy decision metric to Prometheus."""
        if not HAS_PROMETHEUS or prometheus_record_policy is None:
            return

        try:
            prometheus_record_policy(decision=decision, policy_type=policy_type)
        except (ValueError, RuntimeError) as e:
            # Prometheus client errors (invalid labels, unregistered metrics)
            logger.debug("prometheus_metric_failed", error=str(e))

    def get_violations(
        self,
        policy_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PolicyViolation]:
        """Get recorded violations with optional filters."""
        violations = self._violations

        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]

        if violation_type:
            violations = [v for v in violations if v.violation_type == violation_type]

        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]

        return sorted(violations, key=lambda v: v.timestamp, reverse=True)[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get policy evaluation metrics."""
        return {
            **self._metrics,
            "policy_count": len(self._policies),
            "violation_count": len(self._violations),
        }

    def clear_violations(self) -> int:
        """Clear all recorded violations. Returns count cleared."""
        count = len(self._violations)
        self._violations = []
        return count

    def sync_from_store(self, workspace: Optional[str] = None) -> int:
        """
        Sync policies from the persistent ControlPlanePolicyStore.

        This loads control plane policies from the database, enabling
        central policy management and persistence across restarts.

        Args:
            workspace: Optional workspace filter

        Returns:
            Number of policies loaded
        """
        try:
            from aragora.control_plane.policy_store import get_control_plane_policy_store

            store = get_control_plane_policy_store()
            policies = store.list_policies(enabled_only=True, workspace=workspace)

            loaded = 0
            for policy in policies:
                if policy.id not in self._policies:
                    self._policies[policy.id] = policy
                    loaded += 1

            logger.info(
                "policies_synced_from_store",
                loaded=loaded,
                total_in_store=len(policies),
                total_in_manager=len(self._policies),
            )
            return loaded

        except ImportError:
            logger.debug("Control plane policy store not available")
            return 0
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Policy store sync failed: {e}")
            return 0

    def sync_to_store(self) -> int:
        """
        Sync current policies to the persistent store.

        Saves all in-memory policies to the database for persistence.

        Returns:
            Number of policies saved
        """
        try:
            from aragora.control_plane.policy_store import get_control_plane_policy_store

            store = get_control_plane_policy_store()
            saved = 0

            for policy in self._policies.values():
                existing = store.get_policy(policy.id)
                if existing:
                    store.update_policy(policy.id, policy.to_dict())
                else:
                    store.create_policy(policy)
                saved += 1

            logger.info("policies_synced_to_store", saved=saved)
            return saved

        except ImportError:
            logger.debug("Control plane policy store not available")
            return 0
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Policy store sync failed: {e}")
            return 0

    def sync_violations_to_store(self) -> int:
        """
        Sync in-memory violations to the persistent store.

        Returns:
            Number of violations saved
        """
        try:
            from aragora.control_plane.policy_store import get_control_plane_policy_store

            store = get_control_plane_policy_store()
            saved = 0

            for violation in self._violations:
                try:
                    store.create_violation(violation)
                    saved += 1
                except (KeyError, ValueError) as e:
                    logger.debug(
                        "skipping_violation_sync",
                        error=str(e),
                        reason="may already exist or have invalid data",
                    )

            logger.info("violations_synced_to_store", saved=saved)
            return saved

        except ImportError:
            logger.debug("Control plane policy store not available")
            return 0
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Violation store sync failed: {e}")
            return 0


# Pre-built policies for common scenarios
def create_production_policy(
    agent_allowlist: Optional[List[str]] = None,
    allowed_regions: Optional[List[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy for production task restrictions."""
    return ControlPlanePolicy(
        name="production-restrictions",
        description="Restricts production tasks to approved agents and regions",
        task_types=["production-deployment", "production-migration"],
        agent_allowlist=agent_allowlist or [],
        region_constraint=RegionConstraint(
            allowed_regions=allowed_regions or [],
            require_data_residency=True,
        ),
        sla=SLARequirements(
            max_execution_seconds=600.0,
            max_queue_seconds=30.0,
            min_agents_available=2,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=100,
    )


def create_sensitive_data_policy(
    data_regions: List[str],
    blocked_regions: Optional[List[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy for sensitive data handling with residency requirements."""
    return ControlPlanePolicy(
        name="sensitive-data-residency",
        description="Enforces data residency for sensitive task types",
        task_types=["pii-processing", "financial-analysis", "healthcare-analysis"],
        region_constraint=RegionConstraint(
            allowed_regions=data_regions,
            blocked_regions=blocked_regions or [],
            require_data_residency=True,
            allow_cross_region=False,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=90,
    )


def create_agent_tier_policy(
    tier: str,
    agents: List[str],
    task_types: Optional[List[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy restricting certain task types to specific agent tiers."""
    return ControlPlanePolicy(
        name=f"{tier}-agent-tier",
        description=f"Restricts tasks to {tier} tier agents",
        task_types=task_types or [],
        agent_allowlist=agents,
        enforcement_level=EnforcementLevel.HARD,
        priority=50,
    )


def create_sla_policy(
    name: str,
    task_types: List[str],
    max_execution_seconds: float = 300.0,
    max_queue_seconds: float = 60.0,
) -> ControlPlanePolicy:
    """Create a policy with SLA enforcement."""
    return ControlPlanePolicy(
        name=name,
        description=f"SLA requirements for {', '.join(task_types)}",
        task_types=task_types,
        sla=SLARequirements(
            max_execution_seconds=max_execution_seconds,
            max_queue_seconds=max_queue_seconds,
        ),
        enforcement_level=EnforcementLevel.WARN,  # Warn on SLA violation
        priority=30,
    )


# =============================================================================
# Policy Store Sync - Bridges compliance policies to control plane
# =============================================================================


class PolicyStoreSync:
    """
    Syncs policies from the compliance PolicyStore to ControlPlanePolicyManager.

    This enables central policy management where compliance policies define
    constraints that are automatically enforced in the control plane.

    Compliance Policy -> Control Plane Policy Mapping:
    - framework_id='data_residency' -> RegionConstraint policies
    - framework_id='agent_restrictions' -> Agent allowlist/blocklist policies
    - framework_id='sla_requirements' -> SLA enforcement policies
    - level='mandatory' -> EnforcementLevel.HARD
    - level='recommended' -> EnforcementLevel.WARN
    """

    # Mapping of compliance framework IDs to control plane policy generators
    FRAMEWORK_MAPPINGS = {
        "data_residency": "_convert_data_residency_policy",
        "agent_restrictions": "_convert_agent_restriction_policy",
        "sla_requirements": "_convert_sla_policy",
        "task_restrictions": "_convert_task_restriction_policy",
    }

    def __init__(self, policy_manager: ControlPlanePolicyManager):
        self._policy_manager = policy_manager
        self._synced_policy_ids: set = set()

    def sync_from_store(
        self,
        workspace_id: Optional[str] = None,
        enabled_only: bool = True,
        store: Optional[Any] = None,
        replace: bool = True,
    ) -> int:
        """
        Sync policies from compliance store to control plane.

        Args:
            workspace_id: Optional workspace to filter policies
            enabled_only: Only sync enabled policies
            store: Optional compliance policy store instance (for testing)
            replace: If True, clear previously synced policies before loading

        Returns:
            Number of policies synced
        """
        try:
            if store is None:
                from aragora.compliance.policy_store import get_policy_store

                store = get_policy_store()

            if replace:
                self.clear_synced_policies()

            policies = store.list_policies(
                workspace_id=workspace_id,
                enabled_only=enabled_only,
                limit=1000,
            )

            synced = 0
            for policy in policies:
                if self._sync_policy(policy):
                    synced += 1

            logger.info(
                "policy_store_sync_complete",
                total_policies=len(policies),
                synced=synced,
                workspace_id=workspace_id,
            )
            return synced

        except ImportError:
            logger.debug("Compliance policy store not available for sync")
            return 0
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Policy store sync failed: {e}")
            return 0

    def _sync_policy(self, compliance_policy) -> bool:
        """Convert and sync a single compliance policy."""
        # Explicit control plane payloads take priority
        if hasattr(self._policy_manager, "_extract_control_plane_policy"):
            control_policy = self._policy_manager._extract_control_plane_policy(  # type: ignore[attr-defined]
                compliance_policy
            )
            if control_policy:
                self._policy_manager.add_policy(control_policy)
                self._synced_policy_ids.add(control_policy.id)
                return True

        framework_id = compliance_policy.framework_id

        # Check if we have a mapping for this framework
        converter_name = self.FRAMEWORK_MAPPINGS.get(framework_id)
        if not converter_name:
            # Try generic conversion for unmapped frameworks
            converter_name = "_convert_generic_policy"

        converter = getattr(self, converter_name, None)
        if not converter:
            return False

        try:
            control_policy = converter(compliance_policy)
            if control_policy:
                self._policy_manager.add_policy(control_policy)
                self._synced_policy_ids.add(control_policy.id)
                return True
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to convert policy {compliance_policy.id}: {e}")

        return False

    def _convert_data_residency_policy(self, policy) -> Optional[ControlPlanePolicy]:
        """Convert data residency compliance policy to region constraint."""
        # Extract allowed regions from policy rules
        allowed_regions = []
        blocked_regions = []

        for rule in policy.rules:
            if not rule.enabled:
                continue
            metadata = rule.metadata or {}
            if "allowed_regions" in metadata:
                allowed_regions.extend(metadata["allowed_regions"])
            if "blocked_regions" in metadata:
                blocked_regions.extend(metadata["blocked_regions"])

        if not allowed_regions and not blocked_regions:
            return None

        return ControlPlanePolicy(
            id=f"compliance_{policy.id}",
            name=f"[Compliance] {policy.name}",
            description=policy.description or f"Data residency policy from {policy.framework_id}",
            scope=PolicyScope.WORKSPACE if policy.workspace_id != "default" else PolicyScope.GLOBAL,
            workspaces=[policy.workspace_id] if policy.workspace_id != "default" else [],
            region_constraint=RegionConstraint(
                allowed_regions=list(set(allowed_regions)),
                blocked_regions=list(set(blocked_regions)),
                require_data_residency=True,
            ),
            enforcement_level=(
                EnforcementLevel.HARD if policy.level == "mandatory" else EnforcementLevel.WARN
            ),
            enabled=policy.enabled,
            priority=80 if policy.level == "mandatory" else 40,
            created_by=policy.created_by,
            metadata={"source": "compliance_store", "framework_id": policy.framework_id},
        )

    def _convert_agent_restriction_policy(self, policy) -> Optional[ControlPlanePolicy]:
        """Convert agent restriction compliance policy."""
        allowlist = []
        blocklist = []
        task_types = []

        for rule in policy.rules:
            if not rule.enabled:
                continue
            metadata = rule.metadata or {}
            if "agent_allowlist" in metadata:
                allowlist.extend(metadata["agent_allowlist"])
            if "agent_blocklist" in metadata:
                blocklist.extend(metadata["agent_blocklist"])
            if "task_types" in metadata:
                task_types.extend(metadata["task_types"])

        if not allowlist and not blocklist:
            return None

        return ControlPlanePolicy(
            id=f"compliance_{policy.id}",
            name=f"[Compliance] {policy.name}",
            description=policy.description or f"Agent restrictions from {policy.framework_id}",
            scope=PolicyScope.WORKSPACE if policy.workspace_id != "default" else PolicyScope.GLOBAL,
            workspaces=[policy.workspace_id] if policy.workspace_id != "default" else [],
            task_types=list(set(task_types)),
            agent_allowlist=list(set(allowlist)),
            agent_blocklist=list(set(blocklist)),
            enforcement_level=(
                EnforcementLevel.HARD if policy.level == "mandatory" else EnforcementLevel.WARN
            ),
            enabled=policy.enabled,
            priority=70 if policy.level == "mandatory" else 35,
            created_by=policy.created_by,
            metadata={"source": "compliance_store", "framework_id": policy.framework_id},
        )

    def _convert_sla_policy(self, policy) -> Optional[ControlPlanePolicy]:
        """Convert SLA compliance policy."""
        max_execution = None
        max_queue = None
        min_agents = None
        task_types = []

        for rule in policy.rules:
            if not rule.enabled:
                continue
            metadata = rule.metadata or {}
            if "max_execution_seconds" in metadata:
                max_execution = float(metadata["max_execution_seconds"])
            if "max_queue_seconds" in metadata:
                max_queue = float(metadata["max_queue_seconds"])
            if "min_agents_available" in metadata:
                min_agents = int(metadata["min_agents_available"])
            if "task_types" in metadata:
                task_types.extend(metadata["task_types"])

        if max_execution is None and max_queue is None:
            return None

        return ControlPlanePolicy(
            id=f"compliance_{policy.id}",
            name=f"[Compliance] {policy.name}",
            description=policy.description or f"SLA requirements from {policy.framework_id}",
            scope=PolicyScope.WORKSPACE if policy.workspace_id != "default" else PolicyScope.GLOBAL,
            workspaces=[policy.workspace_id] if policy.workspace_id != "default" else [],
            task_types=list(set(task_types)),
            sla=SLARequirements(
                max_execution_seconds=max_execution or 300.0,
                max_queue_seconds=max_queue or 60.0,
                min_agents_available=min_agents,
            ),
            enforcement_level=(
                EnforcementLevel.HARD if policy.level == "mandatory" else EnforcementLevel.WARN
            ),
            enabled=policy.enabled,
            priority=60 if policy.level == "mandatory" else 30,
            created_by=policy.created_by,
            metadata={"source": "compliance_store", "framework_id": policy.framework_id},
        )

    def _convert_task_restriction_policy(self, policy) -> Optional[ControlPlanePolicy]:
        """Convert task restriction compliance policy."""
        task_types = []
        capabilities = []

        for rule in policy.rules:
            if not rule.enabled:
                continue
            metadata = rule.metadata or {}
            if "restricted_task_types" in metadata:
                task_types.extend(metadata["restricted_task_types"])
            if "required_capabilities" in metadata:
                capabilities.extend(metadata["required_capabilities"])

        if not task_types and not capabilities:
            return None

        return ControlPlanePolicy(
            id=f"compliance_{policy.id}",
            name=f"[Compliance] {policy.name}",
            description=policy.description or f"Task restrictions from {policy.framework_id}",
            scope=PolicyScope.WORKSPACE if policy.workspace_id != "default" else PolicyScope.GLOBAL,
            workspaces=[policy.workspace_id] if policy.workspace_id != "default" else [],
            task_types=list(set(task_types)),
            capabilities=list(set(capabilities)),
            enforcement_level=(
                EnforcementLevel.HARD if policy.level == "mandatory" else EnforcementLevel.WARN
            ),
            enabled=policy.enabled,
            priority=50 if policy.level == "mandatory" else 25,
            created_by=policy.created_by,
            metadata={"source": "compliance_store", "framework_id": policy.framework_id},
        )

    def _convert_generic_policy(self, policy) -> Optional[ControlPlanePolicy]:
        """Convert a generic compliance policy with basic mapping."""
        # Only convert if the policy has rules with control-plane-relevant metadata
        has_relevant_rules = False
        for rule in policy.rules:
            if not rule.enabled:
                continue
            metadata = rule.metadata or {}
            if any(
                k in metadata
                for k in [
                    "allowed_regions",
                    "agent_allowlist",
                    "max_execution_seconds",
                    "task_types",
                ]
            ):
                has_relevant_rules = True
                break

        if not has_relevant_rules:
            return None

        # Try each converter to find one that works
        for converter_name in [
            "_convert_data_residency_policy",
            "_convert_agent_restriction_policy",
            "_convert_sla_policy",
            "_convert_task_restriction_policy",
        ]:
            converter = getattr(self, converter_name)
            result = converter(policy)
            if result:
                return result

        return None

    def clear_synced_policies(self) -> int:
        """Remove all policies that were synced from the compliance store."""
        removed = 0
        for policy_id in list(self._synced_policy_ids):
            if self._policy_manager.remove_policy(policy_id):
                removed += 1
                self._synced_policy_ids.discard(policy_id)
        return removed


# Add sync method to ControlPlanePolicyManager
def _sync_from_compliance_store(
    self: ControlPlanePolicyManager,
    workspace_id: Optional[str] = None,
    enabled_only: bool = True,
    store: Optional[Any] = None,
    replace: bool = True,
) -> int:
    """
    Sync policies from the compliance PolicyStore.

    This method bridges compliance policies defined in the policy store
    to control plane policies for runtime enforcement.

    Args:
        workspace_id: Optional workspace filter
        enabled_only: Only sync enabled policies

    Returns:
        Number of policies synced
    """
    if not hasattr(self, "_store_sync"):
        self._store_sync = PolicyStoreSync(self)  # type: ignore[attr-defined]
    return self._store_sync.sync_from_store(  # type: ignore[attr-defined]
        workspace_id=workspace_id,
        enabled_only=enabled_only,
        store=store,
        replace=replace,
    )


# Monkey-patch the method onto the class
ControlPlanePolicyManager.sync_from_compliance_store = _sync_from_compliance_store  # type: ignore[method-assign,assignment]


# =============================================================================
# Policy Conflict Detection
# =============================================================================


@dataclass
class PolicyConflict:
    """Represents a conflict between two policies."""

    policy_a_id: str
    policy_a_name: str
    policy_b_id: str
    policy_b_name: str
    conflict_type: str  # "agent", "region", "overlapping_scope"
    description: str
    severity: str  # "warning", "error"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "policy_a_id": self.policy_a_id,
            "policy_a_name": self.policy_a_name,
            "policy_b_id": self.policy_b_id,
            "policy_b_name": self.policy_b_name,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
        }


class PolicyConflictDetector:
    """
    Detects conflicts between control plane policies.

    Identifies situations where:
    - Two policies have overlapping scope but contradictory agent restrictions
    - Two policies have overlapping scope but contradictory region constraints
    - Policies create impossible-to-satisfy conditions

    Usage:
        detector = PolicyConflictDetector()
        conflicts = detector.detect_conflicts(policies)
        for conflict in conflicts:
            logger.warning(f"Policy conflict: {conflict.description}")
    """

    def detect_conflicts(
        self,
        policies: List[ControlPlanePolicy],
    ) -> List[PolicyConflict]:
        """
        Detect conflicts between a set of policies.

        Args:
            policies: List of policies to check for conflicts

        Returns:
            List of detected conflicts
        """
        conflicts: List[PolicyConflict] = []
        enabled_policies = [p for p in policies if p.enabled]

        for i, policy_a in enumerate(enabled_policies):
            for policy_b in enabled_policies[i + 1 :]:
                # Check for overlapping scope
                if not self._scopes_overlap(policy_a, policy_b):
                    continue

                # Check for agent restriction conflicts
                agent_conflicts = self._check_agent_conflicts(policy_a, policy_b)
                conflicts.extend(agent_conflicts)

                # Check for region constraint conflicts
                region_conflicts = self._check_region_conflicts(policy_a, policy_b)
                conflicts.extend(region_conflicts)

                # Check for enforcement level inconsistencies
                enforcement_conflicts = self._check_enforcement_conflicts(policy_a, policy_b)
                conflicts.extend(enforcement_conflicts)

                # Check for SLA requirement conflicts
                sla_conflicts = self._check_sla_conflicts(policy_a, policy_b)
                conflicts.extend(sla_conflicts)

        return conflicts

    def _scopes_overlap(
        self,
        policy_a: ControlPlanePolicy,
        policy_b: ControlPlanePolicy,
    ) -> bool:
        """Check if two policies have overlapping scopes.

        Even if policies have global scope, they don't overlap if their
        task_types, workspaces, or capabilities are mutually exclusive.
        """
        # Check task type overlap first - this is the primary scope filter
        if policy_a.task_types and policy_b.task_types:
            if not set(policy_a.task_types).intersection(policy_b.task_types):
                return False
        # If only one has task types, they could still overlap

        # Check workspace overlap
        if policy_a.workspaces and policy_b.workspaces:
            if not set(policy_a.workspaces).intersection(policy_b.workspaces):
                return False

        # Check capability overlap
        if policy_a.capabilities and policy_b.capabilities:
            if not set(policy_a.capabilities).intersection(policy_b.capabilities):
                return False

        return True

    def _check_agent_conflicts(
        self,
        policy_a: ControlPlanePolicy,
        policy_b: ControlPlanePolicy,
    ) -> List[PolicyConflict]:
        """Check for conflicting agent restrictions."""
        conflicts: List[PolicyConflict] = []

        # Conflict: A allows an agent that B blocks
        if policy_a.agent_allowlist and policy_b.agent_blocklist:
            blocked_allowed = set(policy_a.agent_allowlist).intersection(policy_b.agent_blocklist)
            if blocked_allowed:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="agent",
                        description=(
                            f"Policy '{policy_a.name}' allows agents {blocked_allowed} "
                            f"but policy '{policy_b.name}' blocks them"
                        ),
                        severity=(
                            "error"
                            if policy_b.enforcement_level == EnforcementLevel.HARD
                            else "warning"
                        ),
                    )
                )

        # Conflict: B allows an agent that A blocks
        if policy_b.agent_allowlist and policy_a.agent_blocklist:
            blocked_allowed = set(policy_b.agent_allowlist).intersection(policy_a.agent_blocklist)
            if blocked_allowed:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="agent",
                        description=(
                            f"Policy '{policy_b.name}' allows agents {blocked_allowed} "
                            f"but policy '{policy_a.name}' blocks them"
                        ),
                        severity=(
                            "error"
                            if policy_a.enforcement_level == EnforcementLevel.HARD
                            else "warning"
                        ),
                    )
                )

        # Conflict: Both have allowlists with no overlap (impossible to satisfy)
        if policy_a.agent_allowlist and policy_b.agent_allowlist:
            overlap = set(policy_a.agent_allowlist).intersection(policy_b.agent_allowlist)
            if not overlap:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="agent",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"non-overlapping agent allowlists - no agent can satisfy both"
                        ),
                        severity="error",
                    )
                )

        return conflicts

    def _check_region_conflicts(
        self,
        policy_a: ControlPlanePolicy,
        policy_b: ControlPlanePolicy,
    ) -> List[PolicyConflict]:
        """Check for conflicting region constraints."""
        conflicts: List[PolicyConflict] = []

        rc_a = policy_a.region_constraint
        rc_b = policy_b.region_constraint

        if not rc_a or not rc_b:
            return conflicts

        # Conflict: A allows a region that B blocks
        if rc_a.allowed_regions and rc_b.blocked_regions:
            blocked_allowed = set(rc_a.allowed_regions).intersection(rc_b.blocked_regions)
            if blocked_allowed:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="region",
                        description=(
                            f"Policy '{policy_a.name}' allows regions {blocked_allowed} "
                            f"but policy '{policy_b.name}' blocks them"
                        ),
                        severity=(
                            "error"
                            if policy_b.enforcement_level == EnforcementLevel.HARD
                            else "warning"
                        ),
                    )
                )

        # Conflict: Both have allowed regions with no overlap
        if rc_a.allowed_regions and rc_b.allowed_regions:
            overlap = set(rc_a.allowed_regions).intersection(rc_b.allowed_regions)
            if not overlap:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="region",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"non-overlapping allowed regions - no region can satisfy both"
                        ),
                        severity="error",
                    )
                )

        return conflicts

    def _check_enforcement_conflicts(
        self,
        policy_a: ControlPlanePolicy,
        policy_b: ControlPlanePolicy,
    ) -> List[PolicyConflict]:
        """Check for inconsistent enforcement levels on similar policies."""
        conflicts: List[PolicyConflict] = []

        # Warning when similar policies have different enforcement levels
        # (can cause confusion about actual behavior)
        if policy_a.enforcement_level != policy_b.enforcement_level:
            # Only warn if both have the same constraints (not just overlapping)
            same_agents = set(policy_a.agent_allowlist) == set(policy_b.agent_allowlist) and set(
                policy_a.agent_blocklist
            ) == set(policy_b.agent_blocklist)
            same_task_types = set(policy_a.task_types) == set(policy_b.task_types)

            if (
                same_agents
                and same_task_types
                and (policy_a.agent_allowlist or policy_a.agent_blocklist)
            ):
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="overlapping_scope",
                        description=(
                            f"Policies '{policy_a.name}' ({policy_a.enforcement_level.value}) "
                            f"and '{policy_b.name}' ({policy_b.enforcement_level.value}) "
                            f"have identical constraints but different enforcement levels"
                        ),
                        severity="warning",
                    )
                )

        return conflicts

    def _check_sla_conflicts(
        self,
        policy_a: ControlPlanePolicy,
        policy_b: ControlPlanePolicy,
    ) -> List[PolicyConflict]:
        """Check for conflicting SLA requirements between overlapping policies.

        Detects situations where two policies with overlapping scope have
        significantly different SLA requirements that could cause confusion
        or impossible-to-satisfy conditions.
        """
        conflicts: List[PolicyConflict] = []

        # Skip if either policy has no SLA requirements
        if not policy_a.sla or not policy_b.sla:
            return conflicts

        sla_a = policy_a.sla
        sla_b = policy_b.sla

        # Check for conflicting max_execution_seconds
        # Flag if one policy requires significantly stricter execution time
        if sla_a.max_execution_seconds > 0 and sla_b.max_execution_seconds > 0:
            ratio = max(sla_a.max_execution_seconds, sla_b.max_execution_seconds) / min(
                sla_a.max_execution_seconds, sla_b.max_execution_seconds
            )
            if ratio > 3.0:  # More than 3x difference is likely a conflict
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="sla_execution_time",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"conflicting execution time limits: {sla_a.max_execution_seconds}s "
                            f"vs {sla_b.max_execution_seconds}s (>{ratio:.1f}x difference)"
                        ),
                        severity="warning",
                    )
                )

        # Check for conflicting max_queue_seconds
        if sla_a.max_queue_seconds > 0 and sla_b.max_queue_seconds > 0:
            ratio = max(sla_a.max_queue_seconds, sla_b.max_queue_seconds) / min(
                sla_a.max_queue_seconds, sla_b.max_queue_seconds
            )
            if ratio > 3.0:
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="sla_queue_time",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"conflicting queue time limits: {sla_a.max_queue_seconds}s "
                            f"vs {sla_b.max_queue_seconds}s (>{ratio:.1f}x difference)"
                        ),
                        severity="warning",
                    )
                )

        # Check for conflicting max_concurrent_tasks
        if sla_a.max_concurrent_tasks != sla_b.max_concurrent_tasks:
            ratio = max(sla_a.max_concurrent_tasks, sla_b.max_concurrent_tasks) / max(
                min(sla_a.max_concurrent_tasks, sla_b.max_concurrent_tasks), 1
            )
            if ratio > 5.0:  # More than 5x difference
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="sla_concurrent_tasks",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"conflicting concurrent task limits: {sla_a.max_concurrent_tasks} "
                            f"vs {sla_b.max_concurrent_tasks}"
                        ),
                        severity="warning",
                    )
                )

        # Check for conflicting response_time_p99_ms
        if sla_a.response_time_p99_ms > 0 and sla_b.response_time_p99_ms > 0:
            ratio = max(sla_a.response_time_p99_ms, sla_b.response_time_p99_ms) / min(
                sla_a.response_time_p99_ms, sla_b.response_time_p99_ms
            )
            if ratio > 5.0:  # More than 5x difference in P99 requirements
                conflicts.append(
                    PolicyConflict(
                        policy_a_id=policy_a.id,
                        policy_a_name=policy_a.name,
                        policy_b_id=policy_b.id,
                        policy_b_name=policy_b.name,
                        conflict_type="sla_response_time",
                        description=(
                            f"Policies '{policy_a.name}' and '{policy_b.name}' have "
                            f"conflicting P99 response time targets: {sla_a.response_time_p99_ms}ms "
                            f"vs {sla_b.response_time_p99_ms}ms"
                        ),
                        severity="warning",
                    )
                )

        # Check for impossible SLA: stricter queue time than execution time
        stricter_queue = min(sla_a.max_queue_seconds, sla_b.max_queue_seconds)
        stricter_exec = min(sla_a.max_execution_seconds, sla_b.max_execution_seconds)
        if stricter_queue > 0 and stricter_exec > 0 and stricter_queue > stricter_exec:
            conflicts.append(
                PolicyConflict(
                    policy_a_id=policy_a.id,
                    policy_a_name=policy_a.name,
                    policy_b_id=policy_b.id,
                    policy_b_name=policy_b.name,
                    conflict_type="sla_impossible",
                    description=(
                        f"Combined SLA requirements create impossible constraint: "
                        f"max queue time ({stricter_queue}s) > max execution time ({stricter_exec}s)"
                    ),
                    severity="error",
                )
            )

        return conflicts


# =============================================================================
# Distributed Policy Cache (Redis-backed)
# =============================================================================


class RedisPolicyCache:
    """
    Redis-backed cache for policy evaluation results.

    Provides fast lookups for frequently evaluated policy decisions,
    reducing repeated policy evaluation overhead in distributed deployments.

    Cache keys are based on the evaluation context hash (task_type + agent + region + workspace).
    Cache entries expire after a configurable TTL.

    Usage:
        cache = RedisPolicyCache(redis_url="redis://localhost:6379")
        await cache.connect()

        # Check cache before evaluation
        cached = await cache.get(task_type="debate", agent_id="claude", region="us-east-1")
        if cached:
            return cached

        # Evaluate and cache result
        result = policy_manager.evaluate_task_dispatch(...)
        await cache.set(result, task_type="debate", agent_id="claude", region="us-east-1")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aragora:policy_cache:",
        ttl_seconds: int = 300,
        enabled: bool = True,
    ):
        """
        Initialize the policy cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for cache keys
            ttl_seconds: Time-to-live for cache entries
            enabled: Whether caching is enabled
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled
        self._redis: Optional[Any] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
        }

    async def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected successfully, False otherwise
        """
        if not self._enabled or not REDIS_AVAILABLE:
            logger.debug("Policy cache disabled or Redis not available")
            return False

        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("policy_cache_connected", redis_url=self._redis_url)
            return True
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning("policy_cache_connection_failed", error=str(e))
            self._redis = None
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _make_cache_key(
        self,
        task_type: str,
        agent_id: str,
        region: str,
        workspace: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> str:
        """Generate a cache key from evaluation context."""
        components = [
            task_type,
            agent_id,
            region,
            workspace or "_default_",
            policy_version or "_current_",
        ]
        key_data = ":".join(components)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"{self._key_prefix}{key_hash}"

    async def get(
        self,
        task_type: str,
        agent_id: str,
        region: str,
        workspace: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> Optional[PolicyEvaluationResult]:
        """
        Get cached evaluation result.

        Args:
            task_type: Task type from evaluation
            agent_id: Agent ID from evaluation
            region: Region from evaluation
            workspace: Workspace from evaluation
            policy_version: Version hash of current policies (for invalidation)

        Returns:
            Cached PolicyEvaluationResult if found and valid, None otherwise
        """
        if not self._redis:
            return None

        try:
            key = self._make_cache_key(task_type, agent_id, region, workspace, policy_version)
            data = await self._redis.get(key)

            if data:
                self._stats["hits"] += 1
                cached_dict = json.loads(data)
                return PolicyEvaluationResult(
                    decision=PolicyDecision(cached_dict["decision"]),
                    allowed=cached_dict["allowed"],
                    policy_id=cached_dict["policy_id"],
                    policy_name=cached_dict["policy_name"],
                    reason=cached_dict["reason"],
                    enforcement_level=EnforcementLevel(cached_dict["enforcement_level"]),
                    task_type=cached_dict.get("task_type"),
                    agent_id=cached_dict.get("agent_id"),
                    region=cached_dict.get("region"),
                    sla_violation=cached_dict.get("sla_violation"),
                )

            self._stats["misses"] += 1
            return None

        except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError) as e:
            self._stats["errors"] += 1
            logger.debug("policy_cache_get_error", error=str(e))
            return None

    async def set(
        self,
        result: PolicyEvaluationResult,
        task_type: str,
        agent_id: str,
        region: str,
        workspace: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> bool:
        """
        Cache an evaluation result.

        Args:
            result: The evaluation result to cache
            task_type: Task type from evaluation
            agent_id: Agent ID from evaluation
            region: Region from evaluation
            workspace: Workspace from evaluation
            policy_version: Version hash of current policies

        Returns:
            True if cached successfully, False otherwise
        """
        if not self._redis:
            return False

        try:
            key = self._make_cache_key(task_type, agent_id, region, workspace, policy_version)
            data = json.dumps(result.to_dict())
            await self._redis.setex(key, self._ttl_seconds, data)
            self._stats["sets"] += 1
            return True

        except (OSError, ConnectionError, TimeoutError, TypeError) as e:
            self._stats["errors"] += 1
            logger.debug("policy_cache_set_error", error=str(e))
            return False

    async def invalidate_all(self) -> int:
        """
        Invalidate all cached policy results.

        Call this after policy changes to ensure fresh evaluations.

        Returns:
            Number of keys deleted
        """
        if not self._redis:
            return 0

        try:
            pattern = f"{self._key_prefix}*"
            deleted = 0
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
                deleted += 1
            logger.info("policy_cache_invalidated", deleted=deleted)
            return deleted
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning("policy_cache_invalidate_error", error=str(e))
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "connected": self._redis is not None,
        }


# =============================================================================
# Continuous Policy Sync Scheduler
# =============================================================================


class PolicySyncScheduler:
    """
    Background scheduler for continuous policy synchronization.

    Periodically syncs policies from the compliance store and control plane
    policy store, ensuring the in-memory policy manager stays up-to-date
    with persistent storage.

    Also runs conflict detection after each sync and can optionally
    invalidate the policy cache when changes are detected.

    Usage:
        manager = ControlPlanePolicyManager()
        cache = RedisPolicyCache(...)
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            policy_cache=cache,
            sync_interval_seconds=60,
        )

        # Start background sync
        await scheduler.start()

        # ... application runs ...

        # Stop on shutdown
        await scheduler.stop()
    """

    def __init__(
        self,
        policy_manager: ControlPlanePolicyManager,
        sync_interval_seconds: float = 60.0,
        policy_cache: Optional[RedisPolicyCache] = None,
        conflict_callback: Optional[Callable[[List[PolicyConflict]], None]] = None,
        sync_from_compliance_store: bool = True,
        sync_from_control_plane_store: bool = True,
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize the sync scheduler.

        Args:
            policy_manager: The policy manager to sync
            sync_interval_seconds: Interval between sync operations
            policy_cache: Optional cache to invalidate on policy changes
            conflict_callback: Callback invoked when conflicts are detected
            sync_from_compliance_store: Whether to sync from compliance store
            sync_from_control_plane_store: Whether to sync from control plane store
            workspace_id: Optional workspace filter for sync
        """
        self._policy_manager = policy_manager
        self._sync_interval = sync_interval_seconds
        self._policy_cache = policy_cache
        self._conflict_callback = conflict_callback
        self._sync_from_compliance = sync_from_compliance_store
        self._sync_from_cp_store = sync_from_control_plane_store
        self._workspace_id = workspace_id

        self._conflict_detector = PolicyConflictDetector()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._last_policy_hash: Optional[str] = None
        self._sync_count = 0
        self._error_count = 0
        self._detected_conflicts: List[PolicyConflict] = []

    async def start(self) -> None:
        """Start the background sync task."""
        if self._running:
            logger.warning("Policy sync scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            "policy_sync_scheduler_started",
            interval_seconds=self._sync_interval,
            workspace_id=self._workspace_id,
        )

    async def stop(self) -> None:
        """Stop the background sync task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("policy_sync_scheduler_stopped")

    async def sync_now(self) -> Dict[str, Any]:
        """
        Trigger an immediate sync.

        Returns:
            Sync result with counts and any detected conflicts
        """
        return await self._do_sync()

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                await self._do_sync()
            except asyncio.CancelledError:
                # Don't swallow cancellation - allow graceful shutdown
                raise
            except Exception as e:
                self._error_count += 1
                logger.error("policy_sync_error", error=str(e))

            await asyncio.sleep(self._sync_interval)

    async def _do_sync(self) -> Dict[str, Any]:
        """Perform a single sync operation."""
        self._sync_count += 1
        synced_compliance = 0
        synced_cp_store = 0
        changes_detected = False

        # Sync from compliance store
        if self._sync_from_compliance:
            try:
                synced_compliance = self._policy_manager.sync_from_compliance_store(
                    workspace_id=self._workspace_id,
                    replace=False,  # Don't replace, merge
                )
            except (ImportError, OSError, ConnectionError, TimeoutError) as e:
                logger.warning("compliance_store_sync_failed", error=str(e))

        # Sync from control plane policy store
        if self._sync_from_cp_store:
            try:
                synced_cp_store = self._policy_manager.sync_from_store(
                    workspace=self._workspace_id,
                )
            except (ImportError, OSError, ConnectionError, TimeoutError) as e:
                logger.warning("control_plane_store_sync_failed", error=str(e))

        # Calculate hash after sync (includes manually added policies)
        current_hash = self._compute_policy_hash()

        # Detect changes by comparing to PREVIOUS sync's hash
        # This catches both: changes from store sync AND manual policy additions
        if self._last_policy_hash is not None:
            changes_detected = self._last_policy_hash != current_hash
        else:
            # First sync - no previous hash to compare
            changes_detected = False

        # Invalidate cache if changes detected
        if changes_detected and self._policy_cache:
            await self._policy_cache.invalidate_all()

        # Run conflict detection
        policies = self._policy_manager.list_policies(enabled_only=True)
        self._detected_conflicts = self._conflict_detector.detect_conflicts(policies)

        if self._detected_conflicts:
            logger.warning(
                "policy_conflicts_detected",
                count=len(self._detected_conflicts),
                conflicts=[c.to_dict() for c in self._detected_conflicts[:5]],  # Log first 5
            )

            if self._conflict_callback:
                try:
                    self._conflict_callback(self._detected_conflicts)
                except Exception as e:  # User callback - any exception possible
                    logger.warning(
                        "conflict_callback_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        self._last_sync = datetime.now(timezone.utc)
        self._last_policy_hash = current_hash

        result = {
            "synced_from_compliance": synced_compliance,
            "synced_from_cp_store": synced_cp_store,
            "changes_detected": changes_detected,
            "conflicts_detected": len(self._detected_conflicts),
            "total_policies": len(policies),
            "sync_time": self._last_sync.isoformat(),
        }

        logger.info("policy_sync_completed", **result)
        return result

    def _compute_policy_hash(self) -> str:
        """Compute a hash of current policy state for change detection."""
        policies = self._policy_manager.list_policies(enabled_only=False)
        policy_data = sorted([p.to_dict() for p in policies], key=lambda p: p["id"])
        data_str = json.dumps(policy_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        return {
            "running": self._running,
            "sync_interval_seconds": self._sync_interval,
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "last_policy_hash": self._last_policy_hash,
            "detected_conflicts": len(self._detected_conflicts),
            "workspace_id": self._workspace_id,
        }

    def get_conflicts(self) -> List[PolicyConflict]:
        """Get currently detected conflicts."""
        return self._detected_conflicts.copy()

    @property
    def policy_version(self) -> Optional[str]:
        """Get the current policy version hash (for cache keys)."""
        return self._last_policy_hash


@dataclass
class PolicyVersion:
    """A snapshot of a policy at a specific version."""

    policy_id: str
    version: int
    policy_data: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    change_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "policy_data": self.policy_data,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_description": self.change_description,
        }


class PolicyHistory:
    """Tracks policy version history for auditing and rollback.

    Maintains a history of policy versions, enabling:
    - Viewing historical versions of any policy
    - Rolling back to previous versions
    - Audit trail of who changed what and when
    """

    def __init__(self, max_versions_per_policy: int = 50):
        """Initialize the policy history tracker.

        Args:
            max_versions_per_policy: Maximum versions to retain per policy
        """
        self._history: Dict[str, List[PolicyVersion]] = {}
        self._max_versions = max_versions_per_policy
        self._lock = asyncio.Lock()

    async def record_version(
        self,
        policy: ControlPlanePolicy,
        change_description: str = "",
        changed_by: Optional[str] = None,
    ) -> PolicyVersion:
        """Record a new version of a policy.

        Args:
            policy: The policy to record
            change_description: Description of what changed
            changed_by: User who made the change

        Returns:
            The recorded PolicyVersion
        """
        async with self._lock:
            policy_id = policy.id

            if policy_id not in self._history:
                self._history[policy_id] = []

            version = PolicyVersion(
                policy_id=policy_id,
                version=policy.version,
                policy_data=policy.to_dict(),
                created_by=changed_by,
                change_description=change_description,
            )

            self._history[policy_id].append(version)

            # Prune old versions
            if len(self._history[policy_id]) > self._max_versions:
                self._history[policy_id] = self._history[policy_id][-self._max_versions :]

            logger.info(
                f"Policy version recorded: {policy.name} v{policy.version} "
                f"by {changed_by or 'system'}"
            )

            return version

    async def get_history(
        self,
        policy_id: str,
        limit: int = 10,
    ) -> List[PolicyVersion]:
        """Get version history for a policy.

        Args:
            policy_id: The policy ID
            limit: Maximum versions to return

        Returns:
            List of PolicyVersions, newest first
        """
        async with self._lock:
            versions = self._history.get(policy_id, [])
            return list(reversed(versions[-limit:]))

    async def get_version(
        self,
        policy_id: str,
        version: int,
    ) -> Optional[PolicyVersion]:
        """Get a specific version of a policy.

        Args:
            policy_id: The policy ID
            version: The version number

        Returns:
            PolicyVersion if found, None otherwise
        """
        async with self._lock:
            versions = self._history.get(policy_id, [])
            for v in versions:
                if v.version == version:
                    return v
            return None

    async def rollback_to_version(
        self,
        policy_id: str,
        version: int,
        rolled_back_by: Optional[str] = None,
    ) -> Optional[ControlPlanePolicy]:
        """Restore a policy to a previous version.

        Args:
            policy_id: The policy ID
            version: The version to restore
            rolled_back_by: User performing the rollback

        Returns:
            New ControlPlanePolicy instance with restored data, or None if not found
        """
        target_version = await self.get_version(policy_id, version)
        if not target_version:
            logger.warning(f"Policy version not found: {policy_id} v{version}")
            return None

        # Get current version number
        history = self._history.get(policy_id, [])
        current_version = max((v.version for v in history), default=0)

        # Create new policy from historical data
        policy_data = target_version.policy_data.copy()
        policy_data["version"] = current_version + 1
        policy_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        policy_data["updated_by"] = rolled_back_by
        policy_data["previous_version_id"] = f"{policy_id}_v{current_version}"
        policy_data["metadata"]["rolled_back_from_version"] = version

        restored_policy = ControlPlanePolicy.from_dict(policy_data)

        # Record the rollback as a new version
        await self.record_version(
            restored_policy,
            change_description=f"Rollback to version {version}",
            changed_by=rolled_back_by,
        )

        logger.info(
            f"Policy rolled back: {policy_id} from v{current_version} to v{version} "
            f"(now v{restored_policy.version}) by {rolled_back_by or 'system'}"
        )

        return restored_policy

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about policy history."""
        total_versions = sum(len(v) for v in self._history.values())
        return {
            "tracked_policies": len(self._history),
            "total_versions": total_versions,
            "max_versions_per_policy": self._max_versions,
            "policies": {policy_id: len(versions) for policy_id, versions in self._history.items()},
        }


# Global policy history instance
_policy_history: Optional[PolicyHistory] = None


def get_policy_history() -> PolicyHistory:
    """Get the global policy history instance."""
    global _policy_history
    if _policy_history is None:
        _policy_history = PolicyHistory()
    return _policy_history


__all__ = [
    # Core classes
    "ControlPlanePolicy",
    "ControlPlanePolicyManager",
    "PolicyEvaluationResult",
    "PolicyViolation",
    "PolicyViolationError",
    "PolicyStoreSync",
    # Constraints
    "SLARequirements",
    "RegionConstraint",
    # Enums
    "PolicyScope",
    "EnforcementLevel",
    "PolicyDecision",
    # Factory functions
    "create_production_policy",
    "create_sensitive_data_policy",
    "create_agent_tier_policy",
    "create_sla_policy",
    # Governance hardening
    "PolicyConflict",
    "PolicyConflictDetector",
    "RedisPolicyCache",
    "PolicySyncScheduler",
    # Versioning and rollback
    "PolicyVersion",
    "PolicyHistory",
    "get_policy_history",
]
