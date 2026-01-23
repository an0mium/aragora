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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from aragora.observability import get_logger

logger = get_logger(__name__)

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
                    f"Queue time {queue_seconds:.1f}s exceeds limit "
                    f"{sla.max_queue_seconds:.1f}s"
                )

        # Check available agents
        if available_agents is not None:
            if available_agents < sla.min_agents_available:
                violations.append(
                    f"Available agents {available_agents} below minimum "
                    f"{sla.min_agents_available}"
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
            except Exception as e:
                logger.warning("violation_callback_failed", error=str(e))

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
        except Exception as e:
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


__all__ = [
    # Core classes
    "ControlPlanePolicy",
    "ControlPlanePolicyManager",
    "PolicyEvaluationResult",
    "PolicyViolation",
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
]
