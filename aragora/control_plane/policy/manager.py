"""
Control Plane Policy Manager.

Manages control plane policies and evaluates them for task dispatch.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable, Optional

from aragora.observability import get_logger

from .types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyViolation,
)

logger = get_logger(__name__)

# Audit logging (optional)
try:
    from aragora.control_plane.audit import log_policy_decision

    HAS_AUDIT = True
except ImportError:
    HAS_AUDIT = False
    log_policy_decision = None  # type: ignore[misc, no-redef]

# Prometheus metrics (optional)
try:
    from aragora.server.prometheus_control_plane import (
        record_policy_decision as prometheus_record_policy,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    prometheus_record_policy = None  # type: ignore[misc, no-redef]


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
        self._policies: dict[str, ControlPlanePolicy] = {}
        self._violations: list[PolicyViolation] = []
        self._violation_callback = violation_callback

        # Metrics
        self._metrics: dict[str, int] = {
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

    def get_policy(self, policy_id: str) -> ControlPlanePolicy | None:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        enabled_only: bool = True,
        task_type: str | None = None,
        workspace: str | None = None,
    ) -> list[ControlPlanePolicy]:
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

    def _extract_control_plane_policy(self, policy: Any) -> ControlPlanePolicy | None:
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
        store: Any | None = None,
        replace: bool = True,
        workspace_id: str | None = None,
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
        capabilities: Optional[list[str]] = None,
        workspace: str | None = None,
        data_region: str | None = None,
        task_id: str | None = None,
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
        execution_seconds: float | None = None,
        queue_seconds: float | None = None,
        available_agents: int | None = None,
        task_id: str | None = None,
        task_type: str | None = None,
        agent_id: str | None = None,
        workspace: str | None = None,
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
        task_type: str | None = None,
        agent_id: str | None = None,
        region: str | None = None,
        sla_violation: str | None = None,
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
        task_id: str | None = None,
        task_type: str | None = None,
        agent_id: str | None = None,
        region: str | None = None,
        workspace_id: str | None = None,
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
        workspace_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        violations: Optional[list[str]] = None,
    ) -> None:
        """Fire audit log entry asynchronously (fire and forget)."""
        if not HAS_AUDIT or log_policy_decision is None:
            return

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
        policy_id: str | None = None,
        violation_type: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
    ) -> list[PolicyViolation]:
        """Get recorded violations with optional filters."""
        violations = self._violations

        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]

        if violation_type:
            violations = [v for v in violations if v.violation_type == violation_type]

        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]

        return sorted(violations, key=lambda v: v.timestamp, reverse=True)[:limit]

    def get_metrics(self) -> dict[str, Any]:
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

    def sync_from_store(self, workspace: str | None = None) -> int:
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
