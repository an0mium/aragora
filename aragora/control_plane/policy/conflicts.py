"""
Control Plane Policy Conflict Detection.

Detects conflicts between control plane policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .types import ControlPlanePolicy, EnforcementLevel

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

    def to_dict(self) -> dict[str, Any]:
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
        policies: list[ControlPlanePolicy],
    ) -> list[PolicyConflict]:
        """
        Detect conflicts between a set of policies.

        Args:
            policies: List of policies to check for conflicts

        Returns:
            List of detected conflicts
        """
        conflicts: list[PolicyConflict] = []
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
    ) -> list[PolicyConflict]:
        """Check for conflicting agent restrictions."""
        conflicts: list[PolicyConflict] = []

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
    ) -> list[PolicyConflict]:
        """Check for conflicting region constraints."""
        conflicts: list[PolicyConflict] = []

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
    ) -> list[PolicyConflict]:
        """Check for inconsistent enforcement levels on similar policies."""
        conflicts: list[PolicyConflict] = []

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
    ) -> list[PolicyConflict]:
        """Check for conflicting SLA requirements between overlapping policies.

        Detects situations where two policies with overlapping scope have
        significantly different SLA requirements that could cause confusion
        or impossible-to-satisfy conditions.
        """
        conflicts: list[PolicyConflict] = []

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
