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

from .cache import RedisPolicyCache, REDIS_AVAILABLE
from .conflicts import PolicyConflict, PolicyConflictDetector
from .factories import (
    create_agent_tier_policy,
    create_production_policy,
    create_sensitive_data_policy,
    create_sla_policy,
)
from .history import PolicyHistory, PolicyVersion, get_policy_history
from .manager import ControlPlanePolicyManager
from .scheduler import PolicySyncScheduler
from .sync import PolicyStoreSync, _apply_monkey_patch
from .types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyScope,
    PolicyViolation,
    PolicyViolationError,
    RegionConstraint,
    SLARequirements,
)

# Apply the monkey-patch to add sync_from_compliance_store to ControlPlanePolicyManager
_apply_monkey_patch()

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
    "REDIS_AVAILABLE",
    "PolicySyncScheduler",
    # Versioning and rollback
    "PolicyVersion",
    "PolicyHistory",
    "get_policy_history",
]
