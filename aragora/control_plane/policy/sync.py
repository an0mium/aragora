"""
Control Plane Policy Store Sync.

Bridges compliance policies to control plane.
"""

from __future__ import annotations

from typing import Any, Protocol

from aragora.observability import get_logger
from aragora.exceptions import REDIS_CONNECTION_ERRORS

from .types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyScope,
    RegionConstraint,
    SLARequirements,
)

logger = get_logger(__name__)


class _PolicyManagerProtocol(Protocol):
    """Protocol defining the policy manager interface needed by PolicyStoreSync."""

    _store_sync: PolicyStoreSync

    def add_policy(self, policy: ControlPlanePolicy) -> None: ...

    def remove_policy(self, policy_id: str) -> bool: ...

    def _extract_control_plane_policy(self, policy: Any) -> ControlPlanePolicy | None: ...


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

    def __init__(self, policy_manager: _PolicyManagerProtocol):
        self._policy_manager = policy_manager
        self._synced_policy_ids: set[str] = set()

    def sync_from_store(
        self,
        workspace_id: str | None = None,
        enabled_only: bool = True,
        store: Any | None = None,
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
        except REDIS_CONNECTION_ERRORS as e:
            logger.warning("Policy store sync failed: %s", e)
            return 0

    def _sync_policy(self, compliance_policy: Any) -> bool:
        """Convert and sync a single compliance policy."""
        # Explicit control plane payloads take priority
        if hasattr(self._policy_manager, "_extract_control_plane_policy"):
            control_policy = self._policy_manager._extract_control_plane_policy(compliance_policy)
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
            logger.debug("Failed to convert policy %s: %s", compliance_policy.id, e)

        return False

    def _convert_data_residency_policy(self, policy: Any) -> ControlPlanePolicy | None:
        """Convert data residency compliance policy to region constraint."""
        # Extract allowed regions from policy rules
        allowed_regions: list[str] = []
        blocked_regions: list[str] = []

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

    def _convert_agent_restriction_policy(self, policy: Any) -> ControlPlanePolicy | None:
        """Convert agent restriction compliance policy."""
        allowlist: list[str] = []
        blocklist: list[str] = []
        task_types: list[str] = []

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

    def _convert_sla_policy(self, policy: Any) -> ControlPlanePolicy | None:
        """Convert SLA compliance policy."""
        max_execution = None
        max_queue = None
        min_agents = None
        task_types: list[str] = []

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

    def _convert_task_restriction_policy(self, policy: Any) -> ControlPlanePolicy | None:
        """Convert task restriction compliance policy."""
        task_types: list[str] = []
        capabilities: list[str] = []

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

    def _convert_generic_policy(self, policy: Any) -> ControlPlanePolicy | None:
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
    self: _PolicyManagerProtocol,
    workspace_id: str | None = None,
    enabled_only: bool = True,
    store: Any | None = None,
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
        object.__setattr__(self, "_store_sync", PolicyStoreSync(self))
    return self._store_sync.sync_from_store(
        workspace_id=workspace_id,
        enabled_only=enabled_only,
        store=store,
        replace=replace,
    )


def _apply_monkey_patch() -> None:
    """Apply the monkey-patch to add sync_from_compliance_store to ControlPlanePolicyManager."""
    from .manager import ControlPlanePolicyManager

    # Monkey-patch the method onto the class using setattr to avoid type errors
    setattr(ControlPlanePolicyManager, "sync_from_compliance_store", _sync_from_compliance_store)
