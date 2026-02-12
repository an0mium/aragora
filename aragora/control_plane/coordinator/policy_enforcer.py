"""
Policy Enforcer for Control Plane Coordinator.

Handles policy evaluation, enforcement, compliance checking,
and policy sync from the compliance store.
"""

from __future__ import annotations

import asyncio
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

from aragora.observability import get_logger

if TYPE_CHECKING:
    from aragora.control_plane.policy import (
        ControlPlanePolicyManager,
        PolicyViolation,
    )

# Policy imports (optional - graceful fallback if not available)
ControlPlanePolicyManagerType: Any = None
try:
    from aragora.control_plane.policy import (
        ControlPlanePolicyManager as ControlPlanePolicyManagerType,
    )

    HAS_POLICY = True
except ImportError:
    ControlPlanePolicyManagerType = Any
    HAS_POLICY = False

logger = get_logger(__name__)


class PolicyEnforcer:
    """
    Handles policy evaluation and enforcement for the control plane.

    Responsibilities:
    - Policy violation handling and notifications
    - SLA compliance evaluation
    - Policy sync from compliance store
    - Audit logging for policy decisions
    """

    def __init__(
        self,
        policy_manager: ControlPlanePolicyManager | None = None,
        violation_callback: Callable[[PolicyViolation], None] | None = None,
        enable_policy_sync: bool = True,
        policy_sync_workspace: str | None = None,
    ):
        """
        Initialize the policy enforcer.

        Args:
            policy_manager: Optional pre-configured ControlPlanePolicyManager
            violation_callback: Callback for handling policy violations
            enable_policy_sync: Whether to enable policy sync from compliance store
            policy_sync_workspace: Workspace ID for policy sync
        """
        self._enable_policy_sync = enable_policy_sync
        self._policy_sync_workspace = policy_sync_workspace
        self._violation_callback = violation_callback

        self._policy_manager: ControlPlanePolicyManager | None = None
        if policy_manager:
            self._policy_manager = policy_manager
        elif HAS_POLICY:
            self._policy_manager = ControlPlanePolicyManagerType(
                violation_callback=self._handle_policy_violation
            )

    @property
    def policy_manager(self) -> ControlPlanePolicyManager | None:
        """Get the Policy Manager if configured."""
        return self._policy_manager

    def set_policy_manager(self, manager: ControlPlanePolicyManager) -> None:
        """Set the Policy Manager."""
        self._policy_manager = manager

    def _schedule_async(self, coro: Any) -> None:
        """Schedule a coroutine regardless of sync/async context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                from aragora.utils.async_utils import run_async

                run_async(coro)
            except Exception as e:
                logger.debug("async_schedule_failed", error=str(e))
        else:
            loop.create_task(coro)

    def _handle_policy_violation(self, violation: PolicyViolation) -> None:
        """Handle policy violations with audit logging and notifications."""
        # Call the external callback if provided
        if self._violation_callback:
            self._violation_callback(violation)

        try:
            enforcement = getattr(violation.enforcement_level, "value", "hard")
            decision = "warn" if enforcement == "warn" else "deny"
        except (AttributeError, TypeError):
            decision = "deny"

        async def _log_violation() -> None:
            try:
                from aragora.control_plane.audit import log_policy_decision
            except ImportError:
                return

            await log_policy_decision(
                policy_id=violation.policy_id,
                decision=decision,
                task_type=violation.task_type or "unknown",
                reason=violation.description,
                workspace_id=violation.workspace_id,
                task_id=violation.task_id,
                agent_id=violation.agent_id,
                violations=[violation.violation_type],
                metadata={
                    "policy_name": violation.policy_name,
                    "region": violation.region,
                },
            )

        self._schedule_async(_log_violation())

        try:
            from aragora.control_plane.notifications import get_default_notification_dispatcher
            from aragora.control_plane.channels import (
                NotificationEventType,
                NotificationPriority,
            )

            dispatcher = get_default_notification_dispatcher()
            if dispatcher:
                title = f"Policy {'Warning' if decision == 'warn' else 'Violation'}: {violation.policy_name}"
                body = (
                    f"Policy `{violation.policy_id}` blocked task `{(violation.task_id or '')[:8]}...` "
                    f"for agent `{violation.agent_id or 'unknown'}`.\n\n"
                    f"Reason: {violation.description}"
                )
                self._schedule_async(
                    dispatcher.dispatch(
                        event_type=NotificationEventType.POLICY_VIOLATION,
                        title=title,
                        body=body,
                        priority=NotificationPriority.HIGH,
                        metadata=violation.to_dict(),
                        workspace_id=violation.workspace_id,
                    )
                )
        except Exception as e:
            logger.debug("policy_notification_failed", error=str(e))

    def _should_sync_policies_from_store(self) -> bool:
        """Determine whether to sync policies from the compliance store.

        Policy sync is controlled by the following environment variables:
        - ARAGORA_POLICY_SYNC_ON_STARTUP / CP_ENABLE_POLICY_SYNC: Master toggle (default: true)
        - ARAGORA_CONTROL_PLANE_POLICY_SOURCE: Override source selection

        In production/staging environments, policy sync from the compliance
        store is enabled by default. In development, it depends on the
        configuration.

        Returns:
            True if policies should be synced from the compliance store
        """
        # Check if policy sync is disabled via config
        if not self._enable_policy_sync:
            return False

        source = os.environ.get("ARAGORA_CONTROL_PLANE_POLICY_SOURCE", "").lower()
        if source in ("inprocess", "local", "memory"):
            return False
        if source in ("compliance", "store", "policy_store"):
            return True

        env = os.environ.get("ARAGORA_ENV", "development").lower()
        return env in ("production", "prod", "staging", "stage")

    def sync_policies_from_store(self) -> int:
        """Sync policies from compliance store on coordinator startup.

        This method is called during coordinator initialization to load
        policies from the compliance store. It handles errors gracefully
        to avoid failing the entire startup sequence.

        The sync behavior is controlled by:
        - ARAGORA_POLICY_SYNC_ON_STARTUP (default: true)
        - ARAGORA_CONTROL_PLANE_POLICY_SOURCE (optional override)
        - ARAGORA_ENV (auto-enables in production/staging)

        Returns:
            Number of policies loaded from the compliance store.
            Returns 0 if sync is disabled or fails.
        """
        # Check if policy manager exists
        if not self._policy_manager:
            logger.debug(
                "policy_sync_skipped",
                reason="no_policy_manager",
            )
            return 0

        if not HAS_POLICY:
            logger.debug(
                "policy_sync_skipped",
                reason="policy_module_not_available",
            )
            return 0

        # Check if sync should be performed
        if not self._should_sync_policies_from_store():
            logger.debug(
                "policy_sync_skipped",
                reason="sync_not_enabled",
                enable_policy_sync=self._enable_policy_sync,
                env=os.environ.get("ARAGORA_ENV", "development"),
            )
            return 0

        try:
            loaded = self._policy_manager.sync_from_compliance_store(
                workspace_id=self._policy_sync_workspace,
            )

            if loaded > 0:
                logger.info(
                    "policy_sync_complete",
                    policies_loaded=loaded,
                    workspace=self._policy_sync_workspace,
                    source="compliance_store",
                )
            else:
                logger.debug(
                    "policy_sync_complete",
                    policies_loaded=0,
                    workspace=self._policy_sync_workspace,
                    note="No policies found in compliance store",
                )

            return loaded

        except ImportError as e:
            # Compliance store module not available - this is expected in some deployments
            logger.debug(
                "policy_sync_skipped",
                reason="compliance_store_not_available",
                error=str(e),
            )
            return 0

        except Exception as e:
            # Log warning but don't fail startup - policy sync is non-critical
            logger.warning(
                "policy_sync_failed",
                error=str(e),
                error_type=type(e).__name__,
                workspace=self._policy_sync_workspace,
                note="Startup will continue without synced policies",
            )
            return 0

    def evaluate_sla_compliance(
        self,
        policy_id: str,
        execution_seconds: float,
        queue_seconds: float | None,
        task_id: str,
        task_type: str,
        agent_id: str | None,
        workspace: str | None,
    ) -> Any:
        """
        Evaluate SLA compliance for a completed task.

        Args:
            policy_id: Policy ID to check against
            execution_seconds: Task execution time
            queue_seconds: Time spent in queue
            task_id: Task identifier
            task_type: Type of task
            agent_id: Agent that executed the task
            workspace: Workspace identifier

        Returns:
            SLA evaluation result from policy manager
        """
        if not self._policy_manager or not HAS_POLICY:
            # Return a mock result that allows the operation
            class AllowedResult:
                allowed = True
                reason = "No policy manager configured"
                enforcement_level = None

            return AllowedResult()

        return self._policy_manager.evaluate_sla_compliance(
            policy_id=policy_id,
            execution_seconds=execution_seconds,
            queue_seconds=queue_seconds,
            task_id=task_id,
            task_type=task_type,
            agent_id=agent_id,
            workspace=workspace,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get policy manager metrics."""
        if self._policy_manager and HAS_POLICY:
            return self._policy_manager.get_metrics()
        return {}
