"""
Unified Approval Enforcer for Gateway, Device, and Computer-Use Actions.

Provides a single enforcement path for all sensitive actions that require
human approval, ensuring:
- Consistent policy evaluation via OpenClaw policy engine
- Unified audit trail for all approval decisions
- Bypass prevention with mandatory enforcement checks
- Integration with both device pairing and computer-use workflows

Stage 6 (#177): Security/approval consolidation.

Usage:
    from aragora.security.approval_enforcer import (
        UnifiedApprovalEnforcer,
        ApprovalDecision,
        enforce_action,
    )

    enforcer = UnifiedApprovalEnforcer(policy=my_policy)
    decision = await enforcer.enforce(action_request)
    if decision.approved:
        # proceed with action
        ...
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EnforcementResult(str, Enum):
    """Outcome of an enforcement decision."""

    ALLOWED = "allowed"
    DENIED = "denied"
    PENDING_APPROVAL = "pending_approval"
    BYPASSED_DETECTED = "bypass_detected"


@dataclass
class EnforcementRequest:
    """Unified request for any action requiring enforcement.

    Normalizes gateway, device, and computer-use action requests into
    a single structure for policy evaluation and audit.
    """

    action_type: str
    actor_id: str
    source: str  # "gateway", "device", "computer_use"
    resource: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    workspace_id: str = "default"
    tenant_id: str | None = None
    roles: list[str] = field(default_factory=list)
    approval_id: str | None = None  # Pre-existing approval token


@dataclass
class EnforcementDecision:
    """Result of an enforcement evaluation."""

    id: str
    result: EnforcementResult
    reason: str
    request: EnforcementRequest
    matched_rule: str | None = None
    approval_request_id: str | None = None
    evaluation_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.result == EnforcementResult.ALLOWED

    @property
    def denied(self) -> bool:
        return self.result in (
            EnforcementResult.DENIED,
            EnforcementResult.BYPASSED_DETECTED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "result": self.result.value,
            "reason": self.reason,
            "source": self.request.source,
            "action_type": self.request.action_type,
            "actor_id": self.request.actor_id,
            "matched_rule": self.matched_rule,
            "approval_request_id": self.approval_request_id,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp,
        }


class UnifiedApprovalEnforcer:
    """Single enforcement path for gateway, device, and computer-use actions.

    Evaluates all sensitive actions against the OpenClaw policy engine and
    routes REQUIRE_APPROVAL decisions to the appropriate approval workflow.
    Emits structured audit events for every decision.
    """

    def __init__(
        self,
        policy: Any | None = None,
        approval_workflow: Any | None = None,
        audit_enabled: bool = True,
    ) -> None:
        self._policy = policy
        self._approval_workflow = approval_workflow
        self._audit_enabled = audit_enabled
        self._decision_log: list[EnforcementDecision] = []
        self._max_log_size = 10_000

    async def enforce(self, request: EnforcementRequest) -> EnforcementDecision:
        """Evaluate an action request and return an enforcement decision.

        This is the single entry point for all sensitive actions. It:
        1. Checks for pre-existing approval tokens
        2. Evaluates the action against the OpenClaw policy
        3. Routes REQUIRE_APPROVAL to the approval workflow
        4. Emits audit events for the decision
        5. Returns the enforcement decision

        Args:
            request: Unified enforcement request

        Returns:
            EnforcementDecision with the outcome
        """
        start = time.time()
        decision_id = str(uuid.uuid4())

        # Step 1: Check pre-existing approval token
        if request.approval_id:
            decision = await self._verify_approval_token(decision_id, request, start)
            if decision:
                await self._record_decision(decision)
                return decision

        # Step 2: Evaluate against policy
        decision = await self._evaluate_policy(decision_id, request, start)

        # Step 3: If policy requires approval, route to workflow
        if decision.result == EnforcementResult.PENDING_APPROVAL:
            decision = await self._route_to_approval(decision, request)

        # Step 4: Emit audit event
        await self._record_decision(decision)

        return decision

    async def wait_for_approval(
        self,
        approval_request_id: str,
        timeout: float | None = None,
    ) -> bool:
        """Wait for an approval decision and return True if approved."""
        if not self._approval_workflow:
            return False

        try:
            from aragora.computer_use.approval import ApprovalStatus
        except ImportError:
            return False

        status = await self._approval_workflow.wait_for_decision(
            approval_request_id,
            timeout=timeout,
        )
        return status == ApprovalStatus.APPROVED

    async def verify_not_bypassed(
        self,
        request: EnforcementRequest,
        claimed_approval_id: str | None = None,
    ) -> EnforcementDecision:
        """Verify that an action has not bypassed the approval flow.

        Used as a secondary check to detect bypass attempts where
        code paths might skip the main enforce() call.

        Args:
            request: The action being performed
            claimed_approval_id: Approval ID claimed by the caller

        Returns:
            EnforcementDecision - BYPASSED_DETECTED if no valid approval
        """
        start = time.time()
        decision_id = str(uuid.uuid4())

        # Check if this action requires approval per policy
        requires_approval = await self._action_requires_approval(request)

        if not requires_approval:
            return EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.ALLOWED,
                reason="Action does not require approval per policy",
                request=request,
                evaluation_time_ms=(time.time() - start) * 1000,
            )

        # Verify the claimed approval
        if not claimed_approval_id:
            decision = EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.BYPASSED_DETECTED,
                reason="Sensitive action performed without approval token",
                request=request,
                evaluation_time_ms=(time.time() - start) * 1000,
            )
            await self._record_decision(decision)
            return decision

        # Verify approval is valid
        valid = await self._is_approval_valid(claimed_approval_id)
        if not valid:
            decision = EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.BYPASSED_DETECTED,
                reason=f"Invalid or expired approval token: {claimed_approval_id}",
                request=request,
                approval_request_id=claimed_approval_id,
                evaluation_time_ms=(time.time() - start) * 1000,
            )
            await self._record_decision(decision)
            return decision

        return EnforcementDecision(
            id=decision_id,
            result=EnforcementResult.ALLOWED,
            reason=f"Valid approval: {claimed_approval_id}",
            request=request,
            approval_request_id=claimed_approval_id,
            evaluation_time_ms=(time.time() - start) * 1000,
        )

    def get_recent_decisions(
        self, limit: int = 100, source: str | None = None
    ) -> list[EnforcementDecision]:
        """Get recent enforcement decisions for audit review."""
        decisions = self._decision_log
        if source:
            decisions = [d for d in decisions if d.request.source == source]
        return decisions[-limit:]

    def get_bypass_attempts(self, limit: int = 100) -> list[EnforcementDecision]:
        """Get detected bypass attempts."""
        return [d for d in self._decision_log if d.result == EnforcementResult.BYPASSED_DETECTED][
            -limit:
        ]

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _verify_approval_token(
        self,
        decision_id: str,
        request: EnforcementRequest,
        start: float,
    ) -> EnforcementDecision | None:
        """Verify a pre-existing approval token."""
        valid = await self._is_approval_valid(request.approval_id)
        if valid:
            return EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.ALLOWED,
                reason=f"Pre-approved via {request.approval_id}",
                request=request,
                approval_request_id=request.approval_id,
                evaluation_time_ms=(time.time() - start) * 1000,
            )
        # Invalid token - fall through to policy evaluation
        logger.warning(
            "Invalid approval token %s for %s action by %s",
            request.approval_id,
            request.action_type,
            request.actor_id,
        )
        return None

    async def _evaluate_policy(
        self,
        decision_id: str,
        request: EnforcementRequest,
        start: float,
    ) -> EnforcementDecision:
        """Evaluate request against the OpenClaw policy engine."""
        if request.details.get("force_approval"):
            return EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.PENDING_APPROVAL,
                reason=request.details.get("force_reason", "Approval required by upstream policy"),
                request=request,
                evaluation_time_ms=(time.time() - start) * 1000,
            )
        if not self._policy:
            # No policy configured - allow by default with audit
            return EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.ALLOWED,
                reason="No policy configured; action allowed by default",
                request=request,
                evaluation_time_ms=(time.time() - start) * 1000,
            )

        try:
            from aragora.gateway.openclaw_policy import (
                ActionRequest,
                ActionType,
                PolicyDecision,
            )

            # Map enforcement request to OpenClaw ActionRequest
            action_type_map = {
                "shell": ActionType.SHELL,
                "file_read": ActionType.FILE_READ,
                "file_write": ActionType.FILE_WRITE,
                "file_delete": ActionType.FILE_DELETE,
                "browser": ActionType.BROWSER,
                "api": ActionType.API,
                "screenshot": ActionType.SCREENSHOT,
                "keyboard": ActionType.KEYBOARD,
                "mouse": ActionType.MOUSE,
            }

            action_type = action_type_map.get(request.action_type)
            if not action_type:
                return EnforcementDecision(
                    id=decision_id,
                    result=EnforcementResult.ALLOWED,
                    reason=f"Unknown action type '{request.action_type}'; not policy-controlled",
                    request=request,
                    evaluation_time_ms=(time.time() - start) * 1000,
                )

            policy_request = ActionRequest(
                action_type=action_type,
                user_id=request.actor_id,
                session_id=request.session_id,
                workspace_id=request.workspace_id,
                path=request.details.get("path"),
                command=request.details.get("command"),
                url=request.details.get("url"),
                roles=request.roles,
                tenant_id=request.tenant_id,
            )

            result = self._policy.evaluate(policy_request)
            eval_time = (time.time() - start) * 1000

            if result.decision == PolicyDecision.ALLOW:
                return EnforcementDecision(
                    id=decision_id,
                    result=EnforcementResult.ALLOWED,
                    reason=result.reason,
                    request=request,
                    matched_rule=result.matched_rule.name if result.matched_rule else None,
                    evaluation_time_ms=eval_time,
                )
            elif result.decision == PolicyDecision.DENY:
                return EnforcementDecision(
                    id=decision_id,
                    result=EnforcementResult.DENIED,
                    reason=result.reason,
                    request=request,
                    matched_rule=result.matched_rule.name if result.matched_rule else None,
                    evaluation_time_ms=eval_time,
                )
            else:  # REQUIRE_APPROVAL
                return EnforcementDecision(
                    id=decision_id,
                    result=EnforcementResult.PENDING_APPROVAL,
                    reason=result.reason,
                    request=request,
                    matched_rule=result.matched_rule.name if result.matched_rule else None,
                    evaluation_time_ms=eval_time,
                )

        except ImportError:
            logger.warning("OpenClaw policy module not available")
            return EnforcementDecision(
                id=decision_id,
                result=EnforcementResult.ALLOWED,
                reason="Policy module unavailable; action allowed",
                request=request,
                evaluation_time_ms=(time.time() - start) * 1000,
            )

    async def _route_to_approval(
        self,
        decision: EnforcementDecision,
        request: EnforcementRequest,
    ) -> EnforcementDecision:
        """Route a REQUIRE_APPROVAL decision to the approval workflow."""
        if not self._approval_workflow:
            # No workflow configured - keep as pending
            return decision

        try:
            from aragora.computer_use.approval import (
                ApprovalCategory,
                ApprovalContext,
                ApprovalPriority,
            )

            # Map source to category
            category_map = {
                "gateway": ApprovalCategory.SYSTEM_MODIFICATION,
                "device": ApprovalCategory.EXTERNAL_SYSTEM,
                "computer_use": ApprovalCategory.DESTRUCTIVE_ACTION,
            }
            category = category_map.get(request.source, ApprovalCategory.UNKNOWN)

            context = ApprovalContext(
                task_id=request.session_id or str(uuid.uuid4()),
                action_type=request.action_type,
                action_details=request.details,
                category=category,
                reason=decision.reason,
                risk_level=request.details.get("risk_level", "medium"),
                user_id=request.actor_id,
                tenant_id=request.tenant_id,
            )

            approval_request = await self._approval_workflow.request_approval(
                context=context,
                priority=ApprovalPriority.HIGH,
            )

            decision.approval_request_id = approval_request.id
            decision.metadata["approval_context"] = {
                "category": category.value,
                "priority": "high",
            }

            return decision

        except ImportError:
            logger.warning("Computer-use approval module not available")
            return decision

    async def _action_requires_approval(self, request: EnforcementRequest) -> bool:
        """Check if an action requires approval per policy."""
        if not self._policy:
            return False

        try:
            from aragora.gateway.openclaw_policy import (
                ActionRequest,
                ActionType,
                PolicyDecision,
            )

            action_type_map = {
                "shell": ActionType.SHELL,
                "file_read": ActionType.FILE_READ,
                "file_write": ActionType.FILE_WRITE,
                "file_delete": ActionType.FILE_DELETE,
                "browser": ActionType.BROWSER,
                "api": ActionType.API,
                "screenshot": ActionType.SCREENSHOT,
                "keyboard": ActionType.KEYBOARD,
                "mouse": ActionType.MOUSE,
            }

            action_type = action_type_map.get(request.action_type)
            if not action_type:
                return False

            policy_request = ActionRequest(
                action_type=action_type,
                user_id=request.actor_id,
                session_id=request.session_id,
                workspace_id=request.workspace_id,
                path=request.details.get("path"),
                command=request.details.get("command"),
                url=request.details.get("url"),
                roles=request.roles,
                tenant_id=request.tenant_id,
            )

            result = self._policy.evaluate(policy_request)
            return result.decision == PolicyDecision.REQUIRE_APPROVAL

        except ImportError:
            return False

    async def _is_approval_valid(self, approval_id: str | None) -> bool:
        """Check if an approval token is valid and not expired."""
        if not approval_id:
            return False

        if not self._approval_workflow:
            return False

        try:
            request = await self._approval_workflow.get_request(approval_id)
            if not request:
                return False

            from aragora.computer_use.approval import ApprovalStatus

            return request.status == ApprovalStatus.APPROVED and not request.is_expired()
        except (ImportError, AttributeError):
            return False

    async def _record_decision(self, decision: EnforcementDecision) -> None:
        """Record decision in log and emit audit event."""
        # Append to in-memory log
        self._decision_log.append(decision)
        if len(self._decision_log) > self._max_log_size:
            self._decision_log = self._decision_log[-self._max_log_size :]

        # Emit structured audit event
        if self._audit_enabled:
            await self._emit_audit_event(decision)

    async def _emit_audit_event(self, decision: EnforcementDecision) -> None:
        """Emit a structured audit event for the enforcement decision."""
        try:
            from aragora.observability.security_audit import (
                audit_rbac_decision,
            )

            granted = decision.result == EnforcementResult.ALLOWED

            await audit_rbac_decision(
                user_id=decision.request.actor_id,
                permission=f"enforce:{decision.request.source}:{decision.request.action_type}",
                granted=granted,
                resource_type=decision.request.source,
                resource_id=decision.request.resource or decision.request.action_type,
                workspace_id=decision.request.workspace_id,
                enforcement_id=decision.id,
                result=decision.result.value,
                reason=decision.reason,
                matched_rule=decision.matched_rule,
                approval_request_id=decision.approval_request_id,
            )
        except (ImportError, TypeError, RuntimeError) as e:
            logger.debug("Audit event emission skipped: %s", e)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_enforcer: UnifiedApprovalEnforcer | None = None


def get_approval_enforcer() -> UnifiedApprovalEnforcer:
    """Get or create the default unified approval enforcer."""
    global _default_enforcer
    if _default_enforcer is None:
        _default_enforcer = UnifiedApprovalEnforcer()
    return _default_enforcer


def set_approval_enforcer(enforcer: UnifiedApprovalEnforcer) -> None:
    """Set the default unified approval enforcer."""
    global _default_enforcer
    _default_enforcer = enforcer


async def enforce_action(request: EnforcementRequest) -> EnforcementDecision:
    """Convenience function to enforce an action via the default enforcer."""
    return await get_approval_enforcer().enforce(request)


__all__ = [
    "EnforcementResult",
    "EnforcementRequest",
    "EnforcementDecision",
    "UnifiedApprovalEnforcer",
    "enforce_action",
    "get_approval_enforcer",
    "set_approval_enforcer",
]
