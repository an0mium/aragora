"""
Policy Engine - Tool access control and approvals.

Enforces policies for agent actions including tool access,
file/network restrictions, and approval workflows.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from .models import (
    ApprovalRequest,
    ApprovalResult,
    Policy,
    PolicyContext,
    PolicyDecision,
    PolicyEffect,
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    """
    Evaluates and enforces policies for agent actions.

    Features:
    - Pattern-based action matching
    - Policy priority and inheritance
    - Explicit approval workflows
    - Audit logging of all decisions
    """

    def __init__(
        self,
        default_effect: PolicyEffect = PolicyEffect.DENY,
        approval_timeout_seconds: float = 300.0,
    ) -> None:
        self._default_effect = default_effect
        self._approval_timeout = approval_timeout_seconds
        self._policies: dict[str, Policy] = {}
        self._pending_approvals: dict[str, ApprovalRequest] = {}
        self._approval_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._decisions_allowed = 0
        self._decisions_denied = 0
        self._decisions_approval_required = 0

    async def add_policy(self, policy: Policy) -> None:
        """Add or update a policy."""
        async with self._lock:
            self._policies[policy.id] = policy
            logger.debug("Added policy %s (%s rules)", policy.id, len(policy.rules))

    async def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        async with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    async def get_policy(self, policy_id: str) -> Policy | None:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    async def list_policies(self) -> list[Policy]:
        """List all policies."""
        return list(self._policies.values())

    async def check(
        self,
        action: str,
        context: PolicyContext,
    ) -> PolicyDecision:
        """
        Check if an action is allowed by policy.

        Args:
            action: Action to check (e.g., "tool.shell.execute")
            context: Context for policy evaluation

        Returns:
            Policy decision
        """
        async with self._lock:
            sorted_policies = sorted(
                [p for p in self._policies.values() if p.enabled],
                key=lambda p: -p.priority,
            )

            for policy in sorted_policies:
                for rule in policy.rules:
                    if self._matches(action, rule.action_pattern, context, rule.conditions):
                        decision = PolicyDecision(
                            allowed=rule.effect == PolicyEffect.ALLOW,
                            effect=rule.effect,
                            matching_rule=rule,
                            matching_policy=policy.id,
                            reason=rule.description or f"Matched rule in policy {policy.id}",
                            requires_approval=rule.effect == PolicyEffect.REQUIRE_APPROVAL,
                        )
                        self._record_decision(decision)
                        return decision

            decision = PolicyDecision(
                allowed=self._default_effect == PolicyEffect.ALLOW,
                effect=self._default_effect,
                reason="No matching policy rule, using default",
            )
            self._record_decision(decision)
            return decision

    def _matches(
        self,
        action: str,
        pattern: str,
        context: PolicyContext,
        conditions: dict[str, Any],
    ) -> bool:
        """Check if an action matches a rule pattern and conditions."""
        if not fnmatch.fnmatch(action, pattern):
            return False

        for key, expected in conditions.items():
            actual = getattr(context, key, None) or context.attributes.get(key)
            if actual is None:
                return False
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    def _record_decision(self, decision: PolicyDecision) -> None:
        """Record a policy decision for metrics."""
        if decision.requires_approval:
            self._decisions_approval_required += 1
        elif decision.allowed:
            self._decisions_allowed += 1
        else:
            self._decisions_denied += 1

    async def require_approval(
        self,
        action: str,
        context: PolicyContext,
        approvers: list[str],
        timeout_seconds: float | None = None,
    ) -> ApprovalResult:
        """
        Request approval for an action.

        Args:
            action: Action requiring approval
            context: Action context
            approvers: List of user IDs who can approve
            timeout_seconds: Custom timeout

        Returns:
            Approval result
        """
        timeout = timeout_seconds or self._approval_timeout
        request_id = str(uuid.uuid4())

        request = ApprovalRequest(
            id=request_id,
            action=action,
            context=context,
            requested_at=datetime.now(timezone.utc),
            requested_by=context.user_id or context.agent_id,
            approvers=approvers,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=timeout),
        )

        event = asyncio.Event()

        async with self._lock:
            self._pending_approvals[request_id] = request
            self._approval_events[request_id] = event

        logger.info("Approval requested: %s for action %s", request_id, action)

        start = datetime.now(timezone.utc)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                request.status = "expired"

        waited = (datetime.now(timezone.utc) - start).total_seconds()

        async with self._lock:
            self._pending_approvals.pop(request_id, None)
            self._approval_events.pop(request_id, None)

        return ApprovalResult(
            approved=request.status == "approved",
            request=request,
            waited_seconds=waited,
        )

    async def approve(
        self,
        request_id: str,
        approver_id: str,
        reason: str = "",
    ) -> bool:
        """Approve a pending request."""
        async with self._lock:
            request = self._pending_approvals.get(request_id)
            if not request:
                return False

            if approver_id not in request.approvers:
                logger.warning("Unauthorized approval attempt: %s for %s", approver_id, request_id)
                return False

            request.status = "approved"
            request.approved_by = approver_id
            request.approved_at = datetime.now(timezone.utc)
            request.reason = reason

            event = self._approval_events.get(request_id)
            if event:
                event.set()

            logger.info("Approved: %s by %s", request_id, approver_id)
            return True

    async def deny(
        self,
        request_id: str,
        denier_id: str,
        reason: str = "",
    ) -> bool:
        """Deny a pending request."""
        async with self._lock:
            request = self._pending_approvals.get(request_id)
            if not request:
                return False

            request.status = "denied"
            request.reason = reason

            event = self._approval_events.get(request_id)
            if event:
                event.set()

            logger.info("Denied: %s by %s", request_id, denier_id)
            return True

    async def list_pending_approvals(
        self,
        approver_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """List pending approval requests."""
        async with self._lock:
            requests = list(self._pending_approvals.values())
            if approver_id:
                requests = [r for r in requests if approver_id in r.approvers]
            return requests

    async def get_stats(self) -> dict[str, Any]:
        """Get policy engine statistics."""
        async with self._lock:
            return {
                "policies": len(self._policies),
                "decisions_allowed": self._decisions_allowed,
                "decisions_denied": self._decisions_denied,
                "decisions_approval_required": self._decisions_approval_required,
                "pending_approvals": len(self._pending_approvals),
            }
