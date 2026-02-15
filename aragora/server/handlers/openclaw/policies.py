"""
Policy enforcement for OpenClaw Gateway.

Stability: STABLE

Contains:
- Policy rule management handlers (mixin class)
- Approval workflow handlers
- Admin operations (health, metrics, audit, stats)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openclaw._base import OpenClawMixinBase
from aragora.server.handlers.openclaw.store import _get_store
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.validation.query_params import safe_query_int

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Policy and Admin Handler Mixin
# =============================================================================


class PolicyHandlerMixin(OpenClawMixinBase):
    """Mixin class providing policy and admin handler methods.

    This mixin is intended to be used with OpenClawGatewayHandler.
    It requires the following methods from the parent class:
    - _get_user_id(handler) -> str
    - _get_tenant_id(handler) -> str | None
    - get_current_user(handler) -> User | None
    """

    # =========================================================================
    # Policy Rule Handlers
    # =========================================================================

    @require_permission("gateway:policy.read")
    @rate_limit(requests_per_minute=60, limiter_name="openclaw_gateway_get_policy")
    def _handle_get_policy_rules(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """Get active policy rules."""
        try:
            store = _get_store()
            rules = store.get_policy_rules() if hasattr(store, "get_policy_rules") else []

            return json_response(
                {
                    "rules": [r.to_dict() if hasattr(r, "to_dict") else r for r in rules],
                    "total": len(rules),
                }
            )
        except (KeyError, ValueError, OSError) as e:
            logger.error("Error getting policy rules: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:policy.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_add_policy")
    def _handle_add_policy_rule(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Add a policy rule."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            name = body.get("name")
            if not name:
                return error_response("name is required", 400)

            action_types = body.get("action_types", [])
            decision = body.get("decision", "deny")
            priority = body.get("priority", 0)
            description = body.get("description", "")
            enabled = body.get("enabled", True)
            config = body.get("config", {})

            if hasattr(store, "add_policy_rule"):
                rule = store.add_policy_rule(
                    name=name,
                    action_types=action_types,
                    decision=decision,
                    priority=priority,
                    description=description,
                    enabled=enabled,
                    config=config,
                )
            else:
                rule = {
                    "name": name,
                    "action_types": action_types,
                    "decision": decision,
                    "priority": priority,
                    "description": description,
                    "enabled": enabled,
                    "config": config,
                }

            # Audit
            store.add_audit_entry(
                action="policy.rule.add",
                actor_id=user_id,
                resource_type="policy_rule",
                resource_id=name,
                result="success",
                details={"decision": decision, "action_types": action_types},
            )

            logger.info("Added policy rule %s", name)
            result = rule.to_dict() if hasattr(rule, "to_dict") else rule
            return json_response(result, status=201)

        except (KeyError, ValueError, TypeError, OSError) as e:
            logger.error("Error adding policy rule: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:policy.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_remove_policy")
    def _handle_remove_policy_rule(self, rule_name: str, handler: Any) -> HandlerResult:
        """Remove a policy rule."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            if hasattr(store, "remove_policy_rule"):
                removed = store.remove_policy_rule(rule_name)
            else:
                removed = True

            # Audit
            store.add_audit_entry(
                action="policy.rule.remove",
                actor_id=user_id,
                resource_type="policy_rule",
                resource_id=rule_name,
                result="success",
            )

            logger.info("Removed policy rule %s", rule_name)
            return json_response({"success": removed, "name": rule_name})

        except Exception as e:
            logger.error("Error removing policy rule %s: %s", rule_name, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # Approval Handlers
    # =========================================================================

    @require_permission("gateway:approvals.read")
    @rate_limit(requests_per_minute=60, limiter_name="openclaw_gateway_list_approvals")
    def _handle_list_approvals(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List pending approval requests."""
        try:
            store = _get_store()
            tenant_id = self._get_tenant_id(handler)
            limit = safe_query_int(query_params, "limit", default=50, max_val=500)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            if hasattr(store, "list_approvals"):
                approvals, total = store.list_approvals(
                    tenant_id=tenant_id,
                    limit=limit,
                    offset=offset,
                )
            else:
                approvals, total = [], 0

            return json_response(
                {
                    "approvals": [a.to_dict() if hasattr(a, "to_dict") else a for a in approvals],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as e:
            logger.error("Error listing approvals: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:approvals.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_approve")
    def _handle_approve_action(
        self, approval_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Approve a pending action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            # Security: approver_id MUST be the authenticated user to prevent impersonation
            # The approver_id in body is ignored - only the authenticated user can approve
            approver_id = user_id
            reason = body.get("reason", "")

            if hasattr(store, "approve_action"):
                result = store.approve_action(
                    approval_id=approval_id,
                    approver_id=approver_id,
                    reason=reason,
                )
                success = result if isinstance(result, bool) else True
            else:
                success = True

            # Audit
            store.add_audit_entry(
                action="approval.approve",
                actor_id=user_id,
                resource_type="approval",
                resource_id=approval_id,
                result="success",
                details={"approver_id": approver_id, "reason": reason},
            )

            logger.info("Approved action %s by %s", approval_id, approver_id)
            return json_response({"success": success, "approval_id": approval_id})

        except Exception as e:
            logger.error("Error approving action %s: %s", approval_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:approvals.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_deny")
    def _handle_deny_action(
        self, approval_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Deny a pending action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            # Security: approver_id MUST be the authenticated user to prevent impersonation
            approver_id = user_id
            reason = body.get("reason", "")

            if hasattr(store, "deny_action"):
                result = store.deny_action(
                    approval_id=approval_id,
                    approver_id=approver_id,
                    reason=reason,
                )
                success = result if isinstance(result, bool) else True
            else:
                success = True

            # Audit
            store.add_audit_entry(
                action="approval.deny",
                actor_id=user_id,
                resource_type="approval",
                resource_id=approval_id,
                result="success",
                details={"approver_id": approver_id, "reason": reason},
            )

            logger.info("Denied action %s by %s", approval_id, approver_id)
            return json_response({"success": success, "approval_id": approval_id})

        except Exception as e:
            logger.error("Error denying action %s: %s", approval_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # Admin Handlers (Health, Metrics, Audit, Stats)
    # =========================================================================

    def _handle_health(self, handler: Any) -> HandlerResult:
        """Get gateway health status.

        Security: Returns only status and timestamp to prevent information
        leakage about internal session/action counts. Detailed metrics
        are available via the authenticated /metrics endpoint.
        """
        try:
            store = _get_store()
            metrics = store.get_metrics()

            # Basic health check
            healthy = True
            status = "healthy"

            # Check for any critical issues
            if metrics["actions"]["running"] > 100:
                status = "degraded"
            if metrics["actions"]["pending"] > 500:
                healthy = False
                status = "unhealthy"

            return json_response(
                {
                    "status": status,
                    "healthy": healthy,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            logger.error("Error getting health: %s", e)
            return json_response(
                {
                    "status": "error",
                    "healthy": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=503,
            )

    @require_permission("gateway:metrics.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_metrics")
    def _handle_metrics(self, handler: Any) -> HandlerResult:
        """Get gateway metrics."""
        try:
            store = _get_store()
            metrics = store.get_metrics()
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

            return json_response(metrics)
        except Exception as e:
            logger.error("Error getting metrics: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:audit.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_audit")
    def _handle_audit(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """Get audit log entries."""
        try:
            store = _get_store()

            # Parse query parameters
            action_filter = query_params.get("action")
            actor_filter = query_params.get("actor_id")
            resource_type = query_params.get("resource_type")
            limit = safe_query_int(query_params, "limit", default=100, max_val=1000)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            entries, total = store.get_audit_log(
                action=action_filter,
                actor_id=actor_filter,
                resource_type=resource_type,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "entries": [e.to_dict() for e in entries],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as e:
            logger.error("Error getting audit log: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:metrics.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_stats")
    def _handle_stats(self, handler: Any) -> HandlerResult:
        """Get proxy statistics."""
        try:
            store = _get_store()
            metrics = store.get_metrics()

            return json_response(
                {
                    "active_sessions": metrics.get("sessions", {}).get("active", 0),
                    "actions_allowed": metrics.get("actions", {}).get("completed", 0),
                    "actions_denied": metrics.get("actions", {}).get("failed", 0),
                    "pending_approvals": metrics.get("actions", {}).get("pending", 0),
                    "policy_rules": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)


__all__ = [
    "PolicyHandlerMixin",
]
