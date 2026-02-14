"""Main AnalyticsDashboardHandler class with routing logic.

This module composes the domain-specific mixins into a single handler class
that routes requests to the appropriate analytics methods.
"""

from __future__ import annotations

from typing import Any

from ._shared import (
    ANALYTICS_STUB_RESPONSES,
    BaseHandler,
    HandlerResult,
    PERM_ANALYTICS_COMPLIANCE,
    PERM_ANALYTICS_COST,
    PERM_ANALYTICS_DELIBERATIONS,
    PERM_ANALYTICS_FLIPS,
    PERM_ANALYTICS_READ,
    PERM_ANALYTICS_TOKENS,
    RBAC_AVAILABLE,
    AuthorizationContext,
    check_permission,
    error_response,
    extract_user_from_request,
    rbac_fail_closed,
    get_string_param,
    json_response,
    logger,
    PermissionDeniedError,
    rate_limit,
    record_rbac_check,
    require_permission,
    strip_version_prefix,
)
from .agents import AgentAnalyticsMixin
from .debates import DebateAnalyticsMixin
from .endpoints import DeliberationAnalyticsMixin
from .usage import UsageAnalyticsMixin


class AnalyticsDashboardHandler(
    DebateAnalyticsMixin,
    AgentAnalyticsMixin,
    UsageAnalyticsMixin,
    DeliberationAnalyticsMixin,
    BaseHandler,
):
    """Handler for analytics dashboard endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def _get_auth_context(self, handler: Any) -> Any | None:
        """Extract authorization context from the request.

        Returns:
            AuthorizationContext if RBAC is available and user is authenticated,
            None otherwise.
        """
        if not RBAC_AVAILABLE or extract_user_from_request is None:
            # SECURITY: In production, deny access when RBAC is unavailable
            # rather than silently skipping permission checks.
            return None

        try:
            # Try to get user info from request
            user_info = extract_user_from_request(handler)
            if not user_info:
                return None

            return AuthorizationContext(
                user_id=user_info.user_id or "anonymous",
                roles={user_info.role} if user_info.role else set(),
                org_id=user_info.org_id,
            )
        except Exception as e:
            logger.debug(f"Could not extract auth context: {e}")
            return None

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        """
        Check if current user has permission. Returns error response if denied.

        Args:
            handler: The HTTP handler
            permission_key: Permission like "analytics:dashboard:read"
            resource_id: Optional resource ID for resource-specific permissions

        Returns:
            None if allowed, error HandlerResult if denied
        """
        if not RBAC_AVAILABLE:
            # SECURITY: Fail closed in production when RBAC module is unavailable
            if rbac_fail_closed():
                return error_response(
                    "Service unavailable: access control module not loaded", 503
                )
            logger.debug(f"RBAC not available, allowing {permission_key}")
            return None

        context = self._get_auth_context(handler)
        if context is None:
            # No auth context means RBAC not configured for this request
            return None

        try:
            decision = check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                logger.warning(
                    f"Permission denied: {permission_key} for user {context.user_id}: {decision.reason}"
                )
                record_rbac_check(permission_key, granted=False)
                return error_response(f"Permission denied: {decision.reason}", 403)
            record_rbac_check(permission_key, granted=True)
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {permission_key} for user {context.user_id}: {e}")
            record_rbac_check(permission_key, granted=False)
            return error_response(f"Permission denied: {str(e)}", 403)

        return None

    ROUTES = [
        "/api/analytics/summary",
        "/api/analytics/trends/findings",
        "/api/analytics/remediation",
        "/api/analytics/agents",
        "/api/analytics/cost",
        "/api/analytics/compliance",
        "/api/analytics/heatmap",
        "/api/analytics/tokens",
        "/api/analytics/tokens/trends",
        "/api/analytics/tokens/providers",
        "/api/analytics/cost/breakdown",
        "/api/analytics/flips/summary",
        "/api/analytics/flips/recent",
        "/api/analytics/flips/consistency",
        "/api/analytics/flips/trends",
        # Deliberation analytics
        "/api/analytics/deliberations",
        "/api/analytics/deliberations/channels",
        "/api/analytics/deliberations/consensus",
        "/api/analytics/deliberations/performance",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        return normalized in self.ROUTES

    @require_permission(PERM_ANALYTICS_READ)
    @rate_limit(requests_per_minute=60)
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests to appropriate methods.

        RBAC permissions are checked at two levels:
        1. Base permission (analytics:dashboard:read) checked by @require_permission decorator
        2. Granular permissions checked within routing for sensitive endpoints
        """
        normalized = strip_version_prefix(path)
        if normalized in ANALYTICS_STUB_RESPONSES:
            user_ctx = self.get_current_user(handler) if handler else None
            workspace_id = get_string_param(query_params, "workspace_id")
            if user_ctx is None or not workspace_id:
                return json_response(ANALYTICS_STUB_RESPONSES[normalized])

        # Basic dashboard analytics - require analytics:dashboard:read (already checked by decorator)
        if normalized == "/api/analytics/summary":
            return self._get_summary(query_params, handler)
        elif normalized == "/api/analytics/trends/findings":
            return self._get_finding_trends(query_params, handler)
        elif normalized == "/api/analytics/remediation":
            return self._get_remediation_metrics(query_params, handler)
        elif normalized == "/api/analytics/agents":
            return self._get_agent_metrics(query_params, handler)
        elif normalized == "/api/analytics/heatmap":
            return self._get_risk_heatmap(query_params, handler)

        # Cost analytics - require analytics:cost:read
        elif normalized == "/api/analytics/cost":
            if error := self._check_permission(handler, PERM_ANALYTICS_COST):
                return error
            return self._get_cost_metrics(query_params, handler)
        elif normalized == "/api/analytics/cost/breakdown":
            if error := self._check_permission(handler, PERM_ANALYTICS_COST):
                return error
            return self._get_cost_breakdown(query_params, handler)

        # Compliance analytics - require analytics:compliance:read
        elif normalized == "/api/analytics/compliance":
            if error := self._check_permission(handler, PERM_ANALYTICS_COMPLIANCE):
                return error
            return self._get_compliance_scorecard(query_params, handler)

        # Token usage analytics - require analytics:tokens:read
        elif normalized == "/api/analytics/tokens":
            if error := self._check_permission(handler, PERM_ANALYTICS_TOKENS):
                return error
            return self._get_token_usage(query_params, handler)
        elif normalized == "/api/analytics/tokens/trends":
            if error := self._check_permission(handler, PERM_ANALYTICS_TOKENS):
                return error
            return self._get_token_trends(query_params, handler)
        elif normalized == "/api/analytics/tokens/providers":
            if error := self._check_permission(handler, PERM_ANALYTICS_TOKENS):
                return error
            return self._get_provider_breakdown(query_params, handler)

        # Flip analytics - require analytics:flips:read
        elif normalized == "/api/analytics/flips/summary":
            if error := self._check_permission(handler, PERM_ANALYTICS_FLIPS):
                return error
            return self._get_flip_summary(query_params, handler)
        elif normalized == "/api/analytics/flips/recent":
            if error := self._check_permission(handler, PERM_ANALYTICS_FLIPS):
                return error
            return self._get_recent_flips(query_params, handler)
        elif normalized == "/api/analytics/flips/consistency":
            if error := self._check_permission(handler, PERM_ANALYTICS_FLIPS):
                return error
            return self._get_agent_consistency(query_params, handler)
        elif normalized == "/api/analytics/flips/trends":
            if error := self._check_permission(handler, PERM_ANALYTICS_FLIPS):
                return error
            return self._get_flip_trends(query_params, handler)

        # Deliberation analytics - require analytics:deliberations:read
        elif normalized == "/api/analytics/deliberations":
            if error := self._check_permission(handler, PERM_ANALYTICS_DELIBERATIONS):
                return error
            return self._get_deliberation_summary(query_params, handler)
        elif normalized == "/api/analytics/deliberations/channels":
            if error := self._check_permission(handler, PERM_ANALYTICS_DELIBERATIONS):
                return error
            return self._get_deliberation_by_channel(query_params, handler)
        elif normalized == "/api/analytics/deliberations/consensus":
            if error := self._check_permission(handler, PERM_ANALYTICS_DELIBERATIONS):
                return error
            return self._get_consensus_rates(query_params, handler)
        elif normalized == "/api/analytics/deliberations/performance":
            if error := self._check_permission(handler, PERM_ANALYTICS_DELIBERATIONS):
                return error
            return self._get_deliberation_performance(query_params, handler)

        return None
