"""
Analytics Dashboard Metrics endpoint handlers.

Provides REST APIs for analytics dashboard showing debate metrics and agent performance:

Debate Analytics:
- GET /api/analytics/debates/overview - Total debates, consensus rate, avg rounds
- GET /api/analytics/debates/trends - Debates over time (daily/weekly/monthly)
- GET /api/analytics/debates/topics - Topic distribution
- GET /api/analytics/debates/outcomes - Win/loss/draw distribution

Agent Performance:
- GET /api/analytics/agents/leaderboard - ELO rankings with win rates
- GET /api/analytics/agents/{agent_id}/performance - Individual agent stats
- GET /api/analytics/agents/comparison - Compare multiple agents
- GET /api/analytics/agents/trends - Agent performance over time

Usage Analytics:
- GET /api/analytics/usage/tokens - Token consumption trends
- GET /api/analytics/usage/costs - Cost breakdown by provider/model
- GET /api/analytics/usage/active_users - Active user counts
"""

from __future__ import annotations

import logging
import re
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.rbac.decorators import require_permission  # noqa: F401

try:
    from aragora.rbac.checker import check_permission  # noqa: F401

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    HandlerResult,
    error_response,
)
from .secure import ForbiddenError, SecureHandler, UnauthorizedError
from .utils.rate_limit import RateLimiter, get_client_ip

# Re-export from submodules for backward compatibility
from ._analytics_metrics_common import (  # noqa: F401
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _group_by_time,
    _parse_time_range,
)
from ._analytics_metrics_agents import AgentAnalyticsMixin  # noqa: F401
from ._analytics_metrics_debates import DebateAnalyticsMixin  # noqa: F401
from ._analytics_metrics_usage import UsageAnalyticsMixin  # noqa: F401

logger = logging.getLogger(__name__)

# Permission required for analytics metrics access
ANALYTICS_METRICS_PERMISSION = "analytics:read"

# Rate limiter for analytics metrics endpoints (60 requests per minute)
_analytics_metrics_limiter = RateLimiter(requests_per_minute=60)


class AnalyticsMetricsHandler(
    DebateAnalyticsMixin,
    AgentAnalyticsMixin,
    UsageAnalyticsMixin,
    SecureHandler,
):
    """Handler for analytics metrics dashboard endpoints.

    Requires authentication and analytics:read permission (RBAC).
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def _validate_org_access(
        self,
        auth_context: Any,
        requested_org_id: str | None,
    ) -> tuple[str | None, HandlerResult | None]:
        """Validate user has access to the requested organization.

        Args:
            auth_context: The user's authorization context
            requested_org_id: The org_id requested in query params

        Returns:
            Tuple of (validated_org_id, error_response or None)
            If error_response is not None, return it immediately.
            If requested_org_id is None, returns user's org_id.
        """
        user_org_id = getattr(auth_context, "org_id", None)
        user_roles = getattr(auth_context, "roles", []) or []

        # Platform admins can access any org
        if "platform_admin" in user_roles or "admin" in user_roles:
            return requested_org_id, None

        # If no org requested, use user's org
        if not requested_org_id:
            return user_org_id, None

        # User can only access their own org
        if user_org_id and requested_org_id != user_org_id:
            return None, error_response(
                "Access denied to organization",
                403,
                code="ORG_ACCESS_DENIED",
            )

        return requested_org_id, None

    ROUTES = [
        # Debate Analytics
        "/api/analytics/debates/overview",
        "/api/analytics/debates/trends",
        "/api/analytics/debates/topics",
        "/api/analytics/debates/outcomes",
        # Agent Performance
        "/api/analytics/agents/leaderboard",
        "/api/analytics/agents/comparison",
        "/api/analytics/agents/trends",
        # Usage Analytics
        "/api/analytics/usage/tokens",
        "/api/analytics/usage/costs",
        "/api/analytics/usage/active_users",
    ]

    # Pattern for agent-specific performance endpoint
    AGENT_PERFORMANCE_PATTERN = re.compile(r"^/api/analytics/agents/([a-zA-Z0-9_-]+)/performance$")

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        if normalized in self.ROUTES:
            return True
        # Check agent performance pattern
        return bool(self.AGENT_PERFORMANCE_PATTERN.match(normalized))

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests to appropriate methods with RBAC."""
        normalized = strip_version_prefix(path)

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _analytics_metrics_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for analytics metrics: {client_ip}")
            return error_response(
                "Rate limit exceeded. Please try again later.",
                429,
            )

        # RBAC: Require authentication and analytics:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, ANALYTICS_METRICS_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401, code="AUTH_REQUIRED")
        except ForbiddenError as e:
            logger.warning(f"Analytics metrics access denied: {e}")
            return error_response(str(e), 403, code="PERMISSION_DENIED")

        # Additional RBAC check via rbac.checker if available
        if not RBAC_AVAILABLE:
            if rbac_fail_closed():
                return error_response("Service unavailable: access control module not loaded", 503)
        elif hasattr(handler, "auth_context"):
            decision = check_permission(handler.auth_context, ANALYTICS_METRICS_PERMISSION)
            if not decision.allowed:
                logger.warning(f"RBAC denied analytics metrics access: {decision.reason}")
                return error_response(
                    decision.reason or "Permission denied",
                    403,
                    code="PERMISSION_DENIED",
                )

        # Debate Analytics
        if normalized == "/api/analytics/debates/overview":
            return self._get_debates_overview(query_params, auth_context)
        elif normalized == "/api/analytics/debates/trends":
            return self._get_debates_trends(query_params, auth_context)
        elif normalized == "/api/analytics/debates/topics":
            return self._get_debates_topics(query_params, auth_context)
        elif normalized == "/api/analytics/debates/outcomes":
            return self._get_debates_outcomes(query_params, auth_context)

        # Agent Performance
        elif normalized == "/api/analytics/agents/leaderboard":
            return self._get_agents_leaderboard(query_params)
        elif normalized == "/api/analytics/agents/comparison":
            return self._get_agents_comparison(query_params)
        elif normalized == "/api/analytics/agents/trends":
            return self._get_agents_trends(query_params)

        # Agent-specific performance
        match = self.AGENT_PERFORMANCE_PATTERN.match(normalized)
        if match:
            agent_id = match.group(1)
            return self._get_agent_performance(agent_id, query_params)

        # Usage Analytics
        if normalized == "/api/analytics/usage/tokens":
            return self._get_usage_tokens(query_params, auth_context)
        elif normalized == "/api/analytics/usage/costs":
            return self._get_usage_costs(query_params, auth_context)
        elif normalized == "/api/analytics/usage/active_users":
            return self._get_active_users(query_params, auth_context)

        return None


__all__ = [
    "AnalyticsMetricsHandler",
    # Re-exports from submodules
    "AgentAnalyticsMixin",
    "DebateAnalyticsMixin",
    "UsageAnalyticsMixin",
    "VALID_GRANULARITIES",
    "VALID_TIME_RANGES",
    "_group_by_time",
    "_parse_time_range",
]
