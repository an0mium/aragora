"""
Cost Dashboard API Handler.

Provides a consolidated cost visibility endpoint for SMB customers:
- GET /api/v1/billing/dashboard - Unified cost dashboard summary

Designed for the SMB tier with focus on simplicity and actionable insights.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import (
    error_response,
    get_string_param,
    json_response,
)
from ..utils.responses import HandlerResult
from ..secure import SecureHandler
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_dashboard_limiter = RateLimiter(requests_per_minute=60)


class CostDashboardHandler(SecureHandler):
    """Handler for the consolidated cost dashboard endpoint.

    Provides a simple, unified view of cost data suitable for SMB
    customers who need quick visibility into spend, budget, and projections.
    """

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    RESOURCE_TYPE = "cost_dashboard"

    ROUTES = [
        "/api/v1/billing/dashboard",
    ]

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
        method: str = "GET",
    ) -> HandlerResult | None:
        """Route dashboard requests."""
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if hasattr(handler, "command"):
            method = handler.command

        if path == "/api/v1/billing/dashboard" and method == "GET":
            return self._get_dashboard(handler, query_params)

        return error_response("Method not allowed", 405)

    @require_permission("billing:read")
    def _get_dashboard(
        self,
        handler: Any,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get the consolidated cost dashboard summary.

        Returns a simple JSON summary with current spend, budget utilization,
        top cost drivers, and projected monthly costs.

        Query params:
            workspace_id: Filter by workspace (optional)
            org_id: Filter by organization (optional)
        """
        workspace_id = get_string_param(query_params, "workspace_id", "")
        org_id = get_string_param(query_params, "org_id", "")

        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            summary = tracker.get_dashboard_summary(
                workspace_id=workspace_id or None,
                org_id=org_id or None,
            )

            return json_response(summary)

        except Exception as e:
            logger.error("Cost dashboard error: %s", e, exc_info=True)
            return error_response(f"Failed to load dashboard: {e}", 500)
