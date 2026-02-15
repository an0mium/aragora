"""Debate statistics handler for aggregate debate metrics.

Endpoints:
- GET /api/v1/debates/stats - Get aggregate debate statistics
- GET /api/v1/debates/stats/agents - Get per-agent statistics
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response, json_response
from aragora.server.validation.query_params import safe_query_int
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)


class DebateStatsHandler(BaseHandler):
    """Handle aggregate debate statistics endpoints.

    All endpoints require ``debates:read`` permission.
    """

    ROUTES = [
        "/api/v1/debates/stats",
        "/api/v1/debates/stats/agents",
    ]

    def can_handle(self, path: str) -> bool:
        stripped = strip_version_prefix(path)
        return stripped in ("/api/debates/stats", "/api/debates/stats/agents")

    @require_permission("debates:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        if handler.command != "GET":
            return error_response("Method not allowed", 405)

        stripped = strip_version_prefix(path)

        if stripped == "/api/debates/stats":
            return self._get_stats(query_params, handler)
        if stripped == "/api/debates/stats/agents":
            return self._get_agent_stats(query_params, handler)

        return None

    def _get_stats(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        period = query_params.get("period", "all")
        if period not in ("all", "day", "week", "month"):
            return error_response("period must be one of: all, day, week, month", 400)

        try:
            from aragora.analytics.debate_analytics import DebateAnalytics

            storage = self.ctx.get("storage")
            if storage is None:
                return error_response("Storage not available", 503)

            service = DebateAnalytics(storage)
            stats = service.get_debate_stats(period=period)
            return json_response(stats.to_dict())
        except Exception as exc:
            logger.error("Failed to get debate stats: %s", exc)
            return error_response("Failed to get debate stats", 500)

    def _get_agent_stats(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        limit = safe_query_int(query_params, "limit", default=20, min_val=1, max_val=100)

        try:
            from aragora.analytics.debate_analytics import DebateAnalytics

            storage = self.ctx.get("storage")
            if storage is None:
                return error_response("Storage not available", 503)

            service = DebateAnalytics(storage)
            agent_stats = service.get_agent_stats(limit=limit)
            return json_response({
                "agents": agent_stats,
                "count": len(agent_stats),
            })
        except Exception as exc:
            logger.error("Failed to get agent stats: %s", exc)
            return error_response("Failed to get agent stats", 500)


__all__ = ["DebateStatsHandler"]
