"""
Analytics Dashboard endpoint handlers.

Endpoints:
- GET  /api/analytics/summary           - Dashboard summary
- GET  /api/analytics/trends/findings   - Finding trends over time
- GET  /api/analytics/remediation       - Remediation metrics
- GET  /api/analytics/agents            - Agent performance metrics
- GET  /api/analytics/cost              - Cost analysis
- GET  /api/analytics/compliance        - Compliance scorecard
- GET  /api/analytics/heatmap           - Risk heatmap data
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Optional, TypeVar

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync context safely.

    Uses asyncio.run() which creates a new event loop, runs the coroutine,
    and closes the loop. Safe to call from sync handlers.
    """
    return asyncio.run(coro)


class AnalyticsDashboardHandler(BaseHandler):
    """Handler for analytics dashboard endpoints."""

    ROUTES = [
        "/api/analytics/summary",
        "/api/analytics/trends/findings",
        "/api/analytics/remediation",
        "/api/analytics/agents",
        "/api/analytics/cost",
        "/api/analytics/compliance",
        "/api/analytics/heatmap",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/analytics/summary":
            return self._get_summary(query_params)
        elif path == "/api/analytics/trends/findings":
            return self._get_finding_trends(query_params)
        elif path == "/api/analytics/remediation":
            return self._get_remediation_metrics(query_params)
        elif path == "/api/analytics/agents":
            return self._get_agent_metrics(query_params)
        elif path == "/api/analytics/cost":
            return self._get_cost_metrics(query_params)
        elif path == "/api/analytics/compliance":
            return self._get_compliance_scorecard(query_params)
        elif path == "/api/analytics/heatmap":
            return self._get_risk_heatmap(query_params)

        return None

    @require_user_auth
    @handle_errors("get analytics summary")
    def _get_summary(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get dashboard summary with key metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d

        Response:
        {
            "workspace_id": "...",
            "time_range": "30d",
            "total_findings": 150,
            "open_findings": 45,
            "critical_findings": 5,
            "resolved_last_period": 23,
            "finding_trend": "down",
            "trend_percentage": -15.5,
            "top_categories": [...],
            "recent_critical": [...]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            summary = _run_async(dashboard.get_summary(workspace_id, time_range))

            return json_response(summary.to_dict())

        except ValueError:
            return error_response(f"Invalid time_range: {time_range_str}", 400)
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return error_response(f"Failed to get summary: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get finding trends")
    def _get_finding_trends(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get finding trends over time.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Time bucket size (hour, day, week, month) - default day

        Response:
        {
            "trends": [
                {
                    "timestamp": "2024-01-15T00:00:00Z",
                    "total": 12,
                    "by_severity": {"critical": 1, "high": 5, ...},
                    "by_category": {...}
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")
        granularity_str = query_params.get("granularity", "day")

        try:
            from aragora.analytics import (
                get_analytics_dashboard,
                TimeRange,
                Granularity,
            )

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)
            granularity = Granularity(granularity_str)

            trends = asyncio.run(
                dashboard.get_finding_trends(workspace_id, time_range, granularity)
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "granularity": granularity_str,
                    "trends": [t.to_dict() for t in trends],
                }
            )

        except ValueError as e:
            return error_response(f"Invalid parameter: {e}", 400)
        except Exception as e:
            logger.error(f"Failed to get finding trends: {e}")
            return error_response(f"Failed to get trends: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get remediation metrics")
    def _get_remediation_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get remediation performance metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "total_resolved": 120,
            "total_open": 45,
            "mttr_hours": 48.5,
            "mttr_by_severity": {"critical": 4.2, "high": 24.1, ...},
            "false_positive_rate": 0.08,
            "accepted_risk_rate": 0.03
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_remediation_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    **metrics.to_dict(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get remediation metrics: {e}")
            return error_response(f"Failed to get metrics: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get agent metrics")
    def _get_agent_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get agent performance metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "agents": [
                {
                    "agent_id": "claude-3-sonnet",
                    "agent_name": "Claude 3.5 Sonnet",
                    "total_findings": 45,
                    "agreement_rate": 0.92,
                    "precision": 0.87,
                    "finding_distribution": {...}
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_agent_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "agents": [m.to_dict() for m in metrics],
                }
            )

        except Exception as e:
            logger.error(f"Failed to get agent metrics: {e}")
            return error_response(f"Failed to get metrics: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get cost metrics")
    def _get_cost_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get cost analysis for audits.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "total_audits": 50,
            "total_cost_usd": 125.50,
            "avg_cost_per_audit": 2.51,
            "cost_by_type": {"security": 45.00, ...},
            "token_usage": {"input": 500000, "output": 100000}
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_cost_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    **metrics.to_dict(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get cost metrics: {e}")
            return error_response(f"Failed to get metrics: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get compliance scorecard")
    def _get_compliance_scorecard(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get compliance scorecard for specified frameworks.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - frameworks: Comma-separated list (SOC2,GDPR,HIPAA,PCI-DSS)

        Response:
        {
            "scores": [
                {
                    "framework": "SOC2",
                    "score": 0.85,
                    "passing_controls": 17,
                    "failing_controls": 3,
                    "critical_gaps": ["access_control"]
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        frameworks_str = query_params.get("frameworks", "SOC2,GDPR,HIPAA,PCI-DSS")
        frameworks = [f.strip() for f in frameworks_str.split(",")]

        try:
            from aragora.analytics import get_analytics_dashboard

            dashboard = get_analytics_dashboard()

            scores = _run_async(dashboard.get_compliance_scorecard(workspace_id, frameworks))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "scores": [s.to_dict() for s in scores],
                }
            )

        except Exception as e:
            logger.error(f"Failed to get compliance scorecard: {e}")
            return error_response(f"Failed to get scorecard: {str(e)}", 500)

    @require_user_auth
    @handle_errors("get risk heatmap")
    def _get_risk_heatmap(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get risk heatmap data (category x severity).

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "cells": [
                {
                    "category": "security",
                    "severity": "critical",
                    "count": 5,
                    "trend": "down"
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            cells = _run_async(dashboard.get_risk_heatmap(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "cells": [c.to_dict() for c in cells],
                }
            )

        except Exception as e:
            logger.error(f"Failed to get risk heatmap: {e}")
            return error_response(f"Failed to get heatmap: {str(e)}", 500)
