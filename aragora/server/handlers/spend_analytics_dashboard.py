"""
Spend Analytics Dashboard Handler.

Provides the five REST endpoints for the Spend Analytics dashboard:

- GET /api/v1/analytics/spend/summary      - Total spend, budget utilization %, trend direction
- GET /api/v1/analytics/spend/trends       - Daily/weekly/monthly spend over time
- GET /api/v1/analytics/spend/by-agent     - Cost breakdown per agent type
- GET /api/v1/analytics/spend/by-decision  - Cost per debate/decision
- GET /api/v1/analytics/spend/budget       - Budget limits, remaining, forecast to exhaustion

These endpoints aggregate data from:
- aragora.billing.cost_tracker.CostTracker (in-memory workspace stats)
- aragora.billing.budget_manager.BudgetManager (budget CRUD and enforcement)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter: 60 requests per minute
_spend_dashboard_limiter = RateLimiter(requests_per_minute=60)


def _get_cost_tracker() -> Any:
    """Lazy import of CostTracker to avoid circular imports."""
    try:
        from aragora.billing.cost_tracker import get_cost_tracker

        return get_cost_tracker()
    except (ImportError, RuntimeError, OSError) as e:
        logger.debug("CostTracker not available: %s", e)
        return None


def _get_budget_manager() -> Any:
    """Lazy import of BudgetManager to avoid circular imports."""
    try:
        from aragora.billing.budget_manager import get_budget_manager

        return get_budget_manager()
    except (ImportError, RuntimeError, OSError) as e:
        logger.debug("BudgetManager not available: %s", e)
        return None


class SpendAnalyticsDashboardHandler(BaseHandler):
    """Handler for the spend analytics dashboard endpoints.

    Aggregates data from billing CostTracker and BudgetManager to provide
    a unified view of organizational spend, agent costs, per-decision costs,
    budget utilization, and trend analysis.
    """

    ROUTES = [
        "/api/analytics/spend/summary",
        "/api/analytics/spend/trends",
        "/api/analytics/spend/by-agent",
        "/api/analytics/spend/by-decision",
        "/api/analytics/spend/budget",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        return normalized in self.ROUTES

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route GET requests to appropriate methods."""
        normalized = strip_version_prefix(path)

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _spend_dashboard_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for spend analytics dashboard: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        route_map: dict[str, Any] = {
            "/api/analytics/spend/summary": self._get_summary,
            "/api/analytics/spend/trends": self._get_trends,
            "/api/analytics/spend/by-agent": self._get_by_agent,
            "/api/analytics/spend/by-decision": self._get_by_decision,
            "/api/analytics/spend/budget": self._get_budget,
        }

        method_fn = route_map.get(normalized)
        if method_fn is not None:
            return method_fn(query_params, handler)

        return None

    # ------------------------------------------------------------------
    # Helper to get workspace_id from query params
    # ------------------------------------------------------------------

    @staticmethod
    def _get_workspace_id(query_params: dict[str, Any]) -> str:
        """Extract workspace_id from query params with default."""
        ws = query_params.get("workspace_id", "default")
        if isinstance(ws, list):
            ws = ws[0] if ws else "default"
        return str(ws)

    @staticmethod
    def _get_org_id(query_params: dict[str, Any]) -> str:
        """Extract org_id from query params with default."""
        org = query_params.get("org_id", "default")
        if isinstance(org, list):
            org = org[0] if org else "default"
        return str(org)

    @staticmethod
    def _get_period(query_params: dict[str, Any]) -> str:
        """Extract period from query params with default."""
        period = query_params.get("period", "daily")
        if isinstance(period, list):
            period = period[0] if period else "daily"
        valid = {"daily", "weekly", "monthly"}
        return str(period) if period in valid else "daily"

    @staticmethod
    def _get_days(query_params: dict[str, Any]) -> int:
        """Extract days from query params with default."""
        days_raw = query_params.get("days", "30")
        if isinstance(days_raw, list):
            days_raw = days_raw[0] if days_raw else "30"
        try:
            days = int(days_raw)
        except (ValueError, TypeError):
            days = 30
        return max(1, min(days, 365))

    # ------------------------------------------------------------------
    # Endpoint: GET /api/v1/analytics/spend/summary
    # ------------------------------------------------------------------

    @handle_errors("get spend summary")
    def _get_summary(
        self,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Return total spend, budget utilization %, and trend direction.

        Query params:
        - workspace_id: Workspace to query (default: "default")
        - org_id: Organization to query (default: "default")
        """
        workspace_id = self._get_workspace_id(query_params)
        org_id = self._get_org_id(query_params)

        tracker = _get_cost_tracker()
        budget_mgr = _get_budget_manager()

        # Total spend from tracker
        total_spend_usd = "0.00"
        total_api_calls = 0
        total_tokens = 0
        if tracker:
            stats = tracker.get_workspace_stats(workspace_id)
            total_spend_usd = stats.get("total_cost_usd", "0")
            total_api_calls = stats.get("total_api_calls", 0)
            total_tokens = stats.get("total_tokens_in", 0) + stats.get("total_tokens_out", 0)

        # Budget utilization
        budget_limit_usd = 0.0
        budget_spent_usd = 0.0
        utilization_pct = 0.0
        if budget_mgr:
            budgets = budget_mgr.get_budgets_for_org(org_id, active_only=True)
            for b in budgets:
                budget_limit_usd += b.amount_usd
                budget_spent_usd += b.spent_usd
            if budget_limit_usd > 0:
                utilization_pct = round((budget_spent_usd / budget_limit_usd) * 100, 1)

        # Trend direction: compare first vs second half of spend if tracker has data
        trend_direction = "stable"
        if tracker:
            dashboard = tracker.get_dashboard_summary(workspace_id=workspace_id, org_id=org_id)
            projected = dashboard.get("projections", {}).get("projected_monthly_usd")
            if projected:
                try:
                    proj_val = float(projected)
                    cur_val = float(total_spend_usd)
                    if proj_val > cur_val * 1.1:
                        trend_direction = "increasing"
                    elif proj_val < cur_val * 0.9:
                        trend_direction = "decreasing"
                except (ValueError, TypeError):
                    pass

        return json_response({
            "total_spend_usd": total_spend_usd,
            "total_api_calls": total_api_calls,
            "total_tokens": total_tokens,
            "budget_limit_usd": round(budget_limit_usd, 2),
            "budget_spent_usd": round(budget_spent_usd, 2),
            "utilization_pct": utilization_pct,
            "trend_direction": trend_direction,
            "avg_cost_per_decision": (
                round(float(total_spend_usd) / max(total_api_calls, 1), 4)
            ),
        })

    # ------------------------------------------------------------------
    # Endpoint: GET /api/v1/analytics/spend/trends
    # ------------------------------------------------------------------

    @handle_errors("get spend trends")
    def _get_trends(
        self,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Return daily/weekly/monthly spend over time.

        Query params:
        - org_id: Organization to query (default: "default")
        - period: "daily" | "weekly" | "monthly" (default: "daily")
        - days: Number of data points to return (default: 30, max: 365)
        """
        org_id = self._get_org_id(query_params)
        period = self._get_period(query_params)
        days = self._get_days(query_params)

        budget_mgr = _get_budget_manager()
        data_points: list[dict[str, Any]] = []

        if budget_mgr:
            # Map period param to budget manager's period param
            period_map = {"daily": "day", "weekly": "week", "monthly": "month"}
            bm_period = period_map.get(period, "day")

            trends = budget_mgr.get_org_spending_trends(
                org_id=org_id,
                period=bm_period,
                limit=days,
            )
            data_points = trends

        return json_response({
            "org_id": org_id,
            "period": period,
            "days": days,
            "data_points": data_points,
        })

    # ------------------------------------------------------------------
    # Endpoint: GET /api/v1/analytics/spend/by-agent
    # ------------------------------------------------------------------

    @handle_errors("get spend by agent")
    def _get_by_agent(
        self,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Return cost breakdown per agent type.

        Query params:
        - workspace_id: Workspace to query (default: "default")
        """
        workspace_id = self._get_workspace_id(query_params)

        tracker = _get_cost_tracker()
        agents: list[dict[str, Any]] = []
        total_usd = "0"

        if tracker:
            stats = tracker.get_workspace_stats(workspace_id)
            cost_by_agent = stats.get("cost_by_agent", {})
            total_usd = stats.get("total_cost_usd", "0")

            try:
                total_float = float(total_usd)
            except (ValueError, TypeError):
                total_float = 0.0

            for agent_name, cost_str in cost_by_agent.items():
                try:
                    cost_val = float(cost_str)
                except (ValueError, TypeError):
                    cost_val = 0.0
                pct = round((cost_val / total_float * 100), 1) if total_float > 0 else 0.0
                agents.append({
                    "agent_name": agent_name,
                    "cost_usd": cost_str,
                    "percentage": pct,
                })

            # Sort by cost descending
            agents.sort(key=lambda x: float(x.get("cost_usd", "0")), reverse=True)

        return json_response({
            "workspace_id": workspace_id,
            "total_usd": total_usd,
            "agents": agents,
        })

    # ------------------------------------------------------------------
    # Endpoint: GET /api/v1/analytics/spend/by-decision
    # ------------------------------------------------------------------

    @handle_errors("get spend by decision")
    def _get_by_decision(
        self,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Return cost per debate/decision.

        Query params:
        - workspace_id: Workspace to query (default: "default")
        - limit: Maximum number of debates to return (default: 20, max: 100)
        """
        workspace_id = self._get_workspace_id(query_params)
        limit_raw = query_params.get("limit", "20")
        if isinstance(limit_raw, list):
            limit_raw = limit_raw[0] if limit_raw else "20"
        try:
            limit = int(limit_raw)
        except (ValueError, TypeError):
            limit = 20
        limit = max(1, min(limit, 100))

        tracker = _get_cost_tracker()
        decisions: list[dict[str, Any]] = []

        if tracker:
            # Pull debate costs from the tracker's internal tracking
            debate_costs = dict(tracker._debate_costs)
            sorted_debates = sorted(debate_costs.items(), key=lambda x: x[1], reverse=True)

            for debate_id, cost in sorted_debates[:limit]:
                decisions.append({
                    "debate_id": debate_id,
                    "cost_usd": str(cost),
                })

        return json_response({
            "workspace_id": workspace_id,
            "decisions": decisions,
            "count": len(decisions),
        })

    # ------------------------------------------------------------------
    # Endpoint: GET /api/v1/analytics/spend/budget
    # ------------------------------------------------------------------

    @handle_errors("get spend budget")
    def _get_budget(
        self,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Return budget limits, remaining, and forecast to exhaustion.

        Query params:
        - org_id: Organization to query (default: "default")
        """
        org_id = self._get_org_id(query_params)
        budget_mgr = _get_budget_manager()

        if not budget_mgr:
            return json_response({
                "org_id": org_id,
                "budgets": [],
                "total_budget_usd": 0.0,
                "total_spent_usd": 0.0,
                "total_remaining_usd": 0.0,
                "utilization_pct": 0.0,
                "forecast_exhaustion_days": None,
            })

        summary = budget_mgr.get_summary(org_id)

        # Forecast exhaustion: days until budget runs out at current rate
        total_budget = summary.get("total_budget_usd", 0)
        total_spent = summary.get("total_spent_usd", 0)
        remaining = summary.get("total_remaining_usd", 0)

        forecast_exhaustion_days = None
        if total_spent > 0 and remaining > 0:
            # Estimate daily rate from budgets
            budgets_list = summary.get("budgets", [])
            active_budgets = [
                b for b in budgets_list
                if b.get("status") == "active" and b.get("period_start")
            ]
            if active_budgets:
                now = datetime.now(timezone.utc).timestamp()
                total_daily_rate = 0.0
                for b in active_budgets:
                    period_start = b.get("period_start", now)
                    days_elapsed = max(1, (now - period_start) / 86400)
                    daily_rate = b.get("spent_usd", 0) / days_elapsed
                    total_daily_rate += daily_rate
                if total_daily_rate > 0:
                    forecast_exhaustion_days = round(remaining / total_daily_rate, 1)

        utilization_pct = round(
            (total_spent / total_budget * 100) if total_budget > 0 else 0, 1
        )

        return json_response({
            "org_id": org_id,
            "budgets": summary.get("budgets", []),
            "total_budget_usd": total_budget,
            "total_spent_usd": total_spent,
            "total_remaining_usd": remaining,
            "utilization_pct": utilization_pct,
            "forecast_exhaustion_days": forecast_exhaustion_days,
        })


__all__ = ["SpendAnalyticsDashboardHandler"]
