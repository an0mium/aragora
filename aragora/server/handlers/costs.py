"""
Cost Visibility API Handler.

Provides API endpoints for tracking and visualizing AI costs:
- Total cost and budget tracking
- Cost breakdown by provider and feature
- Usage timeline data
- Budget alerts and projections
- Optimization suggestions

Endpoints:
- GET /api/costs - Get cost dashboard data
- GET /api/costs/breakdown - Get detailed cost breakdown
- GET /api/costs/timeline - Get usage timeline
- GET /api/costs/alerts - Get budget alerts
- POST /api/costs/budget - Set budget limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CostEntry:
    """A single cost entry."""

    timestamp: datetime
    provider: str
    feature: str
    tokens_input: int
    tokens_output: int
    cost: float
    model: str
    workspace_id: str
    user_id: Optional[str] = None


@dataclass
class BudgetAlert:
    """A budget alert."""

    id: str
    type: str  # budget_warning, spike_detected, limit_reached
    message: str
    severity: str  # critical, warning, info
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class CostSummary:
    """Cost summary data."""

    total_cost: float
    budget: float
    tokens_used: int
    api_calls: int
    last_updated: datetime
    cost_by_provider: List[Dict[str, Any]] = field(default_factory=list)
    cost_by_feature: List[Dict[str, Any]] = field(default_factory=list)
    daily_costs: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_cost_entries: List[CostEntry] = []
_budget_settings: Dict[str, float] = {"default": 500.00}
_alerts: List[BudgetAlert] = []


# =============================================================================
# Cost Tracking
# =============================================================================


def record_cost(
    provider: str,
    feature: str,
    tokens_input: int,
    tokens_output: int,
    cost: float,
    model: str,
    workspace_id: str = "default",
    user_id: Optional[str] = None,
) -> None:
    """Record a cost entry."""
    entry = CostEntry(
        timestamp=datetime.now(timezone.utc),
        provider=provider,
        feature=feature,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        cost=cost,
        model=model,
        workspace_id=workspace_id,
        user_id=user_id,
    )
    _cost_entries.append(entry)

    # Check for budget alerts
    _check_budget_alerts(workspace_id)


def _check_budget_alerts(workspace_id: str) -> None:
    """Check if any budget thresholds have been crossed."""
    budget = _budget_settings.get(workspace_id, _budget_settings.get("default", 500.0))
    total_cost = sum(e.cost for e in _cost_entries if e.workspace_id == workspace_id)

    percentage = (total_cost / budget) * 100 if budget > 0 else 0

    # 80% warning
    if percentage >= 80 and not any(
        a.type == "budget_warning" and a.workspace_id == workspace_id  # type: ignore
        for a in _alerts
        if hasattr(a, "workspace_id")
    ):
        alert = BudgetAlert(
            id=f"alert_{len(_alerts) + 1}",
            type="budget_warning",
            message=f"Budget usage at {percentage:.1f}% - approaching limit",
            severity="warning" if percentage < 90 else "critical",
            timestamp=datetime.now(timezone.utc),
        )
        _alerts.append(alert)


# =============================================================================
# API Handlers
# =============================================================================


async def get_cost_summary(
    workspace_id: str = "default",
    time_range: str = "7d",
) -> CostSummary:
    """Get cost summary data."""
    # Calculate time range
    now = datetime.now(timezone.utc)
    range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    start_date = now - timedelta(days=range_days)

    # Filter entries
    entries = [
        e for e in _cost_entries if e.workspace_id == workspace_id and e.timestamp >= start_date
    ]

    # If no data, generate mock data
    if not entries:
        return _generate_mock_summary(time_range)

    # Calculate totals
    total_cost = sum(e.cost for e in entries)
    total_tokens = sum(e.tokens_input + e.tokens_output for e in entries)
    api_calls = len(entries)
    budget = _budget_settings.get(workspace_id, 500.0)

    # Cost by provider
    provider_costs: Dict[str, float] = {}
    for e in entries:
        provider_costs[e.provider] = provider_costs.get(e.provider, 0) + e.cost

    cost_by_provider = [
        {
            "name": name,
            "cost": cost,
            "percentage": (cost / total_cost * 100) if total_cost > 0 else 0,
        }
        for name, cost in sorted(provider_costs.items(), key=lambda x: -x[1])
    ]

    # Cost by feature
    feature_costs: Dict[str, float] = {}
    for e in entries:
        feature_costs[e.feature] = feature_costs.get(e.feature, 0) + e.cost

    cost_by_feature = [
        {
            "name": name,
            "cost": cost,
            "percentage": (cost / total_cost * 100) if total_cost > 0 else 0,
        }
        for name, cost in sorted(feature_costs.items(), key=lambda x: -x[1])
    ]

    # Daily costs
    daily_costs: Dict[str, Dict[str, float]] = {}
    for e in entries:
        date_key = e.timestamp.strftime("%Y-%m-%d")
        if date_key not in daily_costs:
            daily_costs[date_key] = {"cost": 0, "tokens": 0}
        daily_costs[date_key]["cost"] += e.cost
        daily_costs[date_key]["tokens"] += e.tokens_input + e.tokens_output

    daily_costs_list = [
        {"date": date, "cost": data["cost"], "tokens": int(data["tokens"])}
        for date, data in sorted(daily_costs.items())
    ]

    # Active alerts
    active_alerts = [
        {
            "id": a.id,
            "type": a.type,
            "message": a.message,
            "severity": a.severity,
            "timestamp": a.timestamp.isoformat(),
        }
        for a in _alerts
        if not a.acknowledged
    ]

    return CostSummary(
        total_cost=total_cost,
        budget=budget,
        tokens_used=total_tokens,
        api_calls=api_calls,
        last_updated=now,
        cost_by_provider=cost_by_provider,
        cost_by_feature=cost_by_feature,
        daily_costs=daily_costs_list,
        alerts=active_alerts,
    )


def _generate_mock_summary(time_range: str) -> CostSummary:
    """Generate mock data for demo."""
    now = datetime.now(timezone.utc)

    # Generate daily data
    range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    daily_costs = []
    total_cost = 0
    total_tokens = 0

    for i in range(range_days):
        date = now - timedelta(days=range_days - 1 - i)
        cost = 15 + (i % 5) * 3 + (hash(date.strftime("%Y-%m-%d")) % 10)
        tokens = int(cost * 25000)
        daily_costs.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "cost": round(cost, 2),
                "tokens": tokens,
            }
        )
        total_cost += cost
        total_tokens += tokens

    return CostSummary(
        total_cost=round(total_cost, 2),
        budget=500.00,
        tokens_used=total_tokens,
        api_calls=int(total_cost * 100),
        last_updated=now,
        cost_by_provider=[
            {"name": "Anthropic", "cost": round(total_cost * 0.616, 2), "percentage": 61.6},
            {"name": "OpenAI", "cost": round(total_cost * 0.276, 2), "percentage": 27.6},
            {"name": "Mistral", "cost": round(total_cost * 0.069, 2), "percentage": 6.9},
            {"name": "OpenRouter", "cost": round(total_cost * 0.039, 2), "percentage": 3.9},
        ],
        cost_by_feature=[
            {"name": "Debates", "cost": round(total_cost * 0.432, 2), "percentage": 43.2},
            {"name": "Email Triage", "cost": round(total_cost * 0.255, 2), "percentage": 25.5},
            {"name": "Code Review", "cost": round(total_cost * 0.179, 2), "percentage": 17.9},
            {"name": "Knowledge Work", "cost": round(total_cost * 0.134, 2), "percentage": 13.4},
        ],
        daily_costs=daily_costs,
        alerts=[
            {
                "id": "1",
                "type": "budget_warning",
                "message": "Projected to reach 80% of monthly budget by Jan 25",
                "severity": "warning",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "id": "2",
                "type": "spike_detected",
                "message": "Unusual spike in Debate costs detected (45% above average)",
                "severity": "info",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
            },
        ],
    )


# =============================================================================
# HTTP Handler Class
# =============================================================================


class CostHandler:
    """Handler for cost visibility API endpoints."""

    async def handle_get_costs(self, request: web.Request) -> web.Response:
        """
        GET /api/costs

        Get cost dashboard data.

        Query params:
            - range: Time range (24h, 7d, 30d, 90d)
            - workspace_id: Workspace ID (default: default)
        """
        try:
            time_range = request.query.get("range", "7d")
            workspace_id = request.query.get("workspace_id", "default")

            summary = await get_cost_summary(
                workspace_id=workspace_id,
                time_range=time_range,
            )

            return web.json_response(
                {
                    "totalCost": summary.total_cost,
                    "budget": summary.budget,
                    "tokensUsed": summary.tokens_used,
                    "apiCalls": summary.api_calls,
                    "lastUpdated": summary.last_updated.isoformat(),
                    "costByProvider": summary.cost_by_provider,
                    "costByFeature": summary.cost_by_feature,
                    "dailyCosts": summary.daily_costs,
                    "alerts": summary.alerts,
                }
            )

        except Exception as e:
            logger.exception(f"Failed to get costs: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500,
            )

    async def handle_get_breakdown(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/breakdown

        Get detailed cost breakdown.
        """
        try:
            time_range = request.query.get("range", "7d")
            workspace_id = request.query.get("workspace_id", "default")
            group_by = request.query.get("group_by", "provider")  # provider, feature, model

            summary = await get_cost_summary(workspace_id=workspace_id, time_range=time_range)

            if group_by == "provider":
                breakdown = summary.cost_by_provider
            elif group_by == "feature":
                breakdown = summary.cost_by_feature
            else:
                breakdown = summary.cost_by_provider

            return web.json_response(
                {
                    "groupBy": group_by,
                    "breakdown": breakdown,
                    "total": summary.total_cost,
                }
            )

        except Exception as e:
            logger.exception(f"Failed to get breakdown: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_timeline(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/timeline

        Get usage timeline data.
        """
        try:
            time_range = request.query.get("range", "7d")
            workspace_id = request.query.get("workspace_id", "default")

            summary = await get_cost_summary(workspace_id=workspace_id, time_range=time_range)

            return web.json_response(
                {
                    "timeline": summary.daily_costs,
                    "total": summary.total_cost,
                    "average": summary.total_cost / len(summary.daily_costs)
                    if summary.daily_costs
                    else 0,
                }
            )

        except Exception as e:
            logger.exception(f"Failed to get timeline: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_alerts(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/alerts

        Get budget alerts.
        """
        try:
            _workspace_id = request.query.get("workspace_id", "default")  # noqa: F841

            active_alerts = [
                {
                    "id": a.id,
                    "type": a.type,
                    "message": a.message,
                    "severity": a.severity,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in _alerts
                if not a.acknowledged
            ]

            return web.json_response({"alerts": active_alerts})

        except Exception as e:
            logger.exception(f"Failed to get alerts: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_set_budget(self, request: web.Request) -> web.Response:
        """
        POST /api/costs/budget

        Set budget limits.

        Body:
            - budget: Monthly budget in USD
            - workspace_id: Workspace ID
        """
        try:
            body = await request.json()
            budget = body.get("budget")
            workspace_id = body.get("workspace_id", "default")

            if budget is None or budget < 0:
                return web.json_response(
                    {"error": "Valid budget amount required"},
                    status=400,
                )

            _budget_settings[workspace_id] = float(budget)

            return web.json_response(
                {
                    "success": True,
                    "budget": budget,
                    "workspace_id": workspace_id,
                }
            )

        except Exception as e:
            logger.exception(f"Failed to set budget: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_dismiss_alert(self, request: web.Request) -> web.Response:
        """
        POST /api/costs/alerts/{alert_id}/dismiss

        Dismiss a budget alert.
        """
        try:
            alert_id = request.match_info.get("alert_id")

            for alert in _alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return web.json_response({"success": True})

            return web.json_response(
                {"error": "Alert not found"},
                status=404,
            )

        except Exception as e:
            logger.exception(f"Failed to dismiss alert: {e}")
            return web.json_response({"error": str(e)}, status=500)


def register_routes(app: web.Application) -> None:
    """Register cost visibility routes."""
    handler = CostHandler()

    app.router.add_get("/api/costs", handler.handle_get_costs)
    app.router.add_get("/api/costs/breakdown", handler.handle_get_breakdown)
    app.router.add_get("/api/costs/timeline", handler.handle_get_timeline)
    app.router.add_get("/api/costs/alerts", handler.handle_get_alerts)
    app.router.add_post("/api/costs/budget", handler.handle_set_budget)
    app.router.add_post("/api/costs/alerts/{alert_id}/dismiss", handler.handle_dismiss_alert)
