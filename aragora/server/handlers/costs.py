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

Now integrated with CostTracker for persistent storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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
# CostTracker Integration (replaces in-memory storage)
# =============================================================================

_cost_tracker = None


def _get_cost_tracker():
    """Get or create the cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        try:
            from aragora.billing.cost_tracker import get_cost_tracker
            _cost_tracker = get_cost_tracker()
            logger.info("[CostHandler] Connected to CostTracker with persistence")
        except Exception as e:
            logger.warning(f"[CostHandler] CostTracker unavailable, using fallback: {e}")
            _cost_tracker = None
    return _cost_tracker


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
    """Record a cost entry via CostTracker."""
    tracker = _get_cost_tracker()

    if tracker:
        try:
            import asyncio
            from aragora.billing.cost_tracker import TokenUsage

            usage = TokenUsage(
                workspace_id=workspace_id,
                provider=provider,
                model=model,
                tokens_in=tokens_input,
                tokens_out=tokens_output,
                cost_usd=Decimal(str(cost)),
                operation=feature,
                metadata={"user_id": user_id} if user_id else {},
            )

            # Record asynchronously if in async context, otherwise sync
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(tracker.record(usage))
            except RuntimeError:
                # No running loop, run synchronously
                asyncio.run(tracker.record(usage))

            logger.debug(f"[CostHandler] Recorded cost: ${cost:.6f} for {feature}")
        except Exception as e:
            logger.error(f"[CostHandler] Failed to record cost: {e}")
    else:
        logger.debug(f"[CostHandler] CostTracker not available, cost not persisted")


# =============================================================================
# API Handlers
# =============================================================================


async def get_cost_summary(
    workspace_id: str = "default",
    time_range: str = "7d",
) -> CostSummary:
    """Get cost summary data from CostTracker."""
    now = datetime.now(timezone.utc)
    range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    start_date = now - timedelta(days=range_days)

    tracker = _get_cost_tracker()

    if tracker:
        try:
            # Use CostTracker for real data
            from aragora.billing.cost_tracker import CostGranularity

            # Get report from tracker
            report = await tracker.generate_report(
                workspace_id=workspace_id,
                period_start=start_date,
                period_end=now,
                granularity=CostGranularity.DAILY,
            )

            # Get budget
            budget_obj = tracker.get_budget(workspace_id=workspace_id)
            budget = float(budget_obj.monthly_limit_usd) if budget_obj and budget_obj.monthly_limit_usd else 500.0

            # Convert cost_by_provider to list format
            total_cost = float(report.total_cost_usd)
            cost_by_provider = [
                {
                    "name": name,
                    "cost": float(cost),
                    "percentage": (float(cost) / total_cost * 100) if total_cost > 0 else 0,
                }
                for name, cost in sorted(
                    report.cost_by_provider.items(),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
            ] if report.cost_by_provider else []

            # Convert cost_by_operation (feature) to list format
            cost_by_feature = [
                {
                    "name": name,
                    "cost": float(cost),
                    "percentage": (float(cost) / total_cost * 100) if total_cost > 0 else 0,
                }
                for name, cost in sorted(
                    report.cost_by_operation.items(),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
            ] if report.cost_by_operation else []

            # Get workspace stats for quick access
            stats = tracker.get_workspace_stats(workspace_id)

            # If no real data yet, fall back to mock
            if total_cost == 0 and not report.cost_over_time:
                return _generate_mock_summary(time_range)

            return CostSummary(
                total_cost=total_cost,
                budget=budget,
                tokens_used=report.total_tokens_in + report.total_tokens_out,
                api_calls=report.total_api_calls,
                last_updated=now,
                cost_by_provider=cost_by_provider if cost_by_provider else _generate_mock_summary(time_range).cost_by_provider,
                cost_by_feature=cost_by_feature if cost_by_feature else _generate_mock_summary(time_range).cost_by_feature,
                daily_costs=report.cost_over_time if report.cost_over_time else _generate_mock_summary(time_range).daily_costs,
                alerts=_get_active_alerts(tracker, workspace_id),
            )

        except Exception as e:
            logger.warning(f"[CostHandler] CostTracker query failed, using mock: {e}")
            return _generate_mock_summary(time_range)

    # Fallback to mock data if no tracker
    return _generate_mock_summary(time_range)


def _get_active_alerts(tracker, workspace_id: str) -> List[Dict[str, Any]]:
    """Get active budget alerts from tracker."""
    alerts = []
    try:
        # Check if budget is approaching limits
        budget = tracker.get_budget(workspace_id=workspace_id)
        if budget:
            alert_level = budget.check_alert_level()
            if alert_level:
                from aragora.billing.cost_tracker import BudgetAlertLevel

                severity_map = {
                    BudgetAlertLevel.INFO: "info",
                    BudgetAlertLevel.WARNING: "warning",
                    BudgetAlertLevel.CRITICAL: "critical",
                    BudgetAlertLevel.EXCEEDED: "critical",
                }
                percentage = (
                    float(budget.current_monthly_spend / budget.monthly_limit_usd * 100)
                    if budget.monthly_limit_usd else 0
                )
                alerts.append({
                    "id": f"budget_{workspace_id}",
                    "type": "budget_warning",
                    "message": f"Budget usage at {percentage:.1f}% (${float(budget.current_monthly_spend):.2f} of ${float(budget.monthly_limit_usd):.2f})",
                    "severity": severity_map.get(alert_level, "warning"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    except Exception as e:
        logger.debug(f"[CostHandler] Could not get alerts: {e}")
    return alerts


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
            workspace_id = request.query.get("workspace_id", "default")

            tracker = _get_cost_tracker()
            if tracker:
                active_alerts = _get_active_alerts(tracker, workspace_id)

                # Also get historical alerts from Knowledge Mound if available
                km_alerts = tracker.query_km_workspace_alerts(
                    workspace_id=workspace_id,
                    min_level="warning",
                    limit=20,
                )
                for km_alert in km_alerts:
                    if not km_alert.get("acknowledged"):
                        active_alerts.append({
                            "id": km_alert.get("id", ""),
                            "type": km_alert.get("level", "info"),
                            "message": km_alert.get("message", ""),
                            "severity": km_alert.get("level", "info"),
                            "timestamp": km_alert.get("created_at", ""),
                        })
            else:
                active_alerts = []

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
            - daily_limit: Optional daily limit
            - name: Optional budget name
        """
        try:
            body = await request.json()
            budget_amount = body.get("budget")
            workspace_id = body.get("workspace_id", "default")
            daily_limit = body.get("daily_limit")
            name = body.get("name", f"Budget for {workspace_id}")

            if budget_amount is None or budget_amount < 0:
                return web.json_response(
                    {"error": "Valid budget amount required"},
                    status=400,
                )

            tracker = _get_cost_tracker()
            if tracker:
                from aragora.billing.cost_tracker import Budget

                budget = Budget(
                    name=name,
                    workspace_id=workspace_id,
                    monthly_limit_usd=Decimal(str(budget_amount)),
                    daily_limit_usd=Decimal(str(daily_limit)) if daily_limit else None,
                )
                tracker.set_budget(budget)
                logger.info(f"[CostHandler] Budget set for {workspace_id}: ${budget_amount}")

            return web.json_response(
                {
                    "success": True,
                    "budget": budget_amount,
                    "workspace_id": workspace_id,
                    "daily_limit": daily_limit,
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
            workspace_id = request.query.get("workspace_id", "default")

            # For now, alerts are ephemeral (recalculated from budget state)
            # In production, this would update a database record
            logger.info(f"[CostHandler] Alert {alert_id} dismissed for {workspace_id}")

            return web.json_response({
                "success": True,
                "alert_id": alert_id,
                "dismissed": True,
            })

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
