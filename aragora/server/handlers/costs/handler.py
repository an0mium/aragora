"""
Cost visibility HTTP handler class.

Provides the CostHandler class with all API endpoint methods for cost
tracking, budgets, alerts, recommendations, forecasting, and exports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from aiohttp import web

from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.aiohttp_responses import web_error_response
from aragora.rbac.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.handlers.api_decorators import api_endpoint
from aragora.server.validation.query_params import safe_query_int

from .helpers import (
    _build_export_rows,
    _export_csv_response,
    _get_implementation_difficulty,
    _get_implementation_steps,
    _get_implementation_time,
)
from . import models as _models

logger = logging.getLogger(__name__)


class CostHandler:
    """Handler for cost visibility API endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    @api_endpoint(
        method="GET",
        path="/api/v1/costs",
        summary="Get cost summary",
        description="Fetch cost dashboard summary data including spending, budgets, and alerts.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
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

            summary = await _models.get_cost_summary(
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

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get costs: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/breakdown",
        summary="Get cost breakdown",
        description="Fetch cost breakdown by provider, feature, or model.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_breakdown(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/breakdown

        Get detailed cost breakdown.
        """
        try:
            time_range = request.query.get("range", "7d")
            workspace_id = request.query.get("workspace_id", "default")
            group_by = request.query.get("group_by", "provider")  # provider, feature, model

            summary = await _models.get_cost_summary(
                workspace_id=workspace_id, time_range=time_range
            )

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

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get breakdown: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/timeline",
        summary="Get cost timeline",
        description="Fetch cost timeline data over a specified period.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_timeline(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/timeline

        Get usage timeline data.
        """
        try:
            time_range = request.query.get("range", "7d")
            workspace_id = request.query.get("workspace_id", "default")

            summary = await _models.get_cost_summary(
                workspace_id=workspace_id, time_range=time_range
            )

            return web.json_response(
                {
                    "timeline": summary.daily_costs,
                    "total": summary.total_cost,
                    "average": (
                        summary.total_cost / len(summary.daily_costs) if summary.daily_costs else 0
                    ),
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get timeline: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/alerts",
        summary="Get budget alerts",
        description="Fetch active budget alerts for the workspace.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_alerts(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/alerts

        Get budget alerts.
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")

            tracker = _models._get_cost_tracker()
            if tracker:
                active_alerts = _models._get_active_alerts(tracker, workspace_id)

                # Also get historical alerts from Knowledge Mound if available
                km_alerts = tracker.query_km_workspace_alerts(
                    workspace_id=workspace_id,
                    min_level="warning",
                    limit=20,
                )
                for km_alert in km_alerts:
                    if not km_alert.get("acknowledged"):
                        active_alerts.append(
                            {
                                "id": km_alert.get("id", ""),
                                "type": km_alert.get("level", "info"),
                                "message": km_alert.get("message", ""),
                                "severity": km_alert.get("level", "info"),
                                "timestamp": km_alert.get("created_at", ""),
                            }
                        )
            else:
                active_alerts = []

            return web.json_response({"alerts": active_alerts})

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get alerts: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/budget",
        summary="Set budget limits",
        description="Set or update workspace budget limits.",
    )
    @rate_limit(requests_per_minute=20)  # Write operation
    @require_permission("budget:set")
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
            body, err = await parse_json_body(request, context="set_budget")
            if err:
                return err
            budget_amount = body.get("budget")
            workspace_id = body.get("workspace_id", "default")
            daily_limit = body.get("daily_limit")
            name = body.get("name", f"Budget for {workspace_id}")

            if budget_amount is None or budget_amount < 0:
                return web_error_response("Valid budget amount required", 400)

            tracker = _models._get_cost_tracker()
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

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to set budget: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/alerts/{alert_id}/dismiss",
        summary="Dismiss alert",
        description="Dismiss a budget alert.",
    )
    @rate_limit(requests_per_minute=20)  # Write operation
    @require_permission("costs:read")
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

            return web.json_response(
                {
                    "success": True,
                    "alert_id": alert_id,
                    "dismissed": True,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to dismiss alert: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/recommendations",
        summary="Get recommendations",
        description="Get cost optimization recommendations for the workspace.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_recommendations(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/recommendations

        Get cost optimization recommendations.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - status: Filter by status (pending, applied, dismissed)
            - type: Filter by type (model_downgrade, caching, batching)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            status_filter = request.query.get("status")
            type_filter = request.query.get("type")

            from aragora.billing.optimizer import get_cost_optimizer
            from aragora.billing.recommendations import (
                RecommendationStatus,
                RecommendationType,
            )

            optimizer = get_cost_optimizer()

            # Generate new recommendations if none exist
            existing = optimizer.get_workspace_recommendations(workspace_id)
            if not existing:
                await optimizer.analyze_workspace(workspace_id)

            # Apply filters
            status = RecommendationStatus(status_filter) if status_filter else None
            rec_type = RecommendationType(type_filter) if type_filter else None

            recommendations = optimizer.get_workspace_recommendations(
                workspace_id, status=status, type_filter=rec_type
            )

            summary = optimizer.get_summary(workspace_id)

            return web.json_response(
                {
                    "recommendations": [r.to_dict() for r in recommendations],
                    "summary": summary.to_dict(),
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get recommendations: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/recommendations/{recommendation_id}",
        summary="Get recommendation",
        description="Get a specific cost optimization recommendation.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_recommendation(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/recommendations/{recommendation_id}

        Get a specific recommendation.
        """
        try:
            recommendation_id = request.match_info.get("recommendation_id")

            from aragora.billing.optimizer import get_cost_optimizer

            optimizer = get_cost_optimizer()
            recommendation = optimizer.get_recommendation(recommendation_id)

            if not recommendation:
                return web_error_response("Recommendation not found", 404)

            return web.json_response(recommendation.to_dict())

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get recommendation: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/recommendations/{recommendation_id}/apply",
        summary="Apply recommendation",
        description="Apply a cost optimization recommendation.",
    )
    @rate_limit(requests_per_minute=20)  # Write operation
    @require_permission("costs:write")
    async def handle_apply_recommendation(self, request: web.Request) -> web.Response:
        """
        POST /api/costs/recommendations/{recommendation_id}/apply

        Apply a recommendation.
        """
        try:
            recommendation_id = request.match_info.get("recommendation_id")
            body, err = await parse_json_body(request, context="apply_recommendation")
            if err:
                return err
            user_id = body.get("user_id", "unknown")

            from aragora.billing.optimizer import get_cost_optimizer

            optimizer = get_cost_optimizer()
            success = optimizer.apply_recommendation(recommendation_id, user_id)

            if not success:
                return web_error_response("Recommendation not found", 404)

            recommendation = optimizer.get_recommendation(recommendation_id)

            return web.json_response(
                {
                    "success": True,
                    "recommendation": recommendation.to_dict() if recommendation else None,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to apply recommendation: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/recommendations/{recommendation_id}/dismiss",
        summary="Dismiss recommendation",
        description="Dismiss a cost optimization recommendation.",
    )
    @rate_limit(requests_per_minute=20)  # Write operation
    @require_permission("costs:write")
    async def handle_dismiss_recommendation(self, request: web.Request) -> web.Response:
        """
        POST /api/costs/recommendations/{recommendation_id}/dismiss

        Dismiss a recommendation.
        """
        try:
            recommendation_id = request.match_info.get("recommendation_id")

            from aragora.billing.optimizer import get_cost_optimizer

            optimizer = get_cost_optimizer()
            success = optimizer.dismiss_recommendation(recommendation_id)

            if not success:
                return web_error_response("Recommendation not found", 404)

            return web.json_response({"success": True, "dismissed": True})

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to dismiss recommendation: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/efficiency",
        summary="Get efficiency metrics",
        description="Get cost efficiency metrics including cost per token and model utilization.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_efficiency(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/efficiency

        Get efficiency metrics.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - range: Time range (24h, 7d, 30d)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            time_range = request.query.get("range", "7d")

            tracker = _models._get_cost_tracker()
            if not tracker:
                return web_error_response("Cost tracker not available", 503)

            stats = tracker.get_workspace_stats(workspace_id)

            # Calculate efficiency metrics
            total_tokens = stats.get("total_tokens_in", 0) + stats.get("total_tokens_out", 0)
            total_calls = stats.get("total_api_calls", 0)
            total_cost = float(stats.get("total_cost_usd", "0"))

            cost_per_1k_tokens = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
            tokens_per_call = total_tokens / total_calls if total_calls > 0 else 0
            cost_per_call = total_cost / total_calls if total_calls > 0 else 0

            # Model utilization
            cost_by_model = stats.get("cost_by_model", {})
            model_utilization = []
            for model, cost in cost_by_model.items():
                model_utilization.append(
                    {
                        "model": model,
                        "cost": str(cost),
                        "percentage": (float(cost) / total_cost * 100) if total_cost > 0 else 0,
                    }
                )

            return web.json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range,
                    "metrics": {
                        "cost_per_1k_tokens": round(cost_per_1k_tokens, 4),
                        "tokens_per_call": round(tokens_per_call, 0),
                        "cost_per_call": round(cost_per_call, 4),
                        "total_tokens": total_tokens,
                        "total_calls": total_calls,
                        "total_cost": round(total_cost, 2),
                    },
                    "model_utilization": sorted(model_utilization, key=lambda x: -x["percentage"]),
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get efficiency: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/forecast",
        summary="Get cost forecast",
        description="Get cost forecast for the specified number of days.",
    )
    @rate_limit(requests_per_minute=60)  # Read operation
    @require_permission("costs:read")
    async def handle_get_forecast(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/forecast

        Get cost forecast.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - days: Forecast days (default: 30)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            forecast_days = safe_query_int(
                request.query, "days", default=30, min_val=1, max_val=365
            )

            from aragora.billing.forecaster import get_cost_forecaster

            forecaster = get_cost_forecaster()
            report = await forecaster.generate_forecast(
                workspace_id=workspace_id,
                forecast_days=forecast_days,
            )

            return web.json_response(report.to_dict())

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get forecast: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/forecast/simulate",
        summary="Simulate cost scenario",
        description="Simulate a cost scenario with hypothetical changes.",
    )
    @rate_limit(requests_per_minute=5)  # Expensive: simulation
    @require_permission("costs:read")
    async def handle_simulate_forecast(self, request: web.Request) -> web.Response:
        """
        POST /api/costs/forecast/simulate

        Simulate a cost scenario.

        Body:
            - workspace_id: Workspace ID
            - scenario: Scenario object with name, description, changes
            - days: Days to simulate (default: 30)
        """
        try:
            body, err = await parse_json_body(request, context="simulate_forecast")
            if err:
                return err
            workspace_id = body.get("workspace_id", "default")
            scenario_data = body.get("scenario", {})
            days = body.get("days", 30)

            from aragora.billing.forecaster import SimulationScenario, get_cost_forecaster

            scenario = SimulationScenario(
                name=scenario_data.get("name", "Custom Scenario"),
                description=scenario_data.get("description", ""),
                changes=scenario_data.get("changes", {}),
            )

            forecaster = get_cost_forecaster()
            result = await forecaster.simulate_scenario(
                workspace_id=workspace_id,
                scenario=scenario,
                days=days,
            )

            return web.json_response(result.to_dict())

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to simulate forecast: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/export",
        summary="Export cost data",
        description="Export usage data as CSV or JSON.",
    )
    @rate_limit(requests_per_minute=10)  # Export can be expensive
    @require_permission("costs:read")
    async def handle_export(self, request: web.Request) -> web.Response:
        """
        GET /api/costs/export

        Export usage data as CSV or JSON.

        Query params:
            - format: Export format (csv, json). Default: json
            - range: Time range (24h, 7d, 30d, 90d). Default: 30d
            - workspace_id: Workspace ID (default: default)
            - group_by: Grouping (daily, provider, feature). Default: daily
        """
        try:
            export_format = request.query.get("format", "json")
            time_range = request.query.get("range", "30d")
            workspace_id = request.query.get("workspace_id", "default")
            group_by = request.query.get("group_by", "daily")

            if export_format not in ("csv", "json"):
                return web_error_response("format must be 'csv' or 'json'", 400)

            summary = await _models.get_cost_summary(
                workspace_id=workspace_id,
                time_range=time_range,
            )

            # Build export rows based on grouping
            rows = _build_export_rows(summary, group_by)

            if export_format == "csv":
                return _export_csv_response(rows, workspace_id, time_range)

            # JSON export
            return web.json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range,
                    "group_by": group_by,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "total_cost": summary.total_cost,
                    "total_tokens": summary.tokens_used,
                    "total_api_calls": summary.api_calls,
                    "rows": rows,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to export costs: {e}")
            return web_error_response(str(e), 500)

    # =========================================================================
    # New Endpoints: Usage, Budgets, Constraints, Estimates, Detailed Views
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/usage",
        summary="Get usage tracking",
        description="Get detailed usage tracking data for the workspace.",
    )
    @rate_limit(requests_per_minute=60)
    @require_permission("costs:read")
    async def handle_get_usage(self, request: web.Request) -> web.Response:
        """
        GET /api/v1/costs/usage

        Get detailed usage tracking data.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - range: Time range (24h, 7d, 30d, 90d). Default: 7d
            - group_by: Grouping (provider, model, operation). Default: provider
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            time_range = request.query.get("range", "7d")
            group_by = request.query.get("group_by", "provider")

            tracker = _models._get_cost_tracker()
            if not tracker:
                return web_error_response("Cost tracker not available", 503)

            # Get usage data from cost tracker
            now = datetime.now(timezone.utc)
            range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
            start_date = now - timedelta(days=range_days)

            from aragora.billing.cost_tracker import CostGranularity

            report = await tracker.generate_report(
                workspace_id=workspace_id,
                period_start=start_date,
                period_end=now,
                granularity=CostGranularity.DAILY,
            )

            # Build usage breakdown based on group_by
            usage_breakdown = []
            if group_by == "provider" and report.cost_by_provider:
                for name, cost in report.cost_by_provider.items():
                    usage_breakdown.append(
                        {
                            "name": name,
                            "cost_usd": float(cost),
                            "api_calls": report.calls_by_provider.get(name, 0)
                            if hasattr(report, "calls_by_provider")
                            else 0,
                        }
                    )
            elif group_by == "model" and hasattr(report, "cost_by_model"):
                for name, cost in report.cost_by_model.items():
                    usage_breakdown.append(
                        {
                            "name": name,
                            "cost_usd": float(cost),
                        }
                    )
            elif group_by == "operation" and report.cost_by_operation:
                for name, cost in report.cost_by_operation.items():
                    usage_breakdown.append(
                        {
                            "name": name,
                            "cost_usd": float(cost),
                        }
                    )

            return web.json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range,
                    "group_by": group_by,
                    "total_cost_usd": float(report.total_cost_usd),
                    "total_tokens_in": report.total_tokens_in,
                    "total_tokens_out": report.total_tokens_out,
                    "total_api_calls": report.total_api_calls,
                    "usage": usage_breakdown,
                    "period_start": start_date.isoformat(),
                    "period_end": now.isoformat(),
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get usage: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/budgets",
        summary="List budgets",
        description="List all budgets for the workspace.",
    )
    @rate_limit(requests_per_minute=60)
    @require_permission("costs:read")
    async def handle_list_budgets(self, request: web.Request) -> web.Response:
        """
        GET /api/v1/costs/budgets

        List all budgets for the workspace.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - active_only: Only show active budgets (default: true)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            # Note: active_only filter not yet implemented
            # active_only = request.query.get("active_only", "true").lower() == "true"

            tracker = _models._get_cost_tracker()
            if not tracker:
                return web_error_response("Cost tracker not available", 503)

            # Get budget from tracker
            budget = tracker.get_budget(workspace_id=workspace_id)
            budgets = []

            if budget:
                budget_dict = {
                    "id": f"budget_{workspace_id}",
                    "workspace_id": workspace_id,
                    "name": budget.name
                    if hasattr(budget, "name")
                    else f"Budget for {workspace_id}",
                    "monthly_limit_usd": float(budget.monthly_limit_usd)
                    if budget.monthly_limit_usd
                    else None,
                    "daily_limit_usd": float(budget.daily_limit_usd)
                    if budget.daily_limit_usd
                    else None,
                    "current_monthly_spend": float(budget.current_monthly_spend)
                    if hasattr(budget, "current_monthly_spend")
                    else 0,
                    "current_daily_spend": float(budget.current_daily_spend)
                    if hasattr(budget, "current_daily_spend")
                    else 0,
                    "active": True,
                }
                budgets.append(budget_dict)

            return web.json_response(
                {
                    "budgets": budgets,
                    "count": len(budgets),
                    "workspace_id": workspace_id,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to list budgets: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/budgets",
        summary="Create budget",
        description="Create a new budget for the workspace.",
    )
    @rate_limit(requests_per_minute=20)
    @require_permission("budget:set")
    async def handle_create_budget(self, request: web.Request) -> web.Response:
        """
        POST /api/v1/costs/budgets

        Create a new budget.

        Body:
            - workspace_id: Workspace ID
            - name: Budget name
            - monthly_limit_usd: Monthly spending limit
            - daily_limit_usd: Optional daily limit
            - alert_thresholds: Optional list of alert thresholds (percentages)
        """
        try:
            body, err = await parse_json_body(request, context="create_budget")
            if err:
                return err

            workspace_id = body.get("workspace_id", "default")
            name = body.get("name", f"Budget for {workspace_id}")
            monthly_limit = body.get("monthly_limit_usd")
            daily_limit = body.get("daily_limit_usd")
            alert_thresholds = body.get("alert_thresholds", [50, 75, 90, 100])

            if monthly_limit is None or monthly_limit < 0:
                return web_error_response("Valid monthly_limit_usd required", 400)

            tracker = _models._get_cost_tracker()
            if tracker:
                from aragora.billing.cost_tracker import Budget

                budget = Budget(
                    name=name,
                    workspace_id=workspace_id,
                    monthly_limit_usd=Decimal(str(monthly_limit)),
                    daily_limit_usd=Decimal(str(daily_limit)) if daily_limit else None,
                    alert_threshold_50=50 in alert_thresholds,
                    alert_threshold_75=75 in alert_thresholds,
                    alert_threshold_90=90 in alert_thresholds,
                )
                tracker.set_budget(budget)
                logger.info(f"[CostHandler] Budget created for {workspace_id}: ${monthly_limit}")

            return web.json_response(
                {
                    "success": True,
                    "budget": {
                        "id": f"budget_{workspace_id}",
                        "workspace_id": workspace_id,
                        "name": name,
                        "monthly_limit_usd": monthly_limit,
                        "daily_limit_usd": daily_limit,
                        "alert_thresholds": alert_thresholds,
                    },
                },
                status=201,
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to create budget: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/constraints/check",
        summary="Check cost constraints",
        description="Pre-flight check if an operation would exceed budget constraints.",
    )
    @rate_limit(requests_per_minute=60)
    @require_permission("costs:read")
    async def handle_check_constraints(self, request: web.Request) -> web.Response:
        """
        POST /api/v1/costs/constraints/check

        Pre-flight check if an operation would exceed budget constraints.

        Body:
            - workspace_id: Workspace ID
            - estimated_cost_usd: Estimated cost of the operation
            - operation: Operation type (optional)
        """
        try:
            body, err = await parse_json_body(request, context="check_constraints")
            if err:
                return err

            workspace_id = body.get("workspace_id", "default")
            estimated_cost = body.get("estimated_cost_usd", 0)
            operation = body.get("operation", "unknown")

            if estimated_cost < 0:
                return web_error_response("estimated_cost_usd must be non-negative", 400)

            tracker = _models._get_cost_tracker()
            allowed = True
            reason = "OK"
            remaining_budget = None

            if tracker:
                budget = tracker.get_budget(workspace_id=workspace_id)
                if budget and budget.monthly_limit_usd:
                    current_spend = (
                        float(budget.current_monthly_spend)
                        if hasattr(budget, "current_monthly_spend")
                        else 0
                    )
                    limit = float(budget.monthly_limit_usd)
                    remaining_budget = limit - current_spend

                    if current_spend + estimated_cost > limit:
                        allowed = False
                        reason = f"Would exceed monthly budget (${limit:.2f})"

                    # Check daily limit if set
                    if budget.daily_limit_usd and allowed:
                        daily_spend = (
                            float(budget.current_daily_spend)
                            if hasattr(budget, "current_daily_spend")
                            else 0
                        )
                        daily_limit = float(budget.daily_limit_usd)
                        if daily_spend + estimated_cost > daily_limit:
                            allowed = False
                            reason = f"Would exceed daily budget (${daily_limit:.2f})"

            return web.json_response(
                {
                    "allowed": allowed,
                    "reason": reason,
                    "workspace_id": workspace_id,
                    "estimated_cost_usd": estimated_cost,
                    "operation": operation,
                    "remaining_monthly_budget": remaining_budget,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to check constraints: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/estimate",
        summary="Estimate operation cost",
        description="Estimate the cost of an operation before executing it.",
    )
    @rate_limit(requests_per_minute=60)
    @require_permission("costs:read")
    async def handle_estimate_cost(self, request: web.Request) -> web.Response:
        """
        POST /api/v1/costs/estimate

        Estimate the cost of an operation.

        Body:
            - operation: Operation type (debate, analysis, etc.)
            - tokens_input: Estimated input tokens
            - tokens_output: Estimated output tokens
            - model: Model to use (optional, uses default pricing)
            - provider: Provider (optional)
        """
        try:
            body, err = await parse_json_body(request, context="estimate_cost")
            if err:
                return err

            operation = body.get("operation", "unknown")
            tokens_input = body.get("tokens_input", 0)
            tokens_output = body.get("tokens_output", 0)
            model = body.get("model", "claude-3-opus")
            provider = body.get("provider", "anthropic")

            # Token pricing (per 1M tokens) - simplified pricing table
            pricing = {
                "anthropic": {
                    "claude-3-opus": {"input": 15.00, "output": 75.00},
                    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
                    "claude-3-haiku": {"input": 0.25, "output": 1.25},
                    "default": {"input": 3.00, "output": 15.00},
                },
                "openai": {
                    "gpt-4": {"input": 30.00, "output": 60.00},
                    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
                    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
                    "default": {"input": 10.00, "output": 30.00},
                },
                "default": {"input": 5.00, "output": 15.00},
            }

            provider_pricing = pricing.get(provider, pricing["default"])
            if isinstance(provider_pricing, dict) and "input" not in provider_pricing:
                model_pricing = provider_pricing.get(
                    model, provider_pricing.get("default", pricing["default"])
                )
            else:
                model_pricing = provider_pricing

            # Calculate cost
            input_cost = (tokens_input / 1_000_000) * model_pricing["input"]
            output_cost = (tokens_output / 1_000_000) * model_pricing["output"]
            total_cost = input_cost + output_cost

            return web.json_response(
                {
                    "estimated_cost_usd": round(total_cost, 6),
                    "breakdown": {
                        "input_tokens": tokens_input,
                        "output_tokens": tokens_output,
                        "input_cost_usd": round(input_cost, 6),
                        "output_cost_usd": round(output_cost, 6),
                    },
                    "pricing": {
                        "model": model,
                        "provider": provider,
                        "input_per_1m": model_pricing["input"],
                        "output_per_1m": model_pricing["output"],
                    },
                    "operation": operation,
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to estimate cost: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/forecast/detailed",
        summary="Get detailed forecast",
        description="Get detailed cost forecast with daily breakdowns and confidence intervals.",
    )
    @rate_limit(requests_per_minute=30)
    @require_permission("costs:read")
    async def handle_get_forecast_detailed(self, request: web.Request) -> web.Response:
        """
        GET /api/v1/costs/forecast/detailed

        Get detailed cost forecast with daily breakdowns.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - days: Forecast days (default: 30, max: 90)
            - include_confidence: Include confidence intervals (default: true)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            forecast_days = safe_query_int(request.query, "days", default=30, min_val=1, max_val=90)
            include_confidence = request.query.get("include_confidence", "true").lower() == "true"

            from aragora.billing.forecaster import get_cost_forecaster

            forecaster = get_cost_forecaster()
            report = await forecaster.generate_forecast(
                workspace_id=workspace_id,
                forecast_days=forecast_days,
            )

            # Generate daily breakdowns
            daily_forecasts = []
            base_report = report.to_dict()
            daily_cost = (
                float(base_report.get("projected_cost", 0)) / forecast_days
                if forecast_days > 0
                else 0
            )

            now = datetime.now(timezone.utc)
            for i in range(forecast_days):
                date = now + timedelta(days=i + 1)
                forecast_entry = {
                    "date": date.strftime("%Y-%m-%d"),
                    "projected_cost_usd": round(daily_cost, 2),
                }
                if include_confidence:
                    # Add confidence intervals (simplified: +/- 20%)
                    forecast_entry["confidence_low"] = round(daily_cost * 0.8, 2)
                    forecast_entry["confidence_high"] = round(daily_cost * 1.2, 2)
                daily_forecasts.append(forecast_entry)

            result = {
                "workspace_id": workspace_id,
                "forecast_days": forecast_days,
                "summary": base_report,
                "daily_forecasts": daily_forecasts,
            }

            if include_confidence:
                result["confidence_level"] = 0.80

            return web.json_response(result)

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get detailed forecast: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/costs/recommendations/detailed",
        summary="Get detailed recommendations",
        description="Get detailed cost optimization recommendations with implementation steps.",
    )
    @rate_limit(requests_per_minute=30)
    @require_permission("costs:read")
    async def handle_get_recommendations_detailed(self, request: web.Request) -> web.Response:
        """
        GET /api/v1/costs/recommendations/detailed

        Get detailed cost optimization recommendations.

        Query params:
            - workspace_id: Workspace ID (default: default)
            - include_implementation: Include implementation steps (default: true)
            - min_savings: Minimum savings threshold in USD (default: 0)
        """
        try:
            workspace_id = request.query.get("workspace_id", "default")
            include_implementation = (
                request.query.get("include_implementation", "true").lower() == "true"
            )
            min_savings = float(request.query.get("min_savings", "0"))

            from aragora.billing.optimizer import get_cost_optimizer

            optimizer = get_cost_optimizer()

            # Generate recommendations if none exist
            existing = optimizer.get_workspace_recommendations(workspace_id)
            if not existing:
                await optimizer.analyze_workspace(workspace_id)

            recommendations = optimizer.get_workspace_recommendations(workspace_id)
            summary = optimizer.get_summary(workspace_id)

            # Filter by minimum savings and add implementation details
            detailed_recs = []
            for rec in recommendations:
                rec_dict = rec.to_dict()
                savings = rec_dict.get("estimated_savings_usd", 0)

                if savings >= min_savings:
                    if include_implementation:
                        # Add implementation steps based on recommendation type
                        rec_type = rec_dict.get("type", "")
                        rec_dict["implementation_steps"] = _get_implementation_steps(rec_type)
                        rec_dict["difficulty"] = _get_implementation_difficulty(rec_type)
                        rec_dict["time_to_implement"] = _get_implementation_time(rec_type)

                    detailed_recs.append(rec_dict)

            # Sort by potential savings (descending)
            detailed_recs.sort(key=lambda x: x.get("estimated_savings_usd", 0), reverse=True)

            return web.json_response(
                {
                    "recommendations": detailed_recs,
                    "count": len(detailed_recs),
                    "summary": summary.to_dict(),
                    "workspace_id": workspace_id,
                    "total_potential_savings_usd": sum(
                        r.get("estimated_savings_usd", 0) for r in detailed_recs
                    ),
                }
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to get detailed recommendations: {e}")
            return web_error_response(str(e), 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/costs/alerts",
        summary="Create cost alert",
        description="Create a new cost alert with custom thresholds.",
    )
    @rate_limit(requests_per_minute=20)
    @require_permission("costs:write")
    async def handle_create_alert(self, request: web.Request) -> web.Response:
        """
        POST /api/v1/costs/alerts

        Create a new cost alert.

        Body:
            - workspace_id: Workspace ID
            - name: Alert name
            - type: Alert type (budget_threshold, spike_detection, daily_limit)
            - threshold: Threshold value (percentage for budget, multiplier for spike)
            - notification_channels: List of notification channels (email, slack, webhook)
        """
        try:
            body, err = await parse_json_body(request, context="create_alert")
            if err:
                return err

            workspace_id = body.get("workspace_id", "default")
            name = body.get("name")
            alert_type = body.get("type", "budget_threshold")
            threshold = body.get("threshold", 80)
            notification_channels = body.get("notification_channels", ["email"])

            if not name:
                return web_error_response("Alert name is required", 400)

            if alert_type not in ("budget_threshold", "spike_detection", "daily_limit"):
                return web_error_response(
                    "Invalid alert type. Must be budget_threshold, spike_detection, or daily_limit",
                    400,
                )

            # Generate alert ID
            import uuid

            alert_id = f"alert_{uuid.uuid4().hex[:8]}"

            # In production, this would be stored in a database
            alert = {
                "id": alert_id,
                "workspace_id": workspace_id,
                "name": name,
                "type": alert_type,
                "threshold": threshold,
                "notification_channels": notification_channels,
                "active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"[CostHandler] Created alert {alert_id} for {workspace_id}")

            return web.json_response(
                {
                    "success": True,
                    "alert": alert,
                },
                status=201,
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.exception(f"Failed to create alert: {e}")
            return web_error_response(str(e), 500)
