"""
Cost visibility route registration.

Registers all cost-related HTTP routes on the aiohttp application.
"""

from __future__ import annotations

from aiohttp import web

from .handler import CostHandler


def register_routes(app: web.Application) -> None:
    """Register cost visibility routes."""
    handler = CostHandler()

    # Core cost endpoints (v1 canonical)
    app.router.add_get("/api/v1/costs", handler.handle_get_costs)
    app.router.add_get("/api/v1/costs/breakdown", handler.handle_get_breakdown)
    app.router.add_get("/api/v1/costs/timeline", handler.handle_get_timeline)
    app.router.add_get("/api/v1/costs/alerts", handler.handle_get_alerts)
    app.router.add_post("/api/v1/costs/budget", handler.handle_set_budget)
    app.router.add_post(
        "/api/v1/costs/alerts/{alert_id}/dismiss",
        handler.handle_dismiss_alert,
    )

    # Core cost endpoints (legacy, no version prefix)
    app.router.add_get("/api/costs", handler.handle_get_costs)
    app.router.add_get("/api/costs/breakdown", handler.handle_get_breakdown)
    app.router.add_get("/api/costs/timeline", handler.handle_get_timeline)
    app.router.add_get("/api/costs/alerts", handler.handle_get_alerts)
    app.router.add_post("/api/costs/budget", handler.handle_set_budget)
    app.router.add_post("/api/costs/alerts/{alert_id}/dismiss", handler.handle_dismiss_alert)

    # Optimization recommendations (v1 canonical)
    app.router.add_get("/api/v1/costs/recommendations", handler.handle_get_recommendations)
    app.router.add_get(
        "/api/v1/costs/recommendations/{recommendation_id}",
        handler.handle_get_recommendation,
    )
    app.router.add_post(
        "/api/v1/costs/recommendations/{recommendation_id}/apply",
        handler.handle_apply_recommendation,
    )
    app.router.add_post(
        "/api/v1/costs/recommendations/{recommendation_id}/dismiss",
        handler.handle_dismiss_recommendation,
    )

    # Optimization recommendations (legacy)
    app.router.add_get("/api/costs/recommendations", handler.handle_get_recommendations)
    app.router.add_get(
        "/api/costs/recommendations/{recommendation_id}",
        handler.handle_get_recommendation,
    )
    app.router.add_post(
        "/api/costs/recommendations/{recommendation_id}/apply",
        handler.handle_apply_recommendation,
    )
    app.router.add_post(
        "/api/costs/recommendations/{recommendation_id}/dismiss",
        handler.handle_dismiss_recommendation,
    )

    # Export endpoints (v1 canonical + legacy)
    app.router.add_get("/api/v1/costs/export", handler.handle_export)
    app.router.add_get("/api/costs/export", handler.handle_export)

    # Efficiency metrics (v1 canonical + legacy)
    app.router.add_get("/api/v1/costs/efficiency", handler.handle_get_efficiency)
    app.router.add_get("/api/costs/efficiency", handler.handle_get_efficiency)

    # Forecasting (v1 canonical + legacy)
    app.router.add_get("/api/v1/costs/forecast", handler.handle_get_forecast)
    app.router.add_get("/api/v1/costs/forecast/detailed", handler.handle_get_forecast_detailed)
    app.router.add_post(
        "/api/v1/costs/forecast/simulate",
        handler.handle_simulate_forecast,
    )
    app.router.add_get("/api/costs/forecast", handler.handle_get_forecast)
    app.router.add_get("/api/costs/forecast/detailed", handler.handle_get_forecast_detailed)
    app.router.add_post("/api/costs/forecast/simulate", handler.handle_simulate_forecast)

    # Usage tracking (v1 canonical + legacy)
    app.router.add_get("/api/v1/costs/usage", handler.handle_get_usage)
    app.router.add_get("/api/costs/usage", handler.handle_get_usage)

    # Budget management (v1 canonical + legacy)
    app.router.add_get("/api/v1/costs/budgets", handler.handle_list_budgets)
    app.router.add_post("/api/v1/costs/budgets", handler.handle_create_budget)
    app.router.add_get("/api/costs/budgets", handler.handle_list_budgets)
    app.router.add_post("/api/costs/budgets", handler.handle_create_budget)

    # Constraints and estimates (v1 canonical + legacy)
    app.router.add_post("/api/v1/costs/constraints/check", handler.handle_check_constraints)
    app.router.add_post("/api/v1/costs/estimate", handler.handle_estimate_cost)
    app.router.add_post("/api/costs/constraints/check", handler.handle_check_constraints)
    app.router.add_post("/api/costs/estimate", handler.handle_estimate_cost)

    # Detailed recommendations (v1 canonical + legacy)
    app.router.add_get(
        "/api/v1/costs/recommendations/detailed", handler.handle_get_recommendations_detailed
    )
    app.router.add_get(
        "/api/costs/recommendations/detailed", handler.handle_get_recommendations_detailed
    )

    # Alert creation (v1 canonical + legacy)
    app.router.add_post("/api/v1/costs/alerts", handler.handle_create_alert)
    app.router.add_post("/api/costs/alerts", handler.handle_create_alert)

    # Spend analytics dashboard (v1 canonical + legacy)
    app.router.add_get(
        "/api/v1/costs/analytics/trend", handler.handle_get_spend_trend
    )
    app.router.add_get(
        "/api/v1/costs/analytics/by-agent", handler.handle_get_spend_by_agent
    )
    app.router.add_get(
        "/api/v1/costs/analytics/by-model", handler.handle_get_spend_by_model
    )
    app.router.add_get(
        "/api/v1/costs/analytics/by-debate", handler.handle_get_spend_by_debate
    )
    app.router.add_get(
        "/api/v1/costs/analytics/budget-utilization",
        handler.handle_get_budget_utilization,
    )
    app.router.add_get(
        "/api/costs/analytics/trend", handler.handle_get_spend_trend
    )
    app.router.add_get(
        "/api/costs/analytics/by-agent", handler.handle_get_spend_by_agent
    )
    app.router.add_get(
        "/api/costs/analytics/by-model", handler.handle_get_spend_by_model
    )
    app.router.add_get(
        "/api/costs/analytics/by-debate", handler.handle_get_spend_by_debate
    )
    app.router.add_get(
        "/api/costs/analytics/budget-utilization",
        handler.handle_get_budget_utilization,
    )
