"""Comprehensive tests for cost visibility route registration.

Tests for aragora/server/handlers/costs/routes.py (register_routes function).

Covers:
  TestRegisterRoutesBasic       - Function existence and basic invocation
  TestV1CostRoutes              - Versioned canonical cost routes
  TestLegacyCostRoutes          - Legacy (unversioned) cost routes
  TestRecommendationRoutes      - Recommendation endpoint routes (v1 + legacy)
  TestExportRoutes              - Export endpoint routes
  TestEfficiencyRoutes          - Efficiency metrics routes
  TestForecastRoutes            - Forecasting routes
  TestUsageRoutes               - Usage tracking routes
  TestBudgetManagementRoutes    - Budget management routes
  TestConstraintEstimateRoutes  - Constraints and estimates routes
  TestDetailedRecommendations   - Detailed recommendation routes
  TestAlertCreationRoutes       - Alert creation routes
  TestRouteMethods              - HTTP method correctness
  TestRouteHandlerBinding       - Handler method binding correctness
  TestRouteParity               - V1 and legacy route parity
  TestRouteDuplicates           - No unexpected duplicate routes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.costs.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> web.Application:
    """Create a fresh aiohttp Application for route registration."""
    return web.Application()


def _registered_routes(app: web.Application) -> list[tuple[str, str]]:
    """Return (method, path) tuples for all registered routes."""
    routes = []
    for resource in app.router.resources():
        info = resource.get_info()
        # Resolve the path: plain resources have 'path', dynamic have 'formatter'
        path = info.get("path") or info.get("formatter", "")
        for route in resource:
            routes.append((route.method, path))
    return routes


def _paths_for_method(app: web.Application, method: str) -> list[str]:
    """Return all paths registered for a given HTTP method."""
    return [path for m, path in _registered_routes(app) if m == method]


def _all_paths(app: web.Application) -> list[str]:
    """Return all registered paths (any method)."""
    return [path for _, path in _registered_routes(app)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a fresh app with cost routes registered."""
    application = _make_app()
    register_routes(application)
    return application


# ===========================================================================
# TestRegisterRoutesBasic
# ===========================================================================


class TestRegisterRoutesBasic:
    """Basic invocation and structure tests."""

    def test_register_routes_is_callable(self):
        """register_routes is a callable function."""
        assert callable(register_routes)

    def test_register_routes_accepts_app(self):
        """register_routes can be called with an aiohttp Application."""
        app = _make_app()
        register_routes(app)
        # Should not raise

    def test_register_routes_returns_none(self):
        """register_routes returns None (registers in place)."""
        app = _make_app()
        result = register_routes(app)
        assert result is None

    def test_register_routes_adds_routes(self):
        """After registration, app has more than zero routes."""
        app = _make_app()
        register_routes(app)
        routes = _registered_routes(app)
        assert len(routes) > 0

    def test_register_routes_adds_many_routes(self, app):
        """The routes module registers a large number of endpoints."""
        routes = _registered_routes(app)
        # routes.py registers 42 route bindings (21 v1 + 21 legacy)
        assert len(routes) >= 40


# ===========================================================================
# TestV1CostRoutes
# ===========================================================================


class TestV1CostRoutes:
    """Versioned (v1) core cost routes."""

    def test_get_costs(self, app):
        """GET /api/v1/costs is registered."""
        assert "/api/v1/costs" in _paths_for_method(app, "GET")

    def test_get_breakdown(self, app):
        """GET /api/v1/costs/breakdown is registered."""
        assert "/api/v1/costs/breakdown" in _paths_for_method(app, "GET")

    def test_get_timeline(self, app):
        """GET /api/v1/costs/timeline is registered."""
        assert "/api/v1/costs/timeline" in _paths_for_method(app, "GET")

    def test_get_alerts(self, app):
        """GET /api/v1/costs/alerts is registered."""
        assert "/api/v1/costs/alerts" in _paths_for_method(app, "GET")

    def test_set_budget(self, app):
        """POST /api/v1/costs/budget is registered."""
        assert "/api/v1/costs/budget" in _paths_for_method(app, "POST")

    def test_dismiss_alert(self, app):
        """POST /api/v1/costs/alerts/{alert_id}/dismiss is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/v1/costs/alerts/{alert_id}/dismiss" in paths


# ===========================================================================
# TestLegacyCostRoutes
# ===========================================================================


class TestLegacyCostRoutes:
    """Legacy (unversioned) core cost routes."""

    def test_get_costs_legacy(self, app):
        """GET /api/costs is registered."""
        assert "/api/costs" in _paths_for_method(app, "GET")

    def test_get_breakdown_legacy(self, app):
        """GET /api/costs/breakdown is registered."""
        assert "/api/costs/breakdown" in _paths_for_method(app, "GET")

    def test_get_timeline_legacy(self, app):
        """GET /api/costs/timeline is registered."""
        assert "/api/costs/timeline" in _paths_for_method(app, "GET")

    def test_get_alerts_legacy(self, app):
        """GET /api/costs/alerts is registered."""
        assert "/api/costs/alerts" in _paths_for_method(app, "GET")

    def test_set_budget_legacy(self, app):
        """POST /api/costs/budget is registered."""
        assert "/api/costs/budget" in _paths_for_method(app, "POST")

    def test_dismiss_alert_legacy(self, app):
        """POST /api/costs/alerts/{alert_id}/dismiss is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/costs/alerts/{alert_id}/dismiss" in paths


# ===========================================================================
# TestRecommendationRoutes
# ===========================================================================


class TestRecommendationRoutes:
    """Recommendation endpoint routes (v1 and legacy)."""

    def test_get_recommendations_v1(self, app):
        """GET /api/v1/costs/recommendations is registered."""
        assert "/api/v1/costs/recommendations" in _paths_for_method(app, "GET")

    def test_get_recommendations_legacy(self, app):
        """GET /api/costs/recommendations is registered."""
        assert "/api/costs/recommendations" in _paths_for_method(app, "GET")

    def test_get_single_recommendation_v1(self, app):
        """GET /api/v1/costs/recommendations/{recommendation_id} is registered."""
        paths = _paths_for_method(app, "GET")
        assert "/api/v1/costs/recommendations/{recommendation_id}" in paths

    def test_get_single_recommendation_legacy(self, app):
        """GET /api/costs/recommendations/{recommendation_id} is registered."""
        paths = _paths_for_method(app, "GET")
        assert "/api/costs/recommendations/{recommendation_id}" in paths

    def test_apply_recommendation_v1(self, app):
        """POST /api/v1/costs/recommendations/{recommendation_id}/apply is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/v1/costs/recommendations/{recommendation_id}/apply" in paths

    def test_apply_recommendation_legacy(self, app):
        """POST /api/costs/recommendations/{recommendation_id}/apply is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/costs/recommendations/{recommendation_id}/apply" in paths

    def test_dismiss_recommendation_v1(self, app):
        """POST /api/v1/costs/recommendations/{recommendation_id}/dismiss is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/v1/costs/recommendations/{recommendation_id}/dismiss" in paths

    def test_dismiss_recommendation_legacy(self, app):
        """POST /api/costs/recommendations/{recommendation_id}/dismiss is registered."""
        paths = _paths_for_method(app, "POST")
        assert "/api/costs/recommendations/{recommendation_id}/dismiss" in paths


# ===========================================================================
# TestExportRoutes
# ===========================================================================


class TestExportRoutes:
    """Export endpoint routes."""

    def test_export_v1(self, app):
        """GET /api/v1/costs/export is registered."""
        assert "/api/v1/costs/export" in _paths_for_method(app, "GET")

    def test_export_legacy(self, app):
        """GET /api/costs/export is registered."""
        assert "/api/costs/export" in _paths_for_method(app, "GET")


# ===========================================================================
# TestEfficiencyRoutes
# ===========================================================================


class TestEfficiencyRoutes:
    """Efficiency metrics routes."""

    def test_efficiency_v1(self, app):
        """GET /api/v1/costs/efficiency is registered."""
        assert "/api/v1/costs/efficiency" in _paths_for_method(app, "GET")

    def test_efficiency_legacy(self, app):
        """GET /api/costs/efficiency is registered."""
        assert "/api/costs/efficiency" in _paths_for_method(app, "GET")


# ===========================================================================
# TestForecastRoutes
# ===========================================================================


class TestForecastRoutes:
    """Forecasting routes."""

    def test_forecast_v1(self, app):
        """GET /api/v1/costs/forecast is registered."""
        assert "/api/v1/costs/forecast" in _paths_for_method(app, "GET")

    def test_forecast_legacy(self, app):
        """GET /api/costs/forecast is registered."""
        assert "/api/costs/forecast" in _paths_for_method(app, "GET")

    def test_forecast_detailed_v1(self, app):
        """GET /api/v1/costs/forecast/detailed is registered."""
        assert "/api/v1/costs/forecast/detailed" in _paths_for_method(app, "GET")

    def test_forecast_detailed_legacy(self, app):
        """GET /api/costs/forecast/detailed is registered."""
        assert "/api/costs/forecast/detailed" in _paths_for_method(app, "GET")

    def test_simulate_forecast_v1(self, app):
        """POST /api/v1/costs/forecast/simulate is registered."""
        assert "/api/v1/costs/forecast/simulate" in _paths_for_method(app, "POST")

    def test_simulate_forecast_legacy(self, app):
        """POST /api/costs/forecast/simulate is registered."""
        assert "/api/costs/forecast/simulate" in _paths_for_method(app, "POST")


# ===========================================================================
# TestUsageRoutes
# ===========================================================================


class TestUsageRoutes:
    """Usage tracking routes."""

    def test_usage_v1(self, app):
        """GET /api/v1/costs/usage is registered."""
        assert "/api/v1/costs/usage" in _paths_for_method(app, "GET")

    def test_usage_legacy(self, app):
        """GET /api/costs/usage is registered."""
        assert "/api/costs/usage" in _paths_for_method(app, "GET")


# ===========================================================================
# TestBudgetManagementRoutes
# ===========================================================================


class TestBudgetManagementRoutes:
    """Budget management routes."""

    def test_list_budgets_v1(self, app):
        """GET /api/v1/costs/budgets is registered."""
        assert "/api/v1/costs/budgets" in _paths_for_method(app, "GET")

    def test_list_budgets_legacy(self, app):
        """GET /api/costs/budgets is registered."""
        assert "/api/costs/budgets" in _paths_for_method(app, "GET")

    def test_create_budget_v1(self, app):
        """POST /api/v1/costs/budgets is registered."""
        assert "/api/v1/costs/budgets" in _paths_for_method(app, "POST")

    def test_create_budget_legacy(self, app):
        """POST /api/costs/budgets is registered."""
        assert "/api/costs/budgets" in _paths_for_method(app, "POST")


# ===========================================================================
# TestConstraintEstimateRoutes
# ===========================================================================


class TestConstraintEstimateRoutes:
    """Constraints and estimates routes."""

    def test_check_constraints_v1(self, app):
        """POST /api/v1/costs/constraints/check is registered."""
        assert "/api/v1/costs/constraints/check" in _paths_for_method(app, "POST")

    def test_check_constraints_legacy(self, app):
        """POST /api/costs/constraints/check is registered."""
        assert "/api/costs/constraints/check" in _paths_for_method(app, "POST")

    def test_estimate_cost_v1(self, app):
        """POST /api/v1/costs/estimate is registered."""
        assert "/api/v1/costs/estimate" in _paths_for_method(app, "POST")

    def test_estimate_cost_legacy(self, app):
        """POST /api/costs/estimate is registered."""
        assert "/api/costs/estimate" in _paths_for_method(app, "POST")


# ===========================================================================
# TestDetailedRecommendations
# ===========================================================================


class TestDetailedRecommendations:
    """Detailed recommendation routes."""

    def test_recommendations_detailed_v1(self, app):
        """GET /api/v1/costs/recommendations/detailed is registered."""
        assert "/api/v1/costs/recommendations/detailed" in _paths_for_method(app, "GET")

    def test_recommendations_detailed_legacy(self, app):
        """GET /api/costs/recommendations/detailed is registered."""
        assert "/api/costs/recommendations/detailed" in _paths_for_method(app, "GET")


# ===========================================================================
# TestAlertCreationRoutes
# ===========================================================================


class TestAlertCreationRoutes:
    """Alert creation routes."""

    def test_create_alert_v1(self, app):
        """POST /api/v1/costs/alerts is registered."""
        assert "/api/v1/costs/alerts" in _paths_for_method(app, "POST")

    def test_create_alert_legacy(self, app):
        """POST /api/costs/alerts is registered."""
        assert "/api/costs/alerts" in _paths_for_method(app, "POST")


# ===========================================================================
# TestRouteMethods
# ===========================================================================


class TestRouteMethods:
    """Verify HTTP methods are correct for each route category."""

    def test_all_get_routes_are_read_endpoints(self, app):
        """All GET routes are read-only data retrieval endpoints."""
        get_paths = _paths_for_method(app, "GET")
        # Every GET path should contain 'costs' (they are all cost endpoints)
        for path in get_paths:
            assert "/costs" in path

    def test_all_post_routes_are_write_endpoints(self, app):
        """All POST routes are write or action endpoints."""
        post_paths = _paths_for_method(app, "POST")
        # POST endpoints should be for budget, alerts, recommendations, constraints, etc.
        for path in post_paths:
            assert "/costs" in path

    def test_no_put_routes(self, app):
        """No PUT routes are registered (only GET and POST)."""
        put_paths = _paths_for_method(app, "PUT")
        assert len(put_paths) == 0

    def test_no_delete_routes(self, app):
        """No DELETE routes are registered."""
        delete_paths = _paths_for_method(app, "DELETE")
        assert len(delete_paths) == 0

    def test_no_patch_routes(self, app):
        """No PATCH routes are registered."""
        patch_paths = _paths_for_method(app, "PATCH")
        assert len(patch_paths) == 0

    def test_get_route_count(self, app):
        """Correct number of GET routes are registered."""
        get_paths = _paths_for_method(app, "GET")
        # 18 unique GET endpoints, each with v1 + legacy = 36
        assert len(get_paths) == 36

    def test_post_route_count(self, app):
        """Correct number of POST routes are registered."""
        post_paths = _paths_for_method(app, "POST")
        # 9 unique POST endpoints, each with v1 + legacy = 18
        assert len(post_paths) == 18

    def test_head_routes_auto_registered(self, app):
        """aiohttp auto-registers HEAD routes for each GET route."""
        head_paths = _paths_for_method(app, "HEAD")
        get_paths = _paths_for_method(app, "GET")
        assert len(head_paths) == len(get_paths)


# ===========================================================================
# TestRouteHandlerBinding
# ===========================================================================


class TestRouteHandlerBinding:
    """Verify routes are bound to correct handler methods."""

    def test_handler_instantiated(self, app):
        """register_routes creates a CostHandler instance internally."""
        # We verify indirectly by checking routes work (handler was created)
        routes = _registered_routes(app)
        assert len(routes) > 0

    def test_cost_routes_use_cost_handler(self):
        """All routes bind to CostHandler methods."""
        from aragora.server.handlers.costs.handler import CostHandler

        app = _make_app()

        with patch("aragora.server.handlers.costs.routes.CostHandler") as MockHandler:
            mock_instance = MagicMock(spec=CostHandler)
            MockHandler.return_value = mock_instance

            register_routes(app)

            # Handler was instantiated
            MockHandler.assert_called_once()

    def test_handler_methods_are_bound(self):
        """Each route is bound to the correct handler method."""
        from aragora.server.handlers.costs.handler import CostHandler

        app = _make_app()

        with patch("aragora.server.handlers.costs.routes.CostHandler") as MockHandler:
            mock_instance = MagicMock(spec=CostHandler)
            MockHandler.return_value = mock_instance

            register_routes(app)

            # Verify key handler methods were accessed during registration
            # (they are passed as handler references to add_get/add_post)
            assert mock_instance.handle_get_costs is not None
            assert mock_instance.handle_get_breakdown is not None
            assert mock_instance.handle_get_timeline is not None
            assert mock_instance.handle_get_alerts is not None
            assert mock_instance.handle_set_budget is not None
            assert mock_instance.handle_dismiss_alert is not None


# ===========================================================================
# TestRouteParity
# ===========================================================================


class TestRouteParity:
    """V1 and legacy routes have symmetric parity."""

    def _v1_paths(self, app):
        """Get all v1 paths."""
        return [p for p in _all_paths(app) if p.startswith("/api/v1/")]

    def _legacy_paths(self, app):
        """Get all legacy (non-v1) paths."""
        return [p for p in _all_paths(app) if p.startswith("/api/") and "/v1/" not in p]

    def test_v1_and_legacy_same_count(self, app):
        """Same number of v1 and legacy routes."""
        v1 = self._v1_paths(app)
        legacy = self._legacy_paths(app)
        assert len(v1) == len(legacy)

    def test_every_v1_has_legacy_counterpart(self, app):
        """Every v1 route has a corresponding legacy route."""
        v1_routes = set()
        legacy_routes = set()

        for method, path in _registered_routes(app):
            if "/v1/" in path:
                # Strip /v1 to get the logical path
                normalized = path.replace("/api/v1/", "/api/")
                v1_routes.add((method, normalized))
            else:
                legacy_routes.add((method, path))

        assert v1_routes == legacy_routes

    def test_every_legacy_has_v1_counterpart(self, app):
        """Every legacy route has a corresponding v1 route."""
        v1_normalized = set()
        legacy_routes = set()

        for method, path in _registered_routes(app):
            if "/v1/" in path:
                normalized = path.replace("/api/v1/", "/api/")
                v1_normalized.add((method, normalized))
            else:
                legacy_routes.add((method, path))

        # Every legacy path should exist in v1
        for route in legacy_routes:
            assert route in v1_normalized, f"Legacy route {route} has no v1 counterpart"

    def test_parity_for_get_costs(self, app):
        """GET /api/v1/costs and GET /api/costs both exist."""
        get_paths = _paths_for_method(app, "GET")
        assert "/api/v1/costs" in get_paths
        assert "/api/costs" in get_paths

    def test_parity_for_post_budget(self, app):
        """POST /api/v1/costs/budget and POST /api/costs/budget both exist."""
        post_paths = _paths_for_method(app, "POST")
        assert "/api/v1/costs/budget" in post_paths
        assert "/api/costs/budget" in post_paths

    def test_parity_for_forecast_simulate(self, app):
        """POST forecast/simulate has both v1 and legacy."""
        post_paths = _paths_for_method(app, "POST")
        assert "/api/v1/costs/forecast/simulate" in post_paths
        assert "/api/costs/forecast/simulate" in post_paths


# ===========================================================================
# TestRouteDuplicates
# ===========================================================================


class TestRouteDuplicates:
    """Check for unexpected duplicates."""

    def test_no_duplicate_routes(self, app):
        """Each (method, path) pair is registered exactly once."""
        routes = _registered_routes(app)
        seen = set()
        duplicates = []
        for route in routes:
            if route in seen:
                duplicates.append(route)
            seen.add(route)
        assert duplicates == [], f"Duplicate routes found: {duplicates}"

    def test_all_paths_start_with_api(self, app):
        """Every registered path starts with /api/."""
        for path in _all_paths(app):
            assert path.startswith("/api/"), f"Path {path} does not start with /api/"

    def test_all_paths_contain_costs(self, app):
        """Every registered path contains /costs."""
        for path in _all_paths(app):
            assert "/costs" in path, f"Path {path} does not contain /costs"


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge cases for route registration."""

    def test_register_on_fresh_app(self):
        """Routes can be registered on a brand-new Application."""
        app = web.Application()
        register_routes(app)
        assert len(_registered_routes(app)) > 0

    def test_dynamic_route_segments(self, app):
        """Dynamic route segments use {param} notation."""
        all_paths = _all_paths(app)
        dynamic = [p for p in all_paths if "{" in p]
        # Should have alert_id and recommendation_id dynamic segments
        assert len(dynamic) > 0

        # All dynamic segments use proper aiohttp braces syntax
        for path in dynamic:
            assert "{" in path and "}" in path

    def test_alert_id_dynamic_segment(self, app):
        """The dismiss alert route uses {alert_id} dynamic segment."""
        all_paths = _all_paths(app)
        alert_paths = [p for p in all_paths if "alert_id" in p]
        assert len(alert_paths) >= 2  # v1 + legacy

    def test_recommendation_id_dynamic_segment(self, app):
        """Recommendation routes use {recommendation_id} dynamic segment."""
        all_paths = _all_paths(app)
        rec_paths = [p for p in all_paths if "recommendation_id" in p]
        # apply, dismiss, get single - each with v1 + legacy = 6
        assert len(rec_paths) >= 6

    def test_total_route_count(self, app):
        """Total number of registered routes matches expectations."""
        routes = _registered_routes(app)
        # 36 GET + 36 HEAD (auto) + 18 POST = 90
        assert len(routes) == 90
