"""Comprehensive tests for AnalyticsDashboardHandler (handler.py).

Tests the main handler class that composes all mixins and routes requests:
- __init__ / constructor
- _get_auth_context
- _check_permission (RBAC inline checks)
- ROUTES list completeness
- can_handle() for all routes + version prefix + unknown paths
- handle() routing to every endpoint method
- handle() stub response path (no auth / no workspace_id)
- handle() RBAC permission checks on protected endpoints
- handle() returns None for unknown routes
- Version prefix normalization (v1, v2)
- Edge cases and error paths
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeUserCtx:
    user_id: str = "test_user"
    org_id: str = "org_123"
    role: str = "admin"
    is_authenticated: bool = True
    error_reason: str | None = None


@dataclass
class FakeUserInfo:
    """Fake user info returned by extract_user_from_request."""

    user_id: str = "user_001"
    org_id: str = "org_001"
    role: str = "admin"


@dataclass
class FakeDecision:
    """Fake RBAC decision result."""

    allowed: bool = True
    reason: str | None = None


def _body(result) -> dict:
    """Extract decoded JSON body from a HandlerResult (tuple-style unpacking)."""
    body, _status, _headers = result
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    _body_val, status, _headers = result
    return status


def _make_handler(ctx=None) -> AnalyticsDashboardHandler:
    """Create a fresh AnalyticsDashboardHandler instance."""
    return AnalyticsDashboardHandler(ctx=ctx)


def _mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler with minimal auth."""
    h = MagicMock()
    h.headers = {"Authorization": "Bearer test-token", "Content-Length": "0"}
    h.command = "GET"
    h.path = "/api/analytics/summary"
    return h


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_auth():
    """Patch auth to always succeed for all tests."""
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=FakeUserCtx(),
    ):
        yield


@pytest.fixture
def handler():
    return _make_handler()


@pytest.fixture
def mock_http():
    return _mock_http_handler()


# =========================================================================
# 1. CONSTRUCTOR / __init__
# =========================================================================


class TestConstructor:
    """Tests for AnalyticsDashboardHandler.__init__."""

    def test_default_ctx_is_empty_dict(self):
        h = AnalyticsDashboardHandler()
        assert h.ctx == {}

    def test_ctx_none_becomes_empty_dict(self):
        h = AnalyticsDashboardHandler(ctx=None)
        assert h.ctx == {}

    def test_ctx_preserved(self):
        ctx = {"key": "value", "debug": True}
        h = AnalyticsDashboardHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_ctx_is_same_object(self):
        ctx = {"key": "value"}
        h = AnalyticsDashboardHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_empty_ctx(self):
        h = AnalyticsDashboardHandler(ctx={})
        assert h.ctx == {}


# =========================================================================
# 2. ROUTES LIST
# =========================================================================


class TestRoutesList:
    """Tests for the ROUTES class attribute."""

    def test_routes_is_list(self):
        assert isinstance(AnalyticsDashboardHandler.ROUTES, list)

    def test_routes_count(self):
        """Should have exactly 19 routes."""
        assert len(AnalyticsDashboardHandler.ROUTES) == 19

    def test_routes_all_start_with_api(self):
        for route in AnalyticsDashboardHandler.ROUTES:
            assert route.startswith("/api/analytics/"), (
                f"Route {route!r} does not start with /api/analytics/"
            )

    def test_routes_unique(self):
        assert len(AnalyticsDashboardHandler.ROUTES) == len(set(AnalyticsDashboardHandler.ROUTES))

    # Verify presence of every expected route
    @pytest.mark.parametrize(
        "route",
        [
            "/api/analytics/summary",
            "/api/analytics/trends/findings",
            "/api/analytics/remediation",
            "/api/analytics/agents",
            "/api/analytics/cost",
            "/api/analytics/compliance",
            "/api/analytics/heatmap",
            "/api/analytics/tokens",
            "/api/analytics/tokens/trends",
            "/api/analytics/tokens/providers",
            "/api/analytics/cost/breakdown",
            "/api/analytics/flips/summary",
            "/api/analytics/flips/recent",
            "/api/analytics/flips/consistency",
            "/api/analytics/flips/trends",
            "/api/analytics/deliberations",
            "/api/analytics/deliberations/channels",
            "/api/analytics/deliberations/consensus",
            "/api/analytics/deliberations/performance",
        ],
    )
    def test_route_registered(self, route):
        assert route in AnalyticsDashboardHandler.ROUTES


# =========================================================================
# 3. can_handle() — All routes
# =========================================================================


class TestCanHandle:
    """Tests for can_handle()."""

    @pytest.mark.parametrize("route", AnalyticsDashboardHandler.ROUTES)
    def test_can_handle_all_routes(self, handler, route):
        assert handler.can_handle(route) is True

    @pytest.mark.parametrize("route", AnalyticsDashboardHandler.ROUTES)
    def test_can_handle_versioned_v1(self, handler, route):
        """All routes should work with /api/v1/ prefix."""
        versioned = route.replace("/api/", "/api/v1/")
        assert handler.can_handle(versioned) is True

    @pytest.mark.parametrize("route", AnalyticsDashboardHandler.ROUTES)
    def test_can_handle_versioned_v2(self, handler, route):
        """All routes should work with /api/v2/ prefix."""
        versioned = route.replace("/api/", "/api/v2/")
        assert handler.can_handle(versioned) is True

    def test_cannot_handle_empty(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/") is False

    def test_cannot_handle_unknown(self, handler):
        assert handler.can_handle("/api/analytics/unknown") is False

    def test_cannot_handle_partial(self, handler):
        assert handler.can_handle("/api/analytics") is False

    def test_cannot_handle_extra_suffix(self, handler):
        assert handler.can_handle("/api/analytics/summaryextra") is False

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates/list") is False

    def test_cannot_handle_similar_path(self, handler):
        assert handler.can_handle("/api/analytics/costextra") is False

    def test_cannot_handle_with_trailing_slash(self, handler):
        """Trailing slash changes the path identity."""
        assert handler.can_handle("/api/analytics/summary/") is False


# =========================================================================
# 4. handle() — Routing to stub responses
# =========================================================================


class TestHandleStubResponses:
    """Tests for handle() returning stub responses when no user or no workspace_id."""

    # All routes that should have stub responses (all routes in ANALYTICS_STUB_RESPONSES)
    STUB_ROUTES = [
        ("/api/analytics/summary", "summary"),
        ("/api/analytics/trends/findings", "trends"),
        ("/api/analytics/remediation", "metrics"),
        ("/api/analytics/agents", "agents"),
        ("/api/analytics/cost", "analysis"),
        ("/api/analytics/compliance", "compliance"),
        ("/api/analytics/heatmap", "heatmap"),
        ("/api/analytics/tokens", "summary"),
        ("/api/analytics/tokens/trends", "trends"),
        ("/api/analytics/tokens/providers", "providers"),
        ("/api/analytics/cost/breakdown", "breakdown"),
        ("/api/analytics/flips/summary", "summary"),
        ("/api/analytics/flips/recent", "flips"),
        ("/api/analytics/flips/consistency", "consistency"),
        ("/api/analytics/flips/trends", "trends"),
        ("/api/analytics/deliberations", "summary"),
        ("/api/analytics/deliberations/channels", "channels"),
        ("/api/analytics/deliberations/consensus", "consensus"),
        ("/api/analytics/deliberations/performance", "performance"),
    ]

    @pytest.mark.parametrize("route,key", STUB_ROUTES)
    def test_stub_without_handler(self, handler, route, key):
        """None handler -> stub response for all known routes."""
        result = handler.handle(route, {}, None)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert key in body

    @pytest.mark.parametrize("route,key", STUB_ROUTES)
    def test_stub_without_workspace_id(self, handler, mock_http, route, key):
        """Authenticated but no workspace_id -> stub response."""
        result = handler.handle(route, {}, mock_http)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert key in body

    @pytest.mark.parametrize("route,key", STUB_ROUTES)
    def test_stub_versioned_v1(self, handler, route, key):
        """Versioned v1 path without auth returns stub."""
        versioned = route.replace("/api/", "/api/v1/")
        result = handler.handle(versioned, {}, None)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert key in body


# =========================================================================
# 5. handle() — Routing to actual endpoint methods
# =========================================================================


class TestHandleRoutingToMethods:
    """Tests that handle() routes to the correct internal method."""

    # Basic dashboard analytics (no extra RBAC)
    @patch.object(AnalyticsDashboardHandler, "_get_summary")
    def test_routes_to_get_summary(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/summary", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)
        assert _status(result) == 200

    @patch.object(AnalyticsDashboardHandler, "_get_finding_trends")
    def test_routes_to_get_finding_trends(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/trends/findings", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)
        assert _status(result) == 200

    @patch.object(AnalyticsDashboardHandler, "_get_remediation_metrics")
    def test_routes_to_get_remediation_metrics(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/remediation", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)
        assert _status(result) == 200

    @patch.object(AnalyticsDashboardHandler, "_get_agent_metrics")
    def test_routes_to_get_agent_metrics(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/agents", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)
        assert _status(result) == 200

    @patch.object(AnalyticsDashboardHandler, "_get_risk_heatmap")
    def test_routes_to_get_risk_heatmap(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/heatmap", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)
        assert _status(result) == 200

    # Cost analytics (with RBAC check)
    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_cost_metrics")
    def test_routes_to_get_cost_metrics(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/cost", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_cost_breakdown")
    def test_routes_to_get_cost_breakdown(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/cost/breakdown", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    # Compliance analytics (with RBAC check)
    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_compliance_scorecard")
    def test_routes_to_get_compliance_scorecard(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/compliance", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    # Token usage analytics (with RBAC check)
    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_token_usage")
    def test_routes_to_get_token_usage(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/tokens", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_token_trends")
    def test_routes_to_get_token_trends(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/tokens/trends", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_provider_breakdown")
    def test_routes_to_get_provider_breakdown(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle(
            "/api/analytics/tokens/providers", {"workspace_id": "ws"}, mock_http
        )
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    # Flip analytics (with RBAC check)
    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_flip_summary")
    def test_routes_to_get_flip_summary(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/flips/summary", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_recent_flips")
    def test_routes_to_get_recent_flips(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/flips/recent", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_agent_consistency")
    def test_routes_to_get_agent_consistency(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle(
            "/api/analytics/flips/consistency", {"workspace_id": "ws"}, mock_http
        )
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_flip_trends")
    def test_routes_to_get_flip_trends(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/flips/trends", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    # Deliberation analytics (with RBAC check)
    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_deliberation_summary")
    def test_routes_to_get_deliberation_summary(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle("/api/analytics/deliberations", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_deliberation_by_channel")
    def test_routes_to_get_deliberation_by_channel(
        self, mock_method, mock_perm, handler, mock_http
    ):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle(
            "/api/analytics/deliberations/channels", {"workspace_id": "ws"}, mock_http
        )
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_consensus_rates")
    def test_routes_to_get_consensus_rates(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle(
            "/api/analytics/deliberations/consensus", {"workspace_id": "ws"}, mock_http
        )
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_deliberation_performance")
    def test_routes_to_get_deliberation_performance(
        self, mock_method, mock_perm, handler, mock_http
    ):
        mock_method.return_value = ({"ok": True}, 200, {})
        result = handler.handle(
            "/api/analytics/deliberations/performance", {"workspace_id": "ws"}, mock_http
        )
        mock_method.assert_called_once_with({"workspace_id": "ws"}, mock_http)


# =========================================================================
# 6. handle() — Unknown routes return None
# =========================================================================


class TestHandleUnknownRoutes:
    """Tests that handle() returns None for unrecognized paths."""

    def test_handle_unknown_returns_none(self, handler, mock_http):
        result = handler.handle("/api/unknown", {}, mock_http)
        assert result is None

    def test_handle_empty_returns_none(self, handler, mock_http):
        result = handler.handle("", {}, mock_http)
        assert result is None

    def test_handle_root_returns_none(self, handler, mock_http):
        result = handler.handle("/", {}, mock_http)
        assert result is None

    def test_handle_unrelated_api_returns_none(self, handler, mock_http):
        result = handler.handle("/api/debates/list", {}, mock_http)
        assert result is None

    def test_handle_partial_analytics_returns_none(self, handler, mock_http):
        result = handler.handle("/api/analytics", {}, mock_http)
        assert result is None


# =========================================================================
# 7. handle() — Version prefix handling
# =========================================================================


class TestHandleVersionPrefix:
    """Tests that versioned paths are normalized correctly."""

    @patch.object(AnalyticsDashboardHandler, "_get_summary")
    def test_v1_prefix_routes_summary(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        handler.handle("/api/v1/analytics/summary", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once()

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_cost_metrics")
    def test_v1_prefix_routes_cost(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        handler.handle("/api/v1/analytics/cost", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once()

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_deliberation_summary")
    def test_v2_prefix_routes_deliberations(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        handler.handle("/api/v2/analytics/deliberations", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_called_once()


# =========================================================================
# 8. handle() — RBAC permission checks for protected endpoints
# =========================================================================


class TestHandleRBACChecks:
    """Tests that handle() calls _check_permission for protected endpoints."""

    COST_ROUTES = [
        "/api/analytics/cost",
        "/api/analytics/cost/breakdown",
    ]

    COMPLIANCE_ROUTES = [
        "/api/analytics/compliance",
    ]

    TOKEN_ROUTES = [
        "/api/analytics/tokens",
        "/api/analytics/tokens/trends",
        "/api/analytics/tokens/providers",
    ]

    FLIP_ROUTES = [
        "/api/analytics/flips/summary",
        "/api/analytics/flips/recent",
        "/api/analytics/flips/consistency",
        "/api/analytics/flips/trends",
    ]

    DELIBERATION_ROUTES = [
        "/api/analytics/deliberations",
        "/api/analytics/deliberations/channels",
        "/api/analytics/deliberations/consensus",
        "/api/analytics/deliberations/performance",
    ]

    @pytest.mark.parametrize("route", COST_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_cost_routes_check_cost_permission(self, mock_perm, handler, mock_http, route):
        """Cost routes should check analytics:cost:read."""
        mock_perm.return_value = ({"error": "denied"}, 403, {})
        result = handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_called_once_with(mock_http, "analytics:cost:read")
        assert _status(result) == 403

    @pytest.mark.parametrize("route", COMPLIANCE_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_compliance_routes_check_compliance_permission(
        self, mock_perm, handler, mock_http, route
    ):
        """Compliance routes should check analytics:compliance:read."""
        mock_perm.return_value = ({"error": "denied"}, 403, {})
        result = handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_called_once_with(mock_http, "analytics:compliance:read")
        assert _status(result) == 403

    @pytest.mark.parametrize("route", TOKEN_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_token_routes_check_token_permission(self, mock_perm, handler, mock_http, route):
        """Token routes should check analytics:tokens:read."""
        mock_perm.return_value = ({"error": "denied"}, 403, {})
        result = handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_called_once_with(mock_http, "analytics:tokens:read")
        assert _status(result) == 403

    @pytest.mark.parametrize("route", FLIP_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_flip_routes_check_flips_permission(self, mock_perm, handler, mock_http, route):
        """Flip routes should check analytics:flips:read."""
        mock_perm.return_value = ({"error": "denied"}, 403, {})
        result = handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_called_once_with(mock_http, "analytics:flips:read")
        assert _status(result) == 403

    @pytest.mark.parametrize("route", DELIBERATION_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_deliberation_routes_check_deliberation_permission(
        self, mock_perm, handler, mock_http, route
    ):
        """Deliberation routes should check analytics:deliberations:read."""
        mock_perm.return_value = ({"error": "denied"}, 403, {})
        result = handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_called_once_with(mock_http, "analytics:deliberations:read")
        assert _status(result) == 403

    # Basic routes do NOT go through _check_permission
    BASIC_ROUTES = [
        "/api/analytics/summary",
        "/api/analytics/trends/findings",
        "/api/analytics/remediation",
        "/api/analytics/agents",
        "/api/analytics/heatmap",
    ]

    @pytest.mark.parametrize("route", BASIC_ROUTES)
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_basic_routes_skip_inline_check(self, mock_perm, handler, mock_http, route):
        """Basic dashboard routes should NOT call _check_permission inline."""
        # We need to make the method return something so handle can proceed
        # Use a patch on the underlying method to avoid real execution
        method_name = {
            "/api/analytics/summary": "_get_summary",
            "/api/analytics/trends/findings": "_get_finding_trends",
            "/api/analytics/remediation": "_get_remediation_metrics",
            "/api/analytics/agents": "_get_agent_metrics",
            "/api/analytics/heatmap": "_get_risk_heatmap",
        }[route]
        with patch.object(
            AnalyticsDashboardHandler, method_name, return_value=({"ok": True}, 200, {})
        ):
            handler.handle(route, {"workspace_id": "ws"}, mock_http)
        mock_perm.assert_not_called()


# =========================================================================
# 9. handle() — Permission denied blocks request
# =========================================================================


class TestHandlePermissionDenied:
    """Tests that when _check_permission returns an error, the endpoint method is NOT called."""

    @patch.object(AnalyticsDashboardHandler, "_get_cost_metrics")
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_cost_permission_denied_blocks_method(self, mock_perm, mock_method, handler, mock_http):
        mock_perm.return_value = ({"error": "Permission denied"}, 403, {})
        result = handler.handle("/api/analytics/cost", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_not_called()
        assert _status(result) == 403

    @patch.object(AnalyticsDashboardHandler, "_get_token_usage")
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_token_permission_denied_blocks_method(
        self, mock_perm, mock_method, handler, mock_http
    ):
        mock_perm.return_value = ({"error": "Permission denied"}, 403, {})
        result = handler.handle("/api/analytics/tokens", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_not_called()
        assert _status(result) == 403

    @patch.object(AnalyticsDashboardHandler, "_get_flip_summary")
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_flip_permission_denied_blocks_method(self, mock_perm, mock_method, handler, mock_http):
        mock_perm.return_value = ({"error": "Permission denied"}, 403, {})
        result = handler.handle("/api/analytics/flips/summary", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_not_called()
        assert _status(result) == 403

    @patch.object(AnalyticsDashboardHandler, "_get_deliberation_summary")
    @patch.object(AnalyticsDashboardHandler, "_check_permission")
    def test_deliberation_permission_denied_blocks_method(
        self, mock_perm, mock_method, handler, mock_http
    ):
        mock_perm.return_value = ({"error": "Permission denied"}, 403, {})
        result = handler.handle("/api/analytics/deliberations", {"workspace_id": "ws"}, mock_http)
        mock_method.assert_not_called()
        assert _status(result) == 403


# =========================================================================
# 10. _get_auth_context
# =========================================================================


class TestGetAuthContext:
    """Tests for _get_auth_context method."""

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", False)
    def test_returns_none_when_rbac_unavailable(self):
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request", None)
    def test_returns_none_when_extract_user_is_none(self):
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    def test_returns_none_when_user_info_is_none(self, mock_extract):
        mock_extract.return_value = None
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    @patch("aragora.server.handlers.analytics_dashboard.handler.AuthorizationContext")
    def test_returns_context_with_user_info(self, mock_auth_ctx, mock_extract):
        mock_user = FakeUserInfo(user_id="user_1", org_id="org_1", role="admin")
        mock_extract.return_value = mock_user
        mock_auth_ctx.return_value = "auth_context"
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        mock_auth_ctx.assert_called_once_with(
            user_id="user_1",
            roles={"admin"},
            org_id="org_1",
        )
        assert result == "auth_context"

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    def test_returns_none_on_attribute_error(self, mock_extract):
        mock_extract.side_effect = AttributeError("no attr")
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    def test_returns_none_on_value_error(self, mock_extract):
        mock_extract.side_effect = ValueError("bad value")
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    def test_returns_none_on_type_error(self, mock_extract):
        mock_extract.side_effect = TypeError("bad type")
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    def test_returns_none_on_key_error(self, mock_extract):
        mock_extract.side_effect = KeyError("missing")
        h = _make_handler()
        result = h._get_auth_context(MagicMock())
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    @patch("aragora.server.handlers.analytics_dashboard.handler.AuthorizationContext")
    def test_anonymous_user_id_fallback(self, mock_auth_ctx, mock_extract):
        """When user_id is None, should fall back to 'anonymous'."""
        mock_user = MagicMock()
        mock_user.user_id = None
        mock_user.role = "viewer"
        mock_user.org_id = "org_1"
        mock_extract.return_value = mock_user
        mock_auth_ctx.return_value = "ctx"
        h = _make_handler()
        h._get_auth_context(MagicMock())
        mock_auth_ctx.assert_called_once_with(
            user_id="anonymous",
            roles={"viewer"},
            org_id="org_1",
        )

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.extract_user_from_request")
    @patch("aragora.server.handlers.analytics_dashboard.handler.AuthorizationContext")
    def test_no_role_results_in_empty_roles_set(self, mock_auth_ctx, mock_extract):
        """When role is None/empty, roles set should be empty."""
        mock_user = MagicMock()
        mock_user.user_id = "user_1"
        mock_user.role = None
        mock_user.org_id = "org_1"
        mock_extract.return_value = mock_user
        mock_auth_ctx.return_value = "ctx"
        h = _make_handler()
        h._get_auth_context(MagicMock())
        mock_auth_ctx.assert_called_once_with(
            user_id="user_1",
            roles=set(),
            org_id="org_1",
        )


# =========================================================================
# 11. _check_permission
# =========================================================================


class TestCheckPermission:
    """Tests for _check_permission method."""

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.analytics_dashboard.handler.rbac_fail_closed", return_value=False
    )
    def test_rbac_not_available_not_fail_closed_allows(self, mock_fail, handler):
        """When RBAC is unavailable and fail_closed is False, allow access."""
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is None  # None means allowed

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.analytics_dashboard.handler.rbac_fail_closed", return_value=True
    )
    def test_rbac_not_available_fail_closed_returns_503(self, mock_fail, handler):
        """When RBAC is unavailable and fail_closed is True, return 503."""
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is not None
        assert _status(result) == 503

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    def test_no_auth_context_allows(self, handler):
        """When auth context is None (not configured), allow access."""
        handler._get_auth_context = MagicMock(return_value=None)
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is None

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.check_permission")
    @patch("aragora.server.handlers.analytics_dashboard.handler.record_rbac_check")
    def test_permission_allowed(self, mock_record, mock_check, handler):
        """When permission is allowed, return None and record granted."""
        mock_check.return_value = FakeDecision(allowed=True)
        handler._get_auth_context = MagicMock(return_value=MagicMock(user_id="user_1"))
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is None
        mock_record.assert_called_once_with("analytics:cost:read", granted=True)

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.check_permission")
    @patch("aragora.server.handlers.analytics_dashboard.handler.record_rbac_check")
    def test_permission_denied(self, mock_record, mock_check, handler):
        """When permission is denied, return 403 and record denied."""
        mock_check.return_value = FakeDecision(allowed=False, reason="not admin")
        handler._get_auth_context = MagicMock(return_value=MagicMock(user_id="user_1"))
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is not None
        assert _status(result) == 403
        mock_record.assert_called_once_with("analytics:cost:read", granted=False)

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.check_permission")
    @patch("aragora.server.handlers.analytics_dashboard.handler.record_rbac_check")
    def test_permission_denied_error_exception(self, mock_record, mock_check, handler):
        """When PermissionDeniedError is raised, return 403."""
        from aragora.server.handlers.analytics_dashboard.handler import PermissionDeniedError

        mock_check.side_effect = PermissionDeniedError("forbidden")
        handler._get_auth_context = MagicMock(return_value=MagicMock(user_id="user_1"))
        result = handler._check_permission(MagicMock(), "analytics:cost:read")
        assert result is not None
        assert _status(result) == 403
        mock_record.assert_called_once_with("analytics:cost:read", granted=False)

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.analytics_dashboard.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.analytics_dashboard.handler.check_permission")
    @patch("aragora.server.handlers.analytics_dashboard.handler.record_rbac_check")
    def test_permission_with_resource_id(self, mock_record, mock_check, handler):
        """resource_id should be passed through to check_permission."""
        mock_check.return_value = FakeDecision(allowed=True)
        handler._get_auth_context = MagicMock(return_value=MagicMock(user_id="user_1"))
        handler._check_permission(MagicMock(), "analytics:cost:read", resource_id="ws_123")
        mock_check.assert_called_once()
        args = mock_check.call_args
        assert args[0][1] == "analytics:cost:read"
        assert args[0][2] == "ws_123"


# =========================================================================
# 12. MIXIN COMPOSITION
# =========================================================================


class TestMixinComposition:
    """Tests that all mixin methods are available on the handler."""

    # Debate mixin methods
    def test_has_get_summary(self, handler):
        assert callable(handler._get_summary)

    def test_has_get_finding_trends(self, handler):
        assert callable(handler._get_finding_trends)

    def test_has_get_remediation_metrics(self, handler):
        assert callable(handler._get_remediation_metrics)

    def test_has_get_compliance_scorecard(self, handler):
        assert callable(handler._get_compliance_scorecard)

    def test_has_get_risk_heatmap(self, handler):
        assert callable(handler._get_risk_heatmap)

    # Agent mixin methods
    def test_has_get_agent_metrics(self, handler):
        assert callable(handler._get_agent_metrics)

    def test_has_get_flip_summary(self, handler):
        assert callable(handler._get_flip_summary)

    def test_has_get_recent_flips(self, handler):
        assert callable(handler._get_recent_flips)

    def test_has_get_agent_consistency(self, handler):
        assert callable(handler._get_agent_consistency)

    def test_has_get_flip_trends(self, handler):
        assert callable(handler._get_flip_trends)

    # Usage mixin methods
    def test_has_get_cost_metrics(self, handler):
        assert callable(handler._get_cost_metrics)

    def test_has_get_cost_breakdown(self, handler):
        assert callable(handler._get_cost_breakdown)

    def test_has_get_token_usage(self, handler):
        assert callable(handler._get_token_usage)

    def test_has_get_token_trends(self, handler):
        assert callable(handler._get_token_trends)

    def test_has_get_provider_breakdown(self, handler):
        assert callable(handler._get_provider_breakdown)

    # Deliberation mixin methods
    def test_has_get_deliberation_summary(self, handler):
        assert callable(handler._get_deliberation_summary)

    def test_has_get_deliberation_by_channel(self, handler):
        assert callable(handler._get_deliberation_by_channel)

    def test_has_get_consensus_rates(self, handler):
        assert callable(handler._get_consensus_rates)

    def test_has_get_deliberation_performance(self, handler):
        assert callable(handler._get_deliberation_performance)


# =========================================================================
# 13. INHERITANCE
# =========================================================================


class TestInheritance:
    """Tests for class inheritance chain."""

    def test_inherits_from_base_handler(self):
        from aragora.server.handlers.base import BaseHandler

        assert issubclass(AnalyticsDashboardHandler, BaseHandler)

    def test_inherits_from_debate_mixin(self):
        from aragora.server.handlers.analytics_dashboard.debates import DebateAnalyticsMixin

        assert issubclass(AnalyticsDashboardHandler, DebateAnalyticsMixin)

    def test_inherits_from_agent_mixin(self):
        from aragora.server.handlers.analytics_dashboard.agents import AgentAnalyticsMixin

        assert issubclass(AnalyticsDashboardHandler, AgentAnalyticsMixin)

    def test_inherits_from_usage_mixin(self):
        from aragora.server.handlers.analytics_dashboard.usage import UsageAnalyticsMixin

        assert issubclass(AnalyticsDashboardHandler, UsageAnalyticsMixin)

    def test_inherits_from_deliberation_mixin(self):
        from aragora.server.handlers.analytics_dashboard.endpoints import DeliberationAnalyticsMixin

        assert issubclass(AnalyticsDashboardHandler, DeliberationAnalyticsMixin)


# =========================================================================
# 14. HANDLE QUERY_PARAMS PASSTHROUGH
# =========================================================================


class TestHandleQueryParamsPassthrough:
    """Tests that handle() passes query_params to endpoint methods unchanged."""

    @patch.object(AnalyticsDashboardHandler, "_get_summary")
    def test_query_params_passed_to_summary(self, mock_method, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        params = {"workspace_id": "ws_123", "time_range": "7d", "extra": "value"}
        handler.handle("/api/analytics/summary", params, mock_http)
        mock_method.assert_called_once_with(params, mock_http)

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_token_usage")
    def test_query_params_passed_to_tokens(self, mock_method, mock_perm, handler, mock_http):
        mock_method.return_value = ({"ok": True}, 200, {})
        params = {"workspace_id": "ws_123", "org_id": "org_1", "days": "14"}
        handler.handle("/api/analytics/tokens", params, mock_http)
        mock_method.assert_called_once_with(params, mock_http)


# =========================================================================
# 15. EDGE CASES
# =========================================================================


class TestEdgeCases:
    """Edge cases for the handler."""

    def test_handler_can_be_instantiated_multiple_times(self):
        h1 = _make_handler({"a": 1})
        h2 = _make_handler({"b": 2})
        assert h1.ctx != h2.ctx

    def test_routes_is_class_attribute_not_instance(self):
        """ROUTES should be a class-level attribute."""
        assert "ROUTES" in AnalyticsDashboardHandler.__dict__ or hasattr(
            AnalyticsDashboardHandler, "ROUTES"
        )

    def test_handle_method_exists(self, handler):
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_can_handle_method_exists(self, handler):
        assert hasattr(handler, "can_handle")
        assert callable(handler.can_handle)

    def test_handle_with_none_query_params(self, handler):
        """handle() should work with None handler (returns stub)."""
        result = handler.handle("/api/analytics/summary", {}, None)
        assert result is not None
        assert _status(result) == 200

    def test_handle_with_empty_query_params(self, handler, mock_http):
        """handle() with empty params on stub route returns stub."""
        result = handler.handle("/api/analytics/summary", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch.object(AnalyticsDashboardHandler, "_check_permission", return_value=None)
    @patch.object(AnalyticsDashboardHandler, "_get_flip_trends")
    def test_handle_passes_handler_correctly(self, mock_method, mock_perm, handler, mock_http):
        """The HTTP handler should be passed as the second arg to endpoint methods."""
        mock_method.return_value = ({"ok": True}, 200, {})
        handler.handle("/api/analytics/flips/trends", {"workspace_id": "ws"}, mock_http)
        args = mock_method.call_args[0]
        assert args[1] is mock_http
