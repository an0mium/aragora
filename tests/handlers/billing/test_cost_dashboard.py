"""Comprehensive tests for cost_dashboard handler.

Covers every route and code path of CostDashboardHandler:
- can_handle() route matching
- GET /api/v1/billing/dashboard (success, query params, errors)
- Rate limiting (429)
- Method not allowed (405)
- Permission checks (billing:read)
- Error branches (ImportError, RuntimeError, OSError, ValueError, KeyError, TypeError)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.billing.cost_dashboard import (
    CostDashboardHandler,
    _dashboard_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler for request simulation."""

    def __init__(
        self,
        body: dict | None = None,
        command: str = "GET",
        query_params: dict | None = None,
    ):
        self.command = command
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""
        self._query_params = query_params or {}

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"

    def get(self, key: str, default=None):
        """Support for get_string_param resolution."""
        return self._query_params.get(key, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a CostDashboardHandler with empty context."""
    return CostDashboardHandler(ctx={})


@pytest.fixture
def handler_with_ctx():
    """Create a CostDashboardHandler with populated context."""
    return CostDashboardHandler(ctx={"workspace_id": "ws-1"})


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the rate limiter between tests to avoid cross-test pollution."""
    _dashboard_limiter._buckets.clear()
    yield
    _dashboard_limiter._buckets.clear()


DASHBOARD_PATH = "/api/v1/billing/dashboard"

SAMPLE_SUMMARY = {
    "current_spend": 42.50,
    "budget_total": 100.00,
    "budget_utilization_pct": 42.5,
    "top_cost_drivers": [
        {"name": "claude-3-opus", "spend": 25.00},
        {"name": "gpt-4", "spend": 12.50},
    ],
    "projected_monthly": 85.00,
    "currency": "USD",
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCostDashboardHandlerInit:
    """Test handler initialization."""

    def test_default_ctx_is_empty_dict(self):
        h = CostDashboardHandler()
        assert h.ctx == {}

    def test_ctx_passed_through(self):
        ctx = {"key": "value"}
        h = CostDashboardHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = CostDashboardHandler(ctx=None)
        assert h.ctx == {}

    def test_resource_type(self):
        h = CostDashboardHandler()
        assert h.RESOURCE_TYPE == "cost_dashboard"

    def test_routes_contains_dashboard(self):
        h = CostDashboardHandler()
        assert DASHBOARD_PATH in h.ROUTES


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test route matching via can_handle."""

    def test_matches_dashboard_path(self, handler):
        assert handler.can_handle(DASHBOARD_PATH) is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/billing/unknown") is False

    def test_rejects_empty_string(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/billing") is False

    def test_rejects_trailing_slash(self, handler):
        assert handler.can_handle(DASHBOARD_PATH + "/") is False

    def test_rejects_similar_path(self, handler):
        assert handler.can_handle("/api/v1/billing/dashboards") is False

    def test_rejects_different_version(self, handler):
        assert handler.can_handle("/api/v2/billing/dashboard") is False


# ---------------------------------------------------------------------------
# GET /api/v1/billing/dashboard — success paths
# ---------------------------------------------------------------------------


class TestGetDashboardSuccess:
    """Test successful dashboard retrieval."""

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_basic_success(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = SAMPLE_SUMMARY
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["current_spend"] == 42.50
        assert body["budget_utilization_pct"] == 42.5

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_passes_workspace_id(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"spend": 10}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(
            DASHBOARD_PATH, {"workspace_id": "ws-abc"}, MockHTTPHandler(), method="GET"
        )

        assert _status(result) == 200
        mock_tracker.get_dashboard_summary.assert_called_once_with(
            workspace_id="ws-abc", org_id=None
        )

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_passes_org_id(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"spend": 20}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(
            DASHBOARD_PATH, {"org_id": "org-xyz"}, MockHTTPHandler(), method="GET"
        )

        assert _status(result) == 200
        mock_tracker.get_dashboard_summary.assert_called_once_with(
            workspace_id=None, org_id="org-xyz"
        )

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_passes_both_params(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"spend": 30}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(
            DASHBOARD_PATH,
            {"workspace_id": "ws-1", "org_id": "org-2"},
            MockHTTPHandler(),
            method="GET",
        )

        assert _status(result) == 200
        mock_tracker.get_dashboard_summary.assert_called_once_with(
            workspace_id="ws-1", org_id="org-2"
        )

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_empty_string_params_treated_as_none(self, mock_get_tracker, _mock_ip, handler):
        """Empty string workspace_id and org_id are converted to None."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(
            DASHBOARD_PATH, {"workspace_id": "", "org_id": ""}, MockHTTPHandler(), method="GET"
        )

        assert _status(result) == 200
        mock_tracker.get_dashboard_summary.assert_called_once_with(
            workspace_id=None, org_id=None
        )

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_empty_query_params(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"empty": True}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200
        mock_tracker.get_dashboard_summary.assert_called_once_with(
            workspace_id=None, org_id=None
        )

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_returns_full_summary_structure(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = SAMPLE_SUMMARY
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        body = _body(result)
        assert body["currency"] == "USD"
        assert len(body["top_cost_drivers"]) == 2
        assert body["projected_monthly"] == 85.00


# ---------------------------------------------------------------------------
# GET /api/v1/billing/dashboard — error branches
# ---------------------------------------------------------------------------


class TestGetDashboardErrors:
    """Test error handling in _get_dashboard."""

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_import_error(self, _mock_ip, handler):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("No module named 'aragora.billing.cost_tracker'"),
        ):
            # Patch at module level since the import is inside the function
            result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_runtime_error(self, mock_get_tracker, _mock_ip, handler):
        mock_get_tracker.side_effect = RuntimeError("Database connection failed")

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_os_error(self, mock_get_tracker, _mock_ip, handler):
        mock_get_tracker.side_effect = OSError("Disk full")

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_value_error(self, mock_get_tracker, _mock_ip, handler):
        mock_get_tracker.side_effect = ValueError("Invalid date range")

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_key_error(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.side_effect = KeyError("missing_field")
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_type_error(self, mock_get_tracker, _mock_ip, handler):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.side_effect = TypeError("unhashable type")
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_tracker_get_dashboard_summary_raises_runtime(
        self, mock_get_tracker, _mock_ip, handler
    ):
        """Error from get_dashboard_summary (not get_cost_tracker) is caught."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.side_effect = RuntimeError("query timeout")
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result)["error"]


# ---------------------------------------------------------------------------
# Method not allowed (405)
# ---------------------------------------------------------------------------


class TestMethodNotAllowed:
    """Test that non-GET methods return 405."""

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_post_returns_405(self, _mock_ip, handler):
        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(command="POST"), method="POST")
        assert _status(result) == 405
        assert "Method not allowed" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_put_returns_405(self, _mock_ip, handler):
        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(command="PUT"), method="PUT")
        assert _status(result) == 405

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_delete_returns_405(self, _mock_ip, handler):
        result = handler.handle(
            DASHBOARD_PATH, {}, MockHTTPHandler(command="DELETE"), method="DELETE"
        )
        assert _status(result) == 405

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_patch_returns_405(self, _mock_ip, handler):
        result = handler.handle(
            DASHBOARD_PATH, {}, MockHTTPHandler(command="PATCH"), method="PATCH"
        )
        assert _status(result) == 405

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_handler_command_overrides_method(self, _mock_ip, handler):
        """When handler has .command attribute, that overrides the method kwarg."""
        http = MockHTTPHandler(command="POST")
        result = handler.handle(DASHBOARD_PATH, {}, http, method="GET")
        # handler.command = "POST" overrides method="GET" -> 405
        assert _status(result) == 405


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Test rate limit enforcement."""

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_rate_limit_returns_429(self, mock_ip, handler):
        """When rate limiter denies, handler returns 429."""
        with patch.object(_dashboard_limiter, "is_allowed", return_value=False):
            result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 429
        assert "Rate limit" in _body(result)["error"]

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.2")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_rate_limit_allows_within_limit(self, mock_get_tracker, _mock_ip, handler):
        """Requests within the limit are processed."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"ok": True}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.3")
    def test_rate_limit_per_ip(self, mock_ip, handler):
        """Rate limiting is keyed by client IP."""
        with patch.object(_dashboard_limiter, "is_allowed") as mock_allowed:
            mock_allowed.return_value = False
            handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")
            mock_allowed.assert_called_with("10.0.0.3")


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------


class TestPermissions:
    """Test permission enforcement (billing:read)."""

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    def test_no_auth_returns_error(self, _mock_ip):
        """Without the auto-auth fixture, the require_permission decorator rejects."""
        h = CostDashboardHandler(ctx={})
        result = h.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")
        # The require_permission decorator should block access
        # (returns 401 or 403 depending on implementation)
        assert _status(result) in (401, 403)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_summary_returns_empty_dict(self, mock_get_tracker, _mock_ip, handler):
        """Tracker may return an empty dict — handler should still 200."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200
        assert _body(result) == {}

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_summary_returns_nested_data(self, mock_get_tracker, _mock_ip, handler):
        nested = {"breakdown": {"api": {"cost": 10.0}, "storage": {"cost": 5.0}}}
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = nested
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200
        assert _body(result)["breakdown"]["api"]["cost"] == 10.0

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_summary_with_list_query_param(self, mock_get_tracker, _mock_ip, handler):
        """Query params may arrive as lists (from urllib.parse.parse_qs)."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"ok": True}
        mock_get_tracker.return_value = mock_tracker

        result = handler.handle(
            DASHBOARD_PATH,
            {"workspace_id": ["ws-list"]},
            MockHTTPHandler(),
            method="GET",
        )

        assert _status(result) == 200
        # get_string_param should handle list params gracefully
        mock_tracker.get_dashboard_summary.assert_called_once()

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_handler_without_command_attribute(self, mock_get_tracker, _mock_ip, handler):
        """If handler does not have .command, the method kwarg is used."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"ok": True}
        mock_get_tracker.return_value = mock_tracker

        http = MagicMock(spec=[])  # No attributes at all
        http.client_address = ("127.0.0.1", 12345)
        http.headers = {}

        result = handler.handle(DASHBOARD_PATH, {}, http, method="GET")

        assert _status(result) == 200

    @patch("aragora.server.handlers.billing.cost_dashboard.get_client_ip", return_value="10.0.0.1")
    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_handler_with_ctx(self, mock_get_tracker, _mock_ip, handler_with_ctx):
        """Handler with populated ctx still works."""
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_summary.return_value = {"ws": True}
        mock_get_tracker.return_value = mock_tracker

        result = handler_with_ctx.handle(DASHBOARD_PATH, {}, MockHTTPHandler(), method="GET")

        assert _status(result) == 200

    def test_unknown_path_returns_none_or_405(self, handler):
        """Calling handle() with an unknown path still goes through rate limit then 405."""
        with patch(
            "aragora.server.handlers.billing.cost_dashboard.get_client_ip",
            return_value="10.0.0.1",
        ):
            result = handler.handle("/api/v1/billing/other", {}, MockHTTPHandler(), method="GET")
        # Unknown path doesn't match the route condition -> falls through to 405
        assert _status(result) == 405
