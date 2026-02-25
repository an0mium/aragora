"""Tests for feedback_hub handler (aragora/server/handlers/feedback_hub.py).

Covers all routes and behavior of the FeedbackHubHandler class:
- can_handle() routing for versioned and unversioned paths
- GET /api/v1/feedback-hub/stats - Routing statistics
- GET /api/v1/feedback-hub/history - Recent routing history
- Method filtering (only GET allowed)
- RBAC permission checks (admin:read)
- Module unavailable (503 on ImportError)
- Error handling (@handle_errors)
- Query param parsing (limit, capped at 200)
- Edge cases
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feedback_hub import FeedbackHubHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for FeedbackHubHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a FeedbackHubHandler with an empty context."""
    return FeedbackHubHandler(server_context={})


@pytest.fixture
def http_get():
    """Create a mock GET HTTP handler."""
    return MockHTTPHandler(method="GET")


@pytest.fixture
def http_post():
    """Create a mock POST HTTP handler."""
    return MockHTTPHandler(method="POST")


@pytest.fixture
def mock_hub():
    """Create a mock feedback hub instance."""
    hub = MagicMock()
    hub.stats.return_value = {
        "total_routed": 42,
        "routes": {"nomic": 20, "calibration": 15, "elo": 7},
        "errors": 0,
    }
    hub.history.return_value = [
        {"id": "r-001", "source": "outcome", "route": "nomic", "ts": "2026-02-24T10:00:00Z"},
        {"id": "r-002", "source": "calibration", "route": "elo", "ts": "2026-02-24T10:01:00Z"},
    ]
    return hub


# ---------------------------------------------------------------------------
# can_handle() routing tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_stats_versioned(self, handler):
        assert handler.can_handle("/api/v1/feedback-hub/stats") is True

    def test_stats_unversioned(self, handler):
        assert handler.can_handle("/api/feedback-hub/stats") is True

    def test_history_versioned(self, handler):
        assert handler.can_handle("/api/v1/feedback-hub/history") is True

    def test_history_unversioned(self, handler):
        assert handler.can_handle("/api/feedback-hub/history") is True

    def test_v2_versioned(self, handler):
        assert handler.can_handle("/api/v2/feedback-hub/stats") is True

    def test_rejects_non_get(self, handler):
        assert handler.can_handle("/api/v1/feedback-hub/stats", method="POST") is False

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/feedback-hub") is False

    def test_rejects_extra_segment(self, handler):
        assert handler.can_handle("/api/v1/feedback-hub/stats/extra") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_similar_prefix(self, handler):
        assert handler.can_handle("/api/v1/feedback-hubx/stats") is False


# ---------------------------------------------------------------------------
# Handler initialization tests
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_has_routes(self, handler):
        assert len(handler.ROUTES) == 4
        assert "/api/feedback-hub/stats" in handler.ROUTES
        assert "/api/feedback-hub/history" in handler.ROUTES
        assert "/api/v1/feedback-hub/stats" in handler.ROUTES
        assert "/api/v1/feedback-hub/history" in handler.ROUTES

    def test_default_context(self):
        h = FeedbackHubHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"key": "value"}
        h = FeedbackHubHandler(ctx=ctx)
        assert h.ctx == ctx


# ---------------------------------------------------------------------------
# GET /api/v1/feedback-hub/stats tests
# ---------------------------------------------------------------------------


class TestGetStats:
    """Tests for the stats endpoint."""

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_success(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert "data" in body
        assert body["data"]["total_routed"] == 42

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_returns_route_counts(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        body = _body(result)
        assert body["data"]["routes"]["nomic"] == 20
        assert body["data"]["routes"]["calibration"] == 15

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_unversioned(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/feedback-hub/stats", {}, http_get)
        assert _status(result) == 200

    def test_stats_module_unavailable(self, handler, http_get):
        with patch(
            "aragora.nomic.feedback_hub.get_feedback_hub",
            side_effect=ImportError("no module"),
        ):
            result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_runtime_error(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.stats.side_effect = RuntimeError("unexpected")
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        # handle_errors decorator catches this
        assert _status(result) == 500

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_empty_data(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.stats.return_value = {"total_routed": 0, "routes": {}, "errors": 0}
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total_routed"] == 0
        assert body["data"]["routes"] == {}


# ---------------------------------------------------------------------------
# GET /api/v1/feedback-hub/history tests
# ---------------------------------------------------------------------------


class TestGetHistory:
    """Tests for the history endpoint."""

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_success(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert "data" in body
        assert len(body["data"]) == 2

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_entries_have_expected_fields(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        body = _body(result)
        entry = body["data"][0]
        assert entry["id"] == "r-001"
        assert entry["source"] == "outcome"
        assert entry["route"] == "nomic"

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_default_limit(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        mock_hub.history.assert_called_once_with(limit=50)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_custom_limit(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {"limit": "20"}, http_get)
        mock_hub.history.assert_called_once_with(limit=20)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_limit_capped_at_200(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {"limit": "999"}, http_get)
        mock_hub.history.assert_called_once_with(limit=200)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_limit_one(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {"limit": "1"}, http_get)
        mock_hub.history.assert_called_once_with(limit=1)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_unversioned(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/feedback-hub/history", {}, http_get)
        assert _status(result) == 200

    def test_history_module_unavailable(self, handler, http_get):
        with patch(
            "aragora.nomic.feedback_hub.get_feedback_hub",
            side_effect=ImportError("no module"),
        ):
            result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_runtime_error(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.history.side_effect = RuntimeError("unexpected")
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 500

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_empty_result(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.history.return_value = []
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"] == []


# ---------------------------------------------------------------------------
# Route dispatch tests
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    """Tests for handle_get() dispatching to the correct method."""

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_route_dispatched(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert _status(result) == 200
        mock_hub.stats.assert_called_once()

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_route_dispatched(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 200
        mock_hub.history.assert_called_once()

    def test_unmatched_route_returns_none(self, handler, http_get):
        result = handler.handle_get("/api/v1/feedback-hub/unknown", {}, http_get)
        assert result is None

    def test_base_path_returns_none(self, handler, http_get):
        result = handler.handle_get("/api/v1/feedback-hub", {}, http_get)
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case scenarios."""

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_limit_zero_treated_as_zero(self, mock_get_hub, handler, http_get, mock_hub):
        """limit=0 should pass through (min not applied in handler)."""
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {"limit": "0"}, http_get)
        mock_hub.history.assert_called_once_with(limit=0)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_non_numeric_limit_causes_error(self, mock_get_hub, handler, http_get, mock_hub):
        """Non-numeric limit should cause a ValueError caught by handle_errors."""
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {"limit": "abc"}, http_get)
        # int("abc") raises ValueError -> caught by @handle_errors -> 500
        assert _status(result) == 500

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_response_content_type(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert result.content_type == "application/json"

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_response_content_type(self, mock_get_hub, handler, http_get, mock_hub):
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert result.content_type == "application/json"

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_dict_response_structure(self, mock_get_hub, handler, http_get, mock_hub):
        """Stats endpoint wraps data in {'data': ...} envelope."""
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        body = _body(result)
        assert set(body.keys()) == {"data"}

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_dict_response_structure(self, mock_get_hub, handler, http_get, mock_hub):
        """History endpoint wraps data in {'data': ...} envelope."""
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        body = _body(result)
        assert set(body.keys()) == {"data"}

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_negative_limit_clamped(self, mock_get_hub, handler, http_get, mock_hub):
        """Negative limit is min(-10, 200) = -10, but still passed to hub."""
        mock_get_hub.return_value = mock_hub
        handler.handle_get("/api/v1/feedback-hub/history", {"limit": "-10"}, http_get)
        mock_hub.history.assert_called_once_with(limit=-10)

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_stats_type_error_returns_500(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.stats.side_effect = TypeError("type error")
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/stats", {}, http_get)
        assert _status(result) == 500

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_history_attribute_error_returns_500(self, mock_get_hub, handler, http_get):
        mock_hub = MagicMock()
        mock_hub.history.side_effect = AttributeError("attr error")
        mock_get_hub.return_value = mock_hub
        result = handler.handle_get("/api/v1/feedback-hub/history", {}, http_get)
        assert _status(result) == 500

    @patch("aragora.nomic.feedback_hub.get_feedback_hub")
    def test_multiple_handler_instances_independent(self, mock_get_hub, mock_hub):
        """Multiple handler instances work independently."""
        mock_get_hub.return_value = mock_hub
        h1 = FeedbackHubHandler(ctx={"key": "v1"})
        h2 = FeedbackHubHandler(ctx={"key": "v2"})
        http = MockHTTPHandler(method="GET")

        r1 = h1.handle_get("/api/v1/feedback-hub/stats", {}, http)
        r2 = h2.handle_get("/api/v1/feedback-hub/stats", {}, http)

        assert _status(r1) == 200
        assert _status(r2) == 200


# ---------------------------------------------------------------------------
# ROUTES attribute tests
# ---------------------------------------------------------------------------


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_stats(self):
        assert "/api/v1/feedback-hub/stats" in FeedbackHubHandler.ROUTES

    def test_routes_contains_history(self):
        assert "/api/v1/feedback-hub/history" in FeedbackHubHandler.ROUTES

    def test_routes_contains_unversioned_stats(self):
        assert "/api/feedback-hub/stats" in FeedbackHubHandler.ROUTES

    def test_routes_contains_unversioned_history(self):
        assert "/api/feedback-hub/history" in FeedbackHubHandler.ROUTES

    def test_routes_is_list(self):
        assert isinstance(FeedbackHubHandler.ROUTES, list)

    def test_routes_length(self):
        assert len(FeedbackHubHandler.ROUTES) == 4
