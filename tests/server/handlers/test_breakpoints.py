"""
Tests for aragora.server.handlers.breakpoints - Breakpoints HTTP Handlers.

Tests cover:
- BreakpointsHandler: instantiation, ROUTES, can_handle
- GET /api/v1/breakpoints/pending: success, module unavailable
- GET /api/v1/breakpoints/{id}/status: found, not found, module unavailable
- POST /api/v1/breakpoints/{id}/resolve: success, missing action, invalid action,
  not found, module unavailable, invalid body
- handle routing: rate limiting, resolve via GET returns 405
- handle_post routing: returns None for unmatched paths
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.breakpoints import BreakpointsHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Breakpoint Objects
# ===========================================================================


class MockSnapshot:
    """Mock breakpoint snapshot."""

    def __init__(self):
        self.debate_id = "debate-001"
        self.round_num = 2
        self.task = "Evaluate rate limiter"
        self.current_confidence = 0.75
        self.agent_names = ["claude", "gpt-4"]


class MockTrigger:
    """Mock breakpoint trigger enum value."""

    def __init__(self, value: str = "low_confidence"):
        self.value = value


class MockBreakpoint:
    """Mock breakpoint object."""

    def __init__(
        self,
        breakpoint_id: str = "bp-001",
        status: str = "pending",
        with_snapshot: bool = True,
    ):
        self.breakpoint_id = breakpoint_id
        self.trigger = MockTrigger()
        self.message = "Confidence dropped below threshold"
        self.created_at = "2026-02-14T10:00:00Z"
        self.timeout_minutes = 15
        self.status = status
        self.resolved_at = None
        self.snapshot = MockSnapshot() if with_snapshot else None


class MockBreakpointManager:
    """Mock BreakpointManager."""

    def __init__(self):
        self._breakpoints: dict[str, MockBreakpoint] = {
            "bp-001": MockBreakpoint("bp-001"),
            "bp-002": MockBreakpoint("bp-002"),
        }

    def get_pending_breakpoints(self) -> list[MockBreakpoint]:
        return [bp for bp in self._breakpoints.values() if bp.status == "pending"]

    def get_breakpoint(self, breakpoint_id: str) -> MockBreakpoint | None:
        return self._breakpoints.get(breakpoint_id)

    def resolve_breakpoint(self, breakpoint_id: str, guidance: Any) -> bool:
        bp = self._breakpoints.get(breakpoint_id)
        if bp and bp.status == "pending":
            bp.status = "resolved"
            return True
        return False


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.breakpoints import _breakpoints_limiter

    _breakpoints_limiter._buckets.clear()


@pytest.fixture
def mock_manager():
    """Create a mock breakpoint manager."""
    return MockBreakpointManager()


@pytest.fixture
def handler(mock_manager):
    """Create a BreakpointsHandler with mocked dependencies."""
    with patch(
        "aragora.server.handlers.breakpoints.BreakpointManager",
        return_value=mock_manager,
    ):
        h = BreakpointsHandler()
        h.breakpoint_manager = mock_manager
        yield h


@pytest.fixture
def handler_no_manager():
    """Create a handler with no breakpoint manager available."""
    h = BreakpointsHandler()
    h.breakpoint_manager = None
    return h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestBreakpointsHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, BreakpointsHandler)

    def test_routes_contains_pending(self, handler):
        assert "/api/v1/breakpoints/pending" in handler.ROUTES

    def test_can_handle_pending(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/pending") is True

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/status") is True

    def test_can_handle_resolve(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/resolve") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/api/v1/breakpoints") is False

    def test_cannot_handle_invalid_action(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/delete") is False


# ===========================================================================
# Test GET /api/v1/breakpoints/pending
# ===========================================================================


class TestGetPending:
    """Tests for the pending breakpoints endpoint."""

    def test_get_pending_success(self, handler):
        mock_handler = _make_mock_handler()
        result = handler._get_pending_breakpoints()
        assert result.status_code == 200
        data = _parse_body(result)
        assert "breakpoints" in data
        assert data["count"] == 2
        assert data["breakpoints"][0]["breakpoint_id"] == "bp-001"

    def test_get_pending_includes_snapshot(self, handler):
        result = handler._get_pending_breakpoints()
        data = _parse_body(result)
        bp = data["breakpoints"][0]
        assert bp["snapshot"] is not None
        assert bp["snapshot"]["debate_id"] == "debate-001"
        assert bp["snapshot"]["round_num"] == 2

    def test_get_pending_module_unavailable(self, handler_no_manager):
        result = handler_no_manager._get_pending_breakpoints()
        assert result.status_code == 503

    def test_get_pending_exception(self, handler, mock_manager):
        mock_manager.get_pending_breakpoints = MagicMock(side_effect=RuntimeError("DB down"))
        result = handler._get_pending_breakpoints()
        assert result.status_code == 500


# ===========================================================================
# Test GET /api/v1/breakpoints/{id}/status
# ===========================================================================


class TestGetStatus:
    """Tests for the breakpoint status endpoint."""

    def test_get_status_found(self, handler):
        result = handler._get_breakpoint_status("bp-001")
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["breakpoint_id"] == "bp-001"
        assert data["status"] == "pending"
        assert data["trigger"] == "low_confidence"

    def test_get_status_not_found(self, handler):
        result = handler._get_breakpoint_status("nonexistent")
        assert result.status_code == 404
        data = _parse_body(result)
        assert "error" in data

    def test_get_status_module_unavailable(self, handler_no_manager):
        result = handler_no_manager._get_breakpoint_status("bp-001")
        assert result.status_code == 503

    def test_get_status_includes_snapshot(self, handler):
        result = handler._get_breakpoint_status("bp-001")
        data = _parse_body(result)
        assert data["snapshot"]["debate_id"] == "debate-001"
        assert data["snapshot"]["confidence"] == 0.75

    def test_get_status_exception(self, handler, mock_manager):
        mock_manager.get_breakpoint = MagicMock(side_effect=RuntimeError("DB error"))
        result = handler._get_breakpoint_status("bp-001")
        assert result.status_code == 500


# ===========================================================================
# Test POST /api/v1/breakpoints/{id}/resolve
# ===========================================================================


class TestResolveBreakpoint:
    """Tests for breakpoint resolution endpoint."""

    def test_resolve_success(self, handler):
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {"action": "continue", "message": "Proceed"}
            result = handler._resolve_breakpoint("bp-001", body)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "resolved"
            assert data["action"] == "continue"

    def test_resolve_missing_action(self, handler):
        body = {"message": "No action"}
        result = handler._resolve_breakpoint("bp-001", body)
        assert result.status_code == 400
        data = _parse_body(result)
        assert "action" in data.get("error", "").lower()

    def test_resolve_invalid_action(self, handler):
        body = {"action": "invalid_action"}
        result = handler._resolve_breakpoint("bp-001", body)
        assert result.status_code == 400

    def test_resolve_not_found(self, handler):
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {"action": "abort"}
            result = handler._resolve_breakpoint("nonexistent", body)
            assert result.status_code == 404

    def test_resolve_module_unavailable(self, handler_no_manager):
        body = {"action": "continue"}
        result = handler_no_manager._resolve_breakpoint("bp-001", body)
        assert result.status_code == 503

    def test_resolve_redirect_action(self, handler):
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {
                "action": "redirect",
                "message": "Change direction",
                "redirect_task": "New task",
            }
            result = handler._resolve_breakpoint("bp-001", body)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["action"] == "redirect"

    def test_resolve_inject_action(self, handler):
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {"action": "inject", "message": "Inject data"}
            result = handler._resolve_breakpoint("bp-001", body)
            assert result.status_code == 200

    def test_resolve_exception(self, handler, mock_manager):
        mock_manager.resolve_breakpoint = MagicMock(side_effect=RuntimeError("Resolve failed"))
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {"action": "continue"}
            result = handler._resolve_breakpoint("bp-001", body)
            assert result.status_code == 500


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_pending(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/breakpoints/pending", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_status(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/breakpoints/bp-001/status", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_resolve_via_get_returns_405(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/breakpoints/bp-001/resolve", {}, mock_handler)
        assert result is not None
        assert result.status_code == 405

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/breakpoints/unknown-route", {}, mock_handler)
        assert result is None

    def test_handle_rate_limited(self, handler):
        from aragora.server.handlers.breakpoints import _breakpoints_limiter

        mock_handler = _make_mock_handler()
        with patch.object(_breakpoints_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/breakpoints/pending", {}, mock_handler)
            assert result.status_code == 429


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() method routing."""

    def test_handle_post_resolve(self, handler):
        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_cls:
            mock_cls.return_value = MagicMock()
            body = {"action": "continue", "message": "Go ahead"}
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve", body, _make_mock_handler("POST")
            )
            assert result is not None
            assert result.status_code == 200

    def test_handle_post_unmatched_returns_none(self, handler):
        result = handler.handle_post(
            "/api/v1/breakpoints/something/unknown",
            {},
            _make_mock_handler("POST"),
        )
        assert result is None

    def test_handle_post_no_pattern_match(self, handler):
        result = handler.handle_post(
            "/api/v1/other/path",
            {},
            _make_mock_handler("POST"),
        )
        assert result is None


# ===========================================================================
# Test Lazy Manager Loading
# ===========================================================================


class TestLazyManagerLoading:
    """Tests for the lazy breakpoint manager property."""

    def test_manager_property_setter(self):
        h = BreakpointsHandler()
        mgr = MagicMock()
        h.breakpoint_manager = mgr
        assert h.breakpoint_manager is mgr

    def test_manager_property_deleter(self):
        h = BreakpointsHandler()
        h.breakpoint_manager = MagicMock()
        del h.breakpoint_manager
        assert h._breakpoint_manager is None
        assert h._breakpoint_manager_loaded is False

    def test_manager_lazy_loads_once(self):
        """Manager only attempts load once even if import fails."""
        with patch(
            "aragora.server.handlers.breakpoints.BreakpointManager",
            None,
        ):
            h = BreakpointsHandler()
            # First access tries to load
            result = h.breakpoint_manager
            assert result is None
            # Second access doesn't retry
            assert h._breakpoint_manager_loaded is True
