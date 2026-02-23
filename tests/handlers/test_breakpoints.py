"""Tests for breakpoints handler (aragora/server/handlers/breakpoints.py).

Covers all routes and behavior of the BreakpointsHandler class:
- can_handle() routing for all breakpoint endpoints
- GET /api/v1/breakpoints - List pending breakpoints
- GET /api/v1/breakpoints/pending - List pending breakpoints (alias)
- GET /api/v1/breakpoints/{id}/status - Get breakpoint status
- GET /api/v1/breakpoints/{id}/resolve - Method not allowed (needs POST)
- POST /api/v1/breakpoints/{id}/resolve - Resolve a breakpoint
- Rate limiting
- Input validation (invalid IDs, missing fields, bad actions)
- Error handling (module unavailable, runtime errors)
- Security (path traversal, injection)
- Edge cases
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.breakpoints import BreakpointsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockTrigger(Enum):
    """Mock trigger enum for breakpoints."""

    LOW_CONFIDENCE = "low_confidence"
    CONSENSUS_FAILURE = "consensus_failure"
    HUMAN_REQUESTED = "human_requested"
    ROUND_LIMIT = "round_limit"


class MockSnapshot:
    """Mock debate snapshot."""

    def __init__(
        self,
        debate_id: str = "dbt-001",
        round_num: int = 3,
        task: str = "Evaluate proposal X",
        current_confidence: float = 0.45,
        agent_names: list[str] | None = None,
    ):
        self.debate_id = debate_id
        self.round_num = round_num
        self.task = task
        self.current_confidence = current_confidence
        self.agent_names = agent_names or ["claude", "gpt4", "gemini"]


class MockBreakpoint:
    """Mock breakpoint object."""

    def __init__(
        self,
        breakpoint_id: str = "bp-001",
        trigger: MockTrigger = MockTrigger.LOW_CONFIDENCE,
        message: str = "Confidence below threshold",
        created_at: str = "2026-01-15T10:30:00Z",
        timeout_minutes: int = 30,
        snapshot: MockSnapshot | None = None,
        status: str = "pending",
        resolved_at: str | None = None,
    ):
        self.breakpoint_id = breakpoint_id
        self.trigger = trigger
        self.message = message
        self.created_at = created_at
        self.timeout_minutes = timeout_minutes
        self.snapshot = snapshot if snapshot is not None else MockSnapshot()
        self.status = status
        self.resolved_at = resolved_at


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """Create a mock breakpoint manager."""
    mgr = MagicMock()
    mgr.get_pending_breakpoints.return_value = []
    mgr.get_breakpoint.return_value = None
    mgr.resolve_breakpoint.return_value = False
    return mgr


@pytest.fixture
def handler(mock_manager):
    """Create a BreakpointsHandler with mocked breakpoint manager."""
    h = BreakpointsHandler()
    h.breakpoint_manager = mock_manager
    return h


@pytest.fixture
def handler_no_manager():
    """Create a BreakpointsHandler with no breakpoint manager (module unavailable)."""
    h = BreakpointsHandler()
    h.breakpoint_manager = None
    return h


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler for rate limiting / request context."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


@pytest.fixture
def sample_breakpoint():
    """Create a sample breakpoint for testing."""
    return MockBreakpoint()


@pytest.fixture
def sample_breakpoints():
    """Create a list of sample breakpoints."""
    return [
        MockBreakpoint(
            breakpoint_id="bp-001",
            trigger=MockTrigger.LOW_CONFIDENCE,
            message="Confidence below threshold",
        ),
        MockBreakpoint(
            breakpoint_id="bp-002",
            trigger=MockTrigger.CONSENSUS_FAILURE,
            message="No consensus reached",
            snapshot=MockSnapshot(debate_id="dbt-002", round_num=5),
        ),
        MockBreakpoint(
            breakpoint_id="bp-003",
            trigger=MockTrigger.HUMAN_REQUESTED,
            message="Human intervention requested",
            snapshot=None,
        ),
    ]


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests to prevent cross-test interference."""
    from aragora.server.handlers.breakpoints import _breakpoints_limiter

    _breakpoints_limiter._requests.clear()
    yield
    _breakpoints_limiter._requests.clear()


# ---------------------------------------------------------------------------
# can_handle() routing tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_breakpoints_root(self, handler):
        assert handler.can_handle("/api/v1/breakpoints") is True

    def test_breakpoints_pending(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/pending") is True

    def test_breakpoint_status(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/status") is True

    def test_breakpoint_resolve(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/resolve") is True

    def test_breakpoint_with_underscore_id(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp_001/status") is True

    def test_breakpoint_with_alphanumeric_id(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/abc123/resolve") is True

    def test_unrelated_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_wrong_version_rejected(self, handler):
        assert handler.can_handle("/api/v2/breakpoints") is False

    def test_unknown_action_rejected(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/delete") is False

    def test_too_deep_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/breakpoints/bp-001/status/extra") is False

    def test_empty_path_rejected(self, handler):
        assert handler.can_handle("") is False

    def test_root_path_rejected(self, handler):
        assert handler.can_handle("/") is False


# ---------------------------------------------------------------------------
# Handler initialization tests
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_has_routes(self, handler):
        assert len(handler.ROUTES) == 2
        assert "/api/v1/breakpoints" in handler.ROUTES
        assert "/api/v1/breakpoints/pending" in handler.ROUTES

    def test_has_breakpoint_pattern(self, handler):
        assert handler.BREAKPOINT_PATTERN is not None

    def test_lazy_load_manager_initially_none(self):
        """Manager starts as None before lazy loading."""
        h = BreakpointsHandler()
        assert h._breakpoint_manager is None
        assert h._breakpoint_manager_loaded is False

    def test_setter_sets_loaded_flag(self):
        """Setting the manager also sets the loaded flag."""
        h = BreakpointsHandler()
        h.breakpoint_manager = MagicMock()
        assert h._breakpoint_manager_loaded is True

    def test_deleter_resets_state(self):
        """Deleting the manager resets both manager and loaded flag."""
        h = BreakpointsHandler()
        h.breakpoint_manager = MagicMock()
        del h.breakpoint_manager
        assert h._breakpoint_manager is None
        assert h._breakpoint_manager_loaded is False

    def test_lazy_load_with_breakpoint_manager_class(self):
        """Manager is lazy-loaded from BreakpointManager class when available."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch(
            "aragora.server.handlers.breakpoints.BreakpointManager",
            mock_cls,
        ):
            h = BreakpointsHandler()
            result = h.breakpoint_manager
            assert result is mock_instance
            mock_cls.assert_called_once()

    def test_lazy_load_when_class_is_none(self):
        """Manager remains None when BreakpointManager class is not available."""
        with patch(
            "aragora.server.handlers.breakpoints.BreakpointManager",
            None,
        ):
            h = BreakpointsHandler()
            result = h.breakpoint_manager
            assert result is None
            assert h._breakpoint_manager_loaded is True

    def test_lazy_load_only_once(self):
        """Manager is only lazy-loaded once (cached after first access)."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch(
            "aragora.server.handlers.breakpoints.BreakpointManager",
            mock_cls,
        ):
            h = BreakpointsHandler()
            _ = h.breakpoint_manager
            _ = h.breakpoint_manager
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/v1/breakpoints - List pending breakpoints
# ---------------------------------------------------------------------------


class TestGetPendingBreakpoints:
    """Tests for listing pending breakpoints."""

    def test_empty_list(self, handler, mock_manager, mock_http_handler):
        mock_manager.get_pending_breakpoints.return_value = []
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["breakpoints"] == []
        assert body["count"] == 0

    def test_with_breakpoints(
        self, handler, mock_manager, mock_http_handler, sample_breakpoints
    ):
        mock_manager.get_pending_breakpoints.return_value = sample_breakpoints
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["breakpoints"]) == 3
        assert body["count"] == 3

    def test_breakpoint_fields(
        self, handler, mock_manager, mock_http_handler, sample_breakpoint
    ):
        mock_manager.get_pending_breakpoints.return_value = [sample_breakpoint]
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        body = _body(result)
        bp = body["breakpoints"][0]
        assert bp["breakpoint_id"] == "bp-001"
        assert bp["trigger"] == "low_confidence"
        assert bp["message"] == "Confidence below threshold"
        assert bp["created_at"] == "2026-01-15T10:30:00Z"
        assert bp["timeout_minutes"] == 30

    def test_breakpoint_with_snapshot(
        self, handler, mock_manager, mock_http_handler, sample_breakpoint
    ):
        mock_manager.get_pending_breakpoints.return_value = [sample_breakpoint]
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        body = _body(result)
        snap = body["breakpoints"][0]["snapshot"]
        assert snap is not None
        assert snap["debate_id"] == "dbt-001"
        assert snap["round_num"] == 3
        assert snap["task"] == "Evaluate proposal X"
        assert snap["confidence"] == 0.45
        assert snap["agents"] == ["claude", "gpt4", "gemini"]

    def test_breakpoint_without_snapshot(
        self, handler, mock_manager, mock_http_handler
    ):
        bp = MockBreakpoint(snapshot=None)
        # Override the default fixture snapshot
        bp.snapshot = None
        mock_manager.get_pending_breakpoints.return_value = [bp]
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        body = _body(result)
        assert body["breakpoints"][0]["snapshot"] is None

    def test_pending_alias(self, handler, mock_manager, mock_http_handler):
        """GET /api/v1/breakpoints/pending returns same result as /breakpoints."""
        mock_manager.get_pending_breakpoints.return_value = []
        result = handler.handle(
            "/api/v1/breakpoints/pending", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["breakpoints"] == []
        assert body["count"] == 0

    def test_module_unavailable(
        self, handler_no_manager, mock_http_handler
    ):
        result = handler_no_manager.handle(
            "/api/v1/breakpoints", {}, mock_http_handler
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_pending_module_unavailable(
        self, handler_no_manager, mock_http_handler
    ):
        result = handler_no_manager.handle(
            "/api/v1/breakpoints/pending", {}, mock_http_handler
        )
        assert _status(result) == 503

    def test_attribute_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_pending_breakpoints.side_effect = AttributeError(
            "mock attribute error"
        )
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        assert _status(result) == 500

    def test_runtime_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_pending_breakpoints.side_effect = RuntimeError(
            "mock runtime error"
        )
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        assert _status(result) == 500

    def test_type_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_pending_breakpoints.side_effect = TypeError(
            "mock type error"
        )
        result = handler.handle("/api/v1/breakpoints", {}, mock_http_handler)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/breakpoints/{id}/status - Get breakpoint status
# ---------------------------------------------------------------------------


class TestGetBreakpointStatus:
    """Tests for getting status of a specific breakpoint."""

    def test_breakpoint_found(
        self, handler, mock_manager, mock_http_handler, sample_breakpoint
    ):
        mock_manager.get_breakpoint.return_value = sample_breakpoint
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["breakpoint_id"] == "bp-001"
        assert body["trigger"] == "low_confidence"
        assert body["message"] == "Confidence below threshold"
        assert body["status"] == "pending"

    def test_breakpoint_not_found(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.return_value = None
        result = handler.handle(
            "/api/v1/breakpoints/bp-999/status", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 404
        assert body["error"] == "Breakpoint not found"
        assert body["breakpoint_id"] == "bp-999"

    def test_status_with_snapshot(
        self, handler, mock_manager, mock_http_handler, sample_breakpoint
    ):
        mock_manager.get_breakpoint.return_value = sample_breakpoint
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        snap = body["snapshot"]
        assert snap is not None
        assert snap["debate_id"] == "dbt-001"
        assert snap["round_num"] == 3
        assert snap["task"] == "Evaluate proposal X"
        assert snap["confidence"] == 0.45

    def test_status_without_snapshot(
        self, handler, mock_manager, mock_http_handler
    ):
        bp = MockBreakpoint(snapshot=None)
        bp.snapshot = None
        mock_manager.get_breakpoint.return_value = bp
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        assert body["snapshot"] is None

    def test_resolved_breakpoint(
        self, handler, mock_manager, mock_http_handler
    ):
        bp = MockBreakpoint(
            status="resolved", resolved_at="2026-01-15T11:00:00Z"
        )
        mock_manager.get_breakpoint.return_value = bp
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        assert body["status"] == "resolved"
        assert body["resolved_at"] == "2026-01-15T11:00:00Z"

    def test_breakpoint_without_status_attr(
        self, handler, mock_manager, mock_http_handler
    ):
        """Breakpoint without status attribute defaults to 'pending'."""
        bp = MockBreakpoint()
        del bp.status
        mock_manager.get_breakpoint.return_value = bp
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        assert body["status"] == "pending"

    def test_breakpoint_without_resolved_at_attr(
        self, handler, mock_manager, mock_http_handler
    ):
        """Breakpoint without resolved_at attribute defaults to None."""
        bp = MockBreakpoint()
        del bp.resolved_at
        mock_manager.get_breakpoint.return_value = bp
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        body = _body(result)
        assert body["resolved_at"] is None

    def test_module_unavailable(
        self, handler_no_manager, mock_http_handler
    ):
        result = handler_no_manager.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        assert _status(result) == 503

    def test_attribute_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.side_effect = AttributeError("mock error")
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        assert _status(result) == 500

    def test_runtime_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.side_effect = RuntimeError("mock error")
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        assert _status(result) == 500

    def test_type_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.side_effect = TypeError("mock error")
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/status", {}, mock_http_handler
        )
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/breakpoints/{id}/resolve - Method not allowed
# ---------------------------------------------------------------------------


class TestResolveMethodNotAllowed:
    """Tests for GET on resolve endpoint (should return 405)."""

    def test_get_resolve_returns_405(
        self, handler, mock_http_handler
    ):
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/resolve", {}, mock_http_handler
        )
        assert _status(result) == 405
        body = _body(result)
        assert "POST" in body["error"]

    def test_get_resolve_includes_allow_header(
        self, handler, mock_http_handler
    ):
        result = handler.handle(
            "/api/v1/breakpoints/bp-001/resolve", {}, mock_http_handler
        )
        assert result.headers.get("Allow") == "POST"


# ---------------------------------------------------------------------------
# POST /api/v1/breakpoints/{id}/resolve - Resolve a breakpoint
# ---------------------------------------------------------------------------


class TestResolveBreakpoint:
    """Tests for resolving a breakpoint via POST."""

    def test_successful_resolve_continue(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue", "message": "Proceed with debate"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/dbt001_bp-001/resolve",
                body,
                mock_http_handler,
            )
        resp = _body(result)
        assert _status(result) == 200
        assert resp["breakpoint_id"] == "dbt001_bp-001"
        assert resp["status"] == "resolved"
        assert resp["action"] == "continue"
        assert resp["message"] == "Proceed with debate"

    def test_successful_resolve_abort(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "abort", "message": "Cancel this debate"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 200
        assert _body(result)["action"] == "abort"

    def test_successful_resolve_redirect(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.return_value = True
        body = {
            "action": "redirect",
            "message": "Change topic",
            "redirect_task": "Evaluate alternative proposal",
        }

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 200
        assert _body(result)["action"] == "redirect"

    def test_successful_resolve_inject(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "inject", "message": "Consider this new evidence"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 200
        assert _body(result)["action"] == "inject"

    def test_missing_action(
        self, handler, mock_manager, mock_http_handler
    ):
        body = {"message": "No action specified"}
        result = handler.handle_post(
            "/api/v1/breakpoints/bp-001/resolve",
            body,
            mock_http_handler,
        )
        assert _status(result) == 400
        assert "action" in _body(result)["error"].lower()

    def test_empty_action(
        self, handler, mock_manager, mock_http_handler
    ):
        body = {"action": "", "message": "Empty action"}
        result = handler.handle_post(
            "/api/v1/breakpoints/bp-001/resolve",
            body,
            mock_http_handler,
        )
        assert _status(result) == 400
        assert "action" in _body(result)["error"].lower()

    def test_invalid_action(
        self, handler, mock_manager, mock_http_handler
    ):
        body = {"action": "destroy", "message": "Bad action"}
        result = handler.handle_post(
            "/api/v1/breakpoints/bp-001/resolve",
            body,
            mock_http_handler,
        )
        assert _status(result) == 400
        body_resp = _body(result)
        assert "destroy" in body_resp["error"]
        assert "continue" in body_resp["error"]

    def test_resolve_not_found(
        self, handler, mock_manager, mock_http_handler
    ):
        """Breakpoint doesn't exist or already resolved."""
        mock_manager.resolve_breakpoint.return_value = False
        body = {"action": "continue", "message": "Go on"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-999/resolve",
                body,
                mock_http_handler,
            )
        body_resp = _body(result)
        assert _status(result) == 404
        assert "Failed to resolve" in body_resp["error"]
        assert body_resp["breakpoint_id"] == "bp-999"

    def test_module_unavailable_no_manager(
        self, handler_no_manager, mock_http_handler
    ):
        body = {"action": "continue", "message": "test"}
        result = handler_no_manager.handle_post(
            "/api/v1/breakpoints/bp-001/resolve",
            body,
            mock_http_handler,
        )
        assert _status(result) == 503

    def test_module_unavailable_guidance_class_none(
        self, handler, mock_manager, mock_http_handler
    ):
        """HumanGuidance class is None (import failed)."""
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance",
            None,
        ):
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 503

    def test_default_message_empty_string(
        self, handler, mock_manager, mock_http_handler
    ):
        """Missing message defaults to empty string."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 200
        assert _body(result)["message"] == ""

    def test_reviewer_id_passed(
        self, handler, mock_manager, mock_http_handler
    ):
        """Custom reviewer_id is passed to HumanGuidance."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {
            "action": "continue",
            "message": "test",
            "reviewer_id": "user-42",
        }

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args
            assert call_kwargs[1]["human_id"] == "user-42"

    def test_default_reviewer_id(
        self, handler, mock_manager, mock_http_handler
    ):
        """Missing reviewer_id defaults to 'api_user'."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args
            assert call_kwargs[1]["human_id"] == "api_user"

    def test_debate_id_extracted_from_breakpoint_id(
        self, handler, mock_manager, mock_http_handler
    ):
        """debate_id is extracted from breakpoint_id when it contains underscore."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/dbt001_bp001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args
            assert call_kwargs[1]["debate_id"] == "dbt001"

    def test_debate_id_empty_when_no_underscore(
        self, handler, mock_manager, mock_http_handler
    ):
        """debate_id is empty string when breakpoint_id has no underscore."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args
            assert call_kwargs[1]["debate_id"] == ""

    def test_redirect_task_passed(
        self, handler, mock_manager, mock_http_handler
    ):
        """redirect_task is passed as preferred_direction."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {
            "action": "redirect",
            "message": "redirect",
            "redirect_task": "New task",
        }

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args
            assert call_kwargs[1]["preferred_direction"] == "New task"

    def test_post_non_matching_path_returns_none(
        self, handler, mock_http_handler
    ):
        """POST to non-matching path returns None."""
        result = handler.handle_post(
            "/api/v1/other/endpoint",
            {"action": "continue"},
            mock_http_handler,
        )
        assert result is None

    def test_post_to_status_returns_none(
        self, handler, mock_http_handler
    ):
        """POST to status action returns None (only resolve is handled)."""
        result = handler.handle_post(
            "/api/v1/breakpoints/bp-001/status",
            {"action": "continue"},
            mock_http_handler,
        )
        assert result is None

    def test_attribute_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.side_effect = AttributeError(
            "mock error"
        )
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 500

    def test_runtime_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.side_effect = RuntimeError(
            "mock error"
        )
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 500

    def test_value_error_returns_500(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.resolve_breakpoint.side_effect = ValueError(
            "mock error"
        )
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 500

    def test_import_error_returns_503(
        self, handler, mock_manager, mock_http_handler
    ):
        """ImportError in resolve is caught and returns 503."""
        body = {"action": "continue", "message": "test"}

        # Simulate ImportError by making HumanGuidance constructor raise ImportError
        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.side_effect = ImportError(
                "No module named 'aragora.debate.breakpoints'"
            )
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# Unknown routes
# ---------------------------------------------------------------------------


class TestUnknownRoutes:
    """Tests for handling unknown routes."""

    def test_handle_returns_none_for_unknown(
        self, handler, mock_http_handler
    ):
        result = handler.handle(
            "/api/v1/other/path", {}, mock_http_handler
        )
        assert result is None

    def test_handle_returns_none_for_base_breakpoints_action(
        self, handler, mock_http_handler
    ):
        """A breakpoint ID without an action returns None (doesn't match pattern)."""
        result = handler.handle(
            "/api/v1/breakpoints/bp-001", {}, mock_http_handler
        )
        assert result is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on breakpoints endpoints."""

    def test_rate_limit_exceeded(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.breakpoints._breakpoints_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = handler.handle(
                "/api/v1/breakpoints", {}, mock_http_handler
            )
        assert _status(result) == 429
        assert "rate limit" in _body(result)["error"].lower()

    def test_rate_limit_allowed(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_pending_breakpoints.return_value = []
        result = handler.handle(
            "/api/v1/breakpoints", {}, mock_http_handler
        )
        assert _status(result) == 200

    def test_rate_limiter_config(self):
        """Rate limiter is configured for 60 requests per minute."""
        from aragora.server.handlers.breakpoints import _breakpoints_limiter

        assert _breakpoints_limiter.rpm == 60


# ---------------------------------------------------------------------------
# Input validation (path segment IDs)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for path segment validation."""

    def test_valid_alphanumeric_id(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.return_value = None
        result = handler.handle(
            "/api/v1/breakpoints/abc123/status", {}, mock_http_handler
        )
        # Valid ID, breakpoint not found
        assert _status(result) == 404

    def test_valid_id_with_hyphens(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.return_value = None
        result = handler.handle(
            "/api/v1/breakpoints/bp-001-test/status", {}, mock_http_handler
        )
        assert _status(result) == 404

    def test_valid_id_with_underscores(
        self, handler, mock_manager, mock_http_handler
    ):
        mock_manager.get_breakpoint.return_value = None
        result = handler.handle(
            "/api/v1/breakpoints/bp_001_test/status", {}, mock_http_handler
        )
        assert _status(result) == 404

    def test_post_invalid_id_with_special_chars(
        self, handler, mock_http_handler
    ):
        """IDs with special characters in POST are rejected by the regex pattern."""
        result = handler.handle_post(
            "/api/v1/breakpoints/bp@001/resolve",
            {"action": "continue"},
            mock_http_handler,
        )
        # The regex won't match, so it returns None
        assert result is None


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


class TestSecurity:
    """Tests for security-related edge cases."""

    def test_path_traversal_rejected(self, handler, mock_http_handler):
        """Path traversal attempts don't match the breakpoint pattern."""
        result = handler.handle(
            "/api/v1/breakpoints/../../etc/passwd/status",
            {},
            mock_http_handler,
        )
        # Dots aren't in the regex, so it won't match
        assert result is None

    def test_null_byte_rejected(self, handler, mock_http_handler):
        """Null bytes in path don't match the breakpoint pattern."""
        result = handler.handle(
            "/api/v1/breakpoints/bp%00001/status",
            {},
            mock_http_handler,
        )
        # Percent-encoded chars aren't in the regex
        assert result is None

    def test_script_injection_rejected(self, handler, mock_http_handler):
        """Script tags in path don't match the breakpoint pattern."""
        result = handler.handle(
            "/api/v1/breakpoints/<script>alert(1)</script>/status",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_sql_injection_rejected(self, handler, mock_http_handler):
        """SQL injection in path doesn't match the breakpoint pattern."""
        result = handler.handle(
            "/api/v1/breakpoints/bp'; DROP TABLE breakpoints;--/status",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_long_id_rejected_by_regex(self, handler, mock_http_handler):
        """Very long IDs are not matched by the regex (alphanum + hyphen/underscore only)."""
        # The BREAKPOINT_PATTERN allows [a-zA-Z0-9_-]+, which has no length limit,
        # but SAFE_ID_PATTERN has a 64-char limit enforced by validate_path_segment.
        long_id = "a" * 100
        result = handler.handle(
            f"/api/v1/breakpoints/{long_id}/status",
            {},
            mock_http_handler,
        )
        # Pattern matches but validate_path_segment rejects (>64 chars)
        assert _status(result) == 400

    def test_long_id_rejected_in_post(
        self, handler, mock_http_handler
    ):
        """Very long IDs are rejected in POST as well."""
        long_id = "a" * 100
        result = handler.handle_post(
            f"/api/v1/breakpoints/{long_id}/resolve",
            {"action": "continue"},
            mock_http_handler,
        )
        assert _status(result) == 400

    def test_error_messages_sanitized(
        self, handler, mock_manager, mock_http_handler
    ):
        """Error messages don't leak internal details."""
        mock_manager.get_pending_breakpoints.side_effect = RuntimeError(
            "/internal/path/to/database.db connection failed"
        )
        result = handler.handle(
            "/api/v1/breakpoints", {}, mock_http_handler
        )
        body_resp = _body(result)
        assert "/internal/path" not in body_resp["error"]

    def test_null_handler_ip(self, handler, mock_manager):
        """Handler with no client_address still works (unknown IP)."""
        mock_http = MagicMock()
        mock_http.client_address = None
        mock_http.headers = {}
        mock_manager.get_pending_breakpoints.return_value = []

        result = handler.handle("/api/v1/breakpoints", {}, mock_http)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_breakpoint_in_list(
        self, handler, mock_manager, mock_http_handler
    ):
        bp = MockBreakpoint()
        mock_manager.get_pending_breakpoints.return_value = [bp]
        result = handler.handle(
            "/api/v1/breakpoints", {}, mock_http_handler
        )
        body = _body(result)
        assert body["count"] == 1
        assert len(body["breakpoints"]) == 1

    def test_many_breakpoints(
        self, handler, mock_manager, mock_http_handler
    ):
        """Handler can return many breakpoints."""
        breakpoints = [
            MockBreakpoint(breakpoint_id=f"bp-{i:03d}")
            for i in range(50)
        ]
        mock_manager.get_pending_breakpoints.return_value = breakpoints
        result = handler.handle(
            "/api/v1/breakpoints", {}, mock_http_handler
        )
        body = _body(result)
        assert body["count"] == 50
        assert len(body["breakpoints"]) == 50

    def test_resolve_with_empty_body(
        self, handler, mock_manager, mock_http_handler
    ):
        """Empty body returns 400 for missing action."""
        result = handler.handle_post(
            "/api/v1/breakpoints/bp-001/resolve",
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    def test_resolve_with_extra_fields(
        self, handler, mock_manager, mock_http_handler
    ):
        """Extra fields in body are ignored gracefully."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {
            "action": "continue",
            "message": "go",
            "extra_field": "ignored",
            "another": 42,
        }

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
        assert _status(result) == 200

    def test_different_triggers(
        self, handler, mock_manager, mock_http_handler
    ):
        """Different trigger types are returned correctly."""
        for trigger in MockTrigger:
            bp = MockBreakpoint(trigger=trigger)
            mock_manager.get_pending_breakpoints.return_value = [bp]
            result = handler.handle(
                "/api/v1/breakpoints", {}, mock_http_handler
            )
            body = _body(result)
            assert body["breakpoints"][0]["trigger"] == trigger.value

    def test_concurrent_handler_instances(self, mock_manager):
        """Multiple handler instances work independently."""
        h1 = BreakpointsHandler()
        h1.breakpoint_manager = mock_manager
        h2 = BreakpointsHandler()
        h2.breakpoint_manager = MagicMock()
        h2.breakpoint_manager.get_pending_breakpoints.return_value = [
            MockBreakpoint()
        ]

        mock_manager.get_pending_breakpoints.return_value = []

        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        mock_http.headers = {}

        r1 = h1.handle("/api/v1/breakpoints", {}, mock_http)
        r2 = h2.handle("/api/v1/breakpoints", {}, mock_http)

        assert _body(r1)["count"] == 0
        assert _body(r2)["count"] == 1

    def test_breakpoint_pattern_regex_groups(self, handler):
        """Regex pattern captures ID and action correctly."""
        match = handler.BREAKPOINT_PATTERN.match(
            "/api/v1/breakpoints/test-id_123/resolve"
        )
        assert match is not None
        assert match.group(1) == "test-id_123"
        assert match.group(2) == "resolve"

        match2 = handler.BREAKPOINT_PATTERN.match(
            "/api/v1/breakpoints/abc/status"
        )
        assert match2 is not None
        assert match2.group(1) == "abc"
        assert match2.group(2) == "status"

    def test_all_valid_actions(
        self, handler, mock_manager, mock_http_handler
    ):
        """All four valid actions are accepted."""
        valid_actions = ["continue", "abort", "redirect", "inject"]
        for action in valid_actions:
            mock_manager.resolve_breakpoint.return_value = True
            body = {"action": action, "message": "test"}

            with patch(
                "aragora.server.handlers.breakpoints.HumanGuidance"
            ) as mock_guidance_cls:
                mock_guidance_cls.return_value = MagicMock()
                result = handler.handle_post(
                    "/api/v1/breakpoints/bp-001/resolve",
                    body,
                    mock_http_handler,
                )
            assert _status(result) == 200, f"Action '{action}' failed"
            assert _body(result)["action"] == action

    def test_guidance_receives_correct_action(
        self, handler, mock_manager, mock_http_handler
    ):
        """HumanGuidance is constructed with the correct action."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "redirect", "message": "go elsewhere"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args[1]
            assert call_kwargs["action"] == "redirect"
            assert call_kwargs["reasoning"] == "go elsewhere"

    def test_guidance_id_is_uuid(
        self, handler, mock_manager, mock_http_handler
    ):
        """HumanGuidance is constructed with a UUID guidance_id."""
        mock_manager.resolve_breakpoint.return_value = True
        body = {"action": "continue", "message": "test"}

        with patch(
            "aragora.server.handlers.breakpoints.HumanGuidance"
        ) as mock_guidance_cls:
            mock_guidance_cls.return_value = MagicMock()
            result = handler.handle_post(
                "/api/v1/breakpoints/bp-001/resolve",
                body,
                mock_http_handler,
            )
            call_kwargs = mock_guidance_cls.call_args[1]
            guid = call_kwargs["guidance_id"]
            # UUID format: 8-4-4-4-12 hex chars
            import re

            assert re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                guid,
            )
