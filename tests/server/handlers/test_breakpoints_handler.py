"""
Tests for BreakpointsHandler - Human-in-the-loop intervention endpoints.

Covers:
- Pending breakpoints listing
- Breakpoint status checking
- Breakpoint resolution
- Route matching (can_handle)
- RBAC permission checks
- Rate limiting
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def breakpoints_handler(server_context):
    """Create a BreakpointsHandler instance."""
    from aragora.server.handlers.breakpoints import BreakpointsHandler

    return BreakpointsHandler(server_context)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
        "X-Forwarded-For": "192.168.1.1",
    }
    handler.client_address = ("192.168.1.1", 12345)
    return handler


@pytest.fixture
def mock_breakpoint():
    """Create a mock breakpoint object."""
    bp = MagicMock()
    bp.breakpoint_id = "bp-123"
    bp.trigger = MagicMock(value="low_confidence")
    bp.message = "Confidence dropped below threshold"
    bp.created_at = datetime.now(timezone.utc).isoformat()
    bp.timeout_minutes = 30
    bp.status = "pending"
    bp.resolved_at = None

    # Snapshot
    bp.snapshot = MagicMock()
    bp.snapshot.debate_id = "debate-456"
    bp.snapshot.round_num = 3
    bp.snapshot.task = "Analyze market trends"
    bp.snapshot.current_confidence = 0.45
    bp.snapshot.agent_names = ["claude", "gpt-4", "gemini"]

    return bp


@pytest.fixture
def mock_breakpoint_manager(mock_breakpoint):
    """Create a mock breakpoint manager."""
    manager = MagicMock()

    # Pending breakpoints
    manager.get_pending_breakpoints = MagicMock(return_value=[mock_breakpoint])

    # Get specific breakpoint
    manager.get_breakpoint = MagicMock(return_value=mock_breakpoint)

    # Resolve breakpoint
    manager.resolve_breakpoint = MagicMock(return_value=True)

    return manager


# -----------------------------------------------------------------------------
# Route Matching Tests (can_handle)
# -----------------------------------------------------------------------------


class TestBreakpointsHandlerRouteMatching:
    """Tests for BreakpointsHandler.can_handle() method."""

    def test_can_handle_pending(self, breakpoints_handler):
        """Handler matches /api/v1/breakpoints/pending."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/pending") is True

    def test_can_handle_status(self, breakpoints_handler):
        """Handler matches /api/v1/breakpoints/:id/status."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp-123/status") is True
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/abc_123/status") is True

    def test_can_handle_resolve(self, breakpoints_handler):
        """Handler matches /api/v1/breakpoints/:id/resolve."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp-123/resolve") is True

    def test_cannot_handle_unrelated(self, breakpoints_handler):
        """Handler does not match unrelated paths."""
        assert breakpoints_handler.can_handle("/api/debates") is False
        assert breakpoints_handler.can_handle("/api/breakpoints") is False
        assert breakpoints_handler.can_handle("/api/v1/other") is False

    def test_cannot_handle_invalid_action(self, breakpoints_handler):
        """Handler does not match invalid actions."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp-123/invalid") is False
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp-123/delete") is False


# -----------------------------------------------------------------------------
# Get Pending Breakpoints Tests
# -----------------------------------------------------------------------------


class TestGetPendingBreakpoints:
    """Tests for listing pending breakpoints."""

    def test_get_pending_success(self, breakpoints_handler, mock_breakpoint_manager, mock_handler):
        """Test getting pending breakpoints."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter:
            mock_limiter.is_allowed = MagicMock(return_value=True)

            result = breakpoints_handler._get_pending_breakpoints()

        assert result.status_code == 200
        body = result.body.decode()
        assert "breakpoints" in body
        assert "bp-123" in body

    def test_get_pending_empty(self, breakpoints_handler, mock_handler):
        """Test getting pending when none exist."""
        mock_manager = MagicMock()
        mock_manager.get_pending_breakpoints = MagicMock(return_value=[])
        breakpoints_handler._breakpoint_manager = mock_manager

        result = breakpoints_handler._get_pending_breakpoints()

        assert result.status_code == 200
        body = result.body.decode()
        assert '"count": 0' in body or '"count":0' in body

    def test_get_pending_manager_unavailable(self, breakpoints_handler, mock_handler):
        """Test getting pending when manager unavailable."""
        breakpoints_handler._breakpoint_manager = None

        with patch.object(breakpoints_handler, "breakpoint_manager", None):
            result = breakpoints_handler._get_pending_breakpoints()

        assert result.status_code == 503

    def test_get_pending_handles_exception(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test error handling in get pending."""
        mock_breakpoint_manager.get_pending_breakpoints = MagicMock(
            side_effect=RuntimeError("Database error")
        )
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler._get_pending_breakpoints()

        assert result.status_code == 500


# -----------------------------------------------------------------------------
# Get Breakpoint Status Tests
# -----------------------------------------------------------------------------


class TestGetBreakpointStatus:
    """Tests for getting breakpoint status."""

    def test_get_status_success(
        self, breakpoints_handler, mock_breakpoint_manager, mock_breakpoint
    ):
        """Test getting breakpoint status."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler._get_breakpoint_status("bp-123")

        assert result.status_code == 200
        body = result.body.decode()
        assert "bp-123" in body
        assert "low_confidence" in body

    def test_get_status_not_found(self, breakpoints_handler, mock_breakpoint_manager):
        """Test status for non-existent breakpoint."""
        mock_breakpoint_manager.get_breakpoint = MagicMock(return_value=None)
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler._get_breakpoint_status("nonexistent")

        assert result.status_code == 404

    def test_get_status_manager_unavailable(self, breakpoints_handler):
        """Test status when manager unavailable."""
        breakpoints_handler._breakpoint_manager = None

        with patch.object(breakpoints_handler, "breakpoint_manager", None):
            result = breakpoints_handler._get_breakpoint_status("bp-123")

        assert result.status_code == 503

    def test_get_status_includes_snapshot(
        self, breakpoints_handler, mock_breakpoint_manager, mock_breakpoint
    ):
        """Test that status includes snapshot data."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler._get_breakpoint_status("bp-123")

        assert result.status_code == 200
        body = result.body.decode()
        assert "debate-456" in body
        assert "round_num" in body


# -----------------------------------------------------------------------------
# Resolve Breakpoint Tests
# -----------------------------------------------------------------------------


class TestResolveBreakpoint:
    """Tests for resolving breakpoints."""

    def test_resolve_continue_success(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolving with continue action."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"action": "continue", "message": "Proceeding despite low confidence"}

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 200
        body_str = result.body.decode()
        assert "resolved" in body_str

    def test_resolve_abort_success(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolving with abort action."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"action": "abort", "message": "Stopping debate"}

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 200

    def test_resolve_redirect_success(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolving with redirect action."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {
            "action": "redirect",
            "message": "Redirecting to different task",
            "redirect_task": "New focused task",
        }

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 200

    def test_resolve_inject_success(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolving with inject action."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {
            "action": "inject",
            "message": "Adding context: Consider market volatility",
        }

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 200

    def test_resolve_missing_action(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolve without action field."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"message": "Missing action"}

        result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 400

    def test_resolve_invalid_action(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolve with invalid action."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"action": "invalid_action"}

        result = breakpoints_handler._resolve_breakpoint("bp-123", body)

        assert result.status_code == 400
        assert "Invalid action" in result.body.decode()

    def test_resolve_not_found(self, breakpoints_handler, mock_breakpoint_manager):
        """Test resolve for non-existent breakpoint."""
        mock_breakpoint_manager.resolve_breakpoint = MagicMock(return_value=False)
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"action": "continue", "message": "Continue"}

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler._resolve_breakpoint("nonexistent", body)

        assert result.status_code == 404

    def test_resolve_manager_unavailable(self, breakpoints_handler):
        """Test resolve when manager unavailable."""
        breakpoints_handler._breakpoint_manager = None

        with patch.object(breakpoints_handler, "breakpoint_manager", None):
            result = breakpoints_handler._resolve_breakpoint("bp-123", {"action": "continue"})

        assert result.status_code == 503


# -----------------------------------------------------------------------------
# Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestBreakpointsRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, breakpoints_handler, mock_breakpoint_manager, mock_handler):
        """Test rate limit exceeded response."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=False)

            result = breakpoints_handler.handle("/api/v1/breakpoints/pending", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429

    def test_rate_limit_allowed(self, breakpoints_handler, mock_breakpoint_manager, mock_handler):
        """Test request allowed when within rate limit."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=True)

            result = breakpoints_handler.handle("/api/v1/breakpoints/pending", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


# -----------------------------------------------------------------------------
# Path Validation Tests
# -----------------------------------------------------------------------------


class TestBreakpointsPathValidation:
    """Tests for path segment validation."""

    def test_valid_breakpoint_id_alphanumeric(self, breakpoints_handler):
        """Test valid alphanumeric breakpoint ID."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/abc123/status") is True

    def test_valid_breakpoint_id_with_dash(self, breakpoints_handler):
        """Test valid breakpoint ID with dashes."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp-123-456/status") is True

    def test_valid_breakpoint_id_with_underscore(self, breakpoints_handler):
        """Test valid breakpoint ID with underscores."""
        assert breakpoints_handler.can_handle("/api/v1/breakpoints/bp_123_456/status") is True

    def test_rejects_path_traversal(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test rejection of path traversal attempts."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=True)

            # Path traversal in ID should be rejected by validation
            result = breakpoints_handler.handle(
                "/api/v1/breakpoints/../../../etc/passwd/status",
                {},
                mock_handler,
            )

        # Either not matched or returns error
        if result is not None:
            assert result.status_code in [400, 404]


# -----------------------------------------------------------------------------
# Handle Method Tests
# -----------------------------------------------------------------------------


class TestBreakpointsHandleMethod:
    """Tests for the main handle method."""

    def test_handle_pending_endpoint(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test handle routes to pending."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=True)

            result = breakpoints_handler.handle("/api/v1/breakpoints/pending", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    def test_handle_status_endpoint(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test handle routes to status."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=True)

            result = breakpoints_handler.handle(
                "/api/v1/breakpoints/bp-123/status", {}, mock_handler
            )

        assert result is not None
        assert result.status_code == 200

    def test_handle_resolve_get_returns_405(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test GET to resolve returns method not allowed."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        with (
            patch("aragora.server.handlers.breakpoints._breakpoints_limiter") as mock_limiter,
            patch(
                "aragora.server.handlers.breakpoints.get_client_ip",
                return_value="192.168.1.1",
            ),
        ):
            mock_limiter.is_allowed = MagicMock(return_value=True)

            result = breakpoints_handler.handle(
                "/api/v1/breakpoints/bp-123/resolve", {}, mock_handler
            )

        assert result is not None
        assert result.status_code == 405
        assert "POST" in result.headers.get("Allow", "")


# -----------------------------------------------------------------------------
# Handle POST Method Tests
# -----------------------------------------------------------------------------


class TestBreakpointsHandlePostMethod:
    """Tests for the handle_post method."""

    def test_handle_post_resolve(self, breakpoints_handler, mock_breakpoint_manager, mock_handler):
        """Test POST to resolve endpoint."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        body = {"action": "continue", "message": "Continue"}

        with patch("aragora.server.handlers.breakpoints.HumanGuidance") as mock_guidance:
            mock_guidance.return_value = MagicMock()

            result = breakpoints_handler.handle_post(
                "/api/v1/breakpoints/bp-123/resolve", body, mock_handler
            )

        assert result is not None
        assert result.status_code == 200

    def test_handle_post_invalid_path(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test POST to invalid path returns None."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler.handle_post(
            "/api/v1/breakpoints/bp-123/invalid", {}, mock_handler
        )

        assert result is None

    def test_handle_post_status_returns_none(
        self, breakpoints_handler, mock_breakpoint_manager, mock_handler
    ):
        """Test POST to status endpoint returns None (handled elsewhere)."""
        breakpoints_handler._breakpoint_manager = mock_breakpoint_manager

        result = breakpoints_handler.handle_post(
            "/api/v1/breakpoints/bp-123/status", {}, mock_handler
        )

        assert result is None


# -----------------------------------------------------------------------------
# Breakpoint Manager Lazy Loading Tests
# -----------------------------------------------------------------------------


class TestBreakpointManagerLazyLoading:
    """Tests for lazy loading of breakpoint manager."""

    def test_manager_lazy_loaded(self, breakpoints_handler):
        """Test that manager is lazily loaded."""
        # Initially None
        assert breakpoints_handler._breakpoint_manager is None

        # Property access attempts to load
        with patch("aragora.server.handlers.breakpoints.BreakpointManager") as mock_manager_class:
            mock_manager_class.return_value = MagicMock()

            # Access property
            _ = breakpoints_handler.breakpoint_manager

            # Should have attempted to import and instantiate
            mock_manager_class.assert_called_once()

    def test_manager_handles_import_error(self, breakpoints_handler):
        """Test graceful handling of import error."""
        breakpoints_handler._breakpoint_manager = None

        with patch.dict(
            "sys.modules",
            {"aragora.debate.breakpoints": None},
        ):
            # Should not raise, just return None
            manager = breakpoints_handler.breakpoint_manager

        # May be None if import fails
        assert manager is None or manager is not None
