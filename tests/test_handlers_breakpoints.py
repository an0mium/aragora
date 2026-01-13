"""
Tests for BreakpointsHandler - human-in-the-loop intervention endpoints.

Tests cover:
- GET /api/breakpoints/pending - List pending breakpoints
- POST /api/breakpoints/{id}/resolve - Resolve a breakpoint
- GET /api/breakpoints/{id}/status - Get breakpoint status
- Input validation
- Error handling
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from enum import Enum

from aragora.server.handlers.breakpoints import BreakpointsHandler


# ============================================================================
# Test Fixtures
# ============================================================================


class MockTrigger(Enum):
    """Mock trigger enum for testing."""

    LOW_CONFIDENCE = "low_confidence"
    SAFETY_CONCERN = "safety_concern"
    USER_REQUEST = "user_request"


@pytest.fixture
def mock_snapshot():
    """Create a mock debate snapshot."""
    snapshot = Mock()
    snapshot.debate_id = "debate-123"
    snapshot.round_num = 2
    snapshot.task = "Discuss AI safety"
    snapshot.current_confidence = 0.35
    snapshot.agent_names = ["claude", "gpt4"]
    return snapshot


@pytest.fixture
def mock_breakpoint(mock_snapshot):
    """Create a mock breakpoint object."""
    bp = Mock()
    bp.breakpoint_id = "bp-001"
    bp.trigger = MockTrigger.LOW_CONFIDENCE
    bp.message = "Confidence dropped below threshold"
    bp.created_at = "2026-01-10T10:00:00"
    bp.timeout_minutes = 30
    bp.snapshot = mock_snapshot
    bp.status = "pending"
    bp.resolved_at = None
    return bp


@pytest.fixture
def mock_breakpoint_manager(mock_breakpoint):
    """Create a mock breakpoint manager."""
    manager = Mock()
    manager.get_pending_breakpoints = Mock(return_value=[mock_breakpoint])
    manager.get_breakpoint = Mock(return_value=mock_breakpoint)
    manager.resolve_breakpoint = Mock(return_value=True)
    return manager


@pytest.fixture
def handler():
    """Create a BreakpointsHandler instance."""
    return BreakpointsHandler()


@pytest.fixture
def handler_with_manager(mock_breakpoint_manager):
    """Create a handler with mock manager."""
    h = BreakpointsHandler()
    h._breakpoint_manager = mock_breakpoint_manager
    return h


# ============================================================================
# Route Recognition Tests
# ============================================================================


class TestBreakpointsRouting:
    """Tests for breakpoints route recognition."""

    def test_can_handle_pending_route(self, handler):
        """Test handler recognizes pending route."""
        assert handler.can_handle("/api/breakpoints/pending")

    def test_can_handle_status_route(self, handler):
        """Test handler recognizes status route."""
        assert handler.can_handle("/api/breakpoints/bp-001/status")

    def test_can_handle_resolve_route(self, handler):
        """Test handler recognizes resolve route."""
        assert handler.can_handle("/api/breakpoints/bp-001/resolve")

    def test_can_handle_with_complex_id(self, handler):
        """Test handler handles complex breakpoint IDs."""
        assert handler.can_handle("/api/breakpoints/debate_123_round_2/status")
        assert handler.can_handle("/api/breakpoints/test-bp-abc_123/resolve")

    def test_cannot_handle_unknown_routes(self, handler):
        """Test handler rejects unknown routes."""
        assert not handler.can_handle("/api/breakpoints")
        assert not handler.can_handle("/api/breakpoints/")
        assert not handler.can_handle("/api/breakpoints/bp-001/unknown")
        assert not handler.can_handle("/api/other")


# ============================================================================
# GET /api/breakpoints/pending Tests
# ============================================================================


class TestGetPendingBreakpoints:
    """Tests for listing pending breakpoints."""

    def test_get_pending_success(self, handler_with_manager):
        """Test successful pending breakpoints retrieval."""
        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "breakpoints" in data
        assert "count" in data
        assert data["count"] == 1

        bp = data["breakpoints"][0]
        assert bp["breakpoint_id"] == "bp-001"
        assert bp["trigger"] == "low_confidence"
        assert bp["message"] == "Confidence dropped below threshold"

    def test_pending_includes_snapshot(self, handler_with_manager):
        """Test pending breakpoints include debate snapshot."""
        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)
        data = json.loads(result.body)

        bp = data["breakpoints"][0]
        assert bp["snapshot"] is not None
        assert bp["snapshot"]["debate_id"] == "debate-123"
        assert bp["snapshot"]["round_num"] == 2
        assert bp["snapshot"]["confidence"] == 0.35
        assert "claude" in bp["snapshot"]["agents"]

    def test_pending_no_breakpoints(self, handler_with_manager, mock_breakpoint_manager):
        """Test empty pending list."""
        mock_breakpoint_manager.get_pending_breakpoints.return_value = []

        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)
        data = json.loads(result.body)

        assert data["breakpoints"] == []
        assert data["count"] == 0

    def test_pending_without_manager(self, handler):
        """Test graceful degradation when breakpoint manager not available."""
        result = handler.handle("/api/breakpoints/pending", {}, None)

        # Handler gracefully returns empty list instead of error
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["breakpoints"] == []
        assert data["count"] == 0

    def test_pending_handles_exception(self, handler_with_manager, mock_breakpoint_manager):
        """Test error handling when manager raises exception."""
        mock_breakpoint_manager.get_pending_breakpoints.side_effect = Exception("DB error")

        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data

    def test_pending_breakpoint_without_snapshot(
        self, handler_with_manager, mock_breakpoint_manager
    ):
        """Test breakpoint without snapshot is handled."""
        bp = Mock()
        bp.breakpoint_id = "bp-no-snapshot"
        bp.trigger = MockTrigger.USER_REQUEST
        bp.message = "User requested pause"
        bp.created_at = "2026-01-10T11:00:00"
        bp.timeout_minutes = 60
        bp.snapshot = None

        mock_breakpoint_manager.get_pending_breakpoints.return_value = [bp]

        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)
        data = json.loads(result.body)

        assert data["breakpoints"][0]["snapshot"] is None


# ============================================================================
# GET /api/breakpoints/{id}/status Tests
# ============================================================================


class TestGetBreakpointStatus:
    """Tests for getting breakpoint status."""

    def test_get_status_success(self, handler_with_manager):
        """Test successful status retrieval."""
        result = handler_with_manager.handle("/api/breakpoints/bp-001/status", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["breakpoint_id"] == "bp-001"
        assert data["trigger"] == "low_confidence"
        assert data["status"] == "pending"

    def test_get_status_includes_snapshot(self, handler_with_manager):
        """Test status includes debate snapshot."""
        result = handler_with_manager.handle("/api/breakpoints/bp-001/status", {}, None)
        data = json.loads(result.body)

        assert data["snapshot"] is not None
        assert data["snapshot"]["debate_id"] == "debate-123"

    def test_get_status_not_found(self, handler_with_manager, mock_breakpoint_manager):
        """Test 404 for non-existent breakpoint."""
        mock_breakpoint_manager.get_breakpoint.return_value = None

        result = handler_with_manager.handle("/api/breakpoints/nonexistent/status", {}, None)

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()

    def test_get_status_without_manager(self, handler):
        """Test 404 when manager not available (breakpoint not found)."""
        result = handler.handle("/api/breakpoints/bp-001/status", {}, None)

        # Returns 404 because breakpoint can't be found without manager
        assert result.status_code == 404

    def test_get_status_resolved_breakpoint(self, handler_with_manager, mock_breakpoint):
        """Test status of resolved breakpoint."""
        mock_breakpoint.status = "resolved"
        mock_breakpoint.resolved_at = "2026-01-10T10:30:00"

        result = handler_with_manager.handle("/api/breakpoints/bp-001/status", {}, None)
        data = json.loads(result.body)

        assert data["status"] == "resolved"
        assert data["resolved_at"] == "2026-01-10T10:30:00"

    def test_get_status_handles_exception(self, handler_with_manager, mock_breakpoint_manager):
        """Test error handling for exceptions."""
        mock_breakpoint_manager.get_breakpoint.side_effect = Exception("Connection error")

        result = handler_with_manager.handle("/api/breakpoints/bp-001/status", {}, None)

        assert result.status_code == 500


# ============================================================================
# POST /api/breakpoints/{id}/resolve Tests
# ============================================================================


class TestResolveBreakpoint:
    """Tests for resolving breakpoints."""

    def test_resolve_continue_action(self, handler_with_manager):
        """Test resolving with continue action."""
        body = {
            "action": "continue",
            "message": "Looks fine, proceed",
        }

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "resolved"
        assert data["action"] == "continue"

    def test_resolve_abort_action(self, handler_with_manager):
        """Test resolving with abort action."""
        body = {
            "action": "abort",
            "message": "Safety concerns identified",
        }

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["action"] == "abort"

    def test_resolve_redirect_action(self, handler_with_manager):
        """Test resolving with redirect action."""
        body = {
            "action": "redirect",
            "message": "Topic should focus on safety",
            "redirect_task": "Focus on AI safety implications",
        }

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["action"] == "redirect"

    def test_resolve_inject_action(self, handler_with_manager):
        """Test resolving with inject action."""
        body = {
            "action": "inject",
            "message": "Consider this perspective: ...",
        }

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["action"] == "inject"

    def test_resolve_missing_action(self, handler_with_manager):
        """Test 400 when action is missing."""
        body = {"message": "Some message"}

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "action" in data["error"].lower()

    def test_resolve_invalid_action(self, handler_with_manager):
        """Test 400 for invalid action."""
        body = {"action": "invalid_action"}

        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower()

    def test_resolve_not_found(self, handler_with_manager, mock_breakpoint_manager):
        """Test 404 when breakpoint doesn't exist."""
        mock_breakpoint_manager.resolve_breakpoint.return_value = False

        body = {"action": "continue", "message": "ok"}
        result = handler_with_manager.handle_post(
            "/api/breakpoints/nonexistent/resolve", body, None
        )

        assert result.status_code == 404

    def test_resolve_without_manager(self, handler):
        """Test 404 when manager not available (breakpoint not found)."""
        body = {"action": "continue"}
        result = handler.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        # Returns 404 because breakpoint can't be resolved without manager
        assert result.status_code == 404

    def test_resolve_with_reviewer_id(self, handler_with_manager, mock_breakpoint_manager):
        """Test resolution includes reviewer ID."""
        body = {
            "action": "continue",
            "message": "Approved",
            "reviewer_id": "human_reviewer_1",
        }

        handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        # Verify resolve_breakpoint was called
        assert mock_breakpoint_manager.resolve_breakpoint.called
        call_args = mock_breakpoint_manager.resolve_breakpoint.call_args
        guidance = call_args[0][1]
        assert guidance.human_id == "human_reviewer_1"

    def test_get_resolve_returns_405(self, handler_with_manager):
        """Test GET on resolve endpoint returns 405."""
        result = handler_with_manager.handle("/api/breakpoints/bp-001/resolve", {}, None)

        assert result.status_code == 405
        data = json.loads(result.body)
        assert "POST" in data["error"]


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestBreakpointsValidation:
    """Tests for input validation."""

    def test_valid_breakpoint_id_patterns(self, handler):
        """Test various valid breakpoint ID patterns."""
        valid_ids = [
            "bp-001",
            "debate_123_round_2",
            "test-bp-abc123",
            "simple",
            "with_underscore",
            "with-dash",
        ]

        for bp_id in valid_ids:
            assert handler.can_handle(f"/api/breakpoints/{bp_id}/status"), f"Failed for {bp_id}"

    def test_invalid_breakpoint_id_rejected(self, handler_with_manager):
        """Test invalid breakpoint IDs are rejected."""
        # Try to get status with potentially dangerous ID
        # Note: The pattern should reject IDs with special chars
        # The handler validates IDs after matching the route

        # The route pattern itself limits what can match
        # So these should return None (not handled)
        assert not handler_with_manager.can_handle("/api/breakpoints/..//status")
        assert not handler_with_manager.can_handle("/api/breakpoints/;/status")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestBreakpointsErrorHandling:
    """Tests for error handling scenarios."""

    def test_manager_import_error_handled(self, handler):
        """Test graceful degradation when breakpoint module unavailable."""
        # Handler without manager gracefully returns empty list
        result = handler.handle("/api/breakpoints/pending", {}, None)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["breakpoints"] == []

    def test_exception_during_resolve(self, handler_with_manager, mock_breakpoint_manager):
        """Test error handling during resolve."""
        mock_breakpoint_manager.resolve_breakpoint.side_effect = Exception("Unexpected error")

        body = {"action": "continue", "message": "ok"}
        result = handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# Integration Pattern Tests
# ============================================================================


class TestBreakpointsIntegration:
    """Tests for breakpoint workflow patterns."""

    def test_pending_then_resolve_workflow(self, handler_with_manager, mock_breakpoint_manager):
        """Test typical workflow: list pending → resolve."""
        # Step 1: Get pending breakpoints
        pending_result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)
        assert pending_result.status_code == 200

        pending_data = json.loads(pending_result.body)
        bp_id = pending_data["breakpoints"][0]["breakpoint_id"]

        # Step 2: Resolve the breakpoint
        body = {"action": "continue", "message": "Reviewed and approved"}
        resolve_result = handler_with_manager.handle_post(
            f"/api/breakpoints/{bp_id}/resolve", body, None
        )
        assert resolve_result.status_code == 200

    def test_status_check_after_resolve(
        self, handler_with_manager, mock_breakpoint, mock_breakpoint_manager
    ):
        """Test checking status after resolution."""
        # First resolve
        body = {"action": "abort", "message": "Stopping debate"}
        handler_with_manager.handle_post("/api/breakpoints/bp-001/resolve", body, None)

        # Update mock to reflect resolved state
        mock_breakpoint.status = "resolved"
        mock_breakpoint.resolved_at = "2026-01-10T10:30:00"

        # Check status
        status_result = handler_with_manager.handle("/api/breakpoints/bp-001/status", {}, None)
        data = json.loads(status_result.body)

        assert data["status"] == "resolved"

    def test_multiple_pending_breakpoints(
        self, handler_with_manager, mock_breakpoint_manager, mock_snapshot
    ):
        """Test handling multiple pending breakpoints."""
        bp1 = Mock()
        bp1.breakpoint_id = "bp-001"
        bp1.trigger = MockTrigger.LOW_CONFIDENCE
        bp1.message = "Low confidence"
        bp1.created_at = "2026-01-10T10:00:00"
        bp1.timeout_minutes = 30
        bp1.snapshot = mock_snapshot

        bp2 = Mock()
        bp2.breakpoint_id = "bp-002"
        bp2.trigger = MockTrigger.SAFETY_CONCERN
        bp2.message = "Safety flag raised"
        bp2.created_at = "2026-01-10T10:05:00"
        bp2.timeout_minutes = 15
        bp2.snapshot = mock_snapshot

        mock_breakpoint_manager.get_pending_breakpoints.return_value = [bp1, bp2]

        result = handler_with_manager.handle("/api/breakpoints/pending", {}, None)
        data = json.loads(result.body)

        assert data["count"] == 2
        assert len(data["breakpoints"]) == 2

        # Should have both types
        triggers = [bp["trigger"] for bp in data["breakpoints"]]
        assert "low_confidence" in triggers
        assert "safety_concern" in triggers
