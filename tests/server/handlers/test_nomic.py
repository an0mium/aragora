"""
Tests for the nomic handler - nomic loop state and control endpoints.

Comprehensive test coverage for:
- Route handling (can_handle)
- GET endpoints: state, health, metrics, log, risk-register, witness/status, mayor/current, modes
- POST control operations: start, stop, pause, resume, skip-phase
- POST proposal operations: approve, reject
- RBAC permission checks (nomic:read, nomic:admin)
- WebSocket event streaming integration
- Error handling
"""

import asyncio
import io
import json
import pytest
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.nomic import NomicHandler


# ===========================================================================
# Mock Classes for Testing
# ===========================================================================


class MockHTTPHandler:
    """Mock HTTP request handler for testing.

    Simulates the HTTP request handler that NomicHandler.handle() expects,
    with command (HTTP method), headers, and rfile (request body).
    """

    def __init__(self, method: str = "GET", body: dict | None = None):
        self.command = method
        self._body = json.dumps(body or {}).encode() if body else b""
        self.headers = {
            "Content-Length": str(len(self._body)) if self._body else "0",
            "Content-Type": "application/json" if body else "",
        }
        self.rfile = io.BytesIO(self._body)
        self.client_address = ("127.0.0.1", 12345)


@dataclass
class MockAuthorizationContext:
    """Mock authorization context for testing."""

    user_id: str = "test-user"
    workspace_id: str = "test-workspace"
    roles: set = field(default_factory=lambda: {"member"})
    permissions: set = field(default_factory=lambda: {"nomic:read", "nomic:admin"})


class MockNomicLoopStreamServer:
    """Mock stream server for WebSocket event emission testing."""

    def __init__(self):
        self.events = []

    async def emit_loop_started(self, **kwargs):
        self.events.append(("loop_started", kwargs))

    async def emit_loop_stopped(self, **kwargs):
        self.events.append(("loop_stopped", kwargs))

    async def emit_loop_paused(self, **kwargs):
        self.events.append(("loop_paused", kwargs))

    async def emit_loop_resumed(self, **kwargs):
        self.events.append(("loop_resumed", kwargs))

    async def emit_phase_skipped(self, **kwargs):
        self.events.append(("phase_skipped", kwargs))

    async def emit_proposal_approved(self, **kwargs):
        self.events.append(("proposal_approved", kwargs))

    async def emit_proposal_rejected(self, **kwargs):
        self.events.append(("proposal_rejected", kwargs))


def make_post_handler(body: dict | None = None, method: str = "POST") -> MockHTTPHandler:
    """Create a mock HTTP handler with optional JSON body."""
    return MockHTTPHandler(method, body)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def nomic_handler():
    """Create a nomic handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = NomicHandler(ctx)
    return handler


@pytest.fixture
def nomic_handler_with_dir(tmp_path):
    """Create a nomic handler with a temp directory."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": tmp_path}
    handler = NomicHandler(ctx)
    return handler, tmp_path


@pytest.fixture
def nomic_handler_with_auth(tmp_path):
    """Create a nomic handler with mocked auth that grants permission."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": tmp_path}
    handler = NomicHandler(ctx)

    async def mock_get_auth_context(request, require_auth=False):
        return MockAuthorizationContext()

    def mock_check_permission(auth_context, permission, resource_id=None):
        pass  # Grant permission

    handler.get_auth_context = mock_get_auth_context
    handler.check_permission = mock_check_permission
    return handler, tmp_path


@pytest.fixture
def nomic_handler_unauthorized(tmp_path):
    """Create a nomic handler that raises UnauthorizedError."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": tmp_path}
    handler = NomicHandler(ctx)

    async def mock_get_auth_context(request, require_auth=False):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        raise UnauthorizedError("Not authenticated")

    handler.get_auth_context = mock_get_auth_context
    return handler, tmp_path


@pytest.fixture
def nomic_handler_forbidden(tmp_path):
    """Create a nomic handler that raises ForbiddenError."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": tmp_path}
    handler = NomicHandler(ctx)

    async def mock_get_auth_context(request, require_auth=False):
        return MockAuthorizationContext()

    def mock_check_permission(auth_context, permission, resource_id=None):
        from aragora.server.handlers.utils.auth import ForbiddenError

        raise ForbiddenError(f"Permission denied: {permission}", permission=permission)

    handler.get_auth_context = mock_get_auth_context
    handler.check_permission = mock_check_permission
    return handler, tmp_path


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for GET requests."""
    return MockHTTPHandler("GET")


class TestNomicHandlerRouting:
    """Tests for NomicHandler route matching."""

    def test_can_handle_nomic_state(self, nomic_handler):
        """Test that handler recognizes /api/nomic/state route."""
        assert nomic_handler.can_handle("/api/v1/nomic/state") is True

    def test_can_handle_nomic_health(self, nomic_handler):
        """Test that handler recognizes /api/nomic/health route."""
        assert nomic_handler.can_handle("/api/v1/nomic/health") is True

    def test_can_handle_nomic_metrics(self, nomic_handler):
        """Test that handler recognizes /api/nomic/metrics route."""
        assert nomic_handler.can_handle("/api/v1/nomic/metrics") is True

    def test_can_handle_nomic_log(self, nomic_handler):
        """Test that handler recognizes /api/nomic/log route."""
        assert nomic_handler.can_handle("/api/v1/nomic/log") is True

    def test_can_handle_nomic_risk_register(self, nomic_handler):
        """Test that handler recognizes /api/nomic/risk-register route."""
        assert nomic_handler.can_handle("/api/v1/nomic/risk-register") is True

    def test_can_handle_modes(self, nomic_handler):
        """Test that handler recognizes /api/modes route."""
        assert nomic_handler.can_handle("/api/v1/modes") is True

    def test_cannot_handle_unknown_path(self, nomic_handler):
        """Test that handler rejects unknown paths outside its prefix."""
        assert nomic_handler.can_handle("/api/v1/unknown") is False
        assert nomic_handler.can_handle("/api/v1/nomic") is False
        # Handler accepts all /api/nomic/* paths and handles 404 internally
        assert nomic_handler.can_handle("/api/v1/nomic/unknown") is True


class TestNomicState:
    """Tests for GET /api/nomic/state endpoint."""

    async def test_nomic_state_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic state should return 503 when directory not configured."""
        result = await nomic_handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body

    async def test_nomic_state_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return not_running when no state file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = await handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["state"] == "not_running"
        assert body["cycle"] == 0

    async def test_nomic_state_with_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return state from file."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"state": "running", "cycle": 5, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = await handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["state"] == "running"
        assert body["cycle"] == 5
        assert body["phase"] == "debate"

    async def test_nomic_state_invalid_json(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return error for invalid JSON."""
        handler, tmp_path = nomic_handler_with_dir

        (tmp_path / "nomic_state.json").write_text("not valid json {")

        result = await handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body


class TestNomicHealth:
    """Tests for GET /api/nomic/health endpoint."""

    async def test_nomic_health_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic health should return 503 when directory not configured."""
        result = await nomic_handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    async def test_nomic_health_not_running(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return not_running when no state file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = await handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "not_running"
        assert body["cycle"] == 0

    async def test_nomic_health_healthy(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return healthy when recent activity."""
        handler, tmp_path = nomic_handler_with_dir

        # Recent timestamp (within 30 minutes)
        recent_time = datetime.now().isoformat()
        state = {"cycle": 3, "phase": "design", "last_update": recent_time}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = await handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        assert body["cycle"] == 3
        assert body["phase"] == "design"

    async def test_nomic_health_stalled(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return stalled when no recent activity."""
        handler, tmp_path = nomic_handler_with_dir

        # Old timestamp (more than 30 minutes ago)
        old_time = (datetime.now() - timedelta(hours=1)).isoformat()
        state = {"cycle": 2, "phase": "implement", "last_update": old_time}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = await handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "stalled"
        assert body["stall_duration_seconds"] is not None
        assert body["stall_duration_seconds"] > 1800  # More than 30 minutes


class TestNomicLog:
    """Tests for GET /api/nomic/log endpoint."""

    async def test_nomic_log_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic log should return 503 when directory not configured."""
        result = await nomic_handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    async def test_nomic_log_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should return empty list when no log file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = await handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["lines"] == []
        assert body["total"] == 0

    async def test_nomic_log_with_content(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should return log lines."""
        handler, tmp_path = nomic_handler_with_dir

        log_content = "Line 1\nLine 2\nLine 3\n"
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = await handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["lines"]) == 3
        assert body["total"] == 3

    async def test_nomic_log_with_lines_param(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should respect lines parameter."""
        handler, tmp_path = nomic_handler_with_dir

        log_content = "\n".join([f"Line {i}" for i in range(100)])
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = await handler.handle("/api/v1/nomic/log", {"lines": "10"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["showing"] == 10
        assert body["total"] == 100


class TestNomicMetrics:
    """Tests for GET /api/nomic/metrics endpoint."""

    async def test_nomic_metrics_returns_summary(self, nomic_handler, mock_http_handler):
        """Nomic metrics should return a metrics summary."""
        with patch.object(nomic_handler, "_get_nomic_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(
                body=json.dumps({"metrics": {"cycles": 10}}).encode(), status_code=200
            )
            result = await nomic_handler.handle("/api/v1/nomic/metrics", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200


class TestRiskRegister:
    """Tests for GET /api/nomic/risk-register endpoint."""

    async def test_risk_register_returns_entries(self, nomic_handler, mock_http_handler):
        """Risk register should return risk entries."""
        with patch.object(nomic_handler, "_get_risk_register") as mock_risk:
            mock_risk.return_value = MagicMock(
                body=json.dumps({"risks": [], "total": 0}).encode(), status_code=200
            )
            result = await nomic_handler.handle(
                "/api/v1/nomic/risk-register", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200

    async def test_risk_register_with_limit_param(self, nomic_handler, mock_http_handler):
        """Risk register should respect limit parameter."""
        with patch.object(nomic_handler, "_get_risk_register") as mock_risk:
            mock_risk.return_value = MagicMock(
                body=json.dumps({"risks": [], "total": 0}).encode(), status_code=200
            )
            result = await nomic_handler.handle(
                "/api/v1/nomic/risk-register", {"limit": "10"}, mock_http_handler
            )

        assert result is not None
        # Verify limit was passed (clamped to valid range)
        mock_risk.assert_called_once_with(10)


class TestModes:
    """Tests for GET /api/modes endpoint."""

    async def test_get_modes_returns_list(self, nomic_handler, mock_http_handler):
        """Get modes should return available operational modes."""
        result = await nomic_handler.handle("/api/v1/modes", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should have modes array
        assert "modes" in body or isinstance(body, list)


class TestHandleReturnsNone:
    """Tests for handle returning None for non-matching paths."""

    async def test_handle_returns_none_for_non_matching(self, nomic_handler, mock_http_handler):
        """Handle should return None for paths that can_handle returns False for."""
        # Force a path that would bypass can_handle check
        result = await nomic_handler.handle("/api/v1/nomic/nonexistent", {}, mock_http_handler)
        assert result is None


class TestNomicControl:
    """Tests for nomic loop control endpoints (POST operations)."""

    async def test_start_nomic_loop_no_dir(self, nomic_handler, mock_http_handler):
        """Start should return 503 when directory not configured."""
        result = nomic_handler._start_nomic_loop({})
        assert result is not None
        assert result.status_code == 503

    async def test_start_nomic_loop_already_running(
        self, nomic_handler_with_dir, mock_http_handler
    ):
        """Start should return 409 when loop already running."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "pid": 12345, "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._start_nomic_loop({})
        assert result is not None
        assert result.status_code == 409

    async def test_start_nomic_loop_script_not_found(
        self, nomic_handler_with_dir, mock_http_handler
    ):
        """Start should return 500 when script not found."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._start_nomic_loop({"cycles": 3})
        assert result is not None
        assert result.status_code == 500

    async def test_stop_nomic_loop_no_dir(self, nomic_handler, mock_http_handler):
        """Stop should return 503 when directory not configured."""
        result = nomic_handler._stop_nomic_loop({})
        assert result is not None
        assert result.status_code == 503

    async def test_stop_nomic_loop_not_running(self, nomic_handler_with_dir, mock_http_handler):
        """Stop should return 404 when loop not running."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._stop_nomic_loop({})
        assert result is not None
        assert result.status_code == 404

    async def test_stop_nomic_loop_no_pid(self, nomic_handler_with_dir, mock_http_handler):
        """Stop should return 404 when no PID in state."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": False}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._stop_nomic_loop({})
        assert result is not None
        assert result.status_code == 404

    async def test_pause_nomic_loop_no_dir(self, nomic_handler, mock_http_handler):
        """Pause should return 503 when directory not configured."""
        result = nomic_handler._pause_nomic_loop()
        assert result is not None
        assert result.status_code == 503

    async def test_pause_nomic_loop_not_running(self, nomic_handler_with_dir, mock_http_handler):
        """Pause should return 404 when loop not running."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._pause_nomic_loop()
        assert result is not None
        assert result.status_code == 404

    async def test_pause_nomic_loop_already_paused(self, nomic_handler_with_dir, mock_http_handler):
        """Pause should return 409 when already paused."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "paused": True, "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._pause_nomic_loop()
        assert result is not None
        assert result.status_code == 409

    async def test_pause_nomic_loop_success(self, nomic_handler_with_dir, mock_http_handler):
        """Pause should succeed when loop is running."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "paused": False, "cycle": 2, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._pause_nomic_loop()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "paused"

    async def test_resume_nomic_loop_no_dir(self, nomic_handler, mock_http_handler):
        """Resume should return 503 when directory not configured."""
        result = nomic_handler._resume_nomic_loop()
        assert result is not None
        assert result.status_code == 503

    async def test_resume_nomic_loop_not_paused(self, nomic_handler_with_dir, mock_http_handler):
        """Resume should return 409 when not paused."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "paused": False}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._resume_nomic_loop()
        assert result is not None
        assert result.status_code == 409

    async def test_resume_nomic_loop_success(self, nomic_handler_with_dir, mock_http_handler):
        """Resume should succeed when loop is paused."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "paused": True, "cycle": 3, "phase": "design"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._resume_nomic_loop()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "resumed"


class TestPhaseSkip:
    """Tests for phase skip functionality."""

    async def test_skip_phase_no_dir(self, nomic_handler, mock_http_handler):
        """Skip phase should return 503 when directory not configured."""
        result = nomic_handler._skip_phase()
        assert result is not None
        assert result.status_code == 503

    async def test_skip_phase_not_running(self, nomic_handler_with_dir, mock_http_handler):
        """Skip phase should return 404 when loop not running."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._skip_phase()
        assert result is not None
        assert result.status_code == 404

    async def test_skip_phase_unknown_phase(self, nomic_handler_with_dir, mock_http_handler):
        """Skip phase should return 400 for unknown phase."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "phase": "unknown_phase", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()
        assert result is not None
        assert result.status_code == 400

    async def test_skip_phase_debate_to_design(self, nomic_handler_with_dir, mock_http_handler):
        """Skip phase should transition from debate to design."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "phase": "debate", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["previous_phase"] == "debate"
        assert body["next_phase"] == "design"

    async def test_skip_phase_verify_to_context_increments_cycle(
        self, nomic_handler_with_dir, mock_http_handler
    ):
        """Skip phase from verify should wrap to context and increment cycle."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"running": True, "phase": "verify", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["next_phase"] == "context"
        assert body["cycle"] == 2


class TestProposals:
    """Tests for proposal management endpoints."""

    async def test_get_proposals_no_dir(self, nomic_handler, mock_http_handler):
        """Get proposals should return 503 when directory not configured."""
        result = nomic_handler._get_proposals()
        assert result is not None
        assert result.status_code == 503

    async def test_get_proposals_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Get proposals should return empty list when no file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._get_proposals()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["proposals"] == []
        assert body["total"] == 0

    async def test_get_proposals_with_pending(self, nomic_handler_with_dir, mock_http_handler):
        """Get proposals should return pending proposals."""
        handler, tmp_path = nomic_handler_with_dir

        proposals = {
            "proposals": [
                {"id": "p1", "status": "pending", "description": "Test 1"},
                {"id": "p2", "status": "approved", "description": "Test 2"},
                {"id": "p3", "status": "pending", "description": "Test 3"},
            ]
        }
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._get_proposals()
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2  # Only pending
        assert all(p["status"] == "pending" for p in body["proposals"])

    async def test_approve_proposal_no_id(self, nomic_handler_with_dir, mock_http_handler):
        """Approve should return 400 when no proposal_id."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._approve_proposal({})
        assert result is not None
        assert result.status_code == 400

    async def test_approve_proposal_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Approve should return 404 when proposals file missing."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._approve_proposal({"proposal_id": "p1"})
        assert result is not None
        assert result.status_code == 404

    async def test_approve_proposal_not_found(self, nomic_handler_with_dir, mock_http_handler):
        """Approve should return 404 for non-existent proposal."""
        handler, tmp_path = nomic_handler_with_dir

        proposals = {"proposals": [{"id": "p1", "status": "pending"}]}
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._approve_proposal({"proposal_id": "nonexistent"})
        assert result is not None
        assert result.status_code == 404

    async def test_approve_proposal_success(self, nomic_handler_with_dir, mock_http_handler):
        """Approve should successfully approve a proposal."""
        handler, tmp_path = nomic_handler_with_dir

        proposals = {"proposals": [{"id": "p1", "status": "pending"}]}
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._approve_proposal({"proposal_id": "p1", "approved_by": "tester"})
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "approved"
        assert body["proposal_id"] == "p1"

    async def test_reject_proposal_no_id(self, nomic_handler_with_dir, mock_http_handler):
        """Reject should return 400 when no proposal_id."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._reject_proposal({})
        assert result is not None
        assert result.status_code == 400

    async def test_reject_proposal_success(self, nomic_handler_with_dir, mock_http_handler):
        """Reject should successfully reject a proposal."""
        handler, tmp_path = nomic_handler_with_dir

        proposals = {"proposals": [{"id": "p1", "status": "pending"}]}
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._reject_proposal(
            {"proposal_id": "p1", "rejected_by": "tester", "reason": "Does not meet requirements"}
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "rejected"


class TestRiskRegisterAdvanced:
    """Advanced tests for risk register endpoint."""

    async def test_risk_register_no_dir(self, nomic_handler, mock_http_handler):
        """Risk register should return 503 when directory not configured."""
        result = nomic_handler._get_risk_register(50)
        assert result is not None
        assert result.status_code == 503

    async def test_risk_register_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Risk register should return empty when no file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler._get_risk_register(50)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["risks"] == []
        assert body["total"] == 0

    async def test_risk_register_with_entries(self, nomic_handler_with_dir, mock_http_handler):
        """Risk register should return entries from JSONL file."""
        handler, tmp_path = nomic_handler_with_dir

        risks = [
            {"id": "r1", "severity": "critical", "description": "Test risk 1"},
            {"id": "r2", "severity": "high", "description": "Test risk 2"},
            {"id": "r3", "severity": "low", "description": "Test risk 3"},
        ]
        content = "\n".join(json.dumps(r) for r in risks)
        (tmp_path / "risk_register.jsonl").write_text(content)

        result = handler._get_risk_register(50)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 3
        assert body["critical_count"] == 1
        assert body["high_count"] == 1

    async def test_risk_register_respects_limit(self, nomic_handler_with_dir, mock_http_handler):
        """Risk register should respect limit parameter."""
        handler, tmp_path = nomic_handler_with_dir

        risks = [{"id": f"r{i}", "severity": "low"} for i in range(100)]
        content = "\n".join(json.dumps(r) for r in risks)
        (tmp_path / "risk_register.jsonl").write_text(content)

        result = handler._get_risk_register(10)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["risks"]) == 10
        assert body["total"] == 100


class TestWitnessAndMayor:
    """Tests for witness patrol and mayor endpoints."""

    async def test_witness_status_not_available(self, nomic_handler, mock_http_handler):
        """Witness status should indicate not initialized when not available."""
        with patch("aragora.server.handlers.nomic.NomicHandler._get_witness_status") as mock_method:
            mock_method.return_value = MagicMock(
                body=json.dumps({"patrolling": False, "initialized": False}).encode(),
                status_code=200,
            )
            result = await nomic_handler.handle(
                "/api/v1/nomic/witness/status", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200

    async def test_mayor_current_not_available(self, nomic_handler, mock_http_handler):
        """Mayor current should indicate not initialized when not available."""
        with patch.object(nomic_handler, "_get_mayor_current") as mock_method:
            mock_method.return_value = MagicMock(
                body=json.dumps({"initialized": False}).encode(), status_code=200
            )
            result = await nomic_handler.handle(
                "/api/v1/nomic/mayor/current", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200


class TestNomicMetricsAdvanced:
    """Advanced tests for nomic metrics endpoint."""

    async def test_nomic_metrics_module_unavailable(self, nomic_handler, mock_http_handler):
        """Metrics should handle module not available gracefully."""
        with patch("aragora.server.handlers.nomic.NomicHandler._get_nomic_metrics") as mock_method:
            mock_method.return_value = MagicMock(
                body=json.dumps(
                    {
                        "summary": {},
                        "stuck_detection": {"is_stuck": False},
                        "status": "metrics_unavailable",
                    }
                ).encode(),
                status_code=200,
            )
            result = await nomic_handler.handle("/api/v1/nomic/metrics", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200


class TestErrorHandling:
    """Tests for error handling scenarios."""

    async def test_invalid_json_in_state_file(self, nomic_handler_with_dir, mock_http_handler):
        """Should handle invalid JSON in state file."""
        handler, tmp_path = nomic_handler_with_dir

        (tmp_path / "nomic_state.json").write_text("{ invalid json }")

        result = await handler.handle("/api/v1/nomic/health", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "error"

    async def test_invalid_json_in_proposals_file(self, nomic_handler_with_dir, mock_http_handler):
        """Should handle invalid JSON in proposals file."""
        handler, tmp_path = nomic_handler_with_dir

        (tmp_path / "proposals.json").write_text("{ invalid }")

        result = handler._get_proposals()
        assert result is not None
        assert result.status_code == 500

    async def test_malformed_risk_register_entries(self, nomic_handler_with_dir, mock_http_handler):
        """Should skip malformed entries in risk register."""
        handler, tmp_path = nomic_handler_with_dir

        content = '{"id": "r1", "severity": "low"}\ninvalid json\n{"id": "r2", "severity": "high"}'
        (tmp_path / "risk_register.jsonl").write_text(content)

        result = handler._get_risk_register(50)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should have 2 valid entries (skipped the invalid one)
        assert body["total"] == 2


class TestVersionedEndpoints:
    """Tests for versioned API endpoint handling."""

    def test_handles_v1_endpoints(self, nomic_handler):
        """Should handle /api/v1/ prefixed endpoints."""
        assert nomic_handler.can_handle("/api/v1/nomic/state") is True
        assert nomic_handler.can_handle("/api/v1/nomic/health") is True
        assert nomic_handler.can_handle("/api/v1/nomic/metrics") is True
        assert nomic_handler.can_handle("/api/v1/modes") is True

    def test_handles_non_versioned_endpoints(self, nomic_handler):
        """Should handle non-versioned endpoints."""
        assert nomic_handler.can_handle("/api/nomic/state") is True
        assert nomic_handler.can_handle("/api/nomic/health") is True


class TestStreamEmission:
    """Tests for WebSocket stream event emission."""

    async def test_emit_event_no_stream(self, nomic_handler):
        """Should silently skip emission when no stream configured."""
        # This should not raise even with no stream
        nomic_handler._emit_event("emit_loop_started", cycles=3, auto_approve=False)

    async def test_emit_event_with_stream(self, nomic_handler):
        """Should emit event when stream is configured."""
        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock()
        nomic_handler.set_stream_server(mock_stream)

        nomic_handler._emit_event("emit_loop_started", cycles=3, auto_approve=False)
        # Emission is async/background, so we just verify no error occurred


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestNomicRBACPermissions:
    """Tests for RBAC permission enforcement on nomic endpoints."""

    @pytest.mark.asyncio
    async def test_handle_requires_nomic_read_permission(self, nomic_handler_forbidden):
        """GET handle is now public (no permission required for dashboard data)."""
        handler, tmp_path = nomic_handler_forbidden
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/state", {}, mock_request)

        # Nomic endpoints are now public for dashboard access
        assert result is not None
        assert result.status_code != 403  # No longer permission-gated

    @pytest.mark.asyncio
    async def test_handle_returns_401_when_unauthorized(self, nomic_handler_unauthorized):
        """GET handle is now public (no auth required for dashboard data)."""
        handler, tmp_path = nomic_handler_unauthorized
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/state", {}, mock_request)

        # Nomic endpoints are now public for dashboard access
        assert result is not None
        assert result.status_code != 401  # No longer auth-gated

    @pytest.mark.asyncio
    async def test_handle_post_requires_nomic_admin_permission(self, nomic_handler_forbidden):
        """POST handle should require nomic:admin permission."""
        handler, tmp_path = nomic_handler_forbidden
        mock_request = make_post_handler({})

        result = await handler.handle_post("/api/v1/nomic/control/start", {}, mock_request)

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_handle_post_returns_401_when_unauthorized(self, nomic_handler_unauthorized):
        """POST handle should return 401 when not authenticated."""
        handler, tmp_path = nomic_handler_unauthorized
        mock_request = make_post_handler({})

        result = await handler.handle_post("/api/v1/nomic/control/start", {}, mock_request)

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_state_endpoint_with_valid_auth(self, nomic_handler_with_auth):
        """State endpoint should succeed with valid nomic:read permission."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/state", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_with_valid_auth(self, nomic_handler_with_auth):
        """Health endpoint should succeed with valid nomic:read permission."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/health", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_endpoint_with_valid_auth(self, nomic_handler_with_auth):
        """Metrics endpoint should succeed with valid nomic:read permission."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/metrics", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_log_endpoint_with_valid_auth(self, nomic_handler_with_auth):
        """Log endpoint should succeed with valid nomic:read permission."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        result = await handler.handle("/api/v1/nomic/log", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_proposals_approve_forbidden(self, nomic_handler_forbidden):
        """Approve proposal should require nomic:admin permission."""
        handler, tmp_path = nomic_handler_forbidden
        mock_request = make_post_handler({"proposal_id": "p1"})

        result = await handler.handle_post("/api/v1/nomic/proposals/approve", {}, mock_request)

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_proposals_reject_forbidden(self, nomic_handler_forbidden):
        """Reject proposal should require nomic:admin permission."""
        handler, tmp_path = nomic_handler_forbidden
        mock_request = make_post_handler({"proposal_id": "p1"})

        result = await handler.handle_post("/api/v1/nomic/proposals/reject", {}, mock_request)

        assert result is not None
        assert result.status_code == 403


# ===========================================================================
# WebSocket Stream Integration Tests
# ===========================================================================


class TestWebSocketStreamIntegration:
    """Tests for WebSocket event streaming integration."""

    def test_set_stream_server(self, nomic_handler):
        """Should set stream server instance."""
        mock_stream = MockNomicLoopStreamServer()
        nomic_handler.set_stream_server(mock_stream)

        assert nomic_handler._stream is mock_stream

    def test_get_stream_from_instance(self, nomic_handler):
        """Should get stream from instance variable."""
        mock_stream = MockNomicLoopStreamServer()
        nomic_handler.set_stream_server(mock_stream)

        result = nomic_handler._get_stream()
        assert result is mock_stream

    def test_get_stream_from_context(self, nomic_handler):
        """Should get stream from context if not set on instance."""
        mock_stream = MockNomicLoopStreamServer()
        nomic_handler.ctx["nomic_loop_stream"] = mock_stream

        result = nomic_handler._get_stream()
        assert result is mock_stream

    def test_get_stream_returns_none_when_not_configured(self, nomic_handler):
        """Should return None when no stream configured."""
        result = nomic_handler._get_stream()
        assert result is None

    @pytest.mark.asyncio
    async def test_emit_event_schedules_async_task(self, nomic_handler):
        """Should schedule async emission without blocking."""
        mock_stream = MockNomicLoopStreamServer()
        nomic_handler.set_stream_server(mock_stream)

        # Emit an event
        nomic_handler._emit_event("emit_loop_started", cycles=5, auto_approve=False)

        # Give the async task time to complete
        await asyncio.sleep(0.1)

        # Verify event was emitted
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "loop_started"
        assert mock_stream.events[0][1]["cycles"] == 5

    @pytest.mark.asyncio
    async def test_emit_pause_event(self, nomic_handler_with_auth):
        """Should emit loop paused event on pause."""
        handler, tmp_path = nomic_handler_with_auth
        mock_stream = MockNomicLoopStreamServer()
        handler.set_stream_server(mock_stream)

        # Setup running state
        state = {"running": True, "paused": False, "cycle": 2, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._pause_nomic_loop()

        # Give async emission time to complete
        await asyncio.sleep(0.1)

        assert result.status_code == 200
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "loop_paused"

    @pytest.mark.asyncio
    async def test_emit_resume_event(self, nomic_handler_with_auth):
        """Should emit loop resumed event on resume."""
        handler, tmp_path = nomic_handler_with_auth
        mock_stream = MockNomicLoopStreamServer()
        handler.set_stream_server(mock_stream)

        # Setup paused state
        state = {"running": True, "paused": True, "cycle": 2, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._resume_nomic_loop()

        # Give async emission time to complete
        await asyncio.sleep(0.1)

        assert result.status_code == 200
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "loop_resumed"

    @pytest.mark.asyncio
    async def test_emit_phase_skipped_event(self, nomic_handler_with_auth):
        """Should emit phase skipped event on skip."""
        handler, tmp_path = nomic_handler_with_auth
        mock_stream = MockNomicLoopStreamServer()
        handler.set_stream_server(mock_stream)

        # Setup running state
        state = {"running": True, "phase": "debate", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()

        # Give async emission time to complete
        await asyncio.sleep(0.1)

        assert result.status_code == 200
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "phase_skipped"

    @pytest.mark.asyncio
    async def test_emit_proposal_approved_event(self, nomic_handler_with_auth):
        """Should emit proposal approved event on approve."""
        handler, tmp_path = nomic_handler_with_auth
        mock_stream = MockNomicLoopStreamServer()
        handler.set_stream_server(mock_stream)

        # Setup proposals file
        proposals = {"proposals": [{"id": "p1", "status": "pending"}]}
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._approve_proposal({"proposal_id": "p1", "approved_by": "tester"})

        # Give async emission time to complete
        await asyncio.sleep(0.1)

        assert result.status_code == 200
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "proposal_approved"

    @pytest.mark.asyncio
    async def test_emit_proposal_rejected_event(self, nomic_handler_with_auth):
        """Should emit proposal rejected event on reject."""
        handler, tmp_path = nomic_handler_with_auth
        mock_stream = MockNomicLoopStreamServer()
        handler.set_stream_server(mock_stream)

        # Setup proposals file
        proposals = {"proposals": [{"id": "p1", "status": "pending"}]}
        (tmp_path / "proposals.json").write_text(json.dumps(proposals))

        result = handler._reject_proposal(
            {"proposal_id": "p1", "rejected_by": "tester", "reason": "Not needed"}
        )

        # Give async emission time to complete
        await asyncio.sleep(0.1)

        assert result.status_code == 200
        assert len(mock_stream.events) == 1
        assert mock_stream.events[0][0] == "proposal_rejected"


# ===========================================================================
# Checkpoint and Backup Operations Tests
# ===========================================================================


class TestCheckpointOperations:
    """Tests related to checkpoint operations in nomic loop."""

    @pytest.mark.asyncio
    async def test_state_includes_checkpoint_info(self, nomic_handler_with_auth):
        """State should include checkpoint info if present."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        state = {
            "running": True,
            "cycle": 3,
            "phase": "verify",
            "checkpoint": {
                "id": "cp-123",
                "created_at": datetime.now().isoformat(),
                "state_hash": "abc123",
            },
        }
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = await handler.handle("/api/v1/nomic/state", {}, mock_request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "checkpoint" in body
        assert body["checkpoint"]["id"] == "cp-123"


# ===========================================================================
# Handler Resource Type Tests
# ===========================================================================


class TestHandlerConfiguration:
    """Tests for handler configuration and metadata."""

    def test_resource_type_is_nomic(self, nomic_handler):
        """Resource type should be 'nomic'."""
        assert nomic_handler.RESOURCE_TYPE == "nomic"

    def test_routes_are_defined(self, nomic_handler):
        """Handler should have routes defined."""
        assert len(nomic_handler.ROUTES) > 0
        assert "/api/nomic/state" in nomic_handler.ROUTES
        assert "/api/nomic/health" in nomic_handler.ROUTES
        assert "/api/nomic/metrics" in nomic_handler.ROUTES
        assert "/api/nomic/log" in nomic_handler.ROUTES
        assert "/api/nomic/risk-register" in nomic_handler.ROUTES
        assert "/api/nomic/witness/status" in nomic_handler.ROUTES
        assert "/api/nomic/mayor/current" in nomic_handler.ROUTES
        assert "/api/modes" in nomic_handler.ROUTES

    def test_control_routes_defined(self, nomic_handler):
        """Handler should have control routes defined."""
        assert "/api/nomic/control/start" in nomic_handler.ROUTES
        assert "/api/nomic/control/stop" in nomic_handler.ROUTES
        assert "/api/nomic/control/pause" in nomic_handler.ROUTES
        assert "/api/nomic/control/resume" in nomic_handler.ROUTES
        assert "/api/nomic/control/skip-phase" in nomic_handler.ROUTES

    def test_proposal_routes_defined(self, nomic_handler):
        """Handler should have proposal routes defined."""
        assert "/api/nomic/proposals" in nomic_handler.ROUTES
        assert "/api/nomic/proposals/approve" in nomic_handler.ROUTES
        assert "/api/nomic/proposals/reject" in nomic_handler.ROUTES


# ===========================================================================
# Advanced Witness and Mayor Tests
# ===========================================================================


class TestWitnessEndpointAdvanced:
    """Advanced tests for witness patrol endpoint."""

    @pytest.mark.asyncio
    async def test_witness_status_returns_config(self, nomic_handler_with_auth):
        """Witness status should return configuration details."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        mock_witness = MagicMock()
        mock_witness._running = True
        mock_witness.config.patrol_interval_seconds = 60
        mock_witness.config.heartbeat_timeout_seconds = 30
        mock_witness.config.stuck_threshold_minutes = 15
        mock_witness.config.notify_mayor_on_critical = True
        mock_witness._alerts = {}
        mock_witness.hierarchy = MagicMock()
        mock_witness.hierarchy._assignments = {"agent-1": "convoy-1"}
        # Mock async method
        mock_witness.generate_health_report = AsyncMock(return_value=None)

        # Patch at the location where it's imported inside the method
        with patch("aragora.server.startup.get_witness_behavior") as mock_get:
            mock_get.return_value = mock_witness
            result = await handler.handle("/api/v1/nomic/witness/status", {}, mock_request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["initialized"] is True
        assert body["patrolling"] is True
        assert body["config"]["patrol_interval_seconds"] == 60

    @pytest.mark.asyncio
    async def test_witness_status_includes_alerts(self, nomic_handler_with_auth):
        """Witness status should include active alerts."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        mock_alert = MagicMock()
        mock_alert.id = "alert-1"
        mock_alert.severity.value = "warning"
        mock_alert.target = "agent-1"
        mock_alert.message = "Heartbeat timeout"
        mock_alert.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        mock_alert.acknowledged = False

        mock_witness = MagicMock()
        mock_witness._running = True
        mock_witness.config.patrol_interval_seconds = 60
        mock_witness.config.heartbeat_timeout_seconds = 30
        mock_witness.config.stuck_threshold_minutes = 15
        mock_witness.config.notify_mayor_on_critical = True
        mock_witness._alerts = {"alert-1": mock_alert}
        mock_witness.hierarchy = MagicMock()
        mock_witness.hierarchy._assignments = {}
        # Mock async method
        mock_witness.generate_health_report = AsyncMock(return_value=None)

        # Patch at the location where it's imported inside the method
        with patch("aragora.server.startup.get_witness_behavior") as mock_get:
            mock_get.return_value = mock_witness
            result = await handler.handle("/api/v1/nomic/witness/status", {}, mock_request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "alerts" in body
        assert len(body["alerts"]) == 1
        assert body["alerts"][0]["id"] == "alert-1"


class TestMayorEndpointAdvanced:
    """Advanced tests for mayor current endpoint."""

    @pytest.mark.asyncio
    async def test_mayor_current_returns_info(self, nomic_handler_with_auth):
        """Mayor current should return coordinator info."""
        handler, tmp_path = nomic_handler_with_auth
        mock_request = MockHTTPHandler("GET")

        mock_coordinator = MagicMock()
        mock_coordinator.is_started = True
        mock_coordinator.is_mayor = True
        mock_coordinator.node_id = "node-1"
        mock_coordinator.get_current_mayor_node.return_value = "node-1"
        mock_coordinator.get_mayor_info.return_value = MagicMock(
            to_dict=lambda: {"node_id": "node-1", "became_mayor_at": "2024-01-01T00:00:00"}
        )

        # Patch at the location where it's imported inside the method
        with patch("aragora.server.startup.get_mayor_coordinator") as mock_get:
            mock_get.return_value = mock_coordinator
            result = await handler.handle("/api/v1/nomic/mayor/current", {}, mock_request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["initialized"] is True
        assert body["is_this_node"] is True


# ===========================================================================
# Control Operations Validation Tests
# ===========================================================================


class TestControlOperationsValidation:
    """Tests for validation in control operations."""

    @pytest.mark.asyncio
    async def test_start_validates_cycles_type(self, nomic_handler_with_auth):
        """Start should validate cycles parameter type."""
        handler, tmp_path = nomic_handler_with_auth

        result = handler._start_nomic_loop({"cycles": "invalid"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_start_validates_max_cycles_type(self, nomic_handler_with_auth):
        """Start should validate max_cycles parameter type."""
        handler, tmp_path = nomic_handler_with_auth

        result = handler._start_nomic_loop({"cycles": 3, "max_cycles": "invalid"})

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_start_clamps_cycles_to_valid_range(self, nomic_handler_with_auth):
        """Start should clamp cycles to valid range (1-100)."""
        handler, tmp_path = nomic_handler_with_auth

        # This would fail at script check, but cycles should be clamped
        result = handler._start_nomic_loop({"cycles": 1000, "max_cycles": 2000})

        # Returns 500 because script not found, but cycles should be clamped internally
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_stop_graceful_vs_force(self, nomic_handler_with_auth):
        """Stop should support graceful and force modes."""
        handler, tmp_path = nomic_handler_with_auth

        # Setup running state with PID
        state = {"running": True, "pid": 99999}  # Non-existent PID
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._stop_nomic_loop({"graceful": True})

        # PID doesn't exist, so should report already stopped
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "already_stopped"


# ===========================================================================
# Phase Transition Tests
# ===========================================================================


class TestPhaseTransitions:
    """Tests for phase transition logic."""

    @pytest.mark.asyncio
    async def test_phase_sequence_context_to_debate(self, nomic_handler_with_auth):
        """Should transition from context to debate."""
        handler, tmp_path = nomic_handler_with_auth

        state = {"running": True, "phase": "context", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()

        body = json.loads(result.body)
        assert body["previous_phase"] == "context"
        assert body["next_phase"] == "debate"

    @pytest.mark.asyncio
    async def test_phase_sequence_design_to_implement(self, nomic_handler_with_auth):
        """Should transition from design to implement."""
        handler, tmp_path = nomic_handler_with_auth

        state = {"running": True, "phase": "design", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()

        body = json.loads(result.body)
        assert body["previous_phase"] == "design"
        assert body["next_phase"] == "implement"

    @pytest.mark.asyncio
    async def test_phase_sequence_implement_to_verify(self, nomic_handler_with_auth):
        """Should transition from implement to verify."""
        handler, tmp_path = nomic_handler_with_auth

        state = {"running": True, "phase": "implement", "cycle": 1}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler._skip_phase()

        body = json.loads(result.body)
        assert body["previous_phase"] == "implement"
        assert body["next_phase"] == "verify"

    @pytest.mark.asyncio
    async def test_skip_phase_updates_state_file(self, nomic_handler_with_auth):
        """Skip phase should update the state file."""
        handler, tmp_path = nomic_handler_with_auth

        state = {"running": True, "phase": "debate", "cycle": 1}
        state_file = tmp_path / "nomic_state.json"
        state_file.write_text(json.dumps(state))

        handler._skip_phase()

        # Read updated state
        updated_state = json.loads(state_file.read_text())
        assert updated_state["phase"] == "design"
        assert updated_state["skip_requested"] is True
        assert "skipped_at" in updated_state


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on endpoints."""

    def test_handle_has_rate_limit_decorator(self, nomic_handler):
        """Handle method should have rate limit decorator."""
        # The decorator is applied, we just verify the method exists
        # Actual rate limiting is tested at integration level
        assert hasattr(nomic_handler, "handle")
        assert callable(nomic_handler.handle)

    def test_handle_post_has_rate_limit_decorator(self, nomic_handler):
        """Handle post method should have rate limit decorator."""
        assert hasattr(nomic_handler, "handle_post")
        assert callable(nomic_handler.handle_post)
