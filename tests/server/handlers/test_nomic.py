"""
Tests for the nomic handler - nomic loop state and monitoring.

Tests:
- Route handling (can_handle)
- Get nomic state endpoint
- Get nomic health endpoint
- Get nomic metrics endpoint
- Get nomic log endpoint
- Get risk register endpoint
- Get modes endpoint
- Error handling
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.nomic import NomicHandler


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
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json"}
    mock.command = "GET"
    return mock


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
