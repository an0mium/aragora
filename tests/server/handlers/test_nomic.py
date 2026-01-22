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
from unittest.mock import MagicMock, patch

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

    def test_nomic_state_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic state should return 503 when directory not configured."""
        result = nomic_handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body

    def test_nomic_state_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return not_running when no state file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["state"] == "not_running"
        assert body["cycle"] == 0

    def test_nomic_state_with_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return state from file."""
        handler, tmp_path = nomic_handler_with_dir

        state = {"state": "running", "cycle": 5, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["state"] == "running"
        assert body["cycle"] == 5
        assert body["phase"] == "debate"

    def test_nomic_state_invalid_json(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic state should return error for invalid JSON."""
        handler, tmp_path = nomic_handler_with_dir

        (tmp_path / "nomic_state.json").write_text("not valid json {")

        result = handler.handle("/api/v1/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body


class TestNomicHealth:
    """Tests for GET /api/nomic/health endpoint."""

    def test_nomic_health_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic health should return 503 when directory not configured."""
        result = nomic_handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_nomic_health_not_running(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return not_running when no state file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "not_running"
        assert body["cycle"] == 0

    def test_nomic_health_healthy(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return healthy when recent activity."""
        handler, tmp_path = nomic_handler_with_dir

        # Recent timestamp (within 30 minutes)
        recent_time = datetime.now().isoformat()
        state = {"cycle": 3, "phase": "design", "last_update": recent_time}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        assert body["cycle"] == 3
        assert body["phase"] == "design"

    def test_nomic_health_stalled(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic health should return stalled when no recent activity."""
        handler, tmp_path = nomic_handler_with_dir

        # Old timestamp (more than 30 minutes ago)
        old_time = (datetime.now() - timedelta(hours=1)).isoformat()
        state = {"cycle": 2, "phase": "implement", "last_update": old_time}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/v1/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "stalled"
        assert body["stall_duration_seconds"] is not None
        assert body["stall_duration_seconds"] > 1800  # More than 30 minutes


class TestNomicLog:
    """Tests for GET /api/nomic/log endpoint."""

    def test_nomic_log_no_dir(self, nomic_handler, mock_http_handler):
        """Nomic log should return 503 when directory not configured."""
        result = nomic_handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_nomic_log_no_file(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should return empty list when no log file exists."""
        handler, tmp_path = nomic_handler_with_dir

        result = handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["lines"] == []
        assert body["total"] == 0

    def test_nomic_log_with_content(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should return log lines."""
        handler, tmp_path = nomic_handler_with_dir

        log_content = "Line 1\nLine 2\nLine 3\n"
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = handler.handle("/api/v1/nomic/log", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["lines"]) == 3
        assert body["total"] == 3

    def test_nomic_log_with_lines_param(self, nomic_handler_with_dir, mock_http_handler):
        """Nomic log should respect lines parameter."""
        handler, tmp_path = nomic_handler_with_dir

        log_content = "\n".join([f"Line {i}" for i in range(100)])
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = handler.handle("/api/v1/nomic/log", {"lines": "10"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["showing"] == 10
        assert body["total"] == 100


class TestNomicMetrics:
    """Tests for GET /api/nomic/metrics endpoint."""

    def test_nomic_metrics_returns_summary(self, nomic_handler, mock_http_handler):
        """Nomic metrics should return a metrics summary."""
        with patch.object(nomic_handler, "_get_nomic_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(
                body=json.dumps({"metrics": {"cycles": 10}}).encode(), status_code=200
            )
            result = nomic_handler.handle("/api/v1/nomic/metrics", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200


class TestRiskRegister:
    """Tests for GET /api/nomic/risk-register endpoint."""

    def test_risk_register_returns_entries(self, nomic_handler, mock_http_handler):
        """Risk register should return risk entries."""
        with patch.object(nomic_handler, "_get_risk_register") as mock_risk:
            mock_risk.return_value = MagicMock(
                body=json.dumps({"risks": [], "total": 0}).encode(), status_code=200
            )
            result = nomic_handler.handle("/api/v1/nomic/risk-register", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200

    def test_risk_register_with_limit_param(self, nomic_handler, mock_http_handler):
        """Risk register should respect limit parameter."""
        with patch.object(nomic_handler, "_get_risk_register") as mock_risk:
            mock_risk.return_value = MagicMock(
                body=json.dumps({"risks": [], "total": 0}).encode(), status_code=200
            )
            result = nomic_handler.handle(
                "/api/v1/nomic/risk-register", {"limit": "10"}, mock_http_handler
            )

        assert result is not None
        # Verify limit was passed (clamped to valid range)
        mock_risk.assert_called_once_with(10)


class TestModes:
    """Tests for GET /api/modes endpoint."""

    def test_get_modes_returns_list(self, nomic_handler, mock_http_handler):
        """Get modes should return available operational modes."""
        result = nomic_handler.handle("/api/v1/modes", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should have modes array
        assert "modes" in body or isinstance(body, list)


class TestHandleReturnsNone:
    """Tests for handle returning None for non-matching paths."""

    def test_handle_returns_none_for_non_matching(self, nomic_handler, mock_http_handler):
        """Handle should return None for paths that can_handle returns False for."""
        # Force a path that would bypass can_handle check
        result = nomic_handler.handle("/api/v1/nomic/nonexistent", {}, mock_http_handler)
        assert result is None
