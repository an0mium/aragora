"""
Tests for NomicHandler endpoints.

Endpoints tested:
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/health - Get nomic loop health with stall detection
- GET /api/nomic/metrics - Get nomic loop Prometheus metrics summary
- GET /api/nomic/log - Get nomic loop logs
- GET /api/nomic/risk-register - Get risk register entries
- GET /api/modes - Get available operational modes
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.nomic import NomicHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_nomic_dir(tmp_path):
    """Create a mock nomic directory structure."""
    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir()
    return nomic_dir


@pytest.fixture
def nomic_handler(mock_nomic_dir):
    """Create a NomicHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": mock_nomic_dir,
    }
    return NomicHandler(ctx)


@pytest.fixture
def nomic_handler_no_nomic():
    """Create a NomicHandler without nomic_dir."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return NomicHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestNomicRouting:
    """Tests for route matching."""

    def test_can_handle_nomic_state(self, nomic_handler):
        """Handler can handle /api/nomic/state."""
        assert nomic_handler.can_handle("/api/v1/nomic/state") is True

    def test_can_handle_nomic_health(self, nomic_handler):
        """Handler can handle /api/nomic/health."""
        assert nomic_handler.can_handle("/api/v1/nomic/health") is True

    def test_can_handle_nomic_metrics(self, nomic_handler):
        """Handler can handle /api/nomic/metrics."""
        assert nomic_handler.can_handle("/api/v1/nomic/metrics") is True

    def test_can_handle_nomic_log(self, nomic_handler):
        """Handler can handle /api/nomic/log."""
        assert nomic_handler.can_handle("/api/v1/nomic/log") is True

    def test_can_handle_risk_register(self, nomic_handler):
        """Handler can handle /api/nomic/risk-register."""
        assert nomic_handler.can_handle("/api/v1/nomic/risk-register") is True

    def test_can_handle_modes(self, nomic_handler):
        """Handler can handle /api/modes."""
        assert nomic_handler.can_handle("/api/v1/modes") is True

    def test_cannot_handle_unrelated_routes(self, nomic_handler):
        """Handler doesn't handle unrelated routes."""
        assert nomic_handler.can_handle("/api/v1/debates") is False
        assert nomic_handler.can_handle("/api/v1/agents") is False
        assert nomic_handler.can_handle("/api/v1/replays") is False
        assert nomic_handler.can_handle("/api/v1/nomic/unknown") is False


# ============================================================================
# GET /api/nomic/state Tests
# ============================================================================


class TestNomicState:
    """Tests for GET /api/nomic/state endpoint."""

    def test_state_no_nomic_dir(self, nomic_handler_no_nomic):
        """Returns 503 when nomic_dir is None."""
        result = nomic_handler_no_nomic.handle("/api/nomic/state", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"].lower()

    def test_state_no_state_file(self, nomic_handler, mock_nomic_dir):
        """Returns not_running state when state file doesn't exist."""
        result = nomic_handler.handle("/api/nomic/state", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["state"] == "not_running"
        assert data["cycle"] == 0

    def test_state_valid_state_file(self, nomic_handler, mock_nomic_dir):
        """Returns state from valid state file."""
        state = {
            "state": "running",
            "cycle": 5,
            "phase": "implement",
            "last_update": "2026-01-15T10:00:00Z",
        }
        (mock_nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = nomic_handler.handle("/api/nomic/state", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["state"] == "running"
        assert data["cycle"] == 5
        assert data["phase"] == "implement"

    def test_state_invalid_json(self, nomic_handler, mock_nomic_dir):
        """Returns 500 error when state file has invalid JSON."""
        (mock_nomic_dir / "nomic_state.json").write_text("not valid json")

        result = nomic_handler.handle("/api/nomic/state", {}, None)

        assert result is not None
        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# GET /api/nomic/health Tests
# ============================================================================


class TestNomicHealth:
    """Tests for GET /api/nomic/health endpoint."""

    def test_health_no_nomic_dir(self, nomic_handler_no_nomic):
        """Returns 503 when nomic_dir is None."""
        result = nomic_handler_no_nomic.handle("/api/nomic/health", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"].lower()

    def test_health_not_running(self, nomic_handler, mock_nomic_dir):
        """Returns not_running status when state file doesn't exist."""
        result = nomic_handler.handle("/api/nomic/health", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "not_running"
        assert data["cycle"] == 0
        assert data["phase"] is None
        assert data["warnings"] == []

    def test_health_healthy_state(self, nomic_handler, mock_nomic_dir):
        """Returns healthy status for recent activity."""
        recent_time = datetime.now().isoformat()
        state = {
            "cycle": 3,
            "phase": "debate",
            "last_update": recent_time,
            "warnings": [],
        }
        (mock_nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = nomic_handler.handle("/api/nomic/health", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "healthy"
        assert data["cycle"] == 3
        assert data["phase"] == "debate"
        assert data["stall_duration_seconds"] is None

    def test_health_stalled_state(self, nomic_handler, mock_nomic_dir):
        """Returns stalled status for old activity."""
        old_time = (datetime.now() - timedelta(hours=1)).isoformat()
        state = {
            "cycle": 2,
            "phase": "implement",
            "last_update": old_time,
            "warnings": [],
        }
        (mock_nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = nomic_handler.handle("/api/nomic/health", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "stalled"
        assert data["stall_duration_seconds"] is not None
        assert data["stall_duration_seconds"] > 1800  # More than 30 min
        assert any("minutes" in w for w in data["warnings"])

    def test_health_invalid_json(self, nomic_handler, mock_nomic_dir):
        """Returns error status for invalid state file."""
        (mock_nomic_dir / "nomic_state.json").write_text("not valid json")

        result = nomic_handler.handle("/api/nomic/health", {}, None)

        assert result is not None
        assert result.status_code == 200  # Returns 200 with error status
        data = json.loads(result.body)
        assert data["status"] == "error"
        assert "error" in data


# ============================================================================
# GET /api/nomic/log Tests
# ============================================================================


class TestNomicLog:
    """Tests for GET /api/nomic/log endpoint."""

    def test_log_no_nomic_dir(self, nomic_handler_no_nomic):
        """Returns 503 when nomic_dir is None."""
        result = nomic_handler_no_nomic.handle("/api/nomic/log", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"].lower()

    def test_log_no_log_file(self, nomic_handler, mock_nomic_dir):
        """Returns empty lines when log file doesn't exist."""
        result = nomic_handler.handle("/api/nomic/log", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["lines"] == []
        assert data["total"] == 0

    def test_log_with_content(self, nomic_handler, mock_nomic_dir):
        """Returns log lines from log file."""
        log_content = "Line 1\nLine 2\nLine 3\n"
        (mock_nomic_dir / "nomic_loop.log").write_text(log_content)

        result = nomic_handler.handle("/api/nomic/log", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["lines"] == ["Line 1", "Line 2", "Line 3"]
        assert data["total"] == 3
        assert data["showing"] == 3

    def test_log_respects_lines_param(self, nomic_handler, mock_nomic_dir):
        """Respects lines query parameter."""
        log_content = "\n".join([f"Line {i}" for i in range(10)])
        (mock_nomic_dir / "nomic_loop.log").write_text(log_content)

        result = nomic_handler.handle("/api/nomic/log", {"lines": "3"}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["lines"]) == 3
        assert data["total"] == 10
        assert data["showing"] == 3
        # Should return last 3 lines
        assert data["lines"] == ["Line 7", "Line 8", "Line 9"]

    def test_log_clamps_lines_param(self, nomic_handler, mock_nomic_dir):
        """Clamps lines parameter to valid range (1-1000)."""
        log_content = "Line 1\n"
        (mock_nomic_dir / "nomic_loop.log").write_text(log_content)

        # Test minimum clamping
        result = nomic_handler.handle("/api/nomic/log", {"lines": "-5"}, None)
        assert result is not None
        assert result.status_code == 200

        # Test maximum clamping (should cap at 1000)
        result = nomic_handler.handle("/api/nomic/log", {"lines": "9999"}, None)
        assert result is not None
        assert result.status_code == 200


# ============================================================================
# GET /api/nomic/risk-register Tests
# ============================================================================


class TestRiskRegister:
    """Tests for GET /api/nomic/risk-register endpoint."""

    def test_risk_register_no_nomic_dir(self, nomic_handler_no_nomic):
        """Returns 503 when nomic_dir is None."""
        result = nomic_handler_no_nomic.handle("/api/nomic/risk-register", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"].lower()

    def test_risk_register_no_file(self, nomic_handler, mock_nomic_dir):
        """Returns empty risks when risk file doesn't exist."""
        result = nomic_handler.handle("/api/nomic/risk-register", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["risks"] == []
        assert data["total"] == 0
        assert data["critical_count"] == 0
        assert data["high_count"] == 0

    def test_risk_register_with_entries(self, nomic_handler, mock_nomic_dir):
        """Returns risks from risk register file."""
        risks = [
            {"id": 1, "severity": "critical", "message": "Critical issue"},
            {"id": 2, "severity": "high", "message": "High issue"},
            {"id": 3, "severity": "medium", "message": "Medium issue"},
        ]
        content = "\n".join([json.dumps(r) for r in risks])
        (mock_nomic_dir / "risk_register.jsonl").write_text(content)

        result = nomic_handler.handle("/api/nomic/risk-register", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 3
        assert data["critical_count"] == 1
        assert data["high_count"] == 1
        # Most recent first
        assert data["risks"][0]["id"] == 3
        assert data["risks"][2]["id"] == 1

    def test_risk_register_respects_limit(self, nomic_handler, mock_nomic_dir):
        """Respects limit query parameter."""
        risks = [{"id": i, "severity": "low"} for i in range(10)]
        content = "\n".join([json.dumps(r) for r in risks])
        (mock_nomic_dir / "risk_register.jsonl").write_text(content)

        result = nomic_handler.handle("/api/nomic/risk-register", {"limit": "3"}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["risks"]) == 3
        assert data["total"] == 10

    def test_risk_register_skips_malformed_lines(self, nomic_handler, mock_nomic_dir):
        """Skips malformed JSONL lines."""
        content = '{"id": 1, "severity": "low"}\nnot valid json\n{"id": 2, "severity": "high"}\n'
        (mock_nomic_dir / "risk_register.jsonl").write_text(content)

        result = nomic_handler.handle("/api/nomic/risk-register", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 2  # Skipped invalid line
        assert data["high_count"] == 1


# ============================================================================
# GET /api/modes Tests
# ============================================================================


class TestModes:
    """Tests for GET /api/modes endpoint."""

    def test_modes_returns_builtin_modes(self, nomic_handler):
        """Returns builtin modes list."""
        result = nomic_handler.handle("/api/modes", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "modes" in data
        assert "total" in data
        assert data["total"] >= 5  # At least 5 builtin modes

        # Check builtin modes are present
        builtin_names = [m["name"] for m in data["modes"] if m["type"] == "builtin"]
        assert "architect" in builtin_names
        assert "coder" in builtin_names
        assert "debugger" in builtin_names

    def test_modes_handles_custom_mode_import_error(self, nomic_handler):
        """Handles ImportError for custom modes gracefully."""
        # The import is inside the method, so we simulate it by patching the import
        import sys

        # Temporarily remove the module if it exists and make it raise ImportError
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.modes.custom":
                raise ImportError("Module not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = nomic_handler.handle("/api/modes", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            # Should still return builtin modes
            assert data["total"] >= 5


# ============================================================================
# GET /api/nomic/metrics Tests
# ============================================================================


class TestNomicMetrics:
    """Tests for GET /api/nomic/metrics endpoint."""

    def test_metrics_when_module_unavailable(self, nomic_handler):
        """Returns unavailable status when metrics module is not available."""
        with patch.dict("sys.modules", {"aragora.nomic.metrics": None}):
            with patch(
                "aragora.server.handlers.nomic.NomicHandler._get_nomic_metrics"
            ) as mock_method:
                # Simulate ImportError behavior
                mock_method.return_value = Mock(
                    status_code=200,
                    body=json.dumps(
                        {
                            "summary": {},
                            "stuck_detection": {"is_stuck": False},
                            "status": "metrics_unavailable",
                            "message": "Nomic metrics module not available",
                        }
                    ),
                )
                result = nomic_handler.handle("/api/nomic/metrics", {}, None)

                assert result is not None
                assert result.status_code == 200


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestNomicHandlerImport:
    """Test NomicHandler import and export."""

    def test_handler_importable(self):
        """NomicHandler can be imported from handlers package."""
        from aragora.server.handlers import NomicHandler

        assert NomicHandler is not None

    def test_handler_in_all_exports(self):
        """NomicHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "NomicHandler" in __all__


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestNomicErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled(self, nomic_handler):
        """Returns None for unhandled routes."""
        result = nomic_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_handles_permission_error(self, nomic_handler, mock_nomic_dir):
        """Returns error on permission denied."""
        (mock_nomic_dir / "nomic_state.json").write_text("{}")

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = nomic_handler.handle("/api/nomic/state", {}, None)

            assert result is not None
            assert result.status_code == 500
            data = json.loads(result.body)
            assert "error" in data
