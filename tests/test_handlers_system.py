"""
Tests for SystemHandler endpoints.

Endpoints tested:
- GET /api/health - Health check
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/log - Get nomic loop logs
- GET /api/modes - Get available operational modes
- GET /api/history/cycles - Get cycle history
- GET /api/history/events - Get event history
- GET /api/history/debates - Get debate history
- GET /api/history/summary - Get history summary
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.system import SystemHandler
from aragora.server.handlers.base import clear_cache


@pytest.fixture
def mock_storage():
    """Create a mock storage instance."""
    storage = Mock()
    # Mock both methods for backwards compatibility
    mock_debates = [
        Mock(slug="debate-1", loop_id="loop-001"),
        Mock(slug="debate-2", loop_id="loop-001"),
        Mock(slug="debate-3", loop_id="loop-002"),
    ]
    storage.list_recent.return_value = mock_debates
    storage.list_debates.return_value = mock_debates  # Alias for backwards compat
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_leaderboard.return_value = [
        Mock(agent_name="claude"),
        Mock(agent_name="gpt4"),
    ]
    return elo


@pytest.fixture
def system_handler(mock_storage, mock_elo_system, tmp_path):
    """Create a SystemHandler with mocks."""
    ctx = {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": tmp_path,
    }
    return SystemHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestSystemHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_health(self, system_handler):
        """Should handle /api/health."""
        assert system_handler.can_handle("/api/health") is True

    def test_can_handle_nomic_state(self, system_handler):
        """Should handle /api/nomic/state."""
        assert system_handler.can_handle("/api/nomic/state") is True

    def test_can_handle_nomic_log(self, system_handler):
        """Should handle /api/nomic/log."""
        assert system_handler.can_handle("/api/nomic/log") is True

    def test_can_handle_modes(self, system_handler):
        """Should handle /api/modes."""
        assert system_handler.can_handle("/api/modes") is True

    def test_can_handle_history_cycles(self, system_handler):
        """Should handle /api/history/cycles."""
        assert system_handler.can_handle("/api/history/cycles") is True

    def test_can_handle_history_events(self, system_handler):
        """Should handle /api/history/events."""
        assert system_handler.can_handle("/api/history/events") is True

    def test_can_handle_history_debates(self, system_handler):
        """Should handle /api/history/debates."""
        assert system_handler.can_handle("/api/history/debates") is True

    def test_can_handle_history_summary(self, system_handler):
        """Should handle /api/history/summary."""
        assert system_handler.can_handle("/api/history/summary") is True

    def test_cannot_handle_unknown_routes(self, system_handler):
        """Should not handle unknown routes."""
        assert system_handler.can_handle("/api/unknown") is False
        assert system_handler.can_handle("/api/debates") is False
        assert system_handler.can_handle("/api/leaderboard") is False

    def test_handle_returns_none_for_unknown(self, system_handler):
        """Should return None for unknown paths."""
        result = system_handler.handle("/api/unknown", {}, None)
        assert result is None


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_returns_healthy_status(self, system_handler):
        """Should return healthy status."""
        result = system_handler.handle("/api/health", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "healthy"

    def test_returns_component_status(self, system_handler):
        """Should return component status in checks dict."""
        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert "checks" in data
        assert "database" in data["checks"]
        assert "elo_system" in data["checks"]
        assert "nomic_dir" in data["checks"]

    def test_storage_available(self, system_handler):
        """Should show database as available."""
        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert data["checks"]["database"]["healthy"] is True

    def test_elo_system_available(self, system_handler):
        """Should show ELO system as available."""
        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert data["checks"]["elo_system"]["healthy"] is True

    def test_nomic_dir_available(self, system_handler, tmp_path):
        """Should show nomic_dir as available."""
        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert data["checks"]["nomic_dir"]["healthy"] is True

    def test_returns_version(self, system_handler):
        """Should return version."""
        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert "version" in data

    def test_shows_missing_components(self):
        """Should show missing components as unavailable."""
        handler = SystemHandler({})
        result = handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        assert data["checks"]["database"]["healthy"] is False
        assert data["checks"]["elo_system"]["healthy"] is False


# ============================================================================
# Nomic State Tests
# ============================================================================

class TestNomicStateEndpoint:
    """Tests for /api/nomic/state endpoint."""

    def test_returns_state_from_file(self, system_handler, tmp_path):
        """Should return state from state file."""
        state = {"state": "running", "cycle": 5, "phase": "debate"}
        (tmp_path / "nomic_state.json").write_text(json.dumps(state))

        result = system_handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["state"] == "running"
        assert data["cycle"] == 5

    def test_returns_not_running_when_no_file(self, system_handler):
        """Should return not_running when state file doesn't exist."""
        result = system_handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["state"] == "not_running"

    def test_returns_503_without_nomic_dir(self):
        """Should return 503 without nomic_dir."""
        handler = SystemHandler({})
        result = handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 503

    def test_returns_500_on_invalid_json(self, system_handler, tmp_path):
        """Should return 500 on invalid JSON."""
        (tmp_path / "nomic_state.json").write_text("not valid json {")

        result = system_handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 500


# ============================================================================
# Nomic Log Tests
# ============================================================================

class TestNomicLogEndpoint:
    """Tests for /api/nomic/log endpoint."""

    def test_returns_log_lines(self, system_handler, tmp_path):
        """Should return log lines."""
        log_content = "Line 1\nLine 2\nLine 3\n"
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = system_handler.handle("/api/nomic/log", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["lines"]) == 3
        assert data["total"] == 3

    def test_respects_lines_parameter(self, system_handler, tmp_path):
        """Should respect lines parameter."""
        log_content = "\n".join([f"Line {i}" for i in range(100)])
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = system_handler.handle("/api/nomic/log", {"lines": "10"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["lines"]) == 10
        assert data["showing"] == 10

    def test_caps_lines_at_1000(self, system_handler, tmp_path):
        """Should cap lines at 1000."""
        log_content = "\n".join([f"Line {i}" for i in range(2000)])
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = system_handler.handle("/api/nomic/log", {"lines": "5000"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["showing"] <= 1000

    def test_returns_empty_when_no_log(self, system_handler):
        """Should return empty when log doesn't exist."""
        result = system_handler.handle("/api/nomic/log", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["lines"] == []
        assert data["total"] == 0

    def test_returns_503_without_nomic_dir(self):
        """Should return 503 without nomic_dir."""
        handler = SystemHandler({})
        result = handler.handle("/api/nomic/log", {}, None)

        assert result.status_code == 503

    def test_strips_line_endings(self, system_handler, tmp_path):
        """Should strip line endings from log lines."""
        log_content = "Line with spaces   \nLine with tabs\t\n"
        (tmp_path / "nomic_loop.log").write_text(log_content)

        result = system_handler.handle("/api/nomic/log", {}, None)

        data = json.loads(result.body)
        # Lines should be stripped
        assert all(not line.endswith(('\n', '\r', ' ', '\t')) for line in data["lines"])


# ============================================================================
# Modes Tests
# ============================================================================

class TestModesEndpoint:
    """Tests for /api/modes endpoint."""

    def test_returns_modes_list(self, system_handler, tmp_path):
        """Should return modes list."""
        with patch('aragora.modes.custom.CustomModeLoader') as MockLoader:
            mock_loader = Mock()
            # Mock load_all() which is what the handler actually calls
            mock_mode = Mock()
            mock_mode.name = "custom_mode"
            mock_mode.description = "A custom mode"
            mock_loader.load_all.return_value = [mock_mode]
            MockLoader.return_value = mock_loader

            result = system_handler.handle("/api/modes", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "modes" in data

    def test_returns_builtin_modes_without_nomic_dir(self):
        """Should return builtin modes even without nomic_dir."""
        handler = SystemHandler({})
        result = handler.handle("/api/modes", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should return 5 builtin modes
        assert len(data["modes"]) == 5
        assert all(m["type"] == "builtin" for m in data["modes"])
        mode_names = {m["name"] for m in data["modes"]}
        assert mode_names == {"architect", "coder", "debugger", "orchestrator", "reviewer"}

    def test_returns_builtin_modes_on_custom_loader_exception(self, system_handler):
        """Should return builtin modes even when custom loader fails."""
        with patch('aragora.modes.custom.CustomModeLoader') as MockLoader:
            # Use an exception type that the handler actually catches
            MockLoader.side_effect = OSError("Mode error")

            result = system_handler.handle("/api/modes", {}, None)

            # Should still succeed with builtin modes
            assert result.status_code == 200
            data = json.loads(result.body)
            assert len(data["modes"]) == 5
            assert all(m["type"] == "builtin" for m in data["modes"])


# ============================================================================
# History Cycles Tests
# ============================================================================

class TestHistoryCyclesEndpoint:
    """Tests for /api/history/cycles endpoint."""

    def test_returns_cycles_from_file(self, system_handler, tmp_path):
        """Should return cycles from file."""
        cycles = [
            {"id": "c1", "loop_id": "loop-001"},
            {"id": "c2", "loop_id": "loop-002"},
        ]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))

        result = system_handler.handle("/api/history/cycles", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["cycles"]) == 2

    def test_filters_by_loop_id(self, system_handler, tmp_path):
        """Should filter by loop_id."""
        cycles = [
            {"id": "c1", "loop_id": "loop-001"},
            {"id": "c2", "loop_id": "loop-002"},
        ]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))

        result = system_handler.handle("/api/history/cycles", {"loop_id": "loop-001"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["cycles"]) == 1
        assert data["cycles"][0]["loop_id"] == "loop-001"

    def test_validates_loop_id(self, system_handler):
        """Should validate loop_id parameter."""
        result = system_handler.handle("/api/history/cycles", {"loop_id": "../../../etc"}, None)

        assert result.status_code == 400

    def test_respects_limit(self, system_handler, tmp_path):
        """Should respect limit parameter."""
        cycles = [{"id": f"c{i}", "loop_id": "loop-001"} for i in range(100)]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))

        result = system_handler.handle("/api/history/cycles", {"limit": "10"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["cycles"]) == 10

    def test_returns_empty_when_no_file(self, system_handler):
        """Should return empty when file doesn't exist."""
        result = system_handler.handle("/api/history/cycles", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["cycles"] == []

    def test_handles_exception(self, system_handler, tmp_path):
        """Should return 500 on exception."""
        (tmp_path / "cycles.json").write_text("invalid json")

        result = system_handler.handle("/api/history/cycles", {}, None)

        assert result.status_code == 500


# ============================================================================
# History Events Tests
# ============================================================================

class TestHistoryEventsEndpoint:
    """Tests for /api/history/events endpoint."""

    def test_returns_events_from_file(self, system_handler, tmp_path):
        """Should return events from file."""
        events = [
            {"id": "e1", "type": "start", "loop_id": "loop-001"},
            {"id": "e2", "type": "end", "loop_id": "loop-001"},
        ]
        (tmp_path / "events.json").write_text(json.dumps(events))

        result = system_handler.handle("/api/history/events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["events"]) == 2

    def test_filters_by_loop_id(self, system_handler, tmp_path):
        """Should filter by loop_id."""
        events = [
            {"id": "e1", "loop_id": "loop-001"},
            {"id": "e2", "loop_id": "loop-002"},
        ]
        (tmp_path / "events.json").write_text(json.dumps(events))

        result = system_handler.handle("/api/history/events", {"loop_id": "loop-001"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["events"]) == 1

    def test_validates_loop_id(self, system_handler):
        """Should validate loop_id parameter."""
        result = system_handler.handle("/api/history/events", {"loop_id": "'; DROP TABLE"}, None)

        assert result.status_code == 400

    def test_respects_limit(self, system_handler, tmp_path):
        """Should respect limit parameter."""
        events = [{"id": f"e{i}", "loop_id": "loop-001"} for i in range(200)]
        (tmp_path / "events.json").write_text(json.dumps(events))

        result = system_handler.handle("/api/history/events", {"limit": "50"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["events"]) == 50

    def test_returns_empty_when_no_file(self, system_handler):
        """Should return empty when file doesn't exist."""
        result = system_handler.handle("/api/history/events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["events"] == []


# ============================================================================
# History Debates Tests
# ============================================================================

class TestHistoryDebatesEndpoint:
    """Tests for /api/history/debates endpoint."""

    def test_returns_debates(self, system_handler):
        """Should return debates from storage."""
        result = system_handler.handle("/api/history/debates", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["debates"]) == 3

    def test_filters_by_loop_id(self, system_handler):
        """Should filter by loop_id."""
        result = system_handler.handle("/api/history/debates", {"loop_id": "loop-001"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["debates"]) == 2
        assert all(d["loop_id"] == "loop-001" for d in data["debates"])

    def test_validates_loop_id(self, system_handler):
        """Should validate loop_id parameter."""
        result = system_handler.handle("/api/history/debates", {"loop_id": "../../../etc"}, None)

        assert result.status_code == 400

    def test_returns_empty_without_storage(self):
        """Should return empty without storage."""
        handler = SystemHandler({})
        result = handler.handle("/api/history/debates", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debates"] == []

    def test_handles_exception(self, system_handler, mock_storage):
        """Should return 500 on exception."""
        mock_storage.list_recent.side_effect = Exception("DB error")

        result = system_handler.handle("/api/history/debates", {}, None)

        assert result.status_code == 500


# ============================================================================
# History Summary Tests
# ============================================================================

class TestHistorySummaryEndpoint:
    """Tests for /api/history/summary endpoint."""

    def test_returns_summary(self, system_handler):
        """Should return history summary."""
        result = system_handler.handle("/api/history/summary", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_debates" in data
        assert "total_agents" in data

    def test_counts_debates(self, system_handler):
        """Should count total debates."""
        result = system_handler.handle("/api/history/summary", {}, None)

        data = json.loads(result.body)
        assert data["total_debates"] == 3

    def test_counts_agents(self, system_handler):
        """Should count total agents."""
        result = system_handler.handle("/api/history/summary", {}, None)

        data = json.loads(result.body)
        assert data["total_agents"] == 2

    def test_validates_loop_id(self, system_handler):
        """Should validate loop_id parameter."""
        result = system_handler.handle("/api/history/summary", {"loop_id": "../../etc"}, None)

        assert result.status_code == 400

    def test_handles_missing_components(self):
        """Should handle missing storage and ELO."""
        handler = SystemHandler({})
        result = handler.handle("/api/history/summary", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_debates"] == 0
        assert data["total_agents"] == 0

    def test_handles_exception(self, system_handler, mock_storage):
        """Should return 500 on exception."""
        mock_storage.list_recent.side_effect = Exception("Summary error")

        result = system_handler.handle("/api/history/summary", {}, None)

        assert result.status_code == 500


# ============================================================================
# Limit Cap Tests
# ============================================================================

class TestLimitCaps:
    """Tests for limit parameter capping."""

    def test_cycles_caps_at_200(self, system_handler, tmp_path):
        """Should cap cycles limit at 200."""
        cycles = [{"id": f"c{i}"} for i in range(300)]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))

        result = system_handler.handle("/api/history/cycles", {"limit": "500"}, None)

        data = json.loads(result.body)
        assert len(data["cycles"]) == 200

    def test_events_caps_at_500(self, system_handler, tmp_path):
        """Should cap events limit at 500."""
        events = [{"id": f"e{i}"} for i in range(700)]
        (tmp_path / "events.json").write_text(json.dumps(events))

        result = system_handler.handle("/api/history/events", {"limit": "1000"}, None)

        data = json.loads(result.body)
        assert len(data["events"]) == 500


# ============================================================================
# Edge Cases
# ============================================================================

class TestSystemEdgeCases:
    """Tests for edge cases."""

    def test_empty_state_file(self, system_handler, tmp_path):
        """Should handle empty state file."""
        (tmp_path / "nomic_state.json").write_text("{}")

        result = system_handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data == {}

    def test_empty_log_file(self, system_handler, tmp_path):
        """Should handle empty log file."""
        (tmp_path / "nomic_loop.log").write_text("")

        result = system_handler.handle("/api/nomic/log", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["lines"] == []
        assert data["total"] == 0

    def test_empty_cycles_file(self, system_handler, tmp_path):
        """Should handle empty cycles array."""
        (tmp_path / "cycles.json").write_text("[]")

        result = system_handler.handle("/api/history/cycles", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["cycles"] == []

    def test_nomic_dir_exists_check(self, system_handler, tmp_path):
        """Should add warning when nomic_dir doesn't exist."""
        # Point to non-existent directory
        system_handler.ctx["nomic_dir"] = tmp_path / "nonexistent"

        result = system_handler.handle("/api/health", {}, None)

        data = json.loads(result.body)
        # Non-existent nomic_dir is a warning, not a failure
        assert data["checks"]["nomic_dir"]["healthy"] is True
        assert "warning" in data["checks"]["nomic_dir"]


# ============================================================================
# Auth Endpoint Tests
# ============================================================================

class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_can_handle_auth_stats(self, system_handler):
        """Should handle /api/auth/stats."""
        assert system_handler.can_handle("/api/auth/stats") is True

    def test_can_handle_auth_revoke(self, system_handler):
        """Should handle /api/auth/revoke."""
        assert system_handler.can_handle("/api/auth/revoke") is True

    def test_get_auth_stats(self, system_handler):
        """Should return auth stats."""
        result = system_handler.handle("/api/auth/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "enabled" in data
        assert "rate_limit_per_minute" in data
        assert "ip_rate_limit_per_minute" in data
        assert "token_ttl_seconds" in data
        assert "stats" in data

    def test_revoke_token_invalid_json(self, system_handler):
        """Should reject invalid JSON body."""
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "10"}
        mock_handler.rfile.read.return_value = b"not valid json"

        result = system_handler.handle_post("/api/auth/revoke", {}, mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid JSON" in data["error"]

    def test_revoke_token_missing_token(self, system_handler):
        """Should reject missing token."""
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        result = system_handler.handle_post("/api/auth/revoke", {}, mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "required" in data["error"].lower()

    def test_revoke_token_success(self, system_handler):
        """Should revoke token successfully."""
        mock_handler = Mock()
        body = json.dumps({"token": "test-token-123", "reason": "security"})
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body.encode()

        result = system_handler.handle_post("/api/auth/revoke", {}, mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert "revoked_count" in data

    def test_revoke_token_without_reason(self, system_handler):
        """Should revoke token without reason."""
        mock_handler = Mock()
        body = json.dumps({"token": "another-token"})
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body.encode()

        result = system_handler.handle_post("/api/auth/revoke", {}, mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
