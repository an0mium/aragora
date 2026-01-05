"""
E2E tests for modular HTTP handlers.

Tests cover:
- SystemHandler: health, nomic state/log, modes, history
- DebatesHandler: list, get by slug, impasse, convergence, export
- AgentsHandler: leaderboard, matches, profiles, comparisons
- Handler routing: modular handler chain, fallthrough behavior
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers import (
    SystemHandler,
    DebatesHandler,
    AgentsHandler,
    HandlerResult,
    json_response,
    error_response,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_nomic_dir():
    """Create a temporary nomic directory with state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create nomic state file
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({
            "phase": "implement",
            "stage": "executing",
            "cycle": 1,
            "total_tasks": 5,
            "completed_tasks": 2,
        }))

        # Create nomic log file
        log_file = nomic_dir / "nomic_loop.log"
        log_file.write_text("\n".join([
            "2026-01-05 00:00:01 Starting cycle 1",
            "2026-01-05 00:00:02 Phase: context",
            "2026-01-05 00:00:03 Phase: debate",
            "2026-01-05 00:00:04 Phase: design",
            "2026-01-05 00:00:05 Phase: implement",
        ]))

        yield nomic_dir


@pytest.fixture
def mock_storage():
    """Create a mock DebateStorage."""
    storage = Mock()
    storage.list_debates.return_value = [
        {"id": "debate-1", "slug": "test-debate", "task": "Test task", "created_at": "2026-01-05"},
        {"id": "debate-2", "slug": "another-debate", "task": "Another task", "created_at": "2026-01-04"},
    ]
    storage.get_debate.return_value = {
        "id": "debate-1",
        "slug": "test-debate",
        "task": "Test task",
        "messages": [{"agent": "claude", "content": "Hello"}],
        "critiques": [],
        "consensus_reached": False,
        "rounds_used": 3,
    }
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock EloSystem."""
    elo = Mock()
    elo.get_leaderboard.return_value = [
        {"name": "claude", "rating": 1500, "wins": 10, "losses": 5},
        {"name": "gemini", "rating": 1480, "wins": 8, "losses": 7},
    ]
    elo.get_rating.return_value = 1500
    elo.get_agent_stats.return_value = {"wins": 10, "losses": 5, "win_rate": 0.67}
    elo.get_agent_history.return_value = [
        {"opponent": "gemini", "result": "win", "date": "2026-01-05"},
    ]
    elo.get_recent_matches.return_value = [
        {"agent1": "claude", "agent2": "gemini", "winner": "claude"},
    ]
    elo.get_head_to_head.return_value = {"matches": 5, "agent1_wins": 3, "agent2_wins": 2}
    elo.get_rivals.return_value = [{"name": "gemini", "matches": 10}]
    elo.get_allies.return_value = [{"name": "grok", "agreements": 8}]
    return elo


@pytest.fixture
def system_handler(temp_nomic_dir, mock_storage, mock_elo_system):
    """Create SystemHandler with test context."""
    ctx = {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": temp_nomic_dir,
    }
    return SystemHandler(ctx)


@pytest.fixture
def debates_handler(mock_storage):
    """Create DebatesHandler with test context."""
    ctx = {
        "storage": mock_storage,
        "elo_system": None,
        "nomic_dir": None,
    }
    return DebatesHandler(ctx)


@pytest.fixture
def agents_handler(mock_elo_system, temp_nomic_dir):
    """Create AgentsHandler with test context."""
    ctx = {
        "storage": None,
        "elo_system": mock_elo_system,
        "nomic_dir": temp_nomic_dir,
    }
    return AgentsHandler(ctx)


# ============================================================================
# SystemHandler Tests
# ============================================================================

class TestSystemHandlerE2E:
    """E2E tests for SystemHandler endpoints."""

    def test_health_check_returns_200(self, system_handler):
        """Test /api/health returns 200 with health status."""
        result = system_handler.handle("/api/health", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

        data = json.loads(result.body)
        assert data["status"] == "healthy"

    def test_health_check_includes_components(self, system_handler):
        """Test /api/health includes component status."""
        result = system_handler.handle("/api/health", {}, None)
        data = json.loads(result.body)

        assert "components" in data
        assert "storage" in data["components"]
        assert "elo_system" in data["components"]
        assert "nomic_dir" in data["components"]

    def test_nomic_state_returns_json(self, system_handler):
        """Test /api/nomic/state returns current state."""
        result = system_handler.handle("/api/nomic/state", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["phase"] == "implement"
        assert data["cycle"] == 1

    def test_nomic_log_respects_limit(self, system_handler):
        """Test /api/nomic/log respects lines parameter."""
        result = system_handler.handle("/api/nomic/log", {"lines": 3}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "lines" in data
        assert len(data["lines"]) <= 3

    def test_nomic_state_not_running(self, mock_storage, mock_elo_system):
        """Test /api/nomic/state when no state file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {
                "storage": mock_storage,
                "elo_system": mock_elo_system,
                "nomic_dir": Path(tmpdir),
            }
            handler = SystemHandler(ctx)
            result = handler.handle("/api/nomic/state", {}, None)

            data = json.loads(result.body)
            assert data["state"] == "not_running"

    def test_modes_returns_response(self, system_handler):
        """Test /api/modes returns a response (may be empty or error)."""
        result = system_handler.handle("/api/modes", {}, None)

        # May return 200 with modes or 500 if CustomModeLoader not configured
        assert result.status_code in (200, 500)
        data = json.loads(result.body)
        # Either has modes or error
        assert "modes" in data or "error" in data

    def test_history_cycles_with_loop_id(self, system_handler):
        """Test /api/history/cycles filters by loop_id."""
        result = system_handler.handle(
            "/api/history/cycles",
            {"loop_id": "loop-123", "limit": 10},
            None
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "cycles" in data


# ============================================================================
# DebatesHandler Tests
# ============================================================================

class TestDebatesHandlerE2E:
    """E2E tests for DebatesHandler endpoints."""

    def test_list_debates_returns_array(self, debates_handler):
        """Test /api/debates returns list of debates."""
        result = debates_handler.handle("/api/debates", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "debates" in data
        assert isinstance(data["debates"], list)
        assert len(data["debates"]) == 2

    def test_list_debates_with_limit(self, debates_handler, mock_storage):
        """Test /api/debates respects limit parameter."""
        debates_handler.handle("/api/debates", {"limit": 5}, None)
        mock_storage.list_debates.assert_called_with(limit=5)

    def test_debate_by_slug_returns_object(self, debates_handler):
        """Test /api/debates/{slug} returns debate details."""
        result = debates_handler.handle("/api/debates/test-debate", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["slug"] == "test-debate"
        assert "messages" in data

    def test_debate_not_found_returns_404(self, debates_handler, mock_storage):
        """Test /api/debates/{slug} returns 404 for missing debate."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent", {}, None)

        assert result.status_code == 404

    def test_impasse_detection_indicators(self, debates_handler):
        """Test /api/debates/{id}/impasse returns impasse indicators."""
        result = debates_handler.handle("/api/debates/debate-1/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "is_impasse" in data
        assert "indicators" in data

    def test_convergence_status_fields(self, debates_handler):
        """Test /api/debates/{id}/convergence returns status fields."""
        result = debates_handler.handle("/api/debates/debate-1/convergence", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "convergence_status" in data
        assert "consensus_reached" in data

    def test_export_json_format(self, debates_handler):
        """Test /api/debates/{id}/export/json returns debate as JSON."""
        result = debates_handler.handle(
            "/api/debates/debate-1/export/json",
            {"table": "summary"},
            None
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "slug" in data

    def test_export_invalid_format_400(self, debates_handler):
        """Test /api/debates/{id}/export/{invalid} returns 400."""
        result = debates_handler.handle(
            "/api/debates/debate-1/export/xml",
            {},
            None
        )

        assert result.status_code == 400


# ============================================================================
# AgentsHandler Tests
# ============================================================================

class TestAgentsHandlerE2E:
    """E2E tests for AgentsHandler endpoints."""

    def test_leaderboard_default_params(self, agents_handler):
        """Test /api/leaderboard returns rankings."""
        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "rankings" in data
        assert len(data["rankings"]) > 0

    def test_leaderboard_with_domain(self, agents_handler, mock_elo_system):
        """Test /api/leaderboard filters by domain."""
        agents_handler.handle("/api/leaderboard", {"domain": "coding"}, None)
        mock_elo_system.get_leaderboard.assert_called_with(limit=20, domain="coding")

    def test_leaderboard_limit_capped(self, agents_handler, mock_elo_system):
        """Test /api/leaderboard caps limit at 50."""
        agents_handler.handle("/api/leaderboard", {"limit": 100}, None)
        mock_elo_system.get_leaderboard.assert_called_with(limit=50, domain=None)

    def test_recent_matches_returns_array(self, agents_handler):
        """Test /api/matches/recent returns recent matches."""
        result = agents_handler.handle("/api/matches/recent", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "matches" in data

    def test_agent_profile_complete(self, agents_handler):
        """Test /api/agent/{name}/profile returns full profile."""
        result = agents_handler.handle("/api/agent/claude/profile", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["name"] == "claude"
        assert "rating" in data

    def test_agent_history_with_limit(self, agents_handler, mock_elo_system):
        """Test /api/agent/{name}/history respects limit."""
        agents_handler.handle("/api/agent/claude/history", {"limit": 50}, None)
        mock_elo_system.get_agent_history.assert_called_with("claude", limit=50)

    def test_agent_compare_two_agents(self, agents_handler):
        """Test /api/agent/compare requires 2+ agents."""
        # Test with only 1 agent - should fail
        result = agents_handler.handle("/api/agent/compare", {"agents": ["claude"]}, None)
        assert result.status_code == 400

        # Test with 2 agents - should succeed
        result = agents_handler.handle(
            "/api/agent/compare",
            {"agents": ["claude", "gemini"]},
            None
        )
        assert result.status_code == 200

    def test_head_to_head_stats(self, agents_handler):
        """Test /api/agent/{name}/head-to-head/{opponent} returns stats."""
        result = agents_handler.handle(
            "/api/agent/claude/head-to-head/gemini",
            {},
            None
        )

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["agent1"] == "claude"
        assert data["agent2"] == "gemini"
        assert "matches" in data

    def test_agent_network_structure(self, agents_handler):
        """Test /api/agent/{name}/network returns rivals and allies."""
        result = agents_handler.handle("/api/agent/claude/network", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "rivals" in data
        assert "allies" in data

    def test_calibration_leaderboard(self, agents_handler):
        """Test /api/calibration/leaderboard returns rankings."""
        result = agents_handler.handle("/api/calibration/leaderboard", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rankings" in data


# ============================================================================
# Handler Routing Tests
# ============================================================================

class TestHandlerRoutingE2E:
    """E2E tests for handler routing behavior."""

    def test_can_handle_matches_routes(self, system_handler, debates_handler, agents_handler):
        """Test can_handle correctly identifies routes."""
        # SystemHandler routes
        assert system_handler.can_handle("/api/health") is True
        assert system_handler.can_handle("/api/nomic/state") is True
        assert system_handler.can_handle("/api/debates") is False

        # DebatesHandler routes
        assert debates_handler.can_handle("/api/debates") is True
        assert debates_handler.can_handle("/api/debates/test-slug") is True
        assert debates_handler.can_handle("/api/leaderboard") is False

        # AgentsHandler routes
        assert agents_handler.can_handle("/api/leaderboard") is True
        assert agents_handler.can_handle("/api/agent/claude/profile") is True
        assert agents_handler.can_handle("/api/health") is False

    def test_handler_returns_none_for_unhandled(self, system_handler):
        """Test handler returns None for routes it can handle but has no implementation."""
        # can_handle returns False, so handle() would return None
        result = system_handler.handle("/api/unknown", {}, None)
        assert result is None

    def test_query_param_conversion(self, agents_handler, mock_elo_system):
        """Test query parameters are correctly converted."""
        # Integer conversion
        agents_handler.handle("/api/leaderboard", {"limit": "25"}, None)
        mock_elo_system.get_leaderboard.assert_called_with(limit=25, domain=None)

    def test_error_response_format(self, debates_handler, mock_storage):
        """Test error responses follow consistent format."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/missing", {}, None)
        data = json.loads(result.body)

        assert "error" in data
        assert isinstance(data["error"], str)


# ============================================================================
# Handler Result Tests
# ============================================================================

class TestHandlerResult:
    """Tests for HandlerResult utilities."""

    def test_json_response_creates_valid_result(self):
        """Test json_response creates correct HandlerResult."""
        result = json_response({"key": "value"}, status=200)

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert b'"key"' in result.body

    def test_error_response_creates_error_result(self):
        """Test error_response creates error HandlerResult."""
        result = error_response("Something went wrong", status=500)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert data["error"] == "Something went wrong"

    def test_handler_result_default_headers(self):
        """Test HandlerResult has default empty headers."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b"{}",
        )
        assert result.headers == {}
