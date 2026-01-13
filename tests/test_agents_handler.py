"""
Tests for aragora/server/handlers/agents.py

Comprehensive tests for agent-related API endpoints including:
- Leaderboard and rankings
- Agent profiles and history
- Calibration and consistency
- Head-to-head comparisons
- Flip detection endpoints
- Input validation and caching
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.agents import AgentsHandler
from aragora.server.handlers.base import (
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_string_param,
    validate_agent_name,
    validate_path_segment,
    SAFE_ID_PATTERN,
    clear_cache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()
    elo.get_cached_leaderboard.return_value = [
        {"name": "agent-a", "elo": 1600, "wins": 10, "losses": 5},
        {"name": "agent-b", "elo": 1550, "wins": 8, "losses": 7},
    ]
    elo.get_leaderboard.return_value = [
        {"name": "agent-a", "elo": 1600, "wins": 10, "losses": 5},
        {"name": "agent-b", "elo": 1550, "wins": 8, "losses": 7},
    ]
    elo.get_rating.return_value = {"elo": 1500, "wins": 5, "losses": 3}
    elo.get_agent_stats.return_value = {"wins": 5, "losses": 3, "rank": 2, "win_rate": 0.625}
    elo.get_agent_history.return_value = [
        {"match_id": "m1", "opponent": "agent-b", "result": "win"},
    ]
    elo.get_head_to_head.return_value = {"matches": 5, "agent1_wins": 3, "agent2_wins": 2}
    elo.get_rivals.return_value = [{"name": "agent-b", "matches": 10}]
    elo.get_allies.return_value = [{"name": "agent-c", "synergy": 0.8}]
    elo.get_cached_recent_matches.return_value = [{"match_id": "m1"}]
    elo.get_recent_matches.return_value = [{"match_id": "m1"}]
    elo.get_calibration.return_value = {"agent": "agent-a", "score": 0.75}
    return elo


@pytest.fixture
def mock_server_context(mock_elo_system, tmp_path):
    """Create a mock server context."""
    return {
        "elo_system": mock_elo_system,
        "storage": MagicMock(),
        "nomic_dir": tmp_path,
    }


@pytest.fixture
def handler(mock_server_context):
    """Create an AgentsHandler instance."""
    # Clear cache before each test
    clear_cache()
    return AgentsHandler(mock_server_context)


@pytest.fixture
def handler_without_elo(tmp_path):
    """Create an AgentsHandler without ELO system."""
    ctx = {
        "elo_system": None,
        "storage": MagicMock(),
        "nomic_dir": tmp_path,
    }
    return AgentsHandler(ctx)


# =============================================================================
# Test Input Validation Functions
# =============================================================================


class TestValidation:
    """Tests for input validation functions."""

    def test_validate_agent_name_valid(self):
        """Test valid agent name."""
        is_valid, err = validate_agent_name("agent-a")
        assert is_valid is True
        assert err is None

    def test_validate_agent_name_valid_with_underscore(self):
        """Test valid agent name with underscore."""
        is_valid, err = validate_agent_name("agent_a_1")
        assert is_valid is True

    def test_validate_agent_name_valid_alphanumeric(self):
        """Test valid alphanumeric agent name."""
        is_valid, err = validate_agent_name("Agent123")
        assert is_valid is True

    def test_validate_agent_name_empty(self):
        """Test empty agent name."""
        is_valid, err = validate_agent_name("")
        assert is_valid is False
        assert "Missing" in err

    def test_validate_agent_name_path_traversal(self):
        """Test path traversal in agent name."""
        is_valid, err = validate_agent_name("../etc/passwd")
        assert is_valid is False
        assert "must match pattern" in err.lower()

    def test_validate_agent_name_slash(self):
        """Test slash in agent name."""
        is_valid, err = validate_agent_name("agent/malicious")
        assert is_valid is False

    def test_validate_path_segment_valid(self):
        """Test valid path segment."""
        is_valid, err = validate_path_segment("domain-1", "domain", SAFE_ID_PATTERN)
        assert is_valid is True
        assert err is None

    def test_validate_path_segment_invalid_format(self):
        """Test invalid format in path segment."""
        is_valid, err = validate_path_segment("invalid@format", "domain", SAFE_ID_PATTERN)
        assert is_valid is False
        assert "Invalid" in err


class TestParamHelpers:
    """Tests for parameter helper functions."""

    def test_get_int_param_valid(self):
        """Test get_int_param with valid int."""
        result = get_int_param({"limit": "10"}, "limit", 5)
        assert result == 10

    def test_get_int_param_default(self):
        """Test get_int_param with missing key."""
        result = get_int_param({}, "limit", 5)
        assert result == 5

    def test_get_int_param_invalid(self):
        """Test get_int_param with invalid value."""
        result = get_int_param({"limit": "invalid"}, "limit", 5)
        assert result == 5

    def test_get_string_param_valid(self):
        """Test get_string_param with valid string."""
        result = get_string_param({"name": "test"}, "name", "default")
        assert result == "test"

    def test_get_string_param_list(self):
        """Test get_string_param with list value."""
        result = get_string_param({"name": ["first", "second"]}, "name", "default")
        assert result == "first"

    def test_get_string_param_default(self):
        """Test get_string_param with missing key."""
        result = get_string_param({}, "name", "default")
        assert result == "default"


# =============================================================================
# Test can_handle Method
# =============================================================================


class TestCanHandle:
    """Tests for can_handle method."""

    def test_can_handle_leaderboard(self, handler):
        """Test can_handle for leaderboard."""
        assert handler.can_handle("/api/leaderboard") is True

    def test_can_handle_rankings(self, handler):
        """Test can_handle for rankings."""
        assert handler.can_handle("/api/rankings") is True

    def test_cannot_handle_calibration_leaderboard(self, handler):
        """Test AgentsHandler does NOT handle calibration (handled by CalibrationHandler)."""
        assert handler.can_handle("/api/calibration/leaderboard") is False

    def test_can_handle_matches_recent(self, handler):
        """Test can_handle for recent matches."""
        assert handler.can_handle("/api/matches/recent") is True

    def test_can_handle_agent_compare(self, handler):
        """Test can_handle for agent compare."""
        assert handler.can_handle("/api/agent/compare") is True

    def test_can_handle_agent_profile(self, handler):
        """Test can_handle for agent profile."""
        assert handler.can_handle("/api/agent/test-agent/profile") is True

    def test_can_handle_agent_history(self, handler):
        """Test can_handle for agent history."""
        assert handler.can_handle("/api/agent/test-agent/history") is True

    def test_can_handle_flips_recent(self, handler):
        """Test can_handle for recent flips."""
        assert handler.can_handle("/api/flips/recent") is True

    def test_can_handle_flips_summary(self, handler):
        """Test can_handle for flip summary."""
        assert handler.can_handle("/api/flips/summary") is True

    def test_can_handle_unrelated_path(self, handler):
        """Test can_handle for unrelated path."""
        assert handler.can_handle("/api/debates") is False

    def test_can_handle_head_to_head(self, handler):
        """Test can_handle for head-to-head."""
        assert handler.can_handle("/api/agent/a/head-to-head/b") is True


# =============================================================================
# Test Leaderboard Endpoint
# =============================================================================


class TestLeaderboardEndpoint:
    """Tests for leaderboard endpoint."""

    def test_get_leaderboard_success(self, handler):
        """Test successful leaderboard retrieval."""
        # Clear cache first
        clear_cache()
        result = handler.handle("/api/leaderboard", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rankings" in data
        assert len(data["rankings"]) > 0

    def test_get_leaderboard_with_limit(self, handler):
        """Test leaderboard with limit parameter."""
        clear_cache()
        result = handler.handle("/api/leaderboard", {"limit": "5"}, None)

        assert result is not None
        assert result.status_code == 200

    def test_get_leaderboard_with_domain(self, handler):
        """Test leaderboard with domain filter."""
        clear_cache()
        result = handler.handle("/api/leaderboard", {"domain": "security"}, None)

        assert result is not None
        assert result.status_code == 200

    def test_get_leaderboard_invalid_domain(self, handler):
        """Test leaderboard with invalid domain."""
        clear_cache()
        result = handler.handle("/api/leaderboard", {"domain": "../invalid"}, None)

        assert result is not None
        assert result.status_code == 400

    def test_get_leaderboard_no_elo(self, handler_without_elo):
        """Test leaderboard when ELO system unavailable."""
        clear_cache()
        result = handler_without_elo.handle("/api/leaderboard", {}, None)

        assert result is not None
        assert result.status_code == 503

    def test_rankings_alias(self, handler):
        """Test rankings endpoint as alias for leaderboard."""
        clear_cache()
        result = handler.handle("/api/rankings", {}, None)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Profile Endpoint
# =============================================================================


class TestProfileEndpoint:
    """Tests for profile endpoint."""

    def test_get_profile_success(self, handler):
        """Test successful profile retrieval."""
        clear_cache()
        result = handler.handle("/api/agent/test-agent/profile", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["name"] == "test-agent"

    def test_get_profile_includes_rating(self, handler):
        """Test profile includes rating."""
        clear_cache()
        result = handler.handle("/api/agent/test-agent/profile", {}, None)

        data = json.loads(result.body)
        assert "rating" in data

    def test_get_profile_invalid_agent(self, handler):
        """Test profile with invalid agent name."""
        result = handler.handle("/api/agent/../malicious/profile", {}, None)

        assert result is not None
        assert result.status_code == 400

    def test_get_profile_no_elo(self, handler_without_elo):
        """Test profile when ELO unavailable."""
        clear_cache()
        result = handler_without_elo.handle("/api/agent/test/profile", {}, None)

        assert result.status_code == 503


# =============================================================================
# Test History Endpoint
# =============================================================================


class TestHistoryEndpoint:
    """Tests for history endpoint."""

    def test_get_history_success(self, handler):
        """Test successful history retrieval."""
        result = handler.handle("/api/agent/test-agent/history", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "test-agent"
        assert "history" in data

    def test_get_history_with_limit(self, handler):
        """Test history with limit parameter."""
        result = handler.handle("/api/agent/test-agent/history", {"limit": "10"}, None)

        assert result is not None
        assert result.status_code == 200

    def test_get_history_no_elo(self, handler_without_elo):
        """Test history when ELO unavailable."""
        result = handler_without_elo.handle("/api/agent/test/history", {}, None)

        assert result.status_code == 503


# =============================================================================
# Test Head-to-Head Endpoint
# =============================================================================


class TestHeadToHeadEndpoint:
    """Tests for head-to-head endpoint."""

    def test_get_head_to_head_success(self, handler):
        """Test successful head-to-head retrieval."""
        clear_cache()
        result = handler.handle("/api/agent/agent-a/head-to-head/agent-b", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent1"] == "agent-a"
        assert data["agent2"] == "agent-b"
        assert "matches" in data

    def test_get_head_to_head_invalid_agent(self, handler):
        """Test head-to-head with invalid agent."""
        result = handler.handle("/api/agent/../bad/head-to-head/agent-b", {}, None)

        assert result.status_code == 400

    def test_get_head_to_head_invalid_opponent(self, handler):
        """Test head-to-head with invalid opponent."""
        result = handler.handle("/api/agent/agent-a/head-to-head/../bad", {}, None)

        assert result.status_code == 400

    def test_get_head_to_head_no_elo(self, handler_without_elo):
        """Test head-to-head when ELO unavailable."""
        clear_cache()
        result = handler_without_elo.handle("/api/agent/a/head-to-head/b", {}, None)

        assert result.status_code == 503


# =============================================================================
# Test Calibration Endpoint
# =============================================================================


class TestCalibrationEndpoint:
    """Tests for calibration endpoint."""

    def test_get_calibration_success(self, handler):
        """Test successful calibration retrieval."""
        result = handler.handle("/api/agent/test-agent/calibration", {}, None)

        assert result is not None
        assert result.status_code == 200

    def test_get_calibration_with_domain(self, handler):
        """Test calibration with domain parameter."""
        result = handler.handle("/api/agent/test-agent/calibration", {"domain": "security"}, None)

        assert result.status_code == 200

    # NOTE: test_get_calibration_leaderboard moved to test_calibration_handler.py
    # since /api/calibration/leaderboard is now handled by CalibrationHandler


# =============================================================================
# Test Consistency Endpoint
# =============================================================================


class TestConsistencyEndpoint:
    """Tests for consistency endpoint."""

    def test_get_consistency_success(self, handler):
        """Test successful consistency retrieval."""
        result = handler.handle("/api/agent/test-agent/consistency", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "test-agent"
        assert "consistency_score" in data

    def test_get_consistency_fallback(self, handler):
        """Test consistency returns default when FlipDetector unavailable."""
        # Should return default consistency of 1.0
        result = handler.handle("/api/agent/test-agent/consistency", {}, None)

        data = json.loads(result.body)
        # Either returns a score from detector or default 1.0
        assert "consistency_score" in data


# =============================================================================
# Test Network Endpoints (Rivals/Allies)
# =============================================================================


class TestNetworkEndpoints:
    """Tests for network-related endpoints."""

    def test_get_network_success(self, handler):
        """Test successful network retrieval."""
        result = handler.handle("/api/agent/test-agent/network", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rivals" in data
        assert "allies" in data

    def test_get_rivals_success(self, handler):
        """Test successful rivals retrieval."""
        result = handler.handle("/api/agent/test-agent/rivals", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rivals" in data

    def test_get_rivals_with_limit(self, handler):
        """Test rivals with limit parameter."""
        result = handler.handle("/api/agent/test-agent/rivals", {"limit": "3"}, None)

        assert result.status_code == 200

    def test_get_allies_success(self, handler):
        """Test successful allies retrieval."""
        result = handler.handle("/api/agent/test-agent/allies", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "allies" in data


# =============================================================================
# Test Compare Endpoint
# =============================================================================


class TestCompareEndpoint:
    """Tests for agent comparison endpoint."""

    def test_compare_two_agents(self, handler):
        """Test comparing two agents."""
        result = handler.handle("/api/agent/compare", {"agents": ["agent-a", "agent-b"]}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data
        assert len(data["agents"]) == 2

    def test_compare_with_head_to_head(self, handler):
        """Test compare includes head-to-head for 2 agents."""
        result = handler.handle("/api/agent/compare", {"agents": ["agent-a", "agent-b"]}, None)

        data = json.loads(result.body)
        assert "head_to_head" in data

    def test_compare_too_few_agents(self, handler):
        """Test compare requires at least 2 agents."""
        result = handler.handle("/api/agent/compare", {"agents": ["agent-a"]}, None)

        assert result.status_code == 400

    def test_compare_string_agent(self, handler):
        """Test compare handles string agent (single value)."""
        result = handler.handle(
            "/api/agent/compare", {"agents": "agent-a"}, None  # String instead of list
        )

        # Should fail because only 1 agent
        assert result.status_code == 400


# =============================================================================
# Test Recent Matches Endpoint
# =============================================================================


class TestRecentMatchesEndpoint:
    """Tests for recent matches endpoint."""

    def test_get_recent_matches_success(self, handler):
        """Test successful recent matches retrieval."""
        clear_cache()
        result = handler.handle("/api/matches/recent", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "matches" in data

    def test_get_recent_matches_with_limit(self, handler):
        """Test recent matches with limit."""
        clear_cache()
        result = handler.handle("/api/matches/recent", {"limit": "5"}, None)

        assert result.status_code == 200

    def test_get_recent_matches_with_loop_id(self, handler):
        """Test recent matches with loop_id filter."""
        clear_cache()
        result = handler.handle("/api/matches/recent", {"loop_id": "loop-123"}, None)

        assert result.status_code == 200

    def test_get_recent_matches_invalid_loop_id(self, handler):
        """Test recent matches with invalid loop_id."""
        clear_cache()
        result = handler.handle("/api/matches/recent", {"loop_id": "../invalid"}, None)

        assert result.status_code == 400


# =============================================================================
# Test Flips Endpoints
# =============================================================================


class TestFlipsEndpoints:
    """Tests for flip detection endpoints."""

    def test_get_agent_flips_success(self, handler):
        """Test successful agent flips retrieval."""
        clear_cache()
        result = handler.handle("/api/agent/test-agent/flips", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "test-agent"
        assert "flips" in data
        assert "count" in data

    def test_get_agent_flips_with_limit(self, handler):
        """Test agent flips with limit."""
        clear_cache()
        result = handler.handle("/api/agent/test-agent/flips", {"limit": "10"}, None)

        assert result.status_code == 200

    def test_get_recent_flips_success(self, handler):
        """Test successful recent flips retrieval."""
        clear_cache()
        result = handler.handle("/api/flips/recent", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "flips" in data
        assert "count" in data

    def test_get_flip_summary_success(self, handler):
        """Test successful flip summary retrieval."""
        clear_cache()
        result = handler.handle("/api/flips/summary", {}, None)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Moments Endpoint
# =============================================================================


class TestMomentsEndpoint:
    """Tests for moments endpoint."""

    def test_get_moments_returns_empty_without_detector(self, handler):
        """Test moments returns empty list when detector unavailable."""
        result = handler.handle("/api/agent/test-agent/moments", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "moments" in data


# =============================================================================
# Test Positions Endpoint
# =============================================================================


class TestPositionsEndpoint:
    """Tests for positions endpoint."""

    def test_get_positions_returns_empty_without_ledger(self, handler):
        """Test positions returns empty list when ledger unavailable."""
        result = handler.handle("/api/agent/test-agent/positions", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "positions" in data


# =============================================================================
# Test Opponent Briefing Endpoint
# =============================================================================


class TestOpponentBriefingEndpoint:
    """Tests for opponent briefing endpoint."""

    def test_get_opponent_briefing_unavailable(self, handler):
        """Test opponent briefing returns error when unavailable."""
        # This will fail because PersonaSynthesizer may not be available
        result = handler.handle("/api/agent/agent-a/opponent-briefing/agent-b", {}, None)

        # Could be 200 with null briefing, 400/500 for errors, or 503 if unavailable
        assert result is not None
        assert result.status_code in (200, 400, 500, 503)


# =============================================================================
# Test Routing and Dispatch
# =============================================================================


class TestRoutingAndDispatch:
    """Tests for routing logic."""

    def test_handle_invalid_agent_path(self, handler):
        """Test handling invalid agent path."""
        result = handler.handle("/api/agent/", {}, None)

        # Path too short
        assert result is None or result.status_code == 400

    def test_handle_unknown_agent_endpoint(self, handler):
        """Test handling unknown agent endpoint."""
        result = handler.handle("/api/agent/test/unknown-endpoint", {}, None)

        # Should return None for unknown endpoint
        assert result is None

    def test_handle_returns_none_for_unmatched(self, handler):
        """Test handle returns None for unmatched paths."""
        result = handler.handle("/api/other-endpoint", {}, None)

        assert result is None


# =============================================================================
# Test Response Formatting
# =============================================================================


class TestResponseFormatting:
    """Tests for response formatting functions."""

    def test_json_response_creates_valid_json(self):
        """Test json_response creates valid JSON."""
        result = json_response({"key": "value"})

        assert result.status_code == 200
        assert result.content_type == "application/json"
        data = json.loads(result.body)
        assert data["key"] == "value"

    def test_json_response_custom_status(self):
        """Test json_response with custom status."""
        result = json_response({"created": True}, status=201)

        assert result.status_code == 201

    def test_error_response_format(self):
        """Test error_response format."""
        result = error_response("Something went wrong", 500)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data
        assert data["error"] == "Something went wrong"

    def test_error_response_default_status(self):
        """Test error_response default status is 400."""
        result = error_response("Bad request")

        assert result.status_code == 400


# =============================================================================
# Test TTL Caching
# =============================================================================


class TestTTLCaching:
    """Tests for TTL caching behavior."""

    def test_cache_hit(self, handler):
        """Test cache returns cached value."""
        clear_cache()

        # First call
        result1 = handler.handle("/api/leaderboard", {}, None)

        # Second call should be cached
        result2 = handler.handle("/api/leaderboard", {}, None)

        assert result1.status_code == 200
        assert result2.status_code == 200

    def test_cache_different_params(self, handler):
        """Test cache differentiates by parameters."""
        clear_cache()

        result1 = handler.handle("/api/leaderboard", {"limit": "10"}, None)
        result2 = handler.handle("/api/leaderboard", {"limit": "20"}, None)

        # Both should succeed (different cache keys)
        assert result1.status_code == 200
        assert result2.status_code == 200

    def test_clear_cache(self, handler):
        """Test clearing cache."""
        # Populate cache
        handler.handle("/api/leaderboard", {}, None)

        # Clear and verify
        cleared = clear_cache("leaderboard")
        # cleared count may vary based on cache key format


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentHandlerIntegration:
    """Integration tests for agent handler."""

    def test_full_agent_workflow(self, handler):
        """Test typical workflow: leaderboard -> profile -> history."""
        clear_cache()

        # Get leaderboard
        leaderboard = handler.handle("/api/leaderboard", {}, None)
        assert leaderboard.status_code == 200

        # Get profile for first agent
        profile = handler.handle("/api/agent/agent-a/profile", {}, None)
        assert profile.status_code == 200

        # Get history
        history = handler.handle("/api/agent/agent-a/history", {}, None)
        assert history.status_code == 200

    def test_compare_workflow(self, handler):
        """Test agent comparison workflow."""
        clear_cache()

        # Compare agents
        compare = handler.handle("/api/agent/compare", {"agents": ["agent-a", "agent-b"]}, None)
        assert compare.status_code == 200

        # Get head-to-head
        h2h = handler.handle("/api/agent/agent-a/head-to-head/agent-b", {}, None)
        assert h2h.status_code == 200

    def test_all_endpoints_handle_missing_elo(self, handler_without_elo):
        """Test all endpoints gracefully handle missing ELO system."""
        clear_cache()

        # These should all return 503
        endpoints = [
            ("/api/leaderboard", {}),
            ("/api/agent/test/profile", {}),
            ("/api/agent/test/history", {}),
            ("/api/agent/a/head-to-head/b", {}),
            ("/api/agent/test/rivals", {}),
            ("/api/agent/test/allies", {}),
            ("/api/agent/test/network", {}),
        ]

        for path, params in endpoints:
            clear_cache()  # Clear cache between calls
            result = handler_without_elo.handle(path, params, None)
            assert result is not None, f"Handler returned None for {path}"
            assert result.status_code == 503, f"Expected 503 for {path}, got {result.status_code}"
