"""
Tests for AgentsHandler endpoints.

Endpoints tested:
- GET /api/leaderboard - Get agent rankings
- GET /api/rankings - Get agent rankings (alias)
- GET /api/calibration/leaderboard - Get calibration leaderboard
- GET /api/matches/recent - Get recent matches
- GET /api/agent/compare - Compare multiple agents
- GET /api/agent/{name}/profile - Get agent profile
- GET /api/agent/{name}/history - Get agent match history
- GET /api/agent/{name}/calibration - Get calibration scores
- GET /api/agent/{name}/consistency - Get consistency score
- GET /api/agent/{name}/flips - Get agent flip history
- GET /api/agent/{name}/network - Get relationship network
- GET /api/agent/{name}/rivals - Get top rivals
- GET /api/agent/{name}/allies - Get top allies
- GET /api/agent/{name}/moments - Get significant moments
- GET /api/agent/{name}/positions - Get position history
- GET /api/agent/{name}/head-to-head/{opponent} - Get head-to-head stats
- GET /api/flips/recent - Get recent flips across all agents
- GET /api/flips/summary - Get flip summary for dashboard
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from aragora.server.handlers import (
    AgentsHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.handlers.base import clear_cache
from aragora.utils.async_utils import run_async


# ============================================================================
# Test Fixtures
# ============================================================================


def _wrap_handler(handler: AgentsHandler) -> AgentsHandler:
    """Provide sync wrapper + auth bypass for async handler methods."""
    handler.get_auth_context = AsyncMock(return_value=MagicMock())
    handler.check_permission = MagicMock()
    async_handle = handler.handle

    def _handle_sync(*args, **kwargs):
        return run_async(async_handle(*args, **kwargs))

    handler.handle = _handle_sync  # type: ignore[assignment]
    return handler


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_leaderboard.return_value = [
        {"name": "claude", "elo": 1650, "wins": 10, "losses": 2, "games": 12, "win_rate": 0.83},
        {"name": "gpt4", "elo": 1580, "wins": 8, "losses": 4, "games": 12, "win_rate": 0.67},
    ]
    elo.get_cached_leaderboard.return_value = elo.get_leaderboard.return_value
    elo.get_rating.return_value = 1650
    elo.get_agent_stats.return_value = {
        "rank": 1,
        "wins": 10,
        "losses": 2,
        "win_rate": 0.83,
    }
    elo.get_agent_history.return_value = [
        {"opponent": "gpt4", "result": "win", "elo_change": 15},
        {"opponent": "gemini", "result": "loss", "elo_change": -12},
    ]
    elo.get_elo_history.return_value = [
        (1700000000, 1600),
        (1700003600, 1615),
    ]
    elo.get_recent_matches.return_value = [
        {"agent1": "claude", "agent2": "gpt4", "winner": "claude"},
    ]
    elo.get_cached_recent_matches.return_value = elo.get_recent_matches.return_value
    elo.get_head_to_head.return_value = {"matches": 5, "agent1_wins": 3, "agent2_wins": 2}
    elo.get_rivals.return_value = [{"name": "gpt4", "matches": 5}]
    elo.get_allies.return_value = [{"name": "gemini", "matches": 3}]
    elo.get_calibration.return_value = {"agent": "claude", "score": 0.85}
    return elo


@pytest.fixture
def agents_handler(mock_elo_system):
    """Create an AgentsHandler with mock ELO system."""
    ctx = {"elo_system": mock_elo_system}
    return _wrap_handler(AgentsHandler(ctx))


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch):
    """Disable rate limiting to keep tests deterministic."""
    from aragora.server.handlers.agents import agents as agents_module

    monkeypatch.setattr(agents_module._agent_limiter, "is_allowed", lambda _ip: True)


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestAgentsHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_leaderboard(self, agents_handler):
        """Should handle /api/leaderboard."""
        assert agents_handler.can_handle("/api/v1/leaderboard") is True

    def test_can_handle_rankings(self, agents_handler):
        """Should handle /api/rankings."""
        assert agents_handler.can_handle("/api/v1/rankings") is True

    def test_cannot_handle_calibration_leaderboard(self, agents_handler):
        """Should NOT handle /api/calibration/leaderboard (moved to CalibrationHandler)."""
        assert agents_handler.can_handle("/api/v1/calibration/leaderboard") is False

    def test_can_handle_matches_recent(self, agents_handler):
        """Should handle /api/matches/recent."""
        assert agents_handler.can_handle("/api/v1/matches/recent") is True

    def test_can_handle_agent_compare(self, agents_handler):
        """Should handle /api/agent/compare."""
        assert agents_handler.can_handle("/api/v1/agent/compare") is True

    def test_can_handle_agent_profile(self, agents_handler):
        """Should handle /api/agent/{name}/profile."""
        assert agents_handler.can_handle("/api/v1/agent/claude/profile") is True

    def test_can_handle_agent_history(self, agents_handler):
        """Should handle /api/agent/{name}/history."""
        assert agents_handler.can_handle("/api/v1/agent/claude/history") is True

    def test_can_handle_agent_head_to_head(self, agents_handler):
        """Should handle /api/agent/{name}/head-to-head/{opponent}."""
        assert agents_handler.can_handle("/api/v1/agent/claude/head-to-head/gpt4") is True

    def test_can_handle_flips_recent(self, agents_handler):
        """Should handle /api/flips/recent."""
        assert agents_handler.can_handle("/api/v1/flips/recent") is True

    def test_can_handle_flips_summary(self, agents_handler):
        """Should handle /api/flips/summary."""
        assert agents_handler.can_handle("/api/v1/flips/summary") is True

    def test_cannot_handle_unknown_route(self, agents_handler):
        """Should not handle unknown routes."""
        assert agents_handler.can_handle("/api/v1/debates") is False
        assert agents_handler.can_handle("/api/v1/unknown") is False


# ============================================================================
# Leaderboard Endpoint Tests
# ============================================================================


class TestLeaderboardEndpoint:
    """Tests for /api/leaderboard and /api/rankings endpoints."""

    def test_leaderboard_returns_rankings(self, agents_handler):
        """Should return leaderboard with rankings array."""
        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rankings" in data or "agents" in data
        assert isinstance(data.get("rankings") or data.get("agents"), list)

    def test_rankings_alias_works(self, agents_handler):
        """Should return same data for /api/rankings alias."""
        result = agents_handler.handle("/api/rankings", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rankings" in data or "agents" in data

    def test_leaderboard_respects_limit(self, agents_handler, mock_elo_system):
        """Should respect limit parameter."""
        result = agents_handler.handle("/api/leaderboard", {"limit": "10"}, None)

        assert result.status_code == 200
        # Verify limit was passed (capped at 50)
        mock_elo_system.get_cached_leaderboard.assert_called()

    def test_leaderboard_respects_domain(self, agents_handler, mock_elo_system):
        """Should filter by domain when provided."""
        result = agents_handler.handle("/api/leaderboard", {"domain": "science"}, None)

        assert result.status_code == 200
        # When domain is specified, uses get_leaderboard instead of cached version
        mock_elo_system.get_leaderboard.assert_called()
        call_kwargs = mock_elo_system.get_leaderboard.call_args[1]
        assert call_kwargs["domain"] == "science"

    def test_leaderboard_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"]


# ============================================================================
# Calibration Leaderboard Tests
# ============================================================================
# Note: /api/calibration/leaderboard moved to CalibrationHandler
# See test_handlers_calibration.py for those tests


# ============================================================================
# Recent Matches Tests
# ============================================================================


class TestRecentMatchesEndpoint:
    """Tests for /api/matches/recent endpoint."""

    def test_recent_matches_returns_list(self, agents_handler):
        """Should return recent matches."""
        result = agents_handler.handle("/api/matches/recent", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "matches" in data
        assert isinstance(data["matches"], list)

    def test_recent_matches_respects_limit(self, agents_handler):
        """Should respect limit parameter."""
        result = agents_handler.handle("/api/matches/recent", {"limit": "5"}, None)

        assert result.status_code == 200

    def test_recent_matches_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/matches/recent", {}, None)

        assert result.status_code == 503


# ============================================================================
# Agent Compare Tests
# ============================================================================


class TestAgentCompareEndpoint:
    """Tests for /api/agent/compare endpoint."""

    def test_compare_requires_two_agents(self, agents_handler):
        """Should require at least 2 agents."""
        result = agents_handler.handle("/api/agent/compare", {"agents": "claude"}, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "at least 2 agents" in data["error"]

    def test_compare_returns_profiles(self, agents_handler):
        """Should return agent profiles for comparison."""
        result = agents_handler.handle("/api/agent/compare", {"agents": ["claude", "gpt4"]}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data
        assert len(data["agents"]) == 2

    def test_compare_includes_head_to_head(self, agents_handler):
        """Should include head-to-head stats for 2 agents."""
        result = agents_handler.handle("/api/agent/compare", {"agents": ["claude", "gpt4"]}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "head_to_head" in data

    def test_compare_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/compare", {"agents": ["claude", "gpt4"]}, None)

        assert result.status_code == 503


# ============================================================================
# Agent Profile Tests
# ============================================================================


class TestAgentProfileEndpoint:
    """Tests for /api/agent/{name}/profile endpoint."""

    def test_profile_returns_agent_data(self, agents_handler):
        """Should return agent profile data."""
        result = agents_handler.handle("/api/agent/claude/profile", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["name"] == "claude"
        assert "rating" in data

    def test_profile_invalid_agent_name_returns_400(self, agents_handler):
        """Should return 400 for invalid agent names."""
        # Path traversal attempt
        result = agents_handler.handle("/api/agent/../etc/passwd/profile", {}, None)
        assert result.status_code == 400

    def test_profile_special_chars_returns_400(self, agents_handler):
        """Should reject special characters in agent name."""
        result = agents_handler.handle("/api/agent/test;DROP/profile", {}, None)
        assert result.status_code == 400

    def test_profile_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/claude/profile", {}, None)

        assert result.status_code == 503


# ============================================================================
# Agent History Tests
# ============================================================================


class TestAgentHistoryEndpoint:
    """Tests for /api/agent/{name}/history endpoint."""

    def test_history_returns_matches(self, agents_handler):
        """Should return agent match history."""
        result = agents_handler.handle("/api/agent/claude/history", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "history" in data

    def test_history_respects_limit(self, agents_handler, mock_elo_system):
        """Should respect limit parameter."""
        result = agents_handler.handle("/api/agent/claude/history", {"limit": "10"}, None)

        assert result.status_code == 200
        mock_elo_system.get_elo_history.assert_called_with("claude", limit=10)

    def test_history_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/claude/history", {}, None)

        assert result.status_code == 503


# ============================================================================
# Agent Calibration Tests
# ============================================================================


class TestAgentCalibrationEndpoint:
    """Tests for /api/agent/{name}/calibration endpoint."""

    def test_calibration_returns_data(self, agents_handler):
        """Should return calibration data."""
        result = agents_handler.handle("/api/agent/claude/calibration", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agent" in data or "score" in data

    def test_calibration_accepts_domain(self, agents_handler, mock_elo_system):
        """Should accept domain parameter."""
        result = agents_handler.handle("/api/agent/claude/calibration", {"domain": "science"}, None)

        assert result.status_code == 200
        mock_elo_system.get_calibration.assert_called_with("claude", domain="science")


# ============================================================================
# Agent Consistency Tests
# ============================================================================


class TestAgentConsistencyEndpoint:
    """Tests for /api/agent/{name}/consistency endpoint."""

    def test_consistency_returns_score(self, agents_handler):
        """Should return consistency score."""
        result = agents_handler.handle("/api/agent/claude/consistency", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "consistency_score" in data

    def test_consistency_default_value(self, agents_handler):
        """Should return default value when FlipDetector not available."""
        result = agents_handler.handle("/api/agent/unknown-agent/consistency", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should return some score (default or calculated)
        assert "consistency_score" in data


# ============================================================================
# Agent Network Tests
# ============================================================================


class TestAgentNetworkEndpoint:
    """Tests for /api/agent/{name}/network endpoint."""

    def test_network_returns_structure(self, agents_handler):
        """Should return network with rivals and allies."""
        result = agents_handler.handle("/api/agent/claude/network", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "rivals" in data
        assert "allies" in data

    def test_network_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/claude/network", {}, None)

        assert result.status_code == 503


# ============================================================================
# Agent Rivals/Allies Tests
# ============================================================================


class TestAgentRivalsEndpoint:
    """Tests for /api/agent/{name}/rivals endpoint."""

    def test_rivals_returns_list(self, agents_handler):
        """Should return rivals list."""
        result = agents_handler.handle("/api/agent/claude/rivals", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "rivals" in data
        assert isinstance(data["rivals"], list)


class TestAgentAlliesEndpoint:
    """Tests for /api/agent/{name}/allies endpoint."""

    def test_allies_returns_list(self, agents_handler):
        """Should return allies list."""
        result = agents_handler.handle("/api/agent/claude/allies", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "allies" in data
        assert isinstance(data["allies"], list)


# ============================================================================
# Agent Moments Tests
# ============================================================================


class TestAgentMomentsEndpoint:
    """Tests for /api/agent/{name}/moments endpoint."""

    def test_moments_returns_list(self, agents_handler):
        """Should return moments list."""
        result = agents_handler.handle("/api/agent/claude/moments", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "moments" in data
        assert isinstance(data["moments"], list)


# ============================================================================
# Agent Positions Tests
# ============================================================================


class TestAgentPositionsEndpoint:
    """Tests for /api/agent/{name}/positions endpoint."""

    def test_positions_returns_list(self, agents_handler):
        """Should return positions list."""
        result = agents_handler.handle("/api/agent/claude/positions", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "positions" in data
        assert isinstance(data["positions"], list)


# ============================================================================
# Agent Flips Tests
# ============================================================================


class TestAgentFlipsEndpoint:
    """Tests for /api/agent/{name}/flips endpoint."""

    def test_flips_returns_structure(self, agents_handler):
        """Should return flips with consistency data."""
        result = agents_handler.handle("/api/agent/claude/flips", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "flips" in data
        assert "consistency" in data
        assert "count" in data


# ============================================================================
# Head-to-Head Tests
# ============================================================================


class TestHeadToHeadEndpoint:
    """Tests for /api/agent/{name}/head-to-head/{opponent} endpoint."""

    def test_h2h_returns_stats(self, agents_handler):
        """Should return head-to-head statistics."""
        result = agents_handler.handle("/api/agent/claude/head-to-head/gpt4", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent1"] == "claude"
        assert data["agent2"] == "gpt4"
        assert "matches" in data

    def test_h2h_invalid_opponent_returns_400(self, agents_handler):
        """Should return 400 for invalid opponent name."""
        result = agents_handler.handle("/api/agent/claude/head-to-head/../../../etc", {}, None)

        assert result.status_code == 400

    def test_h2h_unavailable_returns_503(self):
        """Should return 503 when ELO system not available."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/claude/head-to-head/gpt4", {}, None)

        assert result.status_code == 503


# ============================================================================
# Flips Recent/Summary Tests
# ============================================================================


class TestFlipsRecentEndpoint:
    """Tests for /api/flips/recent endpoint."""

    def test_recent_flips_returns_structure(self, agents_handler):
        """Should return recent flips with summary."""
        result = agents_handler.handle("/api/flips/recent", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "flips" in data
        assert "summary" in data
        assert "count" in data


class TestFlipsSummaryEndpoint:
    """Tests for /api/flips/summary endpoint."""

    def test_summary_returns_structure(self, agents_handler):
        """Should return flip summary."""
        result = agents_handler.handle("/api/flips/summary", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should have summary fields
        assert isinstance(data, dict)


# ============================================================================
# Security Tests
# ============================================================================


class TestAgentsSecurity:
    """Tests for security measures."""

    def test_agent_path_traversal_blocked(self, agents_handler):
        """Should block path traversal attempts in agent names."""
        # Test agent names that contain path traversal patterns
        # When path contains slashes, it gets split differently by the handler
        # We test names with '..' that would be extracted as agent name
        dangerous_names = [
            "..passwd",  # starts with ..
            "test..admin",  # contains ..
        ]

        for name in dangerous_names:
            path = f"/api/agent/{name}/profile"
            result = agents_handler.handle(path, {}, None)
            # Handler returns 400 for invalid names, or None if path not matched
            if result is not None:
                assert result.status_code == 400, f"Should block name: {name}"

    def test_agent_name_special_chars_blocked(self, agents_handler):
        """Should block special characters in agent name."""
        dangerous_names = [
            "test; DROP TABLE agents;--",
            "test' OR '1'='1",
            "test<script>alert(1)</script>",
        ]

        for name in dangerous_names:
            path = f"/api/agent/{name}/profile"
            result = agents_handler.handle(path, {}, None)
            assert result.status_code == 400, f"Should block: {name}"

    def test_valid_agent_names_accepted(self, agents_handler):
        """Should accept valid agent names."""
        valid_names = [
            "claude",
            "gpt4",
            "claude-3",
            "claude_opus",
            "agent123",
        ]

        for name in valid_names:
            path = f"/api/agent/{name}/profile"
            result = agents_handler.handle(path, {}, None)
            # Should not return 400 for valid names
            assert result.status_code != 400, f"Should accept: {name}"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestAgentsErrorHandling:
    """Tests for error handling."""

    def test_invalid_agent_path_returns_none(self, agents_handler):
        """Short agent paths should return None (not handled)."""
        result = agents_handler.handle("/api/agent", {}, None)
        # Short path should return 400 or None
        assert result is None or result.status_code == 400

    def test_elo_exception_returns_500(self, agents_handler, mock_elo_system):
        """Should return 500 on ELO system exceptions."""
        mock_elo_system.get_cached_leaderboard.side_effect = Exception("DB error")

        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data

    def test_profile_exception_returns_500(self, agents_handler, mock_elo_system):
        """Should return 500 on profile exceptions."""
        mock_elo_system.get_rating.side_effect = Exception("Rating error")

        result = agents_handler.handle("/api/agent/claude/profile", {}, None)

        assert result.status_code == 500

    def test_history_exception_returns_500(self, agents_handler, mock_elo_system):
        """Should return 500 on history exceptions."""
        mock_elo_system.get_elo_history.side_effect = Exception("History error")

        result = agents_handler.handle("/api/agent/claude/history", {}, None)

        assert result.status_code == 500

    def test_compare_exception_returns_500(self, agents_handler, mock_elo_system):
        """Should return 500 on compare exceptions."""
        mock_elo_system.get_ratings_batch.side_effect = Exception("Compare error")

        result = agents_handler.handle("/api/agent/compare", {"agents": ["a", "b"]}, None)

        assert result.status_code == 500


# ============================================================================
# Limit Cap Tests
# ============================================================================


class TestLimitCaps:
    """Tests for limit parameter capping."""

    def test_leaderboard_caps_at_50(self, agents_handler, mock_elo_system):
        """Should cap leaderboard limit at 50."""
        result = agents_handler.handle("/api/leaderboard", {"limit": "100"}, None)

        assert result.status_code == 200
        # Verify limit was capped
        mock_elo_system.get_cached_leaderboard.assert_called_with(limit=50)

    def test_recent_matches_caps_at_50(self, agents_handler, mock_elo_system):
        """Should cap recent matches limit at 50."""
        result = agents_handler.handle("/api/matches/recent", {"limit": "100"}, None)

        assert result.status_code == 200
        mock_elo_system.get_cached_recent_matches.assert_called_with(limit=50)

    def test_history_caps_at_100(self, agents_handler, mock_elo_system):
        """Should cap history limit at 100."""
        result = agents_handler.handle("/api/agent/claude/history", {"limit": "200"}, None)

        assert result.status_code == 200
        mock_elo_system.get_elo_history.assert_called_with("claude", limit=100)


# ============================================================================
# FlipDetector Mock Tests
# ============================================================================


class TestFlipDetectorIntegration:
    """Tests for FlipDetector integration."""

    def test_leaderboard_with_flip_detector(self, agents_handler, mock_elo_system, tmp_path):
        """Should enhance rankings with consistency when FlipDetector available."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.insights.flip_detector.FlipDetector") as MockDetector:
            mock_detector = Mock()
            mock_score = Mock()
            mock_score.total_positions = 10
            mock_score.total_flips = 1
            mock_detector.get_agents_consistency_batch.return_value = {"claude": mock_score}
            MockDetector.return_value = mock_detector

            result = agents_handler.handle("/api/leaderboard", {}, None)

            assert result.status_code == 200
            # Verify FlipDetector was used
            mock_detector.get_agents_consistency_batch.assert_called()

    def test_consistency_with_flip_detector(self, agents_handler, tmp_path):
        """Should return consistency score from FlipDetector."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.insights.flip_detector.FlipDetector") as MockDetector:
            mock_detector = Mock()
            mock_score = Mock()
            mock_score.agent_name = "claude"
            mock_score.total_positions = 10
            mock_score.total_flips = 2
            mock_score.consistency_score = 0.8
            mock_detector.get_agent_consistency.return_value = mock_score
            MockDetector.return_value = mock_detector

            result = agents_handler.handle("/api/agent/claude/consistency", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "consistency_score" in data

    def test_agent_flips_with_detector(self, agents_handler, tmp_path):
        """Should return flip data from FlipDetector."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.insights.flip_detector.FlipDetector") as MockDetector:
            mock_detector = Mock()
            mock_flip = Mock()
            mock_flip.to_dict.return_value = {"agent": "claude", "topic": "test"}
            mock_detector.detect_flips_for_agent.return_value = [mock_flip]
            mock_score = Mock()
            mock_score.to_dict.return_value = {"consistency_score": 0.9}
            mock_detector.get_agent_consistency.return_value = mock_score
            MockDetector.return_value = mock_detector

            result = agents_handler.handle("/api/agent/claude/flips", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert len(data["flips"]) == 1

    def test_recent_flips_with_detector(self, agents_handler, tmp_path):
        """Should return recent flips from FlipDetector."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.insights.flip_detector.FlipDetector") as MockDetector:
            mock_detector = Mock()
            mock_flip = Mock()
            mock_flip.to_dict.return_value = {"agent": "claude", "topic": "test"}
            mock_detector.get_recent_flips.return_value = [mock_flip]
            mock_detector.get_flip_summary.return_value = {"total_flips": 1}
            MockDetector.return_value = mock_detector

            result = agents_handler.handle("/api/flips/recent", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["count"] == 1

    def test_flip_summary_with_detector(self, agents_handler, tmp_path):
        """Should return summary from FlipDetector."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.insights.flip_detector.FlipDetector") as MockDetector:
            mock_detector = Mock()
            mock_detector.get_flip_summary.return_value = {
                "total_flips": 50,
                "by_type": {"reversal": 30},
                "by_agent": {"claude": 10},
                "recent_24h": 5,
            }
            MockDetector.return_value = mock_detector

            result = agents_handler.handle("/api/flips/summary", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["total_flips"] == 50


# ============================================================================
# MomentDetector and PositionLedger Tests
# ============================================================================


class TestMomentDetectorIntegration:
    """Tests for MomentDetector integration."""

    def test_moments_with_detector(self, agents_handler, mock_elo_system):
        """Should return moments from MomentDetector."""
        with patch("aragora.agents.grounded.MomentDetector") as MockMD:
            from datetime import datetime

            mock_moment = Mock()
            mock_moment.id = "m1"
            mock_moment.moment_type = "upset_win"
            mock_moment.agent_name = "claude"
            mock_moment.description = "Big win"
            mock_moment.significance_score = 0.8
            mock_moment.timestamp = datetime.now()
            mock_moment.debate_id = "d1"

            mock_detector = Mock()
            mock_detector.get_agent_moments.return_value = [mock_moment]
            MockMD.return_value = mock_detector

            result = agents_handler.handle("/api/agent/claude/moments", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert len(data["moments"]) == 1
            assert data["moments"][0]["moment_type"] == "upset_win"

    def test_moments_without_elo_returns_empty(self):
        """Should return empty moments without ELO."""
        handler = _wrap_handler(AgentsHandler({}))
        result = handler.handle("/api/agent/claude/moments", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["moments"] == []


class TestPositionLedgerIntegration:
    """Tests for PositionLedger integration."""

    def test_positions_with_ledger(self, agents_handler, tmp_path):
        """Should return positions from PositionLedger."""
        agents_handler.ctx["nomic_dir"] = tmp_path

        with patch("aragora.agents.grounded.PositionLedger") as MockLedger:
            mock_ledger = Mock()
            mock_ledger.get_agent_positions.return_value = [
                {"topic": "AI safety", "position": "for", "confidence": 0.8}
            ]
            MockLedger.return_value = mock_ledger

            result = agents_handler.handle("/api/agent/claude/positions", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert len(data["positions"]) == 1

    def test_positions_without_nomic_dir_returns_empty(self, agents_handler):
        """Should return empty positions without nomic_dir."""
        agents_handler.ctx["nomic_dir"] = None

        result = agents_handler.handle("/api/agent/claude/positions", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["positions"] == []


# ============================================================================
# Agent Object Type Handling Tests
# ============================================================================


class TestAgentObjectTypes:
    """Tests for handling different agent object types."""

    def test_leaderboard_handles_dict_agents(self, agents_handler, mock_elo_system):
        """Should handle agents returned as dicts."""
        mock_elo_system.get_cached_leaderboard.return_value = [
            {"name": "claude", "elo": 1600, "wins": 10, "losses": 5}
        ]

        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        rankings = data.get("rankings") or data.get("agents")
        assert rankings[0]["name"] == "claude"

    def test_leaderboard_handles_object_agents(self, agents_handler, mock_elo_system):
        """Should handle agents returned as objects."""
        mock_agent = Mock(
            spec=["agent_name", "elo", "wins", "losses", "draws", "win_rate", "games_played"]
        )
        mock_agent.agent_name = "gemini"  # Real AgentRating uses agent_name
        mock_agent.elo = 1550
        mock_agent.wins = 8
        mock_agent.losses = 7
        mock_agent.draws = 0
        mock_agent.win_rate = 0.53
        mock_agent.games_played = 15

        mock_elo_system.get_cached_leaderboard.return_value = [mock_agent]

        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        rankings = data.get("rankings") or data.get("agents")
        assert rankings[0]["name"] == "gemini"
        assert rankings[0]["elo"] == 1550


# ============================================================================
# Domain Validation Tests
# ============================================================================


class TestDomainValidation:
    """Tests for domain parameter validation."""

    def test_rejects_path_traversal_in_domain(self, agents_handler):
        """Should reject path traversal in domain."""
        result = agents_handler.handle("/api/leaderboard", {"domain": "../../../etc"}, None)

        assert result.status_code == 400

    def test_accepts_valid_domain(self, agents_handler, mock_elo_system):
        """Should accept valid domain."""
        result = agents_handler.handle("/api/leaderboard", {"domain": "coding"}, None)

        assert result.status_code == 200

    def test_rejects_special_chars_in_domain(self, agents_handler):
        """Should reject special characters in domain."""
        result = agents_handler.handle("/api/leaderboard", {"domain": "test;drop"}, None)

        assert result.status_code == 400


# ============================================================================
# Loop ID Validation Tests
# ============================================================================


class TestLoopIdValidation:
    """Tests for loop_id parameter validation."""

    def test_accepts_valid_loop_id(self, agents_handler, mock_elo_system):
        """Should accept valid loop_id."""
        result = agents_handler.handle("/api/matches/recent", {"loop_id": "loop-123"}, None)

        assert result.status_code == 200

    def test_rejects_invalid_loop_id(self, agents_handler):
        """Should reject invalid loop_id."""
        result = agents_handler.handle("/api/matches/recent", {"loop_id": "../../etc"}, None)

        assert result.status_code == 400


# ============================================================================
# Compare Agents Edge Cases
# ============================================================================


class TestCompareAgentsEdgeCases:
    """Edge case tests for agent comparison."""

    def test_compare_limits_to_five_agents(self, agents_handler, mock_elo_system):
        """Should limit comparison to 5 agents."""
        result = agents_handler.handle(
            "/api/agent/compare", {"agents": ["a", "b", "c", "d", "e", "f", "g"]}, None
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["agents"]) == 5

    def test_compare_handles_h2h_exception(self, agents_handler, mock_elo_system):
        """Should handle head-to-head exception gracefully."""
        mock_elo_system.get_head_to_head.side_effect = Exception("H2H error")

        result = agents_handler.handle("/api/agent/compare", {"agents": ["claude", "gpt4"]}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should still return agents, just no h2h
        assert "agents" in data


# ============================================================================
# Head-to-Head Fallback Tests
# ============================================================================


class TestHeadToHeadFallback:
    """Tests for head-to-head fallback behavior."""

    def test_h2h_fallback_without_method(self, agents_handler, mock_elo_system):
        """Should fallback when get_head_to_head not available."""
        del mock_elo_system.get_head_to_head

        result = agents_handler.handle("/api/agent/claude/head-to-head/gpt4", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["matches"] == 0


# ============================================================================
# Calibration Fallback Tests
# ============================================================================


class TestCalibrationFallback:
    """Tests for calibration fallback behavior."""

    def test_calibration_fallback_without_method(self, agents_handler, mock_elo_system):
        """Should fallback when get_calibration not available."""
        del mock_elo_system.get_calibration

        result = agents_handler.handle("/api/agent/claude/calibration", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["score"] == 0.5


# ============================================================================
# Cache Behavior Tests
# ============================================================================


class TestCacheBehavior:
    """Tests for caching behavior."""

    def test_leaderboard_uses_cache(self, agents_handler, mock_elo_system):
        """Should use cached leaderboard when no domain."""
        result = agents_handler.handle("/api/leaderboard", {}, None)

        assert result.status_code == 200
        mock_elo_system.get_cached_leaderboard.assert_called()

    def test_leaderboard_bypasses_cache_with_domain(self, agents_handler, mock_elo_system):
        """Should bypass cache when domain specified."""
        result = agents_handler.handle("/api/leaderboard", {"domain": "coding"}, None)

        assert result.status_code == 200
        mock_elo_system.get_leaderboard.assert_called()

    def test_recent_matches_uses_cache(self, agents_handler, mock_elo_system):
        """Should use cached recent matches."""
        result = agents_handler.handle("/api/matches/recent", {}, None)

        assert result.status_code == 200
        mock_elo_system.get_cached_recent_matches.assert_called()


# ============================================================================
# Nomic Dir Handling Tests
# ============================================================================


class TestNomicDirHandling:
    """Tests for nomic_dir handling."""

    def test_flips_without_nomic_dir(self, agents_handler):
        """Should return empty flips without nomic_dir."""
        agents_handler.ctx["nomic_dir"] = None

        result = agents_handler.handle("/api/agent/claude/flips", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["flips"] == []
        assert data["count"] == 0

    def test_recent_flips_without_nomic_dir(self, agents_handler):
        """Should return empty recent flips without nomic_dir."""
        agents_handler.ctx["nomic_dir"] = None

        result = agents_handler.handle("/api/flips/recent", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["flips"] == []

    def test_flip_summary_without_nomic_dir(self, agents_handler):
        """Should return default summary without nomic_dir."""
        agents_handler.ctx["nomic_dir"] = None

        result = agents_handler.handle("/api/flips/summary", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_flips"] == 0

    def test_consistency_without_nomic_dir(self, agents_handler):
        """Should return default consistency without nomic_dir."""
        agents_handler.ctx["nomic_dir"] = None

        result = agents_handler.handle("/api/agent/claude/consistency", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["consistency_score"] == 1.0
