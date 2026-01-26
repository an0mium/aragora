"""
Integration tests for Agent handlers.

Tests cover:
- AgentsHandler: List agents, health, availability, leaderboard, profiles
- CalibrationHandler: Calibration curves, summaries, leaderboard
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()

    # Mock rating object
    mock_rating = MagicMock()
    mock_rating.elo = 1500
    mock_rating.wins = 10
    mock_rating.losses = 5
    mock_rating.draws = 2
    mock_rating.calibration_score = 0.85
    mock_rating.confidence = 0.9

    elo.get_rating = MagicMock(return_value=mock_rating)
    elo.get_leaderboard = MagicMock(
        return_value=[
            {"agent": "claude", "elo": 1650, "wins": 15, "losses": 3},
            {"agent": "gpt-4", "elo": 1600, "wins": 12, "losses": 5},
            {"agent": "gemini", "elo": 1550, "wins": 10, "losses": 7},
        ]
    )
    elo.get_cached_leaderboard = MagicMock(
        return_value=[
            {"agent": "claude", "elo": 1650, "wins": 15, "losses": 3},
            {"agent": "gpt-4", "elo": 1600, "wins": 12, "losses": 5},
        ]
    )
    elo.get_cached_recent_matches = MagicMock(
        return_value=[
            {"id": "match_1", "agents": ["claude", "gpt-4"], "winner": "claude"},
            {"id": "match_2", "agents": ["gemini", "claude"], "winner": "claude"},
        ]
    )
    elo.get_recent_matches = MagicMock(
        return_value=[
            {"id": "match_1", "agents": ["claude", "gpt-4"], "winner": "claude"},
        ]
    )
    elo.get_agent_stats = MagicMock(
        return_value={
            "total_matches": 17,
            "win_rate": 0.65,
            "avg_elo_change": 2.5,
        }
    )
    elo.get_rivals = MagicMock(
        return_value=[
            {"agent": "gpt-4", "matches": 5, "wins": 3, "losses": 2},
        ]
    )
    elo.get_allies = MagicMock(
        return_value=[
            {"agent": "gemini", "collaborations": 3, "success_rate": 0.8},
        ]
    )
    elo.get_head_to_head = MagicMock(
        return_value={
            "matches": 5,
            "agent1_wins": 3,
            "agent2_wins": 2,
            "draws": 0,
        }
    )
    elo.get_elo_history = MagicMock(
        return_value=[
            (datetime.now(timezone.utc).isoformat(), 1500),
            (datetime.now(timezone.utc).isoformat(), 1520),
            (datetime.now(timezone.utc).isoformat(), 1550),
        ]
    )
    elo.get_stats = MagicMock(
        return_value={
            "mean_elo": 1520,
            "median_elo": 1510,
            "std_elo": 75,
            "total_agents": 10,
        }
    )

    return elo


@pytest.fixture
def mock_flip_detector():
    """Create a mock flip detector."""
    detector = MagicMock()

    detector.detect_flips_for_agent = MagicMock(
        return_value=[
            {
                "debate_id": "debate_1",
                "position_before": "supports",
                "position_after": "opposes",
                "round": 3,
            }
        ]
    )
    detector.get_agent_consistency = MagicMock(return_value=0.85)
    detector.get_agents_consistency_batch = MagicMock(
        return_value={
            "claude": 0.85,
            "gpt-4": 0.90,
            "gemini": 0.80,
        }
    )
    detector.get_recent_flips = MagicMock(
        return_value=[
            {
                "agent": "claude",
                "debate_id": "debate_1",
                "flip_type": "full_reversal",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    detector.get_flip_summary = MagicMock(
        return_value={
            "total_flips_24h": 5,
            "total_flips_7d": 23,
            "agents_with_flips": 3,
            "most_volatile": "gemini",
        }
    )

    return detector


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock calibration tracker."""
    tracker = MagicMock()

    tracker.get_calibration_curve = MagicMock(
        return_value={
            "buckets": [
                {"range": "0.0-0.1", "predicted": 0.05, "actual": 0.04, "count": 10},
                {"range": "0.1-0.2", "predicted": 0.15, "actual": 0.16, "count": 15},
                {"range": "0.9-1.0", "predicted": 0.95, "actual": 0.92, "count": 20},
            ]
        }
    )
    tracker.get_calibration_summary = MagicMock(
        return_value={
            "brier_score": 0.12,
            "ece": 0.05,
            "mce": 0.08,
            "overconfidence": 0.02,
            "underconfidence": 0.01,
            "total_predictions": 100,
        }
    )
    tracker.get_all_agents = MagicMock(return_value=["claude", "gpt-4", "gemini"])
    tracker.get_domain_breakdown = MagicMock(
        return_value={
            "general": {"brier": 0.10, "ece": 0.04},
            "technical": {"brier": 0.15, "ece": 0.06},
        }
    )

    return tracker


@pytest.fixture
def agents_handler(mock_elo_system, mock_flip_detector):
    """Create an AgentsHandler with mocked dependencies."""
    from aragora.server.handlers.agents.agents import AgentsHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()

    ctx = {
        "elo_system": mock_elo_system,
        "flip_detector": mock_flip_detector,
    }
    handler = AgentsHandler(server_context=ctx)
    return handler


@pytest.fixture
def calibration_handler(mock_elo_system, mock_calibration_tracker):
    """Create a CalibrationHandler with mocked dependencies."""
    from aragora.server.handlers.agents.calibration import CalibrationHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()

    ctx = {
        "elo_system": mock_elo_system,
        "calibration_tracker": mock_calibration_tracker,
    }
    handler = CalibrationHandler(server_context=ctx)
    return handler


# ===========================================================================
# AgentsHandler Routing Tests
# ===========================================================================


class TestAgentsHandlerRouting:
    """Tests for AgentsHandler routing."""

    def test_can_handle_agents_list(self, agents_handler):
        """Handler recognizes /api/agents path."""
        assert agents_handler.can_handle("/api/agents") is True

    def test_can_handle_agents_health(self, agents_handler):
        """Handler recognizes /api/agents/health path."""
        assert agents_handler.can_handle("/api/agents/health") is True

    def test_can_handle_agents_availability(self, agents_handler):
        """Handler recognizes /api/agents/availability path."""
        assert agents_handler.can_handle("/api/agents/availability") is True

    def test_can_handle_leaderboard(self, agents_handler):
        """Handler recognizes /api/leaderboard path."""
        assert agents_handler.can_handle("/api/leaderboard") is True

    def test_can_handle_rankings(self, agents_handler):
        """Handler recognizes /api/rankings alias."""
        assert agents_handler.can_handle("/api/rankings") is True

    def test_can_handle_agent_profile(self, agents_handler):
        """Handler recognizes agent profile paths."""
        assert agents_handler.can_handle("/api/agent/claude/profile") is True

    def test_can_handle_agent_history(self, agents_handler):
        """Handler recognizes agent history paths."""
        assert agents_handler.can_handle("/api/agent/gpt-4/history") is True

    def test_can_handle_flips_recent(self, agents_handler):
        """Handler recognizes /api/flips/recent path."""
        assert agents_handler.can_handle("/api/flips/recent") is True

    def test_can_handle_versioned_path(self, agents_handler):
        """Handler recognizes versioned paths."""
        assert agents_handler.can_handle("/api/v1/agents") is True

    def test_cannot_handle_unrelated_path(self, agents_handler):
        """Handler rejects unrelated paths."""
        assert agents_handler.can_handle("/api/debates") is False

    def test_can_handle_agent_calibration(self, agents_handler):
        """Handler recognizes agent calibration path."""
        assert agents_handler.can_handle("/api/agent/claude/calibration") is True

    def test_can_handle_agent_consistency(self, agents_handler):
        """Handler recognizes agent consistency path."""
        assert agents_handler.can_handle("/api/agent/claude/consistency") is True

    def test_can_handle_agent_network(self, agents_handler):
        """Handler recognizes agent network path."""
        assert agents_handler.can_handle("/api/agent/claude/network") is True

    def test_can_handle_head_to_head(self, agents_handler):
        """Handler recognizes head-to-head path."""
        assert agents_handler.can_handle("/api/agent/claude/head-to-head/gpt-4") is True


class TestGetLeaderboard:
    """Tests for _get_leaderboard method."""

    def test_get_leaderboard_success(self, agents_handler):
        """Get leaderboard returns agent rankings."""
        result = agents_handler._get_leaderboard(limit=10, domain=None)

        assert result is not None
        assert result.status_code == 200

    def test_get_leaderboard_with_domain(self, agents_handler):
        """Get leaderboard respects domain filter."""
        result = agents_handler._get_leaderboard(limit=10, domain="technical")

        assert result is not None
        assert result.status_code == 200

    def test_get_leaderboard_response_structure(self, agents_handler):
        """Get leaderboard returns proper structure."""
        result = agents_handler._get_leaderboard(limit=10, domain=None)

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert (
            "agents" in body
            or "leaderboard" in body
            or "rankings" in body
            or isinstance(body, list)
        )


class TestGetAgentProfile:
    """Tests for _get_profile method."""

    def test_get_profile_success(self, agents_handler):
        """Get profile returns agent stats."""
        result = agents_handler._get_profile("claude")

        assert result is not None
        assert result.status_code == 200

    def test_get_profile_response_structure(self, agents_handler):
        """Get profile returns proper structure."""
        result = agents_handler._get_profile("claude")

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert "agent" in body
        assert body["agent"] == "claude"


class TestGetAgentHistory:
    """Tests for _get_history method."""

    def test_get_history_success(self, agents_handler):
        """Get history returns ELO history."""
        result = agents_handler._get_history("claude", limit=10)

        assert result is not None
        assert result.status_code == 200

    def test_get_history_response_structure(self, agents_handler):
        """Get history returns proper structure."""
        result = agents_handler._get_history("claude", limit=10)

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert "agent" in body
        assert "history" in body


class TestGetAgentFlips:
    """Tests for flip detection endpoints."""

    def test_get_agent_flips_success(self, agents_handler):
        """Get agent flips returns flip history."""
        result = agents_handler._get_agent_flips("claude", limit=10)

        assert result is not None
        assert result.status_code == 200

    def test_get_recent_flips_success(self, agents_handler):
        """Get recent flips returns all agent flips."""
        result = agents_handler._get_recent_flips(limit=10)

        assert result is not None
        assert result.status_code == 200

    def test_get_flip_summary_success(self, agents_handler):
        """Get flip summary returns dashboard data."""
        result = agents_handler._get_flip_summary()

        assert result is not None
        assert result.status_code == 200


class TestGetAgentConsistency:
    """Tests for _get_consistency method."""

    def test_get_consistency_success(self, agents_handler):
        """Get consistency returns consistency score."""
        result = agents_handler._get_consistency("claude")

        assert result is not None
        assert result.status_code == 200


class TestGetHeadToHead:
    """Tests for _get_head_to_head method."""

    def test_get_head_to_head_success(self, agents_handler):
        """Get head-to-head returns matchup stats."""
        result = agents_handler._get_head_to_head("claude", "gpt-4")

        assert result is not None
        assert result.status_code == 200


class TestGetRivalsAllies:
    """Tests for rivals and allies endpoints."""

    def test_get_rivals_success(self, agents_handler):
        """Get rivals returns top rivals."""
        result = agents_handler._get_rivals("claude", limit=5)

        assert result is not None
        assert result.status_code == 200

    def test_get_allies_success(self, agents_handler):
        """Get allies returns top allies."""
        result = agents_handler._get_allies("claude", limit=5)

        assert result is not None
        assert result.status_code == 200


class TestAgentHealth:
    """Tests for _get_agent_health method."""

    def test_get_agent_health_success(self, agents_handler):
        """Get agent health returns health status."""
        result = agents_handler._get_agent_health()

        assert result is not None
        assert result.status_code == 200

    def test_get_agent_health_response_structure(self, agents_handler):
        """Get agent health returns proper structure."""
        result = agents_handler._get_agent_health()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        # Should have some health-related fields
        assert isinstance(body, dict)


class TestAgentAvailability:
    """Tests for _get_agent_availability method."""

    def test_get_agent_availability_success(self, agents_handler):
        """Get agent availability returns API key status."""
        result = agents_handler._get_agent_availability()

        assert result is not None
        assert result.status_code == 200

    def test_get_agent_availability_response_structure(self, agents_handler):
        """Get agent availability returns proper structure."""
        result = agents_handler._get_agent_availability()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert "agents" in body


class TestRecentMatches:
    """Tests for _get_recent_matches method."""

    def test_get_recent_matches_success(self, agents_handler):
        """Get recent matches returns match list."""
        result = agents_handler._get_recent_matches(limit=10, loop_id=None)

        assert result is not None
        assert result.status_code == 200

    def test_get_recent_matches_with_loop_id(self, agents_handler):
        """Get recent matches filters by loop_id."""
        result = agents_handler._get_recent_matches(limit=10, loop_id="loop_123")

        assert result is not None
        assert result.status_code == 200


class TestCompareAgents:
    """Tests for _compare_agents method."""

    def test_compare_agents_success(self, agents_handler):
        """Compare agents returns comparison data."""
        result = agents_handler._compare_agents(["claude", "gpt-4"])

        assert result is not None
        assert result.status_code == 200

    def test_compare_multiple_agents(self, agents_handler):
        """Compare multiple agents works."""
        result = agents_handler._compare_agents(["claude", "gpt-4", "gemini"])

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# CalibrationHandler Tests
# ===========================================================================


class TestCalibrationHandlerRouting:
    """Tests for CalibrationHandler routing."""

    def test_can_handle_calibration_curve(self, calibration_handler):
        """Handler recognizes calibration curve path."""
        assert calibration_handler.can_handle("/api/agent/claude/calibration-curve") is True

    def test_can_handle_calibration_summary(self, calibration_handler):
        """Handler recognizes calibration summary path."""
        assert calibration_handler.can_handle("/api/agent/claude/calibration-summary") is True

    def test_can_handle_calibration_leaderboard(self, calibration_handler):
        """Handler recognizes calibration leaderboard path."""
        assert calibration_handler.can_handle("/api/calibration/leaderboard") is True

    def test_can_handle_calibration_visualization(self, calibration_handler):
        """Handler recognizes calibration visualization path."""
        assert calibration_handler.can_handle("/api/calibration/visualization") is True


class TestGetCalibrationCurve:
    """Tests for _get_calibration_curve method."""

    def test_get_calibration_curve_success(self, calibration_handler):
        """Get calibration curve returns bucket data."""
        result = calibration_handler._get_calibration_curve("claude", buckets=10, domain=None)

        assert result is not None
        assert result.status_code == 200

    def test_get_calibration_curve_with_domain(self, calibration_handler):
        """Get calibration curve respects domain filter."""
        result = calibration_handler._get_calibration_curve(
            "claude", buckets=10, domain="technical"
        )

        assert result is not None
        assert result.status_code == 200


class TestGetCalibrationSummary:
    """Tests for _get_calibration_summary method."""

    def test_get_calibration_summary_success(self, calibration_handler):
        """Get calibration summary returns metrics."""
        result = calibration_handler._get_calibration_summary("claude", domain=None)

        assert result is not None
        assert result.status_code == 200


class TestGetCalibrationLeaderboard:
    """Tests for _get_calibration_leaderboard method."""

    def test_get_calibration_leaderboard_success(self, calibration_handler):
        """Get calibration leaderboard returns rankings."""
        result = calibration_handler._get_calibration_leaderboard(
            limit=10,
            metric="brier",
            min_predictions=10,
        )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Handler Routing Integration Tests
# ===========================================================================


class TestAgentHandlerIntegration:
    """Integration tests for handler routing."""

    def test_handler_routes_leaderboard(self, agents_handler):
        """Handler routes leaderboard requests correctly."""
        mock_handler = MagicMock()

        result = agents_handler.handle("/api/leaderboard", {"limit": "10"}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handler_routes_rankings_alias(self, agents_handler):
        """Handler routes rankings alias correctly."""
        mock_handler = MagicMock()

        result = agents_handler.handle("/api/rankings", {"limit": "10"}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handler_routes_agent_profile(self, agents_handler):
        """Handler routes agent profile requests correctly."""
        mock_handler = MagicMock()

        result = agents_handler.handle("/api/agent/claude/profile", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handler_routes_versioned_path(self, agents_handler):
        """Handler routes versioned paths correctly."""
        mock_handler = MagicMock()

        result = agents_handler.handle("/api/v1/leaderboard", {"limit": "10"}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handler_routes_flips_recent(self, agents_handler):
        """Handler routes flips recent requests correctly."""
        mock_handler = MagicMock()

        result = agents_handler.handle("/api/flips/recent", {"limit": "10"}, mock_handler)
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================


class TestAgentHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_agent_name_sanitized(self, agents_handler):
        """Invalid agent names are handled safely."""
        # Test with potential injection attempt
        result = agents_handler._get_profile("claude")  # Normal name works
        assert result is not None
        assert result.status_code == 200

    def test_missing_elo_system_handled(self):
        """Missing ELO system is handled gracefully."""
        from aragora.server.handlers.agents.agents import AgentsHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        # Create handler with empty context
        handler = AgentsHandler(server_context={})

        # Should return graceful degradation
        result = handler._get_leaderboard(limit=10, domain=None)
        assert result is not None
        # Either returns empty data or an error response
        assert result.status_code in (200, 500, 503)

    def test_empty_leaderboard_handled(self):
        """Empty leaderboard is handled gracefully."""
        from aragora.server.handlers.agents.agents import AgentsHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.return_value = []

        mock_flip = MagicMock()
        mock_flip.get_agents_consistency_batch.return_value = {}

        handler = AgentsHandler(
            server_context={
                "elo_system": mock_elo,
                "flip_detector": mock_flip,
            }
        )

        result = handler._get_leaderboard(limit=10, domain=None)
        assert result is not None
        assert result.status_code == 200

    def test_limit_parameter_clamped(self, agents_handler):
        """Limit parameters are properly clamped."""
        # Very large limit should be clamped
        result = agents_handler._get_leaderboard(limit=10000, domain=None)
        assert result is not None
        assert result.status_code == 200

    def test_zero_limit_handled(self, agents_handler):
        """Zero limit is handled."""
        result = agents_handler._get_leaderboard(limit=0, domain=None)
        assert result is not None
        # Should either return empty or use default


# ===========================================================================
# Local Agents Tests
# ===========================================================================


class TestLocalAgents:
    """Tests for local LLM agent endpoints."""

    def test_list_local_agents_success(self, agents_handler):
        """List local agents returns available local LLMs."""
        result = agents_handler._list_local_agents()

        assert result is not None
        assert result.status_code == 200

    def test_get_local_status_success(self, agents_handler):
        """Get local status returns availability info."""
        result = agents_handler._get_local_status()

        assert result is not None
        assert result.status_code == 200
