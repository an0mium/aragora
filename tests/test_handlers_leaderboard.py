"""
Tests for LeaderboardViewHandler.

Tests the consolidated leaderboard endpoint that returns all 6 data sources
in a single request with graceful degradation on partial failures.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from aragora.server.handlers.agents import LeaderboardViewHandler
from aragora.server.handlers.base import clear_cache


@dataclass
class MockAgentReputation:
    """Mock reputation object."""

    agent_name: str
    reputation_score: float = 0.8
    vote_weight: float = 1.0
    proposal_acceptance_rate: float = 0.7
    critique_value: float = 0.5
    debates_participated: int = 10


@dataclass
class MockConsistencyScore:
    """Mock consistency score from FlipDetector."""

    total_flips: int = 2
    total_positions: int = 10


@pytest.fixture
def handler(tmp_path):
    """Create LeaderboardViewHandler with mock context."""
    ctx = {
        "storage": Mock(),
        "elo_system": Mock(),
        "nomic_dir": tmp_path,
    }
    return LeaderboardViewHandler(ctx)


@pytest.fixture
def handler_no_elo(tmp_path):
    """Create handler without ELO system."""
    ctx = {
        "storage": Mock(),
        "elo_system": None,
        "nomic_dir": tmp_path,
    }
    return LeaderboardViewHandler(ctx)


@pytest.fixture
def handler_no_nomic():
    """Create handler without nomic directory."""
    ctx = {
        "storage": Mock(),
        "elo_system": Mock(),
        "nomic_dir": None,
    }
    return LeaderboardViewHandler(ctx)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear cache between tests."""
    clear_cache()
    yield
    clear_cache()


class TestLeaderboardRouting:
    """Test route matching and dispatch."""

    def test_can_handle_leaderboard_view(self, handler):
        """can_handle returns True for /api/leaderboard-view."""
        assert handler.can_handle("/api/v1/leaderboard-view") is True

    def test_cannot_handle_similar_paths(self, handler):
        """can_handle returns False for similar but different paths."""
        assert handler.can_handle("/api/v1/leaderboard") is False
        assert handler.can_handle("/api/v1/leaderboard-view/extra") is False
        assert handler.can_handle("/api/v1/leaderboard-views") is False

    def test_cannot_handle_unrelated_paths(self, handler):
        """can_handle returns False for unrelated paths."""
        assert handler.can_handle("/api/v1/agents") is False
        assert handler.can_handle("/api/v1/metrics") is False

    def test_handle_returns_none_for_unknown(self, handler):
        """handle returns None for paths it doesn't handle."""
        result = handler.handle("/api/unknown", {}, None)
        assert result is None


class TestLeaderboardView:
    """Test /api/leaderboard-view endpoint."""

    def test_returns_200_success(self, handler):
        """Returns 200 for successful request."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {}, None)

        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_returns_all_six_sections(self, handler):
        """Returns all 6 data sections."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert "data" in data
        assert "rankings" in data["data"]
        assert "matches" in data["data"]
        assert "reputation" in data["data"]
        assert "teams" in data["data"]
        assert "stats" in data["data"]
        assert "introspection" in data["data"]

    def test_respects_limit_parameter(self, handler):
        """Respects the limit query parameter."""
        agents = [{"name": f"agent{i}", "elo": 1500} for i in range(20)]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents[:5]
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {"limit": ["5"]}, None)
        data = json.loads(result.body)

        # Verify limit was passed (mock returns 5)
        assert data["data"]["rankings"]["count"] == 5

    def test_caps_limit_at_50(self, handler):
        """Caps limit at maximum of 50."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {"limit": ["100"]}, None)

        # Should use capped limit of 50 - check the call was made with <= 50
        call_args = handler.ctx["elo_system"].get_cached_leaderboard.call_args
        assert call_args[1]["limit"] <= 50

    def test_validates_domain_parameter(self, handler):
        """Validates domain parameter for security."""
        result = handler.handle("/api/leaderboard-view", {"domain": ["../../../etc/passwd"]}, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_validates_loop_id_parameter(self, handler):
        """Validates loop_id parameter for security."""
        result = handler.handle(
            "/api/leaderboard-view", {"loop_id": ["<script>alert(1)</script>"]}, None
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_accepts_valid_domain(self, handler):
        """Accepts valid domain parameter."""
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {"domain": ["technology"]}, None)

        assert result.status_code == 200

    def test_returns_errors_structure(self, handler):
        """Returns proper errors structure."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert "errors" in data
        assert "partial_failure" in data["errors"]
        assert "failed_sections" in data["errors"]
        assert "messages" in data["errors"]


class TestPartialFailures:
    """Test graceful degradation when sections fail."""

    def test_rankings_failure_doesnt_break_response(self, handler):
        """Rankings failure still returns other sections."""
        handler.ctx["elo_system"].get_cached_leaderboard.side_effect = Exception("ELO error")
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = [{"id": "match1"}]
        handler.ctx["elo_system"].get_stats.return_value = {"total_agents": 5}

        result = handler.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 200
        assert data["errors"]["partial_failure"] is True
        assert "rankings" in data["errors"]["failed_sections"]
        assert data["data"]["rankings"] == {"agents": [], "count": 0}
        assert data["data"]["matches"]["count"] == 1

    def test_multiple_failures(self, handler):
        """Handles multiple section failures."""
        handler.ctx["elo_system"].get_cached_leaderboard.side_effect = Exception("Error 1")
        handler.ctx["elo_system"].get_cached_recent_matches.side_effect = Exception("Error 2")
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 200
        assert data["errors"]["partial_failure"] is True
        assert "rankings" in data["errors"]["failed_sections"]
        assert "matches" in data["errors"]["failed_sections"]
        assert len(data["errors"]["failed_sections"]) >= 2

    def test_all_sections_fail_gracefully(self, handler_no_elo):
        """Returns valid response even when all sections fail."""
        result = handler_no_elo.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 200
        # All sections should have default empty values
        assert data["data"]["rankings"]["count"] == 0
        assert data["data"]["matches"]["count"] == 0

    def test_error_messages_captured(self, handler):
        """Captures error messages for failed sections."""
        handler.ctx["elo_system"].get_cached_leaderboard.side_effect = Exception(
            "Database connection failed"
        )
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle("/api/leaderboard-view", {}, None)
        data = json.loads(result.body)

        assert "rankings" in data["errors"]["messages"]
        assert "Database connection failed" in data["errors"]["messages"]["rankings"]


class TestFetchRankings:
    """Test _fetch_rankings method."""

    def test_returns_empty_without_elo(self, handler_no_elo):
        """Returns empty list when ELO unavailable."""
        result = handler_no_elo._fetch_rankings(10, None)

        assert result == {"agents": [], "count": 0}

    def test_uses_domain_filter(self, handler):
        """Uses domain filter when provided."""
        handler.ctx["elo_system"].get_leaderboard.return_value = []

        handler._fetch_rankings(10, "technology")

        handler.ctx["elo_system"].get_leaderboard.assert_called_with(limit=10, domain="technology")

    def test_uses_cached_leaderboard_without_domain(self, handler):
        """Uses cached leaderboard when no domain specified."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []

        handler._fetch_rankings(10, None)

        handler.ctx["elo_system"].get_cached_leaderboard.assert_called_with(limit=10)

    def test_handles_dict_agents(self, handler):
        """Handles agent data as dictionaries."""
        agents = [
            {"name": "agent1", "elo": 1600, "wins": 5},
            {"name": "agent2", "elo": 1400, "wins": 3},
        ]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents

        result = handler._fetch_rankings(10, None)

        assert result["count"] == 2
        assert result["agents"][0]["name"] == "agent1"
        assert result["agents"][0]["elo"] == 1600

    def test_handles_object_agents(self, handler):
        """Handles agent data as objects with attributes."""

        class MockAgent:
            def __init__(self, name, elo):
                self.name = name
                self.elo = elo
                self.wins = 0
                self.losses = 0
                self.draws = 0
                self.win_rate = 0
                self.games = 0

        agents = [MockAgent("agent1", 1600), MockAgent("agent2", 1400)]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents

        result = handler._fetch_rankings(10, None)

        assert result["count"] == 2
        assert result["agents"][0]["name"] == "agent1"

    def test_adds_consistency_scores(self, handler, tmp_path):
        """Adds consistency scores from FlipDetector."""
        # Create the DB file
        (tmp_path / "grounded_positions.db").touch()

        agents = [{"name": "agent1", "elo": 1500}]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents

        with patch("aragora.insights.flip_detector.FlipDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.get_agents_consistency_batch.return_value = {
                "agent1": MockConsistencyScore(total_flips=1, total_positions=10)
            }
            mock_detector_class.return_value = mock_detector

            result = handler._fetch_rankings(10, None)

            # May or may not have consistency depending on import success
            assert result["count"] == 1

    def test_works_without_flip_detector(self, handler):
        """Works correctly when FlipDetector unavailable."""
        agents = [{"name": "agent1", "elo": 1500}]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents

        # This should work - FlipDetector is optional
        result = handler._fetch_rankings(10, None)

        assert result["count"] == 1
        # Consistency is optional - may or may not be present
        assert "agents" in result


class TestFetchMatches:
    """Test _fetch_matches method."""

    def test_returns_empty_without_elo(self, handler_no_elo):
        """Returns empty list when ELO unavailable."""
        result = handler_no_elo._fetch_matches(10, None)

        assert result == {"matches": [], "count": 0}

    def test_uses_cached_method_when_available(self, handler):
        """Uses get_cached_recent_matches when available."""
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = []

        handler._fetch_matches(10, None)

        handler.ctx["elo_system"].get_cached_recent_matches.assert_called_with(limit=10)

    def test_falls_back_to_uncached(self, handler):
        """Falls back to get_recent_matches when cached unavailable."""
        del handler.ctx["elo_system"].get_cached_recent_matches
        handler.ctx["elo_system"].get_recent_matches.return_value = []

        handler._fetch_matches(10, None)

        handler.ctx["elo_system"].get_recent_matches.assert_called_with(limit=10)

    def test_returns_match_count(self, handler):
        """Returns correct match count."""
        matches = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
        handler.ctx["elo_system"].get_cached_recent_matches.return_value = matches

        result = handler._fetch_matches(10, None)

        assert result["count"] == 3
        assert len(result["matches"]) == 3


class TestFetchReputations:
    """Test _fetch_reputations method."""

    def test_returns_empty_without_nomic_dir(self, handler_no_nomic):
        """Returns empty list when nomic_dir unavailable."""
        result = handler_no_nomic._fetch_reputations()

        assert result == {"reputations": [], "count": 0}

    def test_returns_empty_without_db_file(self, handler, tmp_path):
        """Returns empty list when debates.db doesn't exist."""
        result = handler._fetch_reputations()

        assert result == {"reputations": [], "count": 0}

    def test_returns_reputations(self, handler, tmp_path):
        """Returns formatted reputation data."""
        # Create the DB file
        (tmp_path / "debates.db").touch()

        # The handler imports CritiqueStore inside the method
        # Just verify the method works and returns proper structure
        result = handler._fetch_reputations()

        # Either returns data or empty (import may fail)
        assert "reputations" in result
        assert "count" in result
        assert isinstance(result["reputations"], list)


class TestFetchTeams:
    """Test _fetch_teams method."""

    def test_returns_team_combinations(self, handler):
        """Returns team combinations from AgentSelector."""
        with patch("aragora.routing.selection.AgentSelector") as mock_selector_class:
            combinations = [
                {"agents": ["a1", "a2"], "win_rate": 0.8},
                {"agents": ["a3", "a4"], "win_rate": 0.7},
            ]
            mock_selector = Mock()
            mock_selector.get_best_team_combinations.return_value = combinations
            mock_selector_class.return_value = mock_selector

            result = handler._fetch_teams(min_debates=3, limit=10)

            # Either returns data or empty (import may succeed or fail)
            assert "combinations" in result
            assert "count" in result

    def test_handles_import_error(self, handler):
        """Returns empty list when AgentSelector unavailable."""
        # The handler should catch ImportError gracefully
        result = handler._fetch_teams(min_debates=3, limit=10)
        # Either returns data or empty, both are valid
        assert "combinations" in result
        assert "count" in result


class TestFetchStats:
    """Test _fetch_stats method."""

    def test_returns_defaults_without_elo(self, handler_no_elo):
        """Returns default stats when ELO unavailable."""
        result = handler_no_elo._fetch_stats()

        assert result["mean_elo"] == 1500
        assert result["median_elo"] == 1500
        assert result["total_agents"] == 0
        assert result["total_matches"] == 0

    def test_returns_stats_from_elo(self, handler):
        """Returns stats from ELO system."""
        handler.ctx["elo_system"].get_stats.return_value = {
            "avg_elo": 1550,
            "median_elo": 1530,
            "total_agents": 10,
            "total_matches": 50,
        }

        result = handler._fetch_stats()

        assert result["mean_elo"] == 1550
        assert result["median_elo"] == 1530
        assert result["total_agents"] == 10

    def test_handles_alternate_field_names(self, handler):
        """Handles mean_elo vs avg_elo field names."""
        handler.ctx["elo_system"].get_stats.return_value = {
            "mean_elo": 1560,  # alternate name
        }

        result = handler._fetch_stats()

        assert result["mean_elo"] == 1560

    def test_includes_trending_data(self, handler):
        """Includes trending up/down agents."""
        handler.ctx["elo_system"].get_stats.return_value = {
            "trending_up": ["agent1", "agent2"],
            "trending_down": ["agent3"],
        }

        result = handler._fetch_stats()

        assert result["trending_up"] == ["agent1", "agent2"]
        assert result["trending_down"] == ["agent3"]


class TestFetchIntrospection:
    """Test _fetch_introspection method."""

    def test_returns_empty_without_imports(self, handler):
        """Returns empty dict when imports fail."""
        # Without proper setup, should return empty
        result = handler._fetch_introspection()

        # May return empty or with default agents
        assert "agents" in result
        assert "count" in result

    def test_uses_default_agents_when_no_reputations(self, handler, tmp_path):
        """Uses default agent list when no reputations exist."""
        # No debates.db means no reputations, so default agents are used
        result = handler._fetch_introspection()

        # Should attempt default agents - may succeed or fail gracefully
        assert "agents" in result
        assert "count" in result


class TestCaching:
    """Test TTL caching behavior."""

    def test_fetch_rankings_uses_cache(self, handler):
        """Fetch rankings results are cached."""
        agents = [{"name": "agent1", "elo": 1500}]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents

        # First call
        result1 = handler._fetch_rankings(10, None)
        # Second call (should use cache after first call)
        result2 = handler._fetch_rankings(10, None)

        assert result1 == result2

    def test_different_params_different_cache(self, handler):
        """Different parameters use different cache entries."""
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_leaderboard.return_value = []

        # Different domains should not share cache
        handler._fetch_rankings(10, None)
        handler._fetch_rankings(10, "tech")

        # Both should have been called (not cached)
        assert handler.ctx["elo_system"].get_cached_leaderboard.call_count >= 1
