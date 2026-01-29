"""Tests for leaderboard view endpoint handler."""
import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler", "get_slack_handler", "get_slack_integration",
    "get_workspace_store", "resolve_workspace", "create_tracked_task",
    "_validate_slack_url", "SLACK_SIGNING_SECRET", "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL", "SLACK_ALLOWED_DOMAINS", "SignatureVerifierMixin",
    "CommandsMixin", "EventsMixin", "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class MockHandler:
    """Mock handler for testing leaderboard functions."""

    def __init__(
        self,
        elo_system: Any = None,
        nomic_dir: Path | None = None,
    ):
        self._elo_system = elo_system
        self._nomic_dir = nomic_dir

    def get_elo_system(self) -> Any:
        return self._elo_system

    def get_nomic_dir(self) -> Path | None:
        return self._nomic_dir


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    from aragora.server.handlers.agents.leaderboard import _leaderboard_limiter
    _leaderboard_limiter._requests.clear()
    yield


class TestLeaderboardViewHandlerRoutes:
    """Tests for LeaderboardViewHandler route configuration."""

    def test_routes_defined(self):
        """Test LeaderboardViewHandler has expected routes."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        routes = LeaderboardViewHandler.ROUTES

        assert "/api/leaderboard-view" in routes

    def test_can_handle_leaderboard_view(self):
        """Test can_handle returns True for leaderboard view."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        assert handler.can_handle("/api/leaderboard-view") is True
        assert handler.can_handle("/api/v1/leaderboard-view") is True

    def test_can_handle_non_leaderboard(self):
        """Test can_handle returns False for non-leaderboard routes."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False


class TestLeaderboardViewHandlerAuth:
    """Tests for LeaderboardViewHandler authentication."""

    @pytest.mark.asyncio
    async def test_requires_authentication(self):
        """Test leaderboard view requires authentication."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler
        from aragora.server.handlers.secure import UnauthorizedError

        handler = LeaderboardViewHandler()
        mock_http_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")
            result = await handler.handle("/api/leaderboard-view", {}, mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_checks_permission(self):
        """Test leaderboard view checks permission."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler
        from aragora.server.handlers.secure import ForbiddenError

        handler = LeaderboardViewHandler()
        mock_http_handler = MagicMock()
        mock_auth_context = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth, \
             patch.object(handler, "check_permission") as mock_check:
            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")
            result = await handler.handle("/api/leaderboard-view", {}, mock_http_handler)

        assert result.status_code == 403


class TestLeaderboardViewHandlerRateLimit:
    """Tests for LeaderboardViewHandler rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit returns 429."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()
        mock_http_handler = MagicMock()

        with patch("aragora.server.handlers.agents.leaderboard.get_client_ip", return_value="127.0.0.1"), \
             patch("aragora.server.handlers.agents.leaderboard._leaderboard_limiter.is_allowed", return_value=False):
            result = await handler.handle("/api/leaderboard-view", {}, mock_http_handler)

        assert result.status_code == 429


class TestGetLeaderboardView:
    """Tests for _get_leaderboard_view method."""

    def test_leaderboard_view_success(self, tmp_path):
        """Test leaderboard view returns all sections."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "_fetch_rankings", return_value={"agents": [], "count": 0}), \
             patch.object(handler, "_fetch_matches", return_value={"matches": [], "count": 0}), \
             patch.object(handler, "_fetch_reputations", return_value={"reputations": [], "count": 0}), \
             patch.object(handler, "_fetch_teams", return_value={"combinations": [], "count": 0}), \
             patch.object(handler, "_fetch_stats", return_value={"mean_elo": 1500, "total_agents": 0, "total_matches": 0, "median_elo": 1500, "rating_distribution": {}, "trending_up": [], "trending_down": []}), \
             patch.object(handler, "_fetch_introspection", return_value={"agents": {}, "count": 0}):
            result = handler._get_leaderboard_view(10, None, None)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "data" in body
        assert "rankings" in body["data"]
        assert "matches" in body["data"]
        assert "reputation" in body["data"]
        assert "teams" in body["data"]
        assert "stats" in body["data"]
        assert "introspection" in body["data"]

    def test_leaderboard_view_partial_failure(self, tmp_path):
        """Test leaderboard view handles partial failures."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "_fetch_rankings", side_effect=RuntimeError("Rankings failed")), \
             patch.object(handler, "_fetch_matches", return_value={"matches": [], "count": 0}), \
             patch.object(handler, "_fetch_reputations", return_value={"reputations": [], "count": 0}), \
             patch.object(handler, "_fetch_teams", return_value={"combinations": [], "count": 0}), \
             patch.object(handler, "_fetch_stats", return_value={"mean_elo": 1500, "total_agents": 0, "total_matches": 0, "median_elo": 1500, "rating_distribution": {}, "trending_up": [], "trending_down": []}), \
             patch.object(handler, "_fetch_introspection", return_value={"agents": {}, "count": 0}):
            result = handler._get_leaderboard_view(10, None, None)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["errors"]["partial_failure"] is True
        assert "rankings" in body["errors"]["failed_sections"]


class TestFetchRankings:
    """Tests for _fetch_rankings method."""

    def test_fetch_rankings_success(self):
        """Test fetch rankings returns agent data."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.return_value = [
            {"agent": "claude", "elo": 1600, "name": "claude"},
            {"agent": "gemini", "elo": 1550, "name": "gemini"},
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo), \
             patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._fetch_rankings(10, None)

        assert "agents" in result
        assert len(result["agents"]) == 2

    def test_fetch_rankings_no_elo(self):
        """Test fetch rankings returns empty when no ELO system."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._fetch_rankings(10, None)

        assert result["agents"] == []
        assert result["count"] == 0

    def test_fetch_rankings_with_domain(self):
        """Test fetch rankings with domain filter."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "claude", "elo": 1600, "name": "claude"}]

        with patch.object(handler, "get_elo_system", return_value=mock_elo), \
             patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._fetch_rankings(10, "coding")

        mock_elo.get_leaderboard.assert_called_once_with(limit=10, domain="coding")


class TestFetchMatches:
    """Tests for _fetch_matches method."""

    def test_fetch_matches_success(self):
        """Test fetch matches returns recent matches."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.return_value = [
            {"winner": "claude", "loser": "gemini", "timestamp": "2024-01-01"},
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._fetch_matches(10, None)

        assert "matches" in result
        assert len(result["matches"]) == 1

    def test_fetch_matches_no_elo(self):
        """Test fetch matches returns empty when no ELO system."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._fetch_matches(10, None)

        assert result["matches"] == []


class TestFetchReputations:
    """Tests for _fetch_reputations method."""

    def test_fetch_reputations_no_nomic_dir(self):
        """Test fetch reputations returns empty when no nomic dir."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._fetch_reputations()

        assert result["reputations"] == []


class TestFetchTeams:
    """Tests for _fetch_teams method."""

    def test_fetch_teams_success(self):
        """Test fetch teams returns team combinations."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = [
            {"agents": ["claude", "gemini"], "win_rate": 0.7},
        ]

        with patch.object(handler, "get_elo_system", return_value=MagicMock()), \
             patch("aragora.server.handlers.agents.leaderboard.AgentSelector", return_value=mock_selector):
            result = handler._fetch_teams(3, 10)

        assert "combinations" in result


class TestFetchStats:
    """Tests for _fetch_stats method."""

    def test_fetch_stats_success(self):
        """Test fetch stats returns statistics."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_elo = MagicMock()
        mock_elo.get_stats.return_value = {
            "avg_elo": 1520,
            "median_elo": 1510,
            "total_agents": 10,
            "total_matches": 50,
            "rating_distribution": {},
            "trending_up": [],
            "trending_down": [],
        }

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._fetch_stats()

        assert result["mean_elo"] == 1520
        assert result["total_agents"] == 10

    def test_fetch_stats_no_elo(self):
        """Test fetch stats returns defaults when no ELO system."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._fetch_stats()

        assert result["mean_elo"] == 1500
        assert result["total_agents"] == 0


class TestFetchIntrospection:
    """Tests for _fetch_introspection method."""

    def test_fetch_introspection_no_nomic_dir(self):
        """Test fetch introspection with default agents when no nomic dir."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent": "claude", "traits": []}

        with patch.object(handler, "get_nomic_dir", return_value=None), \
             patch("aragora.server.handlers.agents.leaderboard.get_agent_introspection", return_value=mock_snapshot):
            result = handler._fetch_introspection()

        assert "agents" in result


class TestSafeFetchSection:
    """Tests for _safe_fetch_section method."""

    def test_safe_fetch_success(self):
        """Test safe fetch stores result on success."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        data: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        handler._safe_fetch_section(
            data, errors, "test_key",
            lambda: {"value": 1},
            {"value": 0}
        )

        assert data["test_key"]["value"] == 1
        assert "test_key" not in errors

    def test_safe_fetch_failure(self):
        """Test safe fetch stores fallback on failure."""
        from aragora.server.handlers.agents.leaderboard import LeaderboardViewHandler

        handler = LeaderboardViewHandler()

        data: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        def failing_fetch():
            raise RuntimeError("Fetch failed")

        handler._safe_fetch_section(
            data, errors, "test_key",
            failing_fetch,
            {"value": 0}
        )

        assert data["test_key"]["value"] == 0
        assert "test_key" in errors
