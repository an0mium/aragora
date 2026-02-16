"""
Tests for agent recommendation and leaderboard handler.

Tests:
- Route handling (can_handle)
- GET /api/v1/agents/recommend (with/without domain)
- GET /api/v1/agents/leaderboard (with/without domain)
- Limit clamping
- ELO system unavailable
- Rate limiting
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.agents.recommendations import (
    AgentRecommendationHandler,
    _recommend_limiter,
)


@pytest.fixture
def handler():
    """Create handler with mocked ELO system."""
    ctx = {"storage": None, "elo_system": MagicMock()}
    h = AgentRecommendationHandler(ctx)
    return h


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP request handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


@pytest.fixture(autouse=True)
def clear_rate_limit():
    """Clear rate limiter state between tests."""
    _recommend_limiter._buckets.clear()


def _make_agent(name, elo=1500, wins=10, losses=5):
    """Create a mock agent rating."""
    agent = MagicMock()
    agent.name = name
    agent.agent_name = name
    agent.elo = elo
    agent.wins = wins
    agent.losses = losses
    agent.draws = 0
    agent.win_rate = wins / max(wins + losses, 1)
    agent.games_played = wins + losses
    agent.matches = wins + losses
    agent.calibration_score = 0.85
    agent.domain_elo = None
    return agent


# ===========================================================================
# Route Matching
# ===========================================================================


class TestRouteMatching:
    def test_can_handle_recommend(self, handler):
        assert handler.can_handle("/api/v1/agents/recommend") is True

    def test_can_handle_leaderboard(self, handler):
        assert handler.can_handle("/api/v1/agents/leaderboard") is True

    def test_cannot_handle_unknown(self, handler):
        assert handler.can_handle("/api/v1/agents/unknown") is False

    def test_cannot_handle_debates(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_agents_root(self, handler):
        assert handler.can_handle("/api/v1/agents") is False


# ===========================================================================
# Recommend Endpoint
# ===========================================================================


class TestRecommendEndpoint:
    def test_recommend_returns_agents(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600), _make_agent("gpt4", 1550)]
        handler.ctx["elo_system"].get_leaderboard.return_value = agents

        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert body["count"] == 2
        assert len(body["recommendations"]) == 2
        assert body["recommendations"][0]["name"] == "claude"

    def test_recommend_with_domain(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600)]
        handler.ctx["elo_system"].get_top_agents_for_domain.return_value = agents

        result = handler.handle(
            "/api/v1/agents/recommend", {"domain": "financial"}, mock_http_handler
        )
        assert result is not None
        body = result[0]
        assert body["domain"] == "financial"
        assert body["count"] == 1
        handler.ctx["elo_system"].get_top_agents_for_domain.assert_called_once_with(
            domain="financial", limit=5
        )

    def test_recommend_default_limit(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        handler.ctx["elo_system"].get_leaderboard.assert_called_once_with(limit=5)

    def test_recommend_custom_limit(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        handler.handle("/api/v1/agents/recommend", {"limit": "10"}, mock_http_handler)
        handler.ctx["elo_system"].get_leaderboard.assert_called_once_with(limit=10)

    def test_recommend_limit_clamped_to_max(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        handler.handle("/api/v1/agents/recommend", {"limit": "100"}, mock_http_handler)
        handler.ctx["elo_system"].get_leaderboard.assert_called_once_with(limit=20)

    def test_recommend_includes_cost_estimate(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600)]
        handler.ctx["elo_system"].get_leaderboard.return_value = agents

        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        body = result[0]
        assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.015

    def test_recommend_includes_calibration(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600)]
        handler.ctx["elo_system"].get_leaderboard.return_value = agents

        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        body = result[0]
        assert body["recommendations"][0]["calibration_score"] == 0.85

    def test_recommend_unknown_agent_cost_is_none(self, handler, mock_http_handler):
        agents = [_make_agent("custom-agent-xyz", 1500)]
        handler.ctx["elo_system"].get_leaderboard.return_value = agents

        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        body = result[0]
        assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] is None

    def test_recommend_invalid_domain(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/agents/recommend",
            {"domain": "'; DROP TABLE agents; --"},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 400

    def test_recommend_elo_unavailable(self, mock_http_handler):
        handler = AgentRecommendationHandler(ctx={"storage": None})
        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        assert result is not None
        assert result[1] == 503

    def test_recommend_handles_exception(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.side_effect = TypeError("db error")
        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        assert result is not None
        assert result[1] == 500


# ===========================================================================
# Leaderboard Endpoint
# ===========================================================================


class TestLeaderboardEndpoint:
    def test_leaderboard_returns_ranked_agents(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600), _make_agent("gpt4", 1550)]
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = agents
        handler.ctx["elo_system"].get_stats.return_value = {
            "total_agents": 10,
            "total_matches": 50,
            "avg_elo": 1500,
        }

        result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert body["count"] == 2
        assert body["leaderboard"][0]["rank"] == 1
        assert body["leaderboard"][1]["rank"] == 2

    def test_leaderboard_with_domain(self, handler, mock_http_handler):
        agents = [_make_agent("claude", 1600)]
        handler.ctx["elo_system"].get_leaderboard.return_value = agents
        handler.ctx["elo_system"].get_stats.return_value = {}

        result = handler.handle(
            "/api/v1/agents/leaderboard", {"domain": "legal"}, mock_http_handler
        )
        assert result is not None
        body = result[0]
        assert body["domain"] == "legal"
        handler.ctx["elo_system"].get_leaderboard.assert_called_once_with(limit=20, domain="legal")

    def test_leaderboard_default_limit(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}
        handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
        handler.ctx["elo_system"].get_cached_leaderboard.assert_called_once_with(limit=20)

    def test_leaderboard_limit_clamped(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {}
        handler.handle("/api/v1/agents/leaderboard", {"limit": "100"}, mock_http_handler)
        handler.ctx["elo_system"].get_cached_leaderboard.assert_called_once_with(limit=50)

    def test_leaderboard_includes_stats(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_cached_leaderboard.return_value = []
        handler.ctx["elo_system"].get_stats.return_value = {
            "total_agents": 42,
            "total_matches": 200,
            "avg_elo": 1520,
        }

        result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
        body = result[0]
        assert body["stats"]["total_agents"] == 42
        assert body["stats"]["total_matches"] == 200
        assert body["stats"]["mean_elo"] == 1520

    def test_leaderboard_elo_unavailable(self, mock_http_handler):
        handler = AgentRecommendationHandler(ctx={"storage": None})
        result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
        assert result is not None
        assert result[1] == 503

    def test_leaderboard_handles_exception(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_cached_leaderboard.side_effect = TypeError("fail")
        result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
        assert result is not None
        assert result[1] == 500


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_allows_normal_traffic(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        assert result is not None
        assert result[1] != 429

    def test_rate_limit_blocks_excessive_traffic(self, handler, mock_http_handler):
        handler.ctx["elo_system"].get_leaderboard.return_value = []
        # Fill up rate limiter
        for _ in range(35):
            _recommend_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
        assert result is not None
        assert result[1] == 429
