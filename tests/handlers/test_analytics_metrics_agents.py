"""Comprehensive tests for AgentAnalyticsMixin endpoints.

Tests the four agent analytics mixin methods from
aragora/server/handlers/_analytics_metrics_agents.py:

- _get_agents_leaderboard: ELO rankings with win rates
- _get_agent_performance: Individual agent stats, ELO history, recent matches
- _get_agents_comparison: Compare multiple agents head-to-head
- _get_agents_trends: Agent performance trends over time

Also tests routing via the AnalyticsMetricsHandler.handle() async method.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics import AnalyticsMetricsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Minimal mock HTTP handler for handle() routing tests."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 54321)
        self.headers: dict[str, str] = {"User-Agent": "test"}
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock agent factory helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_name: str = "claude",
    elo: float = 1500.0,
    wins: int = 50,
    losses: int = 20,
    draws: int = 10,
    win_rate: float = 0.625,
    games_played: int = 80,
    debates_count: int = 80,
    domain_elos: dict[str, float] | None = None,
    calibration_score: float | None = None,
    calibration_accuracy: float | None = None,
) -> MagicMock:
    """Build a mock agent rating object matching what EloSystem.get_rating returns."""
    agent = MagicMock()
    agent.agent_name = agent_name
    agent.elo = elo
    agent.wins = wins
    agent.losses = losses
    agent.draws = draws
    agent.win_rate = win_rate
    agent.games_played = games_played
    agent.debates_count = debates_count
    agent.domain_elos = domain_elos or {}

    if calibration_score is not None:
        agent.calibration_score = calibration_score
    else:
        # Remove attribute so hasattr returns False
        del agent.calibration_score

    if calibration_accuracy is not None:
        agent.calibration_accuracy = calibration_accuracy
    else:
        del agent.calibration_accuracy

    return agent


def _make_elo_system(
    agents: list[MagicMock] | None = None,
    leaderboard: list[MagicMock] | None = None,
    elo_history: list[tuple[str, float]] | None = None,
    recent_matches: list[dict] | None = None,
    head_to_head: dict | None = None,
) -> MagicMock:
    """Build a mock ELO system."""
    elo = MagicMock()
    lb = leaderboard if leaderboard is not None else (agents or [])
    elo.get_leaderboard.return_value = lb
    elo.list_agents.return_value = agents or lb
    elo.get_elo_history.return_value = elo_history or []
    elo.get_recent_matches.return_value = recent_matches or []
    elo.get_head_to_head.return_value = head_to_head or {
        "a_wins": 0,
        "b_wins": 0,
        "draws": 0,
        "total": 0,
    }

    def _get_rating(agent_id):
        for a in agents or lb:
            if a.agent_name == agent_id:
                return a
        raise ValueError(f"Agent not found: {agent_id}")

    elo.get_rating.side_effect = _get_rating
    return elo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AnalyticsMetricsHandler with empty context."""
    return AnalyticsMetricsHandler({})


@pytest.fixture
def http_handler():
    """Mock HTTP handler for async handle() tests."""
    return MockHTTPHandler()


@pytest.fixture
def three_agents():
    """Three mock agents for leaderboard and comparison tests."""
    return [
        _make_agent(
            "claude",
            elo=1650,
            wins=120,
            losses=30,
            draws=10,
            win_rate=0.75,
            games_played=160,
            calibration_score=0.85,
        ),
        _make_agent(
            "gpt-4",
            elo=1580,
            wins=90,
            losses=50,
            draws=20,
            win_rate=0.5625,
            games_played=160,
            calibration_score=0.78,
        ),
        _make_agent(
            "gemini", elo=1520, wins=70, losses=60, draws=30, win_rate=0.4375, games_played=160
        ),
    ]


@pytest.fixture
def mock_elo(three_agents):
    """Mock ELO system with three agents."""
    return _make_elo_system(agents=three_agents)


# ============================================================================
# _get_agents_leaderboard
# ============================================================================


class TestAgentsLeaderboard:
    """Tests for _get_agents_leaderboard."""

    def test_success_with_agents(self, handler, mock_elo, three_agents):
        """Leaderboard returns ranked agents with correct fields."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert _status(result) == 200
        assert len(body["leaderboard"]) == 3
        assert body["total_agents"] == 3
        assert "generated_at" in body

        # Verify first agent
        first = body["leaderboard"][0]
        assert first["rank"] == 1
        assert first["agent_name"] == "claude"
        assert first["elo"] == 1650
        assert first["wins"] == 120
        assert first["losses"] == 30
        assert first["draws"] == 10
        assert first["win_rate"] == 75.0
        assert first["games_played"] == 160

    def test_calibration_score_included_when_present(self, handler, mock_elo):
        """Calibration score is included for agents that have it."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        # claude has calibration_score
        assert body["leaderboard"][0]["calibration_score"] == 0.85
        # gpt-4 has calibration_score
        assert body["leaderboard"][1]["calibration_score"] == 0.78
        # gemini does not have calibration_score
        assert "calibration_score" not in body["leaderboard"][2]

    def test_no_elo_system_returns_empty(self, handler):
        """When ELO system is None, return empty leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert _status(result) == 200
        assert body["leaderboard"] == []
        assert body["total_agents"] == 0
        assert "generated_at" in body

    def test_limit_parameter(self, handler, mock_elo):
        """Limit parameter is passed to get_leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_agents_leaderboard({"limit": "5"})

        mock_elo.get_leaderboard.assert_called_with(limit=5, domain=None)

    def test_limit_default_is_20(self, handler, mock_elo):
        """Default limit is 20."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_agents_leaderboard({})

        mock_elo.get_leaderboard.assert_called_with(limit=20, domain=None)

    def test_limit_max_is_100(self, handler, mock_elo):
        """Limit is capped at 100."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_agents_leaderboard({"limit": "500"})

        mock_elo.get_leaderboard.assert_called_with(limit=100, domain=None)

    def test_domain_filter(self, handler, mock_elo):
        """Domain parameter is passed to get_leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_agents_leaderboard({"domain": "security"})

        mock_elo.get_leaderboard.assert_called_with(limit=20, domain="security")

    def test_rank_numbering_sequential(self, handler, mock_elo):
        """Ranks are sequential starting at 1."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        ranks = [a["rank"] for a in body["leaderboard"]]
        assert ranks == [1, 2, 3]

    def test_elo_rounded_to_zero_decimals(self, handler):
        """ELO values are rounded to 0 decimal places."""
        agent = _make_agent("test", elo=1523.456)
        elo_sys = _make_elo_system(agents=[agent])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert body["leaderboard"][0]["elo"] == 1523.0

    def test_win_rate_percentage(self, handler):
        """Win rate is converted to percentage (0-100)."""
        agent = _make_agent("test", win_rate=0.333)
        elo_sys = _make_elo_system(agents=[agent])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert body["leaderboard"][0]["win_rate"] == 33.3

    def test_domain_included_in_response_when_specified(self, handler, mock_elo):
        """Response includes domain field when domain filter is used."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_agents_leaderboard({"domain": "security"})

        body = _body(result)
        assert body["domain"] == "security"

    def test_domain_none_when_not_specified(self, handler, mock_elo):
        """Response includes domain=None when no domain filter."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert body["domain"] is None

    def test_total_agents_from_list_agents(self, handler):
        """total_agents comes from list_agents() not from leaderboard length."""
        agents = [_make_agent(f"a{i}") for i in range(10)]
        elo_sys = _make_elo_system(agents=agents)
        # Leaderboard returns only top 3
        elo_sys.get_leaderboard.return_value = agents[:3]
        # But list_agents returns all 10
        elo_sys.list_agents.return_value = agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_leaderboard({"limit": "3"})

        body = _body(result)
        assert len(body["leaderboard"]) == 3
        assert body["total_agents"] == 10

    def test_empty_leaderboard(self, handler):
        """Empty leaderboard with active ELO system returns empty list."""
        elo_sys = _make_elo_system(agents=[])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert _status(result) == 200
        assert body["leaderboard"] == []
        assert body["total_agents"] == 0

    def test_invalid_limit_uses_default(self, handler, mock_elo):
        """Non-numeric limit uses default of 20."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_agents_leaderboard({"limit": "invalid"})

        mock_elo.get_leaderboard.assert_called_with(limit=20, domain=None)


# ============================================================================
# _get_agent_performance
# ============================================================================


class TestAgentPerformance:
    """Tests for _get_agent_performance."""

    def test_success_basic(self, handler, three_agents):
        """Returns agent performance with expected fields."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert _status(result) == 200
        assert body["agent_id"] == "claude"
        assert body["agent_name"] == "claude"
        assert body["elo"] == 1650
        assert body["wins"] == 120
        assert body["losses"] == 30
        assert body["draws"] == 10
        assert body["win_rate"] == 75.0
        assert body["games_played"] == 160
        assert body["time_range"] == "30d"
        assert body["rank"] == 1
        assert "generated_at" in body

    def test_no_elo_system_returns_503(self, handler):
        """When ELO system unavailable, returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agent_performance("claude", {})

        assert _status(result) == 503

    def test_agent_not_found_returns_404(self, handler, three_agents):
        """Unknown agent returns 404."""
        elo_sys = _make_elo_system(agents=three_agents)
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("nonexistent", {})

        assert _status(result) == 404

    def test_invalid_time_range_defaults_to_30d(self, handler, three_agents):
        """Invalid time_range silently defaults to 30d."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {"time_range": "invalid"})

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_valid_time_ranges(self, handler, three_agents):
        """All valid time ranges are accepted."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        for tr in ("7d", "14d", "30d", "90d", "180d", "365d", "all"):
            with patch.object(handler, "get_elo_system", return_value=elo_sys):
                result = handler._get_agent_performance("claude", {"time_range": tr})
            assert _status(result) == 200
            assert _body(result)["time_range"] == tr

    def test_elo_change_calculation(self, handler, three_agents):
        """ELO change is computed from current elo minus last history entry."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        # History: latest is at index 0, oldest at end
        elo_sys.get_elo_history.return_value = [
            ("2026-01-20T00:00:00Z", 1640.0),
            ("2026-01-15T00:00:00Z", 1620.0),
            ("2026-01-10T00:00:00Z", 1600.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        # elo_change = current elo (1650) - last history entry elo (1600)
        assert body["elo_change"] == 50.0

    def test_elo_change_zero_with_single_history_entry(self, handler, three_agents):
        """ELO change is 0 when only one history entry."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        elo_sys.get_elo_history.return_value = [("2026-01-20T00:00:00Z", 1650.0)]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert body["elo_change"] == 0.0

    def test_elo_change_zero_with_empty_history(self, handler, three_agents):
        """ELO change is 0 when no history."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert body["elo_change"] == 0.0

    def test_rank_from_leaderboard(self, handler, three_agents):
        """Rank is determined from position in leaderboard."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("gpt-4", {})

        body = _body(result)
        assert body["rank"] == 2

    def test_rank_none_when_not_in_leaderboard(self, handler):
        """Rank is None when agent is not in leaderboard (e.g., too low)."""
        agent = _make_agent("special", elo=1200)
        leaderboard_agents = [_make_agent("top1", elo=1600), _make_agent("top2", elo=1500)]
        elo_sys = _make_elo_system(agents=[agent] + leaderboard_agents)
        elo_sys.get_leaderboard.return_value = leaderboard_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("special", {})

        body = _body(result)
        assert body["rank"] is None

    def test_recent_matches_filtered_for_agent(self, handler, three_agents):
        """Only matches involving the queried agent are returned."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        elo_sys.get_recent_matches.return_value = [
            {"id": "m1", "participants": ["claude", "gpt-4"]},
            {"id": "m2", "participants": ["gpt-4", "gemini"]},
            {"id": "m3", "participants": ["claude", "gemini"]},
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert len(body["recent_matches"]) == 2
        match_ids = {m["id"] for m in body["recent_matches"]}
        assert match_ids == {"m1", "m3"}

    def test_recent_matches_limited_to_5(self, handler, three_agents):
        """At most 5 recent matches are included."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        elo_sys.get_recent_matches.return_value = [
            {"id": f"m{i}", "participants": ["claude", "gpt-4"]} for i in range(10)
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert len(body["recent_matches"]) == 5

    def test_elo_history_in_response(self, handler, three_agents):
        """ELO history is formatted with timestamp and rounded elo."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        elo_sys.get_elo_history.return_value = [
            ("2026-01-20T00:00:00Z", 1645.67),
            ("2026-01-15T00:00:00Z", 1630.33),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert len(body["elo_history"]) == 2
        assert body["elo_history"][0]["timestamp"] == "2026-01-20T00:00:00Z"
        assert body["elo_history"][0]["elo"] == 1646.0
        assert body["elo_history"][1]["elo"] == 1630.0

    def test_domain_performance_included(self, handler):
        """Domain performance is included when agent has domain_elos."""
        agent = _make_agent(
            "claude", elo=1650, domain_elos={"security": 1700.3, "performance": 1620.7}
        )
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert "domain_performance" in body
        assert body["domain_performance"]["security"]["elo"] == 1700.0
        assert body["domain_performance"]["performance"]["elo"] == 1621.0

    def test_no_domain_performance_when_empty(self, handler):
        """domain_performance not present when agent has no domain_elos."""
        agent = _make_agent("claude", domain_elos={})
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert "domain_performance" not in body

    def test_calibration_score_included(self, handler):
        """Calibration score included when agent has it."""
        agent = _make_agent("claude", calibration_score=0.857)
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert body["calibration_score"] == 0.86

    def test_calibration_accuracy_included(self, handler):
        """Calibration accuracy included when agent has it."""
        agent = _make_agent("claude", calibration_score=0.85, calibration_accuracy=0.923)
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert body["calibration_accuracy"] == 0.92

    def test_no_calibration_when_absent(self, handler):
        """Calibration fields not present when agent lacks them."""
        agent = _make_agent("claude")
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert "calibration_score" not in body
        assert "calibration_accuracy" not in body

    def test_debates_count_in_response(self, handler):
        """debates_count field is present in the response."""
        agent = _make_agent("claude", debates_count=200)
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("claude", {})

        body = _body(result)
        assert body["debates_count"] == 200

    def test_get_rating_raises_key_error(self, handler):
        """KeyError from get_rating returns 404."""
        elo_sys = MagicMock()
        elo_sys.get_rating.side_effect = KeyError("not found")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("unknown", {})

        assert _status(result) == 404


# ============================================================================
# _get_agents_comparison
# ============================================================================


class TestAgentsComparison:
    """Tests for _get_agents_comparison."""

    def test_success_two_agents(self, handler, three_agents):
        """Compare two agents with head-to-head stats."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.return_value = {
            "a_wins": 15,
            "b_wins": 10,
            "draws": 5,
            "total": 30,
        }
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        body = _body(result)
        assert _status(result) == 200
        assert body["agents"] == ["claude", "gpt-4"]
        assert len(body["comparison"]) == 2
        assert "head_to_head" in body
        assert "generated_at" in body

        # Check comparison data
        claude_data = body["comparison"][0]
        assert claude_data["agent_name"] == "claude"
        assert claude_data["elo"] == 1650
        assert claude_data["win_rate"] == 75.0

    def test_head_to_head_stats(self, handler, three_agents):
        """Head-to-head stats are keyed correctly."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.return_value = {"a_wins": 8, "b_wins": 5, "draws": 2, "total": 15}
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        body = _body(result)
        h2h_key = "claude_vs_gpt-4"
        assert h2h_key in body["head_to_head"]
        h2h = body["head_to_head"][h2h_key]
        assert h2h["claude_wins"] == 8
        assert h2h["gpt-4_wins"] == 5
        assert h2h["draws"] == 2
        assert h2h["total_matches"] == 15

    def test_three_agents_comparison(self, handler, three_agents):
        """Three agents produce 3 head-to-head pairs."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.return_value = {"a_wins": 5, "b_wins": 3, "draws": 1, "total": 9}
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4,gemini"})

        body = _body(result)
        assert len(body["comparison"]) == 3
        # 3 agents: C(3,2) = 3 pairs
        assert len(body["head_to_head"]) == 3
        expected_keys = {"claude_vs_gpt-4", "claude_vs_gemini", "gpt-4_vs_gemini"}
        assert set(body["head_to_head"].keys()) == expected_keys

    def test_missing_agents_param_returns_400(self, handler):
        """No agents parameter returns 400."""
        with patch.object(handler, "get_elo_system", return_value=MagicMock()):
            result = handler._get_agents_comparison({})

        assert _status(result) == 400
        body = _body(result)
        assert "agents parameter is required" in body.get("error", body.get("message", ""))

    def test_empty_agents_param_returns_400(self, handler):
        """Empty agents parameter returns 400."""
        with patch.object(handler, "get_elo_system", return_value=MagicMock()):
            result = handler._get_agents_comparison({"agents": ""})

        assert _status(result) == 400

    def test_single_agent_returns_400(self, handler):
        """Single agent returns 400 (need at least 2)."""
        with patch.object(handler, "get_elo_system", return_value=MagicMock()):
            result = handler._get_agents_comparison({"agents": "claude"})

        assert _status(result) == 400
        body = _body(result)
        assert "At least 2 agents" in body.get("error", body.get("message", ""))

    def test_more_than_10_agents_returns_400(self, handler):
        """More than 10 agents returns 400."""
        agents_str = ",".join(f"agent-{i}" for i in range(11))
        with patch.object(handler, "get_elo_system", return_value=MagicMock()):
            result = handler._get_agents_comparison({"agents": agents_str})

        assert _status(result) == 400
        body = _body(result)
        assert "Maximum 10 agents" in body.get("error", body.get("message", ""))

    def test_exactly_10_agents_accepted(self, handler):
        """Exactly 10 agents is accepted."""
        agents = [_make_agent(f"agent-{i}") for i in range(10)]
        elo_sys = _make_elo_system(agents=agents)
        agents_str = ",".join(f"agent-{i}" for i in range(10))
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": agents_str})

        assert _status(result) == 200

    def test_no_elo_system_returns_503(self, handler):
        """No ELO system returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        assert _status(result) == 503

    def test_agent_not_found_shows_error(self, handler, three_agents):
        """Unknown agent gets error entry in comparison."""
        elo_sys = _make_elo_system(agents=three_agents)
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,nonexistent"})

        body = _body(result)
        assert _status(result) == 200
        # claude succeeds
        assert body["comparison"][0]["agent_name"] == "claude"
        assert "error" not in body["comparison"][0]
        # nonexistent fails
        assert body["comparison"][1]["agent_name"] == "nonexistent"
        assert body["comparison"][1]["error"] == "Agent not found"

    def test_head_to_head_error_silenced(self, handler, three_agents):
        """Errors in head-to-head computation are silently logged."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.side_effect = RuntimeError("h2h error")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        body = _body(result)
        assert _status(result) == 200
        assert body["head_to_head"] == {}

    def test_whitespace_in_agent_names_trimmed(self, handler, three_agents):
        """Agent names are trimmed of whitespace."""
        elo_sys = _make_elo_system(agents=three_agents)
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": " claude , gpt-4 "})

        body = _body(result)
        assert body["agents"] == ["claude", "gpt-4"]

    def test_empty_entries_filtered(self, handler, three_agents):
        """Empty entries from trailing comma are filtered out."""
        elo_sys = _make_elo_system(agents=three_agents)
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4,"})

        body = _body(result)
        assert body["agents"] == ["claude", "gpt-4"]

    def test_calibration_score_in_comparison(self, handler, three_agents):
        """Calibration score included in comparison when available."""
        elo_sys = _make_elo_system(agents=three_agents)
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gemini"})

        body = _body(result)
        # claude has calibration_score
        assert body["comparison"][0]["calibration_score"] == 0.85
        # gemini does not
        assert "calibration_score" not in body["comparison"][1]

    def test_head_to_head_type_error_silenced(self, handler, three_agents):
        """TypeError in head-to-head is silently handled."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.side_effect = TypeError("bad type")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        assert _status(result) == 200
        assert _body(result)["head_to_head"] == {}

    def test_head_to_head_attribute_error_silenced(self, handler, three_agents):
        """AttributeError in head-to-head is silently handled."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_head_to_head.side_effect = AttributeError("no attr")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        assert _status(result) == 200
        assert _body(result)["head_to_head"] == {}


# ============================================================================
# _get_agents_trends
# ============================================================================


class TestAgentsTrends:
    """Tests for _get_agents_trends."""

    def test_success_with_specified_agents(self, handler, three_agents):
        """Returns trends for specified agents."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1650.0),
            ((now - timedelta(days=1)).isoformat(), 1640.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude,gpt-4"})

        body = _body(result)
        assert _status(result) == 200
        assert body["agents"] == ["claude", "gpt-4"]
        assert body["time_range"] == "30d"
        assert body["granularity"] == "daily"
        assert "claude" in body["trends"]
        assert "gpt-4" in body["trends"]
        assert "generated_at" in body

    def test_default_agents_top_5(self, handler):
        """Without agents param, defaults to top 5 from leaderboard."""
        agents = [_make_agent(f"agent-{i}") for i in range(7)]
        elo_sys = _make_elo_system(agents=agents)
        elo_sys.get_leaderboard.return_value = agents[:5]
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({})

        body = _body(result)
        assert len(body["agents"]) == 5
        elo_sys.get_leaderboard.assert_called_with(limit=5)

    def test_no_elo_system_returns_503(self, handler):
        """No ELO system returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agents_trends({"agents": "claude"})

        assert _status(result) == 503

    def test_invalid_time_range_defaults_to_30d(self, handler, three_agents):
        """Invalid time_range defaults to 30d."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "time_range": "bogus"})

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_invalid_granularity_defaults_to_daily(self, handler, three_agents):
        """Invalid granularity defaults to daily."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "hourly"})

        body = _body(result)
        assert body["granularity"] == "daily"

    def test_daily_granularity(self, handler, three_agents):
        """Daily granularity uses YYYY-MM-DD format."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1650.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "daily"})

        body = _body(result)
        assert body["granularity"] == "daily"
        if body["trends"]["claude"]:
            period = body["trends"]["claude"][0]["period"]
            assert len(period) == 10  # YYYY-MM-DD

    def test_weekly_granularity(self, handler, three_agents):
        """Weekly granularity uses YYYY-Www format."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1650.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "weekly"})

        body = _body(result)
        assert body["granularity"] == "weekly"
        if body["trends"]["claude"]:
            period = body["trends"]["claude"][0]["period"]
            assert "-W" in period

    def test_monthly_granularity(self, handler, three_agents):
        """Monthly granularity uses YYYY-MM format."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1650.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "monthly"})

        body = _body(result)
        assert body["granularity"] == "monthly"
        if body["trends"]["claude"]:
            period = body["trends"]["claude"][0]["period"]
            assert len(period) == 7  # YYYY-MM

    def test_trends_limited_to_10_agents(self, handler):
        """At most 10 agents are processed for trends."""
        agents = [_make_agent(f"agent-{i}") for i in range(15)]
        elo_sys = _make_elo_system(agents=agents)
        elo_sys.get_elo_history.return_value = []
        agents_str = ",".join(f"agent-{i}" for i in range(15))
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": agents_str})

        body = _body(result)
        # trends should have at most 10 entries
        assert len(body["trends"]) <= 10

    def test_elo_history_with_datetime_objects(self, handler, three_agents):
        """ELO history with datetime objects (not strings) works."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now, 1650.0),
            (now - timedelta(days=1), 1640.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        assert _status(result) == 200
        assert len(body["trends"]["claude"]) >= 1

    def test_period_grouping_takes_latest_elo(self, handler, three_agents):
        """When multiple entries in same period, latest ELO wins."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        # Two entries on the same day
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1660.0),
            ((now - timedelta(hours=2)).isoformat(), 1640.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "daily"})

        body = _body(result)
        # Should have one period with the latest elo
        assert len(body["trends"]["claude"]) == 1
        assert body["trends"]["claude"][0]["elo"] == 1660.0

    def test_trends_sorted_by_period(self, handler, three_agents):
        """Trend data points are sorted by period."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            ((now - timedelta(days=5)).isoformat(), 1620.0),
            ((now - timedelta(days=2)).isoformat(), 1640.0),
            (now.isoformat(), 1660.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "daily"})

        body = _body(result)
        periods = [dp["period"] for dp in body["trends"]["claude"]]
        assert periods == sorted(periods)

    def test_agent_error_returns_empty_trend(self, handler, three_agents):
        """Error getting agent trends returns empty list for that agent."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.side_effect = RuntimeError("ELO history unavailable")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        assert _status(result) == 200
        assert body["trends"]["claude"] == []

    def test_agent_value_error_returns_empty_trend(self, handler, three_agents):
        """ValueError getting agent history returns empty list."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.side_effect = ValueError("agent not found")
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        assert body["trends"]["claude"] == []

    def test_invalid_timestamp_in_history_skipped(self, handler, three_agents):
        """Entries with invalid timestamps are skipped."""
        elo_sys = _make_elo_system(agents=three_agents)
        now = datetime.now(timezone.utc)
        elo_sys.get_elo_history.return_value = [
            ("not-a-timestamp", 1650.0),
            (now.isoformat(), 1660.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        # Only one valid entry should be present
        assert len(body["trends"]["claude"]) == 1

    def test_z_suffix_timestamp_parsed(self, handler, three_agents):
        """ISO timestamps with Z suffix are parsed correctly."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            ("2026-02-01T12:00:00Z", 1650.0),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude", "granularity": "daily"})

        body = _body(result)
        assert len(body["trends"]["claude"]) == 1
        assert body["trends"]["claude"][0]["period"] == "2026-02-01"

    def test_all_valid_time_ranges(self, handler, three_agents):
        """All valid time ranges return 200."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        for tr in ("7d", "14d", "30d", "90d", "180d", "365d", "all"):
            with patch.object(handler, "get_elo_system", return_value=elo_sys):
                result = handler._get_agents_trends({"agents": "claude", "time_range": tr})
            assert _status(result) == 200
            assert _body(result)["time_range"] == tr

    def test_all_valid_granularities(self, handler, three_agents):
        """All valid granularities return 200."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        for g in ("daily", "weekly", "monthly"):
            with patch.object(handler, "get_elo_system", return_value=elo_sys):
                result = handler._get_agents_trends({"agents": "claude", "granularity": g})
            assert _status(result) == 200
            assert _body(result)["granularity"] == g

    def test_empty_history_returns_empty_trends(self, handler, three_agents):
        """Empty ELO history returns empty trends list."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        assert body["trends"]["claude"] == []

    def test_elo_rounded_in_trends(self, handler, three_agents):
        """ELO values in trends are rounded."""
        now = datetime.now(timezone.utc)
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = [
            (now.isoformat(), 1647.89),
        ]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "claude"})

        body = _body(result)
        assert body["trends"]["claude"][0]["elo"] == 1648.0


# ============================================================================
# Async handle() routing tests for agent endpoints
# ============================================================================


class TestAgentHandleRouting:
    """Tests for routing agent endpoints through the async handle() method."""

    @pytest.mark.asyncio
    async def test_route_agents_leaderboard(self, handler, mock_elo, http_handler):
        """handle() routes /api/v1/analytics/agents/leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/analytics/agents/leaderboard", {}, http_handler)

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "leaderboard" in body

    @pytest.mark.asyncio
    async def test_route_agents_leaderboard_unversioned(self, handler, mock_elo, http_handler):
        """handle() routes unversioned /api/analytics/agents/leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/analytics/agents/leaderboard", {}, http_handler)

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_agents_comparison(self, handler, mock_elo, http_handler):
        """handle() routes /api/v1/analytics/agents/comparison."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/comparison",
                {"agents": "claude,gpt-4"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "comparison" in body

    @pytest.mark.asyncio
    async def test_route_agents_trends(self, handler, mock_elo, http_handler):
        """handle() routes /api/v1/analytics/agents/trends."""
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/trends",
                {"agents": "claude"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "trends" in body

    @pytest.mark.asyncio
    async def test_route_agent_performance(self, handler, mock_elo, http_handler):
        """handle() routes /api/v1/analytics/agents/{agent_id}/performance."""
        mock_elo.get_leaderboard.return_value = list(mock_elo.list_agents.return_value)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "claude"

    @pytest.mark.asyncio
    async def test_route_agent_performance_with_dashes(self, handler, http_handler):
        """Agent IDs with dashes are matched by the regex."""
        agent = _make_agent("gpt-4")
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = await handler.handle(
                "/api/v1/analytics/agents/gpt-4/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_route_agent_performance_with_underscores(self, handler, http_handler):
        """Agent IDs with underscores are matched."""
        agent = _make_agent("my_agent_1")
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = await handler.handle(
                "/api/v1/analytics/agents/my_agent_1/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "my_agent_1"


# ============================================================================
# can_handle() routing tests for agent endpoints
# ============================================================================


class TestAgentCanHandle:
    """Tests for can_handle() with agent analytics routes."""

    def test_leaderboard_routes(self, handler):
        """Leaderboard routes are recognized."""
        assert handler.can_handle("/api/analytics/agents/leaderboard")
        assert handler.can_handle("/api/v1/analytics/agents/leaderboard")

    def test_comparison_routes(self, handler):
        """Comparison routes are recognized."""
        assert handler.can_handle("/api/analytics/agents/comparison")
        assert handler.can_handle("/api/v1/analytics/agents/comparison")

    def test_trends_routes(self, handler):
        """Trends routes are recognized."""
        assert handler.can_handle("/api/analytics/agents/trends")
        assert handler.can_handle("/api/v1/analytics/agents/trends")

    def test_agent_performance_pattern(self, handler):
        """Agent performance pattern is recognized."""
        assert handler.can_handle("/api/analytics/agents/claude/performance")
        assert handler.can_handle("/api/v1/analytics/agents/claude/performance")
        assert handler.can_handle("/api/analytics/agents/gpt-4/performance")
        assert handler.can_handle("/api/analytics/agents/my_agent_1/performance")

    def test_invalid_agent_performance_path(self, handler):
        """Invalid agent performance paths are not matched."""
        assert not handler.can_handle("/api/analytics/agents/performance")
        assert not handler.can_handle("/api/analytics/agents//performance")

    def test_unknown_agent_route(self, handler):
        """Unknown agent analytics route not handled."""
        assert not handler.can_handle("/api/analytics/agents/unknown-endpoint")
        # But agent-like patterns with /performance are matched
        assert handler.can_handle("/api/analytics/agents/unknown-endpoint/performance")


# ============================================================================
# Edge cases
# ============================================================================


class TestAgentEdgeCases:
    """Edge case tests for agent analytics."""

    def test_leaderboard_single_agent(self, handler):
        """Leaderboard works with a single agent."""
        agent = _make_agent("only-one", elo=1500)
        elo_sys = _make_elo_system(agents=[agent])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_leaderboard({})

        body = _body(result)
        assert len(body["leaderboard"]) == 1
        assert body["leaderboard"][0]["rank"] == 1
        assert body["total_agents"] == 1

    def test_performance_with_all_optional_fields(self, handler):
        """Performance includes all optional fields when present."""
        agent = _make_agent(
            "complete",
            elo=1700,
            domain_elos={"security": 1750.5, "dev": 1680.3},
            calibration_score=0.91,
            calibration_accuracy=0.88,
        )
        elo_sys = _make_elo_system(agents=[agent])
        elo_sys.get_leaderboard.return_value = [agent]
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agent_performance("complete", {})

        body = _body(result)
        assert body["calibration_score"] == 0.91
        assert body["calibration_accuracy"] == 0.88
        assert "domain_performance" in body
        assert body["domain_performance"]["security"]["elo"] == 1750.0
        assert body["domain_performance"]["dev"]["elo"] == 1680.0

    def test_comparison_all_agents_not_found(self, handler):
        """Comparison where all agents are not found still returns 200."""
        elo_sys = _make_elo_system(agents=[])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_comparison({"agents": "missing-a,missing-b"})

        body = _body(result)
        assert _status(result) == 200
        assert all("error" in c for c in body["comparison"])

    def test_trends_multiple_agents_mixed_success(self, handler):
        """Trends with some agents failing and some succeeding."""
        now = datetime.now(timezone.utc)
        agent_ok = _make_agent("ok-agent")
        agent_fail = _make_agent("fail-agent")
        elo_sys = _make_elo_system(agents=[agent_ok, agent_fail])

        call_count = [0]

        def _mock_history(agent_name, limit=100):
            call_count[0] += 1
            if agent_name == "fail-agent":
                raise ValueError("history not available")
            return [(now.isoformat(), 1500.0)]

        elo_sys.get_elo_history.side_effect = _mock_history
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = handler._get_agents_trends({"agents": "ok-agent,fail-agent"})

        body = _body(result)
        assert _status(result) == 200
        assert len(body["trends"]["ok-agent"]) == 1
        assert body["trends"]["fail-agent"] == []

    def test_generated_at_always_present(self, handler, mock_elo):
        """All responses include generated_at timestamp."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            lb = handler._get_agents_leaderboard({})

        mock_elo.get_leaderboard.return_value = list(mock_elo.list_agents.return_value)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            perf = handler._get_agent_performance("claude", {})

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            comp = handler._get_agents_comparison({"agents": "claude,gpt-4"})

        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            trends = handler._get_agents_trends({"agents": "claude"})

        for result in (lb, perf, comp, trends):
            assert "generated_at" in _body(result)

    def test_performance_elo_history_limit_50(self, handler, three_agents):
        """get_elo_history called with limit=50 for performance."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            handler._get_agent_performance("claude", {})

        elo_sys.get_elo_history.assert_called_with("claude", limit=50)

    def test_trends_elo_history_limit_100(self, handler, three_agents):
        """get_elo_history called with limit=100 for trends."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            handler._get_agents_trends({"agents": "claude"})

        elo_sys.get_elo_history.assert_called_with("claude", limit=100)

    def test_performance_recent_matches_limit_10(self, handler, three_agents):
        """get_recent_matches called with limit=10."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            handler._get_agent_performance("claude", {})

        elo_sys.get_recent_matches.assert_called_with(limit=10)

    def test_performance_leaderboard_limit_100(self, handler, three_agents):
        """get_leaderboard called with limit=100 for rank calculation."""
        elo_sys = _make_elo_system(agents=three_agents)
        elo_sys.get_leaderboard.return_value = three_agents
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            handler._get_agent_performance("claude", {})

        elo_sys.get_leaderboard.assert_called_with(limit=100)
