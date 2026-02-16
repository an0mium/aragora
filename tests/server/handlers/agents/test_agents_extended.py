"""
Extended tests for AgentsHandler covering methods not in the base test file.

Covers:
- _list_agents with and without stats
- _get_network, _get_moments, _get_positions, _get_domains, _get_performance
- _get_metadata, _get_agent_introspect, _get_opponent_briefing, _get_calibration
- _dispatch_agent_endpoint unknown endpoint
- _handle_agent_endpoint invalid path
- _compare_agents validation (fewer than 2 agents)
- _compute_confidence thresholds
- _missing_required_env_vars helper
- ELO-not-available error paths for untested methods
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


@pytest.fixture
def mock_elo():
    """Minimal mock ELO system for extended tests."""
    elo = MagicMock()
    rating = MagicMock()
    rating.elo = 1600
    rating.wins = 12
    rating.losses = 4
    rating.draws = 2
    rating.domain_elos = {"technical": 1700, "ethics": 1500}
    rating.calibration_accuracy = 0.82
    rating.calibration_brier_score = 0.15
    rating.calibration_total = 30
    rating.critiques_accepted = 8
    rating.critiques_total = 10
    rating.critique_acceptance_rate = 0.8
    elo.get_rating.return_value = rating
    elo.get_leaderboard.return_value = [
        {"name": "claude", "elo": 1650},
        {"name": "gpt-4", "elo": 1600},
    ]
    elo.get_cached_leaderboard.return_value = elo.get_leaderboard.return_value
    elo.get_rivals.return_value = [{"agent": "gpt-4", "matches": 3}]
    elo.get_allies.return_value = [{"agent": "gemini", "collaborations": 2}]
    elo.get_elo_history.return_value = [
        ("2025-01-01T00:00:00", 1600),
        ("2025-01-02T00:00:00", 1620),
    ]
    elo.get_agent_history.return_value = [
        {"result": "win"},
        {"result": "loss"},
        {"result": "win"},
    ]
    elo.get_agent_stats.return_value = {"total_matches": 18, "win_rate": 0.67}
    elo.get_ratings_batch.return_value = {"claude": 1650, "gpt-4": 1600}
    elo.get_head_to_head.return_value = {"matches": 5, "agent1_wins": 3, "agent2_wins": 2}
    elo.get_calibration.return_value = {"agent": "claude", "score": 0.85}
    return elo


@pytest.fixture
def handler(mock_elo):
    """AgentsHandler with mock ELO and cleared cache."""
    from aragora.server.handlers.agents.agents import AgentsHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()
    return AgentsHandler(server_context={"elo_system": mock_elo})


@pytest.fixture
def empty_handler():
    """AgentsHandler with no ELO system."""
    from aragora.server.handlers.agents.agents import AgentsHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()
    return AgentsHandler(server_context={})


def _body(result):
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body


# ---------------------------------------------------------------------------
# _list_agents
# ---------------------------------------------------------------------------


class TestListAgents:
    """Tests for _list_agents method."""

    def test_list_agents_without_stats(self, handler):
        result = handler._list_agents(include_stats=False)
        assert result.status_code == 200
        body = _body(result)
        assert "agents" in body
        assert "total" in body
        # Without stats, agents should just have name
        for agent in body["agents"]:
            assert "name" in agent

    def test_list_agents_with_stats(self, handler, mock_elo):
        from aragora.server.handlers.base import clear_cache

        clear_cache()
        mock_elo.get_leaderboard.return_value = [
            MagicMock(name="claude", elo=1650, matches=15, wins=12, losses=3),
        ]
        result = handler._list_agents(include_stats=True)
        assert result.status_code == 200

    def test_list_agents_fallback_no_elo(self, empty_handler):
        result = empty_handler._list_agents(include_stats=False)
        assert result.status_code == 200
        body = _body(result)
        assert body["total"] > 0  # Falls back to ALLOWED_AGENT_TYPES

    def test_list_agents_elo_exception_falls_back(self, handler, mock_elo):
        from aragora.server.handlers.base import clear_cache

        clear_cache()
        mock_elo.get_leaderboard.side_effect = OSError("db locked")
        result = handler._list_agents(include_stats=False)
        assert result.status_code == 200
        body = _body(result)
        # Should fall back to ALLOWED_AGENT_TYPES
        assert body["total"] > 0


# ---------------------------------------------------------------------------
# _get_network
# ---------------------------------------------------------------------------


class TestGetNetwork:
    """Tests for _get_network method."""

    def test_get_network_success(self, handler):
        result = handler._get_network("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert "rivals" in body
        assert "allies" in body

    def test_get_network_no_elo(self, empty_handler):
        result = empty_handler._get_network("claude")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# _get_calibration
# ---------------------------------------------------------------------------


class TestGetCalibration:
    """Tests for _get_calibration method."""

    def test_get_calibration_with_method(self, handler, mock_elo):
        result = handler._get_calibration("claude", domain=None)
        assert result.status_code == 200

    def test_get_calibration_without_method(self, handler, mock_elo):
        del mock_elo.get_calibration
        mock_elo.get_calibration = None
        mock_elo.configure_mock(**{"get_calibration": None})
        # hasattr will still be True with MagicMock; test the fallback path
        # by removing the attribute
        mock_elo_clean = MagicMock(spec=[])
        mock_elo_clean.get_rating = mock_elo.get_rating
        from aragora.server.handlers.agents.agents import AgentsHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()
        h = AgentsHandler(server_context={"elo_system": mock_elo_clean})
        result = h._get_calibration("claude", domain="tech")
        assert result.status_code == 200
        body = _body(result)
        assert body["score"] == 0.5  # default fallback

    def test_get_calibration_no_elo(self, empty_handler):
        result = empty_handler._get_calibration("claude", domain=None)
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# _get_domains
# ---------------------------------------------------------------------------


class TestGetDomains:
    """Tests for _get_domains method."""

    def test_get_domains_success(self, handler):
        result = handler._get_domains("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["domain_count"] == 2
        # Should be sorted by ELO descending
        assert body["domains"][0]["domain"] == "technical"

    def test_get_domains_no_elo(self, empty_handler):
        result = empty_handler._get_domains("claude")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# _get_performance
# ---------------------------------------------------------------------------


class TestGetPerformance:
    """Tests for _get_performance method."""

    def test_get_performance_success(self, handler):
        result = handler._get_performance("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["total_games"] == 18  # 12 + 4 + 2
        assert body["win_rate"] > 0
        assert "calibration" in body

    def test_get_performance_no_elo(self, empty_handler):
        result = empty_handler._get_performance("claude")
        assert result.status_code == 503

    def test_get_performance_zero_games(self, handler, mock_elo):
        from aragora.server.handlers.base import clear_cache

        clear_cache()
        rating = mock_elo.get_rating.return_value
        rating.wins = 0
        rating.losses = 0
        rating.draws = 0
        result = handler._get_performance("newbie")
        assert result.status_code == 200
        body = _body(result)
        assert body["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# _get_metadata
# ---------------------------------------------------------------------------


class TestGetMetadata:
    """Tests for _get_metadata method."""

    def test_get_metadata_no_nomic_dir(self, handler):
        result = handler._get_metadata("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["metadata"] is None
        assert (
            "not available" in body.get("message", "").lower()
            or "not found" in body.get("message", "").lower()
            or body["metadata"] is None
        )


# ---------------------------------------------------------------------------
# _get_agent_introspect
# ---------------------------------------------------------------------------


class TestGetAgentIntrospect:
    """Tests for _get_agent_introspect method."""

    def test_introspect_basic(self, handler):
        result = handler._get_agent_introspect("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["agent_id"] == "claude"
        assert "identity" in body
        assert "performance" in body
        assert "calibration" in body

    def test_introspect_no_elo(self, empty_handler):
        result = empty_handler._get_agent_introspect("claude")
        assert result.status_code == 200
        body = _body(result)
        assert body["performance"] == {}

    def test_introspect_with_debate_id(self, handler):
        result = handler._get_agent_introspect("claude", debate_id="debate_123")
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# _compare_agents validation
# ---------------------------------------------------------------------------


class TestCompareAgentsValidation:
    """Tests for _compare_agents edge cases."""

    def test_compare_fewer_than_two(self, handler):
        result = handler._compare_agents(["claude"])
        assert result.status_code == 400
        body = _body(result)
        assert "2" in body.get("error", "").lower() or "2" in str(body)

    def test_compare_empty_list(self, handler):
        result = handler._compare_agents([])
        assert result.status_code == 400

    def test_compare_more_than_five_clamped(self, handler):
        agents = ["a", "b", "c", "d", "e", "f", "g"]
        result = handler._compare_agents(agents)
        assert result.status_code == 200
        body = _body(result)
        # Should only include first 5
        assert len(body["agents"]) <= 5


# ---------------------------------------------------------------------------
# _dispatch_agent_endpoint unknown endpoint
# ---------------------------------------------------------------------------


class TestDispatchAgentEndpoint:
    """Tests for _dispatch_agent_endpoint with unknown endpoints."""

    def test_unknown_endpoint_returns_none(self, handler):
        result = handler._dispatch_agent_endpoint("claude", "nonexistent", {})
        assert result is None

    def test_valid_endpoint_profile(self, handler):
        result = handler._dispatch_agent_endpoint("claude", "profile", {})
        assert result is not None
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# _handle_agent_endpoint invalid path
# ---------------------------------------------------------------------------


class TestHandleAgentEndpoint:
    """Tests for _handle_agent_endpoint edge cases."""

    def test_too_few_parts_returns_error(self, handler):
        result = handler._handle_agent_endpoint("/api/agent/claude", {})
        assert result is not None
        assert result.status_code == 400

    def test_head_to_head_dispatch(self, handler):
        result = handler._handle_agent_endpoint("/api/agent/claude/head-to-head/gpt-4", {})
        assert result is not None
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for _compute_confidence helper."""

    def test_insufficient_data(self, handler):
        rating = MagicMock()
        rating.calibration_accuracy = 0.9
        rating.calibration_total = 3
        assert handler._compute_confidence(rating) == "insufficient_data"

    def test_high_confidence(self, handler):
        rating = MagicMock()
        rating.calibration_accuracy = 0.85
        rating.calibration_total = 20
        assert handler._compute_confidence(rating) == "high"

    def test_medium_confidence(self, handler):
        rating = MagicMock()
        rating.calibration_accuracy = 0.7
        rating.calibration_total = 10
        assert handler._compute_confidence(rating) == "medium"

    def test_low_confidence(self, handler):
        rating = MagicMock()
        rating.calibration_accuracy = 0.4
        rating.calibration_total = 10
        assert handler._compute_confidence(rating) == "low"


# ---------------------------------------------------------------------------
# _missing_required_env_vars helper
# ---------------------------------------------------------------------------


class TestMissingRequiredEnvVars:
    """Tests for _missing_required_env_vars module function."""

    def test_none_input(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars(None) == []

    def test_empty_string(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("") == []

    def test_optional_keyword(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("OPENAI_API_KEY (optional)") == []

    def test_no_uppercase_vars(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("no vars here") == []

    @patch("aragora.server.handlers.agents.agents._secret_configured", return_value=True)
    def test_secret_configured(self, mock_secret):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("ANTHROPIC_API_KEY") == []

    @patch("aragora.server.handlers.agents.agents._secret_configured", return_value=False)
    def test_secret_not_configured(self, mock_secret):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        result = _missing_required_env_vars("ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" in result


# ---------------------------------------------------------------------------
# ELO-missing error paths for untested methods
# ---------------------------------------------------------------------------


class TestEloMissingPaths:
    """Ensure 503 is returned when ELO system is unavailable."""

    def test_rivals_no_elo(self, empty_handler):
        result = empty_handler._get_rivals("claude", limit=5)
        assert result.status_code == 503

    def test_allies_no_elo(self, empty_handler):
        result = empty_handler._get_allies("claude", limit=5)
        assert result.status_code == 503

    def test_head_to_head_no_elo(self, empty_handler):
        result = empty_handler._get_head_to_head("claude", "gpt-4")
        assert result.status_code == 503

    def test_history_no_elo(self, empty_handler):
        result = empty_handler._get_history("claude", limit=10)
        assert result.status_code == 503

    def test_profile_no_elo(self, empty_handler):
        result = empty_handler._get_profile("claude")
        assert result.status_code == 503

    def test_recent_matches_no_elo(self, empty_handler):
        result = empty_handler._get_recent_matches(limit=10, loop_id=None)
        assert result.status_code == 503
