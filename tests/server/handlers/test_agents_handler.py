"""
Tests for agent-related endpoint handlers.

Tests:
- AgentsHandler initialization
- Route matching (can_handle)
- Path parsing and validation
- Endpoint dispatching
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.agents.agents import AgentsHandler


class TestAgentsHandlerInit:
    """Tests for AgentsHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = AgentsHandler({})
        assert hasattr(handler, "ctx")

    def test_routes_constant_is_list(self):
        """ROUTES should be a list."""
        assert isinstance(AgentsHandler.ROUTES, list)

    def test_routes_includes_core_endpoints(self):
        """ROUTES should include core agent endpoints."""
        routes = AgentsHandler.ROUTES
        assert "/api/agents" in routes
        assert "/api/leaderboard" in routes
        assert "/api/rankings" in routes
        assert "/api/matches/recent" in routes
        assert "/api/agent/compare" in routes


class TestAgentsHandlerCanHandle:
    """Tests for can_handle method."""

    def test_can_handle_agents_list(self):
        """Should handle /api/agents."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agents") is True

    def test_can_handle_local_agents(self):
        """Should handle /api/agents/local."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agents/local") is True

    def test_can_handle_local_status(self):
        """Should handle /api/agents/local/status."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agents/local/status") is True

    def test_can_handle_leaderboard(self):
        """Should handle /api/leaderboard."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/leaderboard") is True

    def test_can_handle_rankings(self):
        """Should handle /api/rankings."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/rankings") is True

    def test_can_handle_recent_matches(self):
        """Should handle /api/matches/recent."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/matches/recent") is True

    def test_can_handle_agent_compare(self):
        """Should handle /api/agent/compare."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agent/compare") is True

    def test_can_handle_agent_profile(self):
        """Should handle /api/agent/{name}/* paths."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agent/claude/profile") is True

    def test_can_handle_agent_history(self):
        """Should handle agent history endpoint."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/agent/claude/history") is True

    def test_can_handle_flips_recent(self):
        """Should handle /api/flips/recent."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/flips/recent") is True

    def test_can_handle_flips_summary(self):
        """Should handle /api/flips/summary."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/flips/summary") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unknown paths."""
        handler = AgentsHandler({})
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/other") is False
        assert handler.can_handle("/api/users") is False


class TestAgentsHandlerDispatch:
    """Tests for endpoint dispatching."""

    def test_dispatch_profile_endpoint(self):
        """Should recognize profile endpoint."""
        handler = AgentsHandler({})
        # Mock the _get_profile method
        handler._get_profile = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "profile", {})
        handler._get_profile.assert_called_once_with("claude")

    def test_dispatch_history_endpoint(self):
        """Should recognize history endpoint."""
        handler = AgentsHandler({})
        handler._get_history = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "history", {"limit": ["10"]})
        handler._get_history.assert_called()

    def test_dispatch_calibration_endpoint(self):
        """Should recognize calibration endpoint."""
        handler = AgentsHandler({})
        handler._get_calibration = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "calibration", {})
        handler._get_calibration.assert_called()

    def test_dispatch_consistency_endpoint(self):
        """Should recognize consistency endpoint."""
        handler = AgentsHandler({})
        handler._get_consistency = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "consistency", {})
        handler._get_consistency.assert_called_once_with("claude")

    def test_dispatch_flips_endpoint(self):
        """Should recognize flips endpoint."""
        handler = AgentsHandler({})
        handler._get_agent_flips = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "flips", {})
        handler._get_agent_flips.assert_called()

    def test_dispatch_network_endpoint(self):
        """Should recognize network endpoint."""
        handler = AgentsHandler({})
        handler._get_network = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "network", {})
        handler._get_network.assert_called_once_with("claude")

    def test_dispatch_rivals_endpoint(self):
        """Should recognize rivals endpoint."""
        handler = AgentsHandler({})
        handler._get_rivals = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "rivals", {})
        handler._get_rivals.assert_called()

    def test_dispatch_allies_endpoint(self):
        """Should recognize allies endpoint."""
        handler = AgentsHandler({})
        handler._get_allies = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "allies", {})
        handler._get_allies.assert_called()

    def test_dispatch_moments_endpoint(self):
        """Should recognize moments endpoint."""
        handler = AgentsHandler({})
        handler._get_moments = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "moments", {})
        handler._get_moments.assert_called()

    def test_dispatch_positions_endpoint(self):
        """Should recognize positions endpoint."""
        handler = AgentsHandler({})
        handler._get_positions = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "positions", {})
        handler._get_positions.assert_called()

    def test_dispatch_domains_endpoint(self):
        """Should recognize domains endpoint."""
        handler = AgentsHandler({})
        handler._get_domains = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "domains", {})
        handler._get_domains.assert_called_once_with("claude")

    def test_dispatch_performance_endpoint(self):
        """Should recognize performance endpoint."""
        handler = AgentsHandler({})
        handler._get_performance = MagicMock(return_value={"status": 200})

        result = handler._dispatch_agent_endpoint("claude", "performance", {})
        handler._get_performance.assert_called_once_with("claude")

    def test_dispatch_unknown_returns_none(self):
        """Unknown endpoint should return None."""
        handler = AgentsHandler({})
        result = handler._dispatch_agent_endpoint("claude", "unknown_endpoint", {})
        assert result is None


class TestAgentsHandlerHandle:
    """Tests for main handle method routing."""

    def test_handle_agents_list(self):
        """Should route /api/agents to _list_agents."""
        handler = AgentsHandler({})
        handler._list_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents", {}, None)
        handler._list_agents.assert_called_once()

    def test_handle_local_agents(self):
        """Should route /api/agents/local to _list_local_agents."""
        handler = AgentsHandler({})
        handler._list_local_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents/local", {}, None)
        handler._list_local_agents.assert_called_once()

    def test_handle_local_status(self):
        """Should route /api/agents/local/status to _get_local_status."""
        handler = AgentsHandler({})
        handler._get_local_status = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents/local/status", {}, None)
        handler._get_local_status.assert_called_once()

    def test_handle_leaderboard(self):
        """Should route /api/leaderboard to _get_leaderboard."""
        handler = AgentsHandler({})
        handler._get_leaderboard = MagicMock(return_value={"status": 200})

        handler.handle("/api/leaderboard", {}, None)
        handler._get_leaderboard.assert_called()

    def test_handle_rankings(self):
        """Should route /api/rankings to _get_leaderboard."""
        handler = AgentsHandler({})
        handler._get_leaderboard = MagicMock(return_value={"status": 200})

        handler.handle("/api/rankings", {}, None)
        handler._get_leaderboard.assert_called()

    def test_handle_recent_matches(self):
        """Should route /api/matches/recent to _get_recent_matches."""
        handler = AgentsHandler({})
        handler._get_recent_matches = MagicMock(return_value={"status": 200})

        handler.handle("/api/matches/recent", {}, None)
        handler._get_recent_matches.assert_called()

    def test_handle_agent_compare(self):
        """Should route /api/agent/compare to _compare_agents."""
        handler = AgentsHandler({})
        handler._compare_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agent/compare", {"agents": ["claude", "gpt"]}, None)
        handler._compare_agents.assert_called()

    def test_handle_flips_recent(self):
        """Should route /api/flips/recent to _get_recent_flips."""
        handler = AgentsHandler({})
        handler._get_recent_flips = MagicMock(return_value={"status": 200})

        handler.handle("/api/flips/recent", {}, None)
        handler._get_recent_flips.assert_called()

    def test_handle_flips_summary(self):
        """Should route /api/flips/summary to _get_flip_summary."""
        handler = AgentsHandler({})
        handler._get_flip_summary = MagicMock(return_value={"status": 200})

        handler.handle("/api/flips/summary", {}, None)
        handler._get_flip_summary.assert_called_once()


class TestAgentsHandlerAgentEndpoint:
    """Tests for _handle_agent_endpoint method."""

    def test_invalid_short_path(self):
        """Short paths should return error."""
        handler = AgentsHandler({})
        result = handler._handle_agent_endpoint("/api/agent", {})
        assert result.status_code == 400

    def test_head_to_head_routing(self):
        """Should route head-to-head endpoint correctly."""
        handler = AgentsHandler({})
        handler._get_head_to_head = MagicMock(return_value={"status": 200})
        handler.extract_path_param = MagicMock(return_value=("claude", None))

        # Need to mock extract_path_param to return agent and opponent
        with patch.object(handler, 'extract_path_param', side_effect=[
            ("claude", None),  # First call returns agent name
            ("gpt", None),     # Second call returns opponent name
        ]):
            result = handler._handle_agent_endpoint(
                "/api/agent/claude/head-to-head/gpt", {}
            )
            handler._get_head_to_head.assert_called_once_with("claude", "gpt")

    def test_opponent_briefing_routing(self):
        """Should route opponent-briefing endpoint correctly."""
        handler = AgentsHandler({})
        handler._get_opponent_briefing = MagicMock(return_value={"status": 200})

        with patch.object(handler, 'extract_path_param', side_effect=[
            ("claude", None),
            ("gpt", None),
        ]):
            result = handler._handle_agent_endpoint(
                "/api/agent/claude/opponent-briefing/gpt", {}
            )
            handler._get_opponent_briefing.assert_called_once_with("claude", "gpt")


class TestCompareAgents:
    """Tests for _compare_agents method."""

    def test_requires_at_least_two_agents(self):
        """Should require at least 2 agents."""
        handler = AgentsHandler({})
        result = handler._compare_agents(["claude"])
        assert result.status_code == 400

    def test_empty_list_returns_error(self):
        """Empty agent list should return error."""
        handler = AgentsHandler({})
        result = handler._compare_agents([])
        assert result.status_code == 400

    def test_single_agent_returns_error(self):
        """Single agent should return error."""
        handler = AgentsHandler({})
        result = handler._compare_agents(["single_agent"])
        assert result.status_code == 400


class TestListAgentsIncludeStats:
    """Tests for include_stats parameter in list agents."""

    def test_include_stats_default_false(self):
        """include_stats should default to false."""
        handler = AgentsHandler({})
        handler._list_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents", {}, None)
        handler._list_agents.assert_called_with(False)

    def test_include_stats_true(self):
        """include_stats=true should pass True."""
        handler = AgentsHandler({})
        handler._list_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents", {"include_stats": ["true"]}, None)
        handler._list_agents.assert_called_with(True)

    def test_include_stats_false(self):
        """include_stats=false should pass False."""
        handler = AgentsHandler({})
        handler._list_agents = MagicMock(return_value={"status": 200})

        handler.handle("/api/agents", {"include_stats": ["false"]}, None)
        handler._list_agents.assert_called_with(False)
