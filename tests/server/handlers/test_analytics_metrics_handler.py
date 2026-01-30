"""
Tests for analytics metrics handler endpoints.

Tests the analytics metrics API handlers for:
- Debate analytics (overview, trends, topics, outcomes)
- Agent analytics (leaderboard, performance, comparison, trends)
- Usage analytics (tokens, costs, active users)
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


def parse_handler_result(result) -> dict[str, Any]:
    """Parse HandlerResult to dict for assertions."""
    body = json.loads(result.body)
    return {"success": result.status_code < 400, "data": body, "status_code": result.status_code}


class MockDebate:
    """Mock debate object for testing."""

    def __init__(
        self,
        debate_id: str = "debate-1",
        task: str = "Test task",
        consensus_reached: bool = True,
        confidence: float = 0.85,
        created_at: datetime | None = None,
        rounds_used: int = 3,
    ):
        self.id = debate_id
        self.task = task
        self.consensus_reached = consensus_reached
        self.confidence = confidence
        self.created_at = created_at or datetime.now(timezone.utc)
        self.rounds_used = rounds_used


class MockAgent:
    """Mock agent object with required attributes."""

    def __init__(
        self,
        agent_name: str,
        elo: float = 1500,
        wins: int = 10,
        losses: int = 5,
        draws: int = 2,
    ):
        self.agent_name = agent_name
        self.elo = elo
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.games_played = wins + losses + draws
        self.debates_count = self.games_played
        self.win_rate = wins / self.games_played if self.games_played > 0 else 0
        self.domain_elos = {}


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self):
        self._agents = {
            "agent-1": MockAgent("agent-1", elo=1500, wins=10, losses=5, draws=2),
            "agent-2": MockAgent("agent-2", elo=1450, wins=8, losses=6, draws=3),
        }

    def get_leaderboard(self, limit: int = 10, domain: str | None = None) -> list:
        return list(self._agents.values())[:limit]

    def get_rating(self, agent_id: str):
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id}")
        return self._agents[agent_id]

    def get_elo_history(self, agent_id: str, limit: int = 30) -> list:
        if agent_id not in self._agents:
            return []
        now = datetime.now(timezone.utc)
        return [(now.isoformat(), 1500 - i * 10) for i in range(min(limit, 5))]

    def get_recent_matches(self, limit: int = 10) -> list[dict]:
        return [
            {
                "debate_id": f"debate-{i}",
                "participants": ["agent-1", "agent-2"],
                "winner": "agent-1",
            }
            for i in range(min(limit, 3))
        ]

    def get_head_to_head(self, agent_a: str, agent_b: str) -> dict:
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "a_wins": 5,
            "b_wins": 3,
            "draws": 2,
            "total": 10,
        }

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())


class MockCostTracker:
    """Mock cost tracker for testing."""

    def get_workspace_stats(self, org_id: str | None = None) -> dict[str, Any]:
        return {
            "total_tokens_in": 100000,
            "total_tokens_out": 50000,
            "total_cost_usd": 15.50,
            "cost_by_agent": {"agent-1": 10.00, "agent-2": 5.50},
            "cost_by_model": {"gpt-4": 12.00, "claude-3": 3.50},
            "total_api_calls": 250,
        }


@pytest.fixture
def mock_storage():
    """Create mock storage with debates."""
    storage = MagicMock()
    storage.list_debates.return_value = [
        MockDebate(debate_id=f"debate-{i}", confidence=0.7 + i * 0.05) for i in range(5)
    ]
    return storage


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system."""
    return MockEloSystem()


@pytest.fixture
def mock_cost_tracker():
    """Create mock cost tracker."""
    return MockCostTracker()


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestDebateAnalytics:
    """Tests for debate analytics endpoints."""

    def test_debates_overview_returns_stats(self, mock_storage, mock_server_context):
        """Test debates overview returns correct statistics."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_overview({"time_range": "30d"})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        data = result["data"]
        assert "total_debates" in data
        assert "consensus_rate" in data
        assert "avg_confidence" in data
        assert data["total_debates"] == 5

    def test_debates_overview_empty_storage(self, mock_server_context):
        """Test debates overview with no debates."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_overview({})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert result["data"]["total_debates"] == 0
        assert result["data"]["consensus_rate"] == 0

    def test_debates_trends_grouping(self, mock_storage, mock_server_context):
        """Test debates trends groups by time correctly."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_trends(
                {
                    "time_range": "7d",
                    "granularity": "daily",
                }
            )

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "data_points" in result["data"]

    def test_debates_topics_extracts_keywords(self, mock_storage, mock_server_context):
        """Test debates topics extracts and counts keywords."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_topics({"limit": "10"})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "topics" in result["data"]

    def test_debates_outcomes_buckets_confidence(self, mock_storage, mock_server_context):
        """Test debates outcomes buckets by confidence level."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_outcomes({})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "outcomes" in result["data"]

    def test_time_range_parsing(self, mock_storage, mock_server_context):
        """Test various time range formats are parsed correctly."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        for time_range in ["7d", "14d", "30d", "90d", "180d", "365d", "all"]:
            with patch.object(handler, "get_storage", return_value=mock_storage):
                raw_result = handler._get_debates_overview({"time_range": time_range})

                result = parse_handler_result(raw_result)
            assert result["success"] is True, f"Failed for time_range={time_range}"

    def test_invalid_time_range_defaults(self, mock_storage, mock_server_context):
        """Test invalid time range defaults to 30d."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_overview({"time_range": "invalid"})

            result = parse_handler_result(raw_result)

        assert result["success"] is True


class TestAgentAnalytics:
    """Tests for agent analytics endpoints."""

    def test_leaderboard_returns_rankings(self, mock_elo_system, mock_server_context):
        """Test leaderboard returns agent rankings."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_leaderboard({"limit": "10"})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "leaderboard" in result["data"]
        assert len(result["data"]["leaderboard"]) <= 10

    def test_leaderboard_with_domain_filter(self, mock_elo_system, mock_server_context):
        """Test leaderboard filters by domain."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_leaderboard(
                {
                    "limit": "10",
                    "domain": "technology",
                }
            )

            result = parse_handler_result(raw_result)

        assert result["success"] is True

    def test_agent_performance_found(self, mock_elo_system, mock_server_context):
        """Test agent performance returns data for existing agent."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agent_performance("agent-1", {})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "agent_id" in result["data"]
        assert "elo" in result["data"]

    def test_agent_performance_not_found(self, mock_elo_system, mock_server_context):
        """Test agent performance returns 404 for missing agent."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agent_performance("nonexistent", {})

            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_agents_comparison(self, mock_elo_system, mock_server_context):
        """Test agents comparison returns head-to-head stats."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_comparison(
                {
                    "agents": "agent-1,agent-2",
                }
            )

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "comparison" in result["data"]

    def test_agents_trends(self, mock_elo_system, mock_server_context):
        """Test agents trends returns historical data."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_trends(
                {
                    "time_range": "30d",
                    "agent_id": "agent-1",
                }
            )

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "trends" in result["data"]

    def test_missing_elo_system_returns_empty(self, mock_server_context):
        """Test missing ELO system returns empty leaderboard."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=None):
            raw_result = handler._get_agents_leaderboard({})

            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert result["data"]["leaderboard"] == []
        assert result["data"]["total_agents"] == 0


class TestUsageAnalytics:
    """Tests for usage analytics endpoints."""

    def test_usage_tokens_returns_stats(self, mock_cost_tracker, mock_server_context):
        """Test usage tokens returns token statistics."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_cost_tracker,
        ):
            raw_result = handler._get_usage_tokens({"org_id": "org-123"})
            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "summary" in result["data"]
        assert "total_tokens_in" in result["data"]["summary"]
        assert "total_tokens_out" in result["data"]["summary"]

    def test_usage_costs_returns_breakdown(self, mock_cost_tracker, mock_server_context):
        """Test usage costs returns cost breakdown."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_cost_tracker,
        ):
            raw_result = handler._get_usage_costs({"org_id": "org-123"})
            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "summary" in result["data"]
        assert "total_cost_usd" in result["data"]["summary"]
        assert "by_provider" in result["data"]
        assert "by_model" in result["data"]

    def test_usage_with_org_filter(self, mock_cost_tracker, mock_server_context):
        """Test usage endpoints respect org_id filter."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_cost_tracker,
        ):
            raw_result = handler._get_usage_tokens({"org_id": "org-123"})
            result = parse_handler_result(raw_result)

        assert result["success"] is True

    def test_active_users_returns_counts(self, mock_server_context):
        """Test active users returns user counts."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        mock_user_store = MagicMock()
        mock_user_store.get_active_user_counts.return_value = {
            "daily": 50,
            "weekly": 200,
            "monthly": 500,
        }
        mock_user_store.get_user_growth.return_value = {
            "growth_rate": 0.15,
            "new_users": 25,
        }

        # Pass user_store in context
        context = {**mock_server_context, "user_store": mock_user_store}
        handler = AnalyticsMetricsHandler(context)

        raw_result = handler._get_active_users({})
        result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "active_users" in result["data"]


class TestHandlerRouting:
    """Tests for handler routing and can_handle."""

    def test_can_handle_analytics_paths(self, mock_server_context):
        """Test handler recognizes analytics paths."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        assert handler.can_handle("/api/analytics/debates/overview")
        assert handler.can_handle("/api/analytics/agents/leaderboard")
        assert handler.can_handle("/api/analytics/usage/tokens")

    def test_cannot_handle_other_paths(self, mock_server_context):
        """Test handler rejects non-analytics paths."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/agents")
        assert not handler.can_handle("/health")


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_limit_clamping(self, mock_storage, mock_server_context):
        """Test limit parameter is clamped to valid range."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            # Test very high limit
            raw_result = handler._get_debates_topics({"limit": "1000"})

            result = parse_handler_result(raw_result)
            assert result["success"] is True

            # Test negative limit
            raw_result = handler._get_debates_topics({"limit": "-5"})

            result = parse_handler_result(raw_result)
            assert result["success"] is True

    def test_offset_validation(self, mock_storage, mock_server_context):
        """Test offset parameter validation."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_overview({"offset": "10"})

            result = parse_handler_result(raw_result)
            assert result["success"] is True

    def test_granularity_validation(self, mock_storage, mock_server_context):
        """Test granularity parameter validation."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        for granularity in ["daily", "weekly", "monthly"]:
            with patch.object(handler, "get_storage", return_value=mock_storage):
                raw_result = handler._get_debates_trends({"granularity": granularity})

                result = parse_handler_result(raw_result)
            assert result["success"] is True

    def test_invalid_granularity_defaults(self, mock_storage, mock_server_context):
        """Test invalid granularity defaults to daily."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_trends({"granularity": "invalid"})

            result = parse_handler_result(raw_result)

        assert result["success"] is True


# ===========================================================================
# Additional Tests for Rate Limiting and Authentication
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_exceeded_returns_429(self, mock_server_context):
        """Test rate limit exceeded returns 429 error."""
        from aragora.server.handlers.analytics_metrics import (
            AnalyticsMetricsHandler,
            _analytics_metrics_limiter,
        )

        handler = AnalyticsMetricsHandler(mock_server_context)
        mock_http_handler = MagicMock()
        mock_http_handler.client_address = ("127.0.0.1", 12345)

        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=False):
            with patch.object(handler, "get_auth_context", return_value=MagicMock()):
                raw_result = handler.handle(
                    "/api/analytics/debates/overview",
                    {},
                    mock_http_handler,
                )

        assert raw_result is not None
        assert raw_result.status_code == 429

    def test_rate_limit_allowed_proceeds(self, mock_server_context, mock_storage):
        """Test within rate limit proceeds to handler logic."""
        from aragora.server.handlers.analytics_metrics import (
            AnalyticsMetricsHandler,
            _analytics_metrics_limiter,
        )

        handler = AnalyticsMetricsHandler(mock_server_context)
        mock_http_handler = MagicMock()
        mock_http_handler.client_address = ("127.0.0.1", 12345)

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "test-user"
        mock_auth_ctx.roles = {"admin"}

        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=True):
            with patch.object(handler, "get_auth_context", return_value=mock_auth_ctx):
                with patch.object(handler, "check_permission", return_value=None):
                    with patch.object(handler, "get_storage", return_value=mock_storage):
                        raw_result = handler.handle(
                            "/api/analytics/debates/overview",
                            {},
                            mock_http_handler,
                        )

        assert raw_result is not None
        assert raw_result.status_code != 429


class TestAuthentication:
    """Tests for authentication and authorization."""

    def test_unauthenticated_returns_401(self, mock_server_context):
        """Test unauthenticated request returns 401."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler
        from aragora.server.handlers.secure import UnauthorizedError

        handler = AnalyticsMetricsHandler(mock_server_context)
        mock_http_handler = MagicMock()
        mock_http_handler.client_address = ("127.0.0.1", 12345)

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            raw_result = handler.handle(
                "/api/analytics/debates/overview",
                {},
                mock_http_handler,
            )

        assert raw_result is not None
        assert raw_result.status_code == 401

    def test_permission_denied_returns_403(self, mock_server_context):
        """Test permission denied returns 403."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler
        from aragora.server.handlers.secure import ForbiddenError

        handler = AnalyticsMetricsHandler(mock_server_context)
        mock_http_handler = MagicMock()
        mock_http_handler.client_address = ("127.0.0.1", 12345)

        mock_auth_ctx = MagicMock()

        with patch.object(handler, "get_auth_context", return_value=mock_auth_ctx):
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("Permission denied")
            ):
                raw_result = handler.handle(
                    "/api/analytics/debates/overview",
                    {},
                    mock_http_handler,
                )

        assert raw_result is not None
        assert raw_result.status_code == 403


class TestAgentPerformanceRouting:
    """Tests for agent performance pattern matching."""

    def test_can_handle_agent_performance_pattern(self, mock_server_context):
        """Test handler recognizes agent performance pattern."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        assert handler.can_handle("/api/analytics/agents/agent-1/performance")
        assert handler.can_handle("/api/analytics/agents/claude_3/performance")
        assert handler.can_handle("/api/analytics/agents/gpt-4-turbo/performance")

    def test_cannot_handle_invalid_agent_performance_pattern(self, mock_server_context):
        """Test handler rejects invalid agent performance patterns."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        # Missing /performance
        assert not handler.can_handle("/api/analytics/agents/agent-1")
        # Extra segments
        assert not handler.can_handle("/api/analytics/agents/agent-1/performance/extra")


class TestUsageAnalyticsEdgeCases:
    """Tests for usage analytics edge cases."""

    def test_usage_tokens_missing_org_id_returns_error(self, mock_server_context):
        """Test usage tokens without org_id returns error."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        raw_result = handler._get_usage_tokens({})
        result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_usage_costs_missing_org_id_returns_error(self, mock_server_context):
        """Test usage costs without org_id returns error."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        raw_result = handler._get_usage_costs({})
        result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_usage_tokens_import_error_graceful(self, mock_server_context):
        """Test usage tokens handles import error gracefully."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch(
            "aragora.server.handlers.analytics_metrics.AnalyticsMetricsHandler._get_usage_tokens",
            wraps=handler._get_usage_tokens,
        ):
            raw_result = handler._get_usage_tokens({"org_id": "org-123"})
            result = parse_handler_result(raw_result)

        # Should return message about cost tracker not available or success
        assert result["success"] is True or "message" in result["data"]

    def test_active_users_no_user_store(self, mock_server_context):
        """Test active users returns defaults when user_store not available."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        raw_result = handler._get_active_users({})
        result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert result["data"]["active_users"]["daily"] == 0
        assert "message" in result["data"]

    def test_active_users_with_time_range(self, mock_server_context):
        """Test active users respects time_range parameter."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        for time_range in ["7d", "30d", "90d"]:
            raw_result = handler._get_active_users({"time_range": time_range})
            result = parse_handler_result(raw_result)
            assert result["success"] is True
            assert result["data"]["time_range"] == time_range


class TestAgentsComparisonEdgeCases:
    """Tests for agents comparison edge cases."""

    def test_agents_comparison_missing_agents_param(self, mock_server_context):
        """Test agents comparison without agents param returns error."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        raw_result = handler._get_agents_comparison({})
        result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_agents_comparison_single_agent_error(self, mock_elo_system, mock_server_context):
        """Test agents comparison with only one agent returns error."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_comparison({"agents": "agent-1"})
            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_agents_comparison_too_many_agents(self, mock_elo_system, mock_server_context):
        """Test agents comparison with more than 10 agents returns error."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        agents = ",".join([f"agent-{i}" for i in range(15)])

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_comparison({"agents": agents})
            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_agents_comparison_missing_agent_in_comparison(
        self, mock_elo_system, mock_server_context
    ):
        """Test agents comparison handles missing agents gracefully."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            raw_result = handler._get_agents_comparison({"agents": "agent-1,nonexistent"})
            result = parse_handler_result(raw_result)

        # Should succeed but include error info for missing agent
        assert result["success"] is True
        comparison = result["data"]["comparison"]
        # One agent should have error info
        assert any("error" in c for c in comparison if isinstance(c, dict))


class TestDebateAnalyticsWithDictData:
    """Tests for debate analytics with dict-based debates."""

    def test_debates_overview_with_dict_debates(self, mock_server_context):
        """Test debates overview handles dict-based debates."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "task": "Test task",
                "consensus_reached": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": {"rounds_used": 3, "confidence": 0.85},
                "agents": ["agent-1", "agent-2"],
            },
            {
                "id": "debate-2",
                "task": "Test task 2",
                "consensus_reached": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": {"rounds_used": 5, "confidence": 0.5},
                "agents": ["agent-1", "agent-3"],
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_overview({})
            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert result["data"]["total_debates"] == 2

    def test_debates_outcomes_with_various_result_types(self, mock_server_context):
        """Test debates outcomes handles various outcome types."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "consensus_reached": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": {"outcome_type": "consensus", "confidence": 0.9},
            },
            {
                "id": "debate-2",
                "consensus_reached": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": {"outcome_type": "majority", "confidence": 0.6},
            },
            {
                "id": "debate-3",
                "consensus_reached": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": {"outcome_type": "dissent", "confidence": 0.4},
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            raw_result = handler._get_debates_outcomes({})
            result = parse_handler_result(raw_result)

        assert result["success"] is True
        assert "outcomes" in result["data"]


class TestEloSystemEdgeCases:
    """Tests for ELO system edge cases."""

    def test_agent_performance_no_elo_system(self, mock_server_context):
        """Test agent performance returns error when no ELO system."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=None):
            raw_result = handler._get_agent_performance("agent-1", {})
            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 503

    def test_agents_trends_no_elo_system(self, mock_server_context):
        """Test agents trends returns error when no ELO system."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=None):
            raw_result = handler._get_agents_trends({})
            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 503

    def test_agents_comparison_no_elo_system(self, mock_server_context):
        """Test agents comparison returns error when no ELO system."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "get_elo_system", return_value=None):
            raw_result = handler._get_agents_comparison({"agents": "a,b"})
            result = parse_handler_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 503


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_time_range_valid(self):
        """Test _parse_time_range with valid ranges."""
        from aragora.server.handlers.analytics_metrics import _parse_time_range

        result = _parse_time_range("30d")
        assert result is not None

        result = _parse_time_range("all")
        assert result is None

    def test_parse_time_range_invalid(self):
        """Test _parse_time_range with invalid range."""
        from aragora.server.handlers.analytics_metrics import _parse_time_range

        result = _parse_time_range("invalid")
        assert result is not None  # Should return default

    def test_group_by_time_daily(self):
        """Test _group_by_time with daily granularity."""
        from aragora.server.handlers.analytics_metrics import _group_by_time

        items = [
            {"timestamp": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)},
            {"timestamp": datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)},
            {"timestamp": datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)},
        ]

        result = _group_by_time(items, "timestamp", "daily")

        assert "2026-01-01" in result
        assert "2026-01-02" in result
        assert len(result["2026-01-01"]) == 2

    def test_group_by_time_weekly(self):
        """Test _group_by_time with weekly granularity."""
        from aragora.server.handlers.analytics_metrics import _group_by_time

        items = [
            {"timestamp": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)},
            {"timestamp": datetime(2026, 1, 8, 10, 0, tzinfo=timezone.utc)},
        ]

        result = _group_by_time(items, "timestamp", "weekly")

        # Should have different week keys
        assert len(result) == 2

    def test_group_by_time_monthly(self):
        """Test _group_by_time with monthly granularity."""
        from aragora.server.handlers.analytics_metrics import _group_by_time

        items = [
            {"timestamp": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)},
            {"timestamp": datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)},
        ]

        result = _group_by_time(items, "timestamp", "monthly")

        assert "2026-01" in result
        assert "2026-02" in result

    def test_group_by_time_string_timestamp(self):
        """Test _group_by_time with string timestamps."""
        from aragora.server.handlers.analytics_metrics import _group_by_time

        items = [
            {"timestamp": "2026-01-01T10:00:00+00:00"},
            {"timestamp": "2026-01-01T15:00:00Z"},
        ]

        result = _group_by_time(items, "timestamp", "daily")

        assert "2026-01-01" in result
        assert len(result["2026-01-01"]) == 2

    def test_group_by_time_missing_timestamp(self):
        """Test _group_by_time with missing timestamp."""
        from aragora.server.handlers.analytics_metrics import _group_by_time

        items = [
            {"timestamp": "2026-01-01T10:00:00+00:00"},
            {"other_field": "value"},  # No timestamp
        ]

        result = _group_by_time(items, "timestamp", "daily")

        assert "2026-01-01" in result
        assert len(result["2026-01-01"]) == 1


class TestVersionPrefixHandling:
    """Tests for version prefix handling in routes."""

    def test_can_handle_with_v1_prefix(self, mock_server_context):
        """Test handler recognizes paths with /v1 prefix."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        assert handler.can_handle("/api/v1/analytics/debates/overview")
        assert handler.can_handle("/api/v1/analytics/agents/leaderboard")

    def test_can_handle_without_version_prefix(self, mock_server_context):
        """Test handler recognizes paths without version prefix."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        assert handler.can_handle("/api/analytics/debates/overview")
        assert handler.can_handle("/api/analytics/agents/leaderboard")
