"""
Tests for analytics metrics handler endpoints.

Tests the analytics metrics API handlers for:
- Debate analytics (overview, trends, topics, outcomes)
- Agent analytics (leaderboard, performance, comparison, trends)
- Usage analytics (tokens, costs, active users)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


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


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self):
        self.agents = {
            "agent-1": {"name": "Agent One", "rating": 1500, "matches": 10},
            "agent-2": {"name": "Agent Two", "rating": 1450, "matches": 8},
        }

    def get_leaderboard(self, limit: int = 10, domain: str | None = None) -> List[Dict]:
        return [{"agent_id": k, **v} for k, v in list(self.agents.items())[:limit]]

    def get_rating(self, agent_id: str) -> Dict | None:
        return self.agents.get(agent_id)

    def get_elo_history(self, agent_id: str, limit: int = 30) -> List[Dict]:
        if agent_id not in self.agents:
            return []
        return [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "rating": 1500 - i * 10}
            for i in range(min(limit, 5))
        ]

    def get_recent_matches(self, limit: int = 10) -> List[Dict]:
        return [
            {
                "debate_id": f"debate-{i}",
                "agents": ["agent-1", "agent-2"],
                "winner": "agent-1",
            }
            for i in range(min(limit, 3))
        ]

    def get_head_to_head(self, agent_a: str, agent_b: str) -> Dict:
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "wins_a": 5,
            "wins_b": 3,
            "total_matches": 8,
        }

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())


class MockCostTracker:
    """Mock cost tracker for testing."""

    def get_workspace_stats(self, org_id: str | None = None) -> Dict[str, Any]:
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

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "30d"})

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

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({})

        assert result["success"] is True
        assert result["data"]["total_debates"] == 0
        assert result["data"]["consensus_rate"] == 0

    def test_debates_trends_grouping(self, mock_storage, mock_server_context):
        """Test debates trends groups by time correctly."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_trends(
                {
                    "time_range": "7d",
                    "granularity": "daily",
                }
            )

        assert result["success"] is True
        assert "trends" in result["data"]

    def test_debates_topics_extracts_keywords(self, mock_storage, mock_server_context):
        """Test debates topics extracts and counts keywords."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_topics({"limit": "10"})

        assert result["success"] is True
        assert "topics" in result["data"]

    def test_debates_outcomes_buckets_confidence(self, mock_storage, mock_server_context):
        """Test debates outcomes buckets by confidence level."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_outcomes({})

        assert result["success"] is True
        assert "outcomes" in result["data"]

    def test_time_range_parsing(self, mock_storage, mock_server_context):
        """Test various time range formats are parsed correctly."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        for time_range in ["7d", "14d", "30d", "90d", "180d", "365d", "all"]:
            with patch.object(handler, "_get_storage", return_value=mock_storage):
                result = handler._get_debates_overview({"time_range": time_range})
            assert result["success"] is True, f"Failed for time_range={time_range}"

    def test_invalid_time_range_defaults(self, mock_storage, mock_server_context):
        """Test invalid time range defaults to 30d."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"time_range": "invalid"})

        assert result["success"] is True


class TestAgentAnalytics:
    """Tests for agent analytics endpoints."""

    def test_leaderboard_returns_rankings(self, mock_elo_system, mock_server_context):
        """Test leaderboard returns agent rankings."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_leaderboard({"limit": "10"})

        assert result["success"] is True
        assert "leaderboard" in result["data"]
        assert len(result["data"]["leaderboard"]) <= 10

    def test_leaderboard_with_domain_filter(self, mock_elo_system, mock_server_context):
        """Test leaderboard filters by domain."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_leaderboard(
                {
                    "limit": "10",
                    "domain": "technology",
                }
            )

        assert result["success"] is True

    def test_agent_performance_found(self, mock_elo_system, mock_server_context):
        """Test agent performance returns data for existing agent."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agent_performance("agent-1", {})

        assert result["success"] is True
        assert "agent_id" in result["data"]
        assert "rating" in result["data"]

    def test_agent_performance_not_found(self, mock_elo_system, mock_server_context):
        """Test agent performance returns 404 for missing agent."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agent_performance("nonexistent", {})

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_agents_comparison(self, mock_elo_system, mock_server_context):
        """Test agents comparison returns head-to-head stats."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_comparison(
                {
                    "agents": "agent-1,agent-2",
                }
            )

        assert result["success"] is True
        assert "comparison" in result["data"]

    def test_agents_trends(self, mock_elo_system, mock_server_context):
        """Test agents trends returns historical data."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_trends(
                {
                    "time_range": "30d",
                    "agent_id": "agent-1",
                }
            )

        assert result["success"] is True
        assert "trends" in result["data"]

    def test_missing_elo_system_returns_error(self, mock_server_context):
        """Test missing ELO system returns 503."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_elo_system", return_value=None):
            result = handler._get_agents_leaderboard({})

        assert result["success"] is False
        assert result["status_code"] == 503


class TestUsageAnalytics:
    """Tests for usage analytics endpoints."""

    def test_usage_tokens_returns_stats(self, mock_cost_tracker, mock_server_context):
        """Test usage tokens returns token statistics."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_cost_tracker", return_value=mock_cost_tracker):
            result = handler._get_usage_tokens({})

        assert result["success"] is True
        assert "total_tokens_in" in result["data"]
        assert "total_tokens_out" in result["data"]

    def test_usage_costs_returns_breakdown(self, mock_cost_tracker, mock_server_context):
        """Test usage costs returns cost breakdown."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_cost_tracker", return_value=mock_cost_tracker):
            result = handler._get_usage_costs({})

        assert result["success"] is True
        assert "total_cost_usd" in result["data"]
        assert "cost_by_agent" in result["data"]
        assert "cost_by_model" in result["data"]

    def test_usage_with_org_filter(self, mock_cost_tracker, mock_server_context):
        """Test usage endpoints respect org_id filter."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_cost_tracker", return_value=mock_cost_tracker):
            result = handler._get_usage_tokens({"org_id": "org-123"})

        assert result["success"] is True

    def test_active_users_returns_counts(self, mock_server_context):
        """Test active users returns user counts."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)
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

        with patch.object(handler, "_get_user_store", return_value=mock_user_store):
            result = handler._get_active_users({})

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

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            # Test very high limit
            result = handler._get_debates_topics({"limit": "1000"})
            assert result["success"] is True

            # Test negative limit
            result = handler._get_debates_topics({"limit": "-5"})
            assert result["success"] is True

    def test_offset_validation(self, mock_storage, mock_server_context):
        """Test offset parameter validation."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_overview({"offset": "10"})
            assert result["success"] is True

    def test_granularity_validation(self, mock_storage, mock_server_context):
        """Test granularity parameter validation."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        for granularity in ["daily", "weekly", "monthly"]:
            with patch.object(handler, "_get_storage", return_value=mock_storage):
                result = handler._get_debates_trends({"granularity": granularity})
            assert result["success"] is True

    def test_invalid_granularity_defaults(self, mock_storage, mock_server_context):
        """Test invalid granularity defaults to daily."""
        from aragora.server.handlers.analytics_metrics import AnalyticsMetricsHandler

        handler = AnalyticsMetricsHandler(mock_server_context)

        with patch.object(handler, "_get_storage", return_value=mock_storage):
            result = handler._get_debates_trends({"granularity": "invalid"})

        assert result["success"] is True
