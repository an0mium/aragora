"""Tests for Analytics namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestAnalyticsCoreMetrics:
    """Tests for core analytics metrics."""

    def test_disagreements_no_period(self, client: AragoraClient, mock_request) -> None:
        """Get disagreement analytics without period filter."""
        mock_request.return_value = {"patterns": [], "total_disagreements": 42}

        result = client.analytics.disagreements()

        mock_request.assert_called_once_with("GET", "/api/v1/analytics/disagreements", params={})
        assert result["total_disagreements"] == 42

    def test_disagreements_with_period(self, client: AragoraClient, mock_request) -> None:
        """Get disagreement analytics for a specific period."""
        mock_request.return_value = {"patterns": []}

        client.analytics.disagreements(period="7d")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/analytics/disagreements", params={"period": "7d"}
        )

    def test_role_rotation(self, client: AragoraClient, mock_request) -> None:
        """Get role rotation analytics."""
        mock_request.return_value = {"rotations": []}

        client.analytics.role_rotation(period="30d")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/analytics/role-rotation", params={"period": "30d"}
        )

    def test_early_stops(self, client: AragoraClient, mock_request) -> None:
        """Get early stop analytics."""
        mock_request.return_value = {"early_stops": 5, "total_debates": 100}

        result = client.analytics.early_stops()

        mock_request.assert_called_once_with("GET", "/api/v1/analytics/early-stops", params={})
        assert result["early_stops"] == 5

    def test_consensus_quality(self, client: AragoraClient, mock_request) -> None:
        """Get consensus quality metrics."""
        mock_request.return_value = {"average_quality": 0.87, "hollow_consensus_rate": 0.03}

        result = client.analytics.consensus_quality(period="90d")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/analytics/consensus-quality", params={"period": "90d"}
        )
        assert result["average_quality"] == 0.87

    def test_ranking_stats(self, client: AragoraClient, mock_request) -> None:
        """Get ranking statistics."""
        mock_request.return_value = {"top_agent": "claude", "average_elo": 1500}

        result = client.analytics.ranking_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/analytics/ranking")
        assert result["top_agent"] == "claude"

    def test_memory_stats(self, client: AragoraClient, mock_request) -> None:
        """Get memory system statistics."""
        mock_request.return_value = {"total_memories": 10000, "utilization": 0.65}

        result = client.analytics.memory_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/analytics/memory")
        assert result["total_memories"] == 10000


class TestAnalyticsDashboard:
    """Tests for dashboard overview analytics."""

    def test_get_summary(self, client: AragoraClient, mock_request) -> None:
        """Get dashboard summary."""
        mock_request.return_value = {
            "total_debates": 500,
            "active_agents": 12,
            "consensus_rate": 0.85,
        }

        result = client.analytics.get_summary()

        mock_request.assert_called_once_with("GET", "/api/analytics/summary", params={})
        assert result["total_debates"] == 500

    def test_get_summary_with_filters(self, client: AragoraClient, mock_request) -> None:
        """Get dashboard summary filtered by workspace and time range."""
        mock_request.return_value = {"total_debates": 50}

        client.analytics.get_summary(workspace_id="ws_1", time_range="7d")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["workspace_id"] == "ws_1"
        assert call_params["time_range"] == "7d"

    def test_get_risk_heatmap(self, client: AragoraClient, mock_request) -> None:
        """Get risk heatmap data."""
        mock_request.return_value = {"cells": []}

        client.analytics.get_risk_heatmap(time_range="30d")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["time_range"] == "30d"


class TestAnalyticsDebates:
    """Tests for debate analytics."""

    def test_debates_overview(self, client: AragoraClient, mock_request) -> None:
        """Get debates overview metrics."""
        mock_request.return_value = {
            "total": 1000,
            "consensus_rate": 0.82,
            "average_rounds": 3.5,
        }

        result = client.analytics.debates_overview()

        mock_request.assert_called_once_with("GET", "/api/analytics/debates/overview")
        assert result["consensus_rate"] == 0.82

    def test_debate_trends(self, client: AragoraClient, mock_request) -> None:
        """Get debate trends over time."""
        mock_request.return_value = {"data_points": []}

        client.analytics.debate_trends(time_range="30d", granularity="day")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["time_range"] == "30d"
        assert call_params["granularity"] == "day"

    def test_debate_topics(self, client: AragoraClient, mock_request) -> None:
        """Get topic distribution."""
        mock_request.return_value = {"topics": [{"name": "architecture", "count": 50}]}

        result = client.analytics.debate_topics(limit=5)

        mock_request.assert_called_once_with(
            "GET", "/api/analytics/debates/topics", params={"limit": 5}
        )
        assert result["topics"][0]["name"] == "architecture"

    def test_debate_outcomes(self, client: AragoraClient, mock_request) -> None:
        """Get debate outcome distribution."""
        mock_request.return_value = {"consensus": 80, "no_consensus": 15, "cancelled": 5}

        result = client.analytics.debate_outcomes()

        mock_request.assert_called_once_with("GET", "/api/analytics/debates/outcomes", params={})
        assert result["consensus"] == 80


class TestAnalyticsAgents:
    """Tests for agent analytics."""

    def test_agent_leaderboard(self, client: AragoraClient, mock_request) -> None:
        """Get agent leaderboard."""
        mock_request.return_value = {
            "agents": [
                {"name": "claude", "elo": 1650},
                {"name": "gpt-4", "elo": 1620},
            ]
        }

        result = client.analytics.agent_leaderboard(limit=10)

        mock_request.assert_called_once_with(
            "GET", "/api/analytics/agents/leaderboard", params={"limit": 10}
        )
        assert result["agents"][0]["name"] == "claude"

    def test_agent_leaderboard_by_domain(self, client: AragoraClient, mock_request) -> None:
        """Get agent leaderboard filtered by domain."""
        mock_request.return_value = {"agents": []}

        client.analytics.agent_leaderboard(domain="coding")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["domain"] == "coding"

    def test_agent_performance(self, client: AragoraClient, mock_request) -> None:
        """Get individual agent performance."""
        mock_request.return_value = {"win_rate": 0.72, "debates": 200}

        result = client.analytics.agent_performance("claude")

        mock_request.assert_called_once_with(
            "GET", "/api/analytics/agents/claude/performance", params={}
        )
        assert result["win_rate"] == 0.72

    def test_compare_agents(self, client: AragoraClient, mock_request) -> None:
        """Compare multiple agents."""
        mock_request.return_value = {"comparison": []}

        client.analytics.compare_agents(["claude", "gpt-4", "gemini"])

        mock_request.assert_called_once_with(
            "GET",
            "/api/analytics/agents/comparison",
            params={"agents": "claude,gpt-4,gemini"},
        )

    def test_calibration_stats(self, client: AragoraClient, mock_request) -> None:
        """Get calibration statistics."""
        mock_request.return_value = {"well_calibrated_count": 10}

        client.analytics.calibration_stats(agent="claude")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["agent"] == "claude"


class TestAnalyticsUsage:
    """Tests for usage and cost analytics."""

    def test_token_usage(self, client: AragoraClient, mock_request) -> None:
        """Get token usage data."""
        mock_request.return_value = {"total_tokens": 1000000, "cost": 25.50}

        result = client.analytics.token_usage(time_range="7d")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["time_range"] == "7d"
        assert result["total_tokens"] == 1000000

    def test_cost_breakdown(self, client: AragoraClient, mock_request) -> None:
        """Get cost breakdown by provider."""
        mock_request.return_value = {
            "providers": [
                {"name": "anthropic", "cost": 15.00},
                {"name": "openai", "cost": 10.50},
            ]
        }

        result = client.analytics.cost_breakdown(org_id="org_1")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["org_id"] == "org_1"
        assert len(result["providers"]) == 2

    def test_active_users(self, client: AragoraClient, mock_request) -> None:
        """Get active user counts."""
        mock_request.return_value = {"dau": 50, "mau": 200}

        result = client.analytics.active_users(time_range="30d")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["time_range"] == "30d"
        assert result["dau"] == 50


class TestAnalyticsFlips:
    """Tests for flip detection analytics."""

    def test_flip_summary(self, client: AragoraClient, mock_request) -> None:
        """Get flip detection summary."""
        mock_request.return_value = {"total_flips": 42, "flip_rate": 0.05}

        result = client.analytics.flip_summary()

        mock_request.assert_called_once_with("GET", "/api/analytics/flips/summary")
        assert result["total_flips"] == 42

    def test_recent_flips(self, client: AragoraClient, mock_request) -> None:
        """Get recent flips."""
        mock_request.return_value = {"flips": []}

        client.analytics.recent_flips(limit=10, agent="claude")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["limit"] == 10
        assert call_params["agent"] == "claude"

    def test_agent_consistency(self, client: AragoraClient, mock_request) -> None:
        """Get agent consistency scores."""
        mock_request.return_value = {"scores": {"claude": 0.95, "gpt-4": 0.88}}

        result = client.analytics.agent_consistency(agents=["claude", "gpt-4"])

        call_params = mock_request.call_args[1]["params"]
        assert call_params["agents"] == "claude,gpt-4"
        assert result["scores"]["claude"] == 0.95


class TestAnalyticsDeliberations:
    """Tests for deliberation analytics."""

    def test_deliberation_summary(self, client: AragoraClient, mock_request) -> None:
        """Get deliberation summary."""
        mock_request.return_value = {"total": 150, "average_duration": 45.0}

        result = client.analytics.deliberation_summary(days=7)

        mock_request.assert_called_once_with(
            "GET", "/api/analytics/deliberations", params={"days": 7}
        )
        assert result["total"] == 150

    def test_consensus_rates(self, client: AragoraClient, mock_request) -> None:
        """Get consensus rates by agent team."""
        mock_request.return_value = {"rates": []}

        client.analytics.consensus_rates(org_id="org_1", days=14)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["org_id"] == "org_1"
        assert call_params["days"] == 14


class TestAsyncAnalytics:
    """Tests for async analytics API."""

    @pytest.mark.asyncio
    async def test_async_disagreements(self, mock_async_request) -> None:
        """Get disagreements asynchronously."""
        mock_async_request.return_value = {"patterns": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.analytics.disagreements(period="7d")

            mock_async_request.assert_called_once_with(
                "GET", "/api/v1/analytics/disagreements", params={"period": "7d"}
            )

    @pytest.mark.asyncio
    async def test_async_debates_overview(self, mock_async_request) -> None:
        """Get debates overview asynchronously."""
        mock_async_request.return_value = {"total": 1000}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.analytics.debates_overview()

            mock_async_request.assert_called_once_with("GET", "/api/analytics/debates/overview")
            assert result["total"] == 1000

    @pytest.mark.asyncio
    async def test_async_agent_leaderboard(self, mock_async_request) -> None:
        """Get agent leaderboard asynchronously."""
        mock_async_request.return_value = {"agents": [{"name": "claude", "elo": 1650}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.analytics.agent_leaderboard(limit=5)

            assert result["agents"][0]["elo"] == 1650

    @pytest.mark.asyncio
    async def test_async_flip_summary(self, mock_async_request) -> None:
        """Get flip summary asynchronously."""
        mock_async_request.return_value = {"total_flips": 42}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.analytics.flip_summary()

            mock_async_request.assert_called_once_with("GET", "/api/analytics/flips/summary")
            assert result["total_flips"] == 42
