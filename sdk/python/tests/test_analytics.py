"""Tests for Analytics namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

class TestAnalyticsDashboardOverview:
    """Tests for dashboard summary and overview endpoints."""

    def test_get_summary_no_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_debates": 500,
                "active_agents": 12,
                "consensus_rate": 0.85,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.get_summary()
            mock_request.assert_called_once_with("GET", "/api/analytics/summary", params={})
            assert result["total_debates"] == 500
            assert result["consensus_rate"] == 0.85
            client.close()

    def test_get_summary_with_workspace_and_time_range(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_debates": 50}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.analytics.get_summary(workspace_id="ws_123", time_range="7d")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/summary",
                params={"workspace_id": "ws_123", "time_range": "7d"},
            )
            client.close()

    def test_debates_overview(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total": 1000,
                "consensus_rate": 0.82,
                "average_rounds": 3.5,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.debates_overview()
            mock_request.assert_called_once_with("GET", "/api/analytics/debates/overview")
            assert result["total"] == 1000
            assert result["average_rounds"] == 3.5
            client.close()

class TestAnalyticsTimeSeries:
    """Tests for time-series and trend data endpoints."""

    def test_get_finding_trends_all_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "data_points": [{"date": "2025-01-01", "count": 12}],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.get_finding_trends(
                workspace_id="ws_456", time_range="30d", granularity="day"
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/trends/findings",
                params={
                    "workspace_id": "ws_456",
                    "time_range": "30d",
                    "granularity": "day",
                },
            )
            assert result["data_points"][0]["count"] == 12
            client.close()

    def test_debate_trends_with_optional_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data_points": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.analytics.debate_trends(time_range="30d", granularity="day")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/debates/trends",
                params={"time_range": "30d", "granularity": "day"},
            )
            client.close()

    def test_debate_topics_with_limit(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "topics": [{"name": "architecture", "count": 50}],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.debate_topics(limit=5)
            mock_request.assert_called_once_with(
                "GET", "/api/analytics/debates/topics", params={"limit": 5}
            )
            assert result["topics"][0]["name"] == "architecture"
            client.close()

class TestAnalyticsAgentPerformance:
    """Tests for agent analytics and comparison endpoints."""

    def test_agent_leaderboard_with_domain(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agents": [{"name": "claude", "elo": 1650}],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.agent_leaderboard(limit=5, domain="security")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/agents/leaderboard",
                params={"limit": 5, "domain": "security"},
            )
            assert result["agents"][0]["elo"] == 1650
            client.close()

    def test_compare_agents(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"comparison": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.analytics.compare_agents(["claude", "gpt-4", "gemini"])
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/agents/comparison",
                params={"agents": "claude,gpt-4,gemini"},
            )
            client.close()

class TestAnalyticsReportsAndCosts:
    """Tests for usage tracking, cost breakdown, and report-style endpoints."""

    def test_token_usage_all_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_tokens": 1_500_000}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.token_usage(
                org_id="org_42", time_range="7d", granularity="hour"
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/usage/tokens",
                params={"org_id": "org_42", "time_range": "7d", "granularity": "hour"},
            )
            assert result["total_tokens"] == 1_500_000
            client.close()

    def test_cost_breakdown(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "by_provider": {"anthropic": 12.50, "openai": 8.30},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.cost_breakdown(time_range="30d")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/usage/costs",
                params={"time_range": "30d"},
            )
            assert result["by_provider"]["anthropic"] == 12.50
            client.close()

    def test_deliberation_summary_default_days(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total": 80, "avg_duration": 4.5}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.analytics.deliberation_summary()
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/deliberations",
                params={"days": 30},
            )
            assert result["total"] == 80
            client.close()

class TestAsyncAnalytics:
    """Tests for async analytics methods."""

    @pytest.mark.asyncio
    async def test_async_disagreements_with_period(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"patterns": [{"pair": "claude-gpt"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.analytics.disagreements(period="30d")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/analytics/disagreements", params={"period": "30d"}
            )
            assert len(result["patterns"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_async_debates_overview(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total": 200, "consensus_rate": 0.85}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.analytics.debates_overview()
            mock_request.assert_called_once_with("GET", "/api/analytics/debates/overview")
            assert result["consensus_rate"] == 0.85
            await client.close()

    @pytest.mark.asyncio
    async def test_async_agent_leaderboard(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"agents": [{"id": "claude", "elo": 1700}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.analytics.agent_leaderboard(limit=10, domain="coding")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/agents/leaderboard",
                params={"limit": 10, "domain": "coding"},
            )
            assert result["agents"][0]["elo"] == 1700
            await client.close()

    @pytest.mark.asyncio
    async def test_async_cost_breakdown(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"by_provider": {"anthropic": 25.00}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.analytics.cost_breakdown(org_id="org_1", time_range="7d")
            mock_request.assert_called_once_with(
                "GET",
                "/api/analytics/usage/costs",
                params={"org_id": "org_1", "time_range": "7d"},
            )
            assert result["by_provider"]["anthropic"] == 25.00
            await client.close()
