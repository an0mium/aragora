"""Tests for CostManagementAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.cost_management import (
    Budget,
    CostAlert,
    CostBreakdownItem,
    CostForecast,
    CostManagementAPI,
    CostRecommendation,
    CostSummary,
    DailyCost,
    EfficiencyMetrics,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> CostManagementAPI:
    return CostManagementAPI(mock_client)


SAMPLE_BREAKDOWN_ITEM = {"name": "anthropic", "cost": 42.5, "percentage": 65.0}

SAMPLE_DAILY_COST = {"date": "2026-02-10", "cost": 12.3, "tokens": 50000}

SAMPLE_ALERT = {
    "id": "alert-001",
    "type": "budget_warning",
    "message": "Approaching 80% of monthly budget",
    "severity": "warning",
    "timestamp": "2026-02-10T14:00:00Z",
    "acknowledged": False,
}

SAMPLE_RECOMMENDATION = {
    "id": "rec-001",
    "type": "model_downgrade",
    "title": "Switch to smaller model for summaries",
    "description": "Use claude-haiku instead of claude-opus for summary tasks",
    "estimated_savings": 15.0,
    "effort": "low",
    "status": "pending",
    "created_at": "2026-02-01T08:00:00Z",
}

SAMPLE_SUMMARY = {
    "totalCost": 150.75,
    "budget": 500.0,
    "tokensUsed": 2000000,
    "apiCalls": 1500,
    "lastUpdated": "2026-02-10T15:00:00Z",
    "costByProvider": [SAMPLE_BREAKDOWN_ITEM],
    "costByFeature": [{"name": "debate", "cost": 100.0, "percentage": 66.4}],
    "dailyCosts": [SAMPLE_DAILY_COST],
    "alerts": [SAMPLE_ALERT],
}

SAMPLE_EFFICIENCY = {
    "metrics": {
        "cost_per_1k_tokens": 0.015,
        "tokens_per_call": 1200.0,
        "cost_per_call": 0.018,
        "total_tokens": 2000000,
        "total_calls": 1500,
        "total_cost": 150.75,
    },
    "model_utilization": [{"model": "claude-opus", "calls": 800, "cost": 100.0}],
}

SAMPLE_FORECAST = {
    "workspace_id": "ws-123",
    "forecast_days": 30,
    "projected_cost": 450.0,
    "confidence_interval": [400.0, 500.0],
    "trend": "increasing",
    "daily_projections": [{"date": "2026-02-11", "cost": 15.0}],
}

SAMPLE_BUDGET_RESPONSE = {
    "workspace_id": "ws-123",
    "budget": 500.0,
    "daily_limit": 25.0,
}


class TestGetSummary:
    def test_get_summary(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_SUMMARY
        result = api.get_summary()
        assert isinstance(result, CostSummary)
        assert result.total_cost == 150.75
        assert result.budget == 500.0
        assert result.tokens_used == 2000000
        assert result.api_calls == 1500
        assert result.last_updated == "2026-02-10T15:00:00Z"
        mock_client._get.assert_called_once_with(
            "/api/costs", params={"workspace_id": "default", "range": "7d"}
        )

    def test_get_summary_custom_params(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_SUMMARY
        api.get_summary(workspace_id="ws-custom", time_range="30d")
        mock_client._get.assert_called_once_with(
            "/api/costs", params={"workspace_id": "ws-custom", "range": "30d"}
        )

    def test_get_summary_parses_nested(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_SUMMARY
        result = api.get_summary()
        assert len(result.cost_by_provider) == 1
        assert result.cost_by_provider[0].name == "anthropic"
        assert len(result.cost_by_feature) == 1
        assert result.cost_by_feature[0].name == "debate"
        assert len(result.daily_costs) == 1
        assert result.daily_costs[0].date == "2026-02-10"
        assert len(result.alerts) == 1
        assert result.alerts[0].id == "alert-001"

    @pytest.mark.asyncio
    async def test_get_summary_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        result = await api.get_summary_async()
        assert isinstance(result, CostSummary)
        assert result.total_cost == 150.75

    def test_get_summary_empty_data(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_summary()
        assert result.total_cost == 0.0
        assert result.budget == 0.0
        assert result.tokens_used == 0
        assert result.api_calls == 0
        assert result.last_updated == ""
        assert result.cost_by_provider == []
        assert result.cost_by_feature == []
        assert result.daily_costs == []
        assert result.alerts == []


class TestGetBreakdown:
    def test_get_breakdown_default(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {
            "breakdown": [SAMPLE_BREAKDOWN_ITEM],
            "total": 42.5,
        }
        items, total = api.get_breakdown()
        assert len(items) == 1
        assert isinstance(items[0], CostBreakdownItem)
        assert items[0].name == "anthropic"
        assert items[0].cost == 42.5
        assert items[0].percentage == 65.0
        assert total == 42.5
        mock_client._get.assert_called_once_with(
            "/api/costs/breakdown",
            params={"workspace_id": "default", "range": "7d", "group_by": "provider"},
        )

    def test_get_breakdown_custom_group(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"breakdown": [], "total": 0.0}
        api.get_breakdown(group_by="model", time_range="24h", workspace_id="ws-x")
        params = mock_client._get.call_args[1]["params"]
        assert params["group_by"] == "model"
        assert params["range"] == "24h"
        assert params["workspace_id"] == "ws-x"

    def test_get_breakdown_empty(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        items, total = api.get_breakdown()
        assert items == []
        assert total == 0.0

    @pytest.mark.asyncio
    async def test_get_breakdown_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"breakdown": [SAMPLE_BREAKDOWN_ITEM], "total": 42.5}
        )
        items, total = await api.get_breakdown_async()
        assert len(items) == 1
        assert total == 42.5


class TestGetTimeline:
    def test_get_timeline(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "timeline": [SAMPLE_DAILY_COST],
            "total": 12.3,
            "average": 12.3,
        }
        daily, total, average = api.get_timeline()
        assert len(daily) == 1
        assert isinstance(daily[0], DailyCost)
        assert daily[0].date == "2026-02-10"
        assert daily[0].cost == 12.3
        assert daily[0].tokens == 50000
        assert total == 12.3
        assert average == 12.3

    def test_get_timeline_custom_params(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"timeline": [], "total": 0.0, "average": 0.0}
        api.get_timeline(workspace_id="ws-y", time_range="90d")
        mock_client._get.assert_called_once_with(
            "/api/costs/timeline",
            params={"workspace_id": "ws-y", "range": "90d"},
        )

    def test_get_timeline_empty(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        daily, total, average = api.get_timeline()
        assert daily == []
        assert total == 0.0
        assert average == 0.0

    @pytest.mark.asyncio
    async def test_get_timeline_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"timeline": [SAMPLE_DAILY_COST], "total": 12.3, "average": 12.3}
        )
        daily, total, average = await api.get_timeline_async()
        assert len(daily) == 1
        assert total == 12.3


class TestAlerts:
    def test_get_alerts(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"alerts": [SAMPLE_ALERT]}
        alerts = api.get_alerts()
        assert len(alerts) == 1
        assert isinstance(alerts[0], CostAlert)
        assert alerts[0].id == "alert-001"
        assert alerts[0].type == "budget_warning"
        assert alerts[0].severity == "warning"
        assert alerts[0].acknowledged is False
        mock_client._get.assert_called_once_with(
            "/api/costs/alerts", params={"workspace_id": "default"}
        )

    def test_get_alerts_custom_workspace(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"alerts": []}
        api.get_alerts(workspace_id="ws-z")
        mock_client._get.assert_called_once_with(
            "/api/costs/alerts", params={"workspace_id": "ws-z"}
        )

    def test_get_alerts_empty(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        alerts = api.get_alerts()
        assert alerts == []

    @pytest.mark.asyncio
    async def test_get_alerts_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"alerts": [SAMPLE_ALERT]})
        alerts = await api.get_alerts_async()
        assert len(alerts) == 1
        assert alerts[0].id == "alert-001"

    def test_dismiss_alert(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.dismiss_alert("alert-001")
        assert result is True
        mock_client._post.assert_called_once_with(
            "/api/costs/alerts/alert-001/dismiss",
            {"workspace_id": "default"},
        )

    def test_dismiss_alert_custom_workspace(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        api.dismiss_alert("alert-002", workspace_id="ws-abc")
        mock_client._post.assert_called_once_with(
            "/api/costs/alerts/alert-002/dismiss",
            {"workspace_id": "ws-abc"},
        )

    @pytest.mark.asyncio
    async def test_dismiss_alert_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.dismiss_alert_async("alert-001")
        assert result is True


class TestBudget:
    def test_set_budget_minimal(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_BUDGET_RESPONSE
        result = api.set_budget(500.0)
        assert isinstance(result, Budget)
        assert result.workspace_id == "ws-123"
        assert result.monthly_limit == 500.0
        assert result.daily_limit == 25.0
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["budget"] == 500.0
        assert body["workspace_id"] == "default"
        assert "daily_limit" not in body
        assert "name" not in body

    def test_set_budget_with_daily_limit(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_BUDGET_RESPONSE
        api.set_budget(500.0, daily_limit=25.0)
        body = mock_client._post.call_args[0][1]
        assert body["daily_limit"] == 25.0

    def test_set_budget_with_name(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_BUDGET_RESPONSE
        api.set_budget(500.0, name="Q1 Budget")
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Q1 Budget"

    def test_set_budget_custom_workspace(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_BUDGET_RESPONSE
        api.set_budget(1000.0, workspace_id="ws-custom")
        body = mock_client._post.call_args[0][1]
        assert body["workspace_id"] == "ws-custom"

    def test_set_budget_fallback_values(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        result = api.set_budget(300.0, workspace_id="ws-x")
        assert result.workspace_id == "ws-x"
        assert result.monthly_limit == 300.0
        assert result.daily_limit is None

    @pytest.mark.asyncio
    async def test_set_budget_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_BUDGET_RESPONSE)
        result = await api.set_budget_async(500.0, daily_limit=25.0, name="Monthly")
        assert isinstance(result, Budget)
        assert result.monthly_limit == 500.0


class TestRecommendations:
    def test_get_recommendations(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"recommendations": [SAMPLE_RECOMMENDATION]}
        recs = api.get_recommendations()
        assert len(recs) == 1
        assert isinstance(recs[0], CostRecommendation)
        assert recs[0].id == "rec-001"
        assert recs[0].type == "model_downgrade"
        assert recs[0].title == "Switch to smaller model for summaries"
        assert recs[0].estimated_savings == 15.0
        assert recs[0].effort == "low"
        assert recs[0].status == "pending"
        mock_client._get.assert_called_once_with(
            "/api/costs/recommendations", params={"workspace_id": "default"}
        )

    def test_get_recommendations_with_filters(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"recommendations": []}
        api.get_recommendations(status="applied", type_filter="caching")
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "applied"
        assert params["type"] == "caching"

    def test_get_recommendations_no_filters(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"recommendations": []}
        api.get_recommendations()
        params = mock_client._get.call_args[1]["params"]
        assert "status" not in params
        assert "type" not in params

    def test_get_recommendations_empty(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        recs = api.get_recommendations()
        assert recs == []

    @pytest.mark.asyncio
    async def test_get_recommendations_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"recommendations": [SAMPLE_RECOMMENDATION]}
        )
        recs = await api.get_recommendations_async()
        assert len(recs) == 1
        assert recs[0].id == "rec-001"

    def test_apply_recommendation(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        applied = {**SAMPLE_RECOMMENDATION, "status": "applied"}
        mock_client._post.return_value = {"recommendation": applied}
        result = api.apply_recommendation("rec-001")
        assert isinstance(result, CostRecommendation)
        assert result.status == "applied"
        mock_client._post.assert_called_once_with("/api/costs/recommendations/rec-001/apply", {})

    def test_apply_recommendation_with_user(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"recommendation": SAMPLE_RECOMMENDATION}
        api.apply_recommendation("rec-001", user_id="user-42")
        body = mock_client._post.call_args[0][1]
        assert body["user_id"] == "user-42"

    def test_apply_recommendation_flat_response(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_RECOMMENDATION
        result = api.apply_recommendation("rec-001")
        assert result.id == "rec-001"

    @pytest.mark.asyncio
    async def test_apply_recommendation_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"recommendation": SAMPLE_RECOMMENDATION})
        result = await api.apply_recommendation_async("rec-001")
        assert isinstance(result, CostRecommendation)

    def test_dismiss_recommendation(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        result = api.dismiss_recommendation("rec-001")
        assert result is True
        mock_client._post.assert_called_once_with("/api/costs/recommendations/rec-001/dismiss", {})

    @pytest.mark.asyncio
    async def test_dismiss_recommendation_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.dismiss_recommendation_async("rec-001")
        assert result is True


class TestEfficiency:
    def test_get_efficiency(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_EFFICIENCY
        result = api.get_efficiency()
        assert isinstance(result, EfficiencyMetrics)
        assert result.cost_per_1k_tokens == 0.015
        assert result.tokens_per_call == 1200.0
        assert result.cost_per_call == 0.018
        assert result.total_tokens == 2000000
        assert result.total_calls == 1500
        assert result.total_cost == 150.75
        assert len(result.model_utilization) == 1
        mock_client._get.assert_called_once_with(
            "/api/costs/efficiency",
            params={"workspace_id": "default", "range": "7d"},
        )

    def test_get_efficiency_custom_params(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_EFFICIENCY
        api.get_efficiency(workspace_id="ws-eff", time_range="24h")
        mock_client._get.assert_called_once_with(
            "/api/costs/efficiency",
            params={"workspace_id": "ws-eff", "range": "24h"},
        )

    def test_get_efficiency_flat_data(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        flat = {
            "cost_per_1k_tokens": 0.02,
            "tokens_per_call": 800.0,
            "cost_per_call": 0.016,
            "total_tokens": 100000,
            "total_calls": 500,
            "total_cost": 50.0,
        }
        mock_client._get.return_value = flat
        result = api.get_efficiency()
        assert result.cost_per_1k_tokens == 0.02
        assert result.total_tokens == 100000

    @pytest.mark.asyncio
    async def test_get_efficiency_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_EFFICIENCY)
        result = await api.get_efficiency_async()
        assert isinstance(result, EfficiencyMetrics)
        assert result.cost_per_1k_tokens == 0.015


class TestForecast:
    def test_get_forecast(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_FORECAST
        result = api.get_forecast()
        assert isinstance(result, CostForecast)
        assert result.workspace_id == "ws-123"
        assert result.forecast_days == 30
        assert result.projected_cost == 450.0
        assert result.confidence_interval == (400.0, 500.0)
        assert result.trend == "increasing"
        assert len(result.daily_projections) == 1
        mock_client._get.assert_called_once_with(
            "/api/costs/forecast",
            params={"workspace_id": "default", "days": 30},
        )

    def test_get_forecast_custom_params(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_FORECAST
        api.get_forecast(workspace_id="ws-fc", days=90)
        mock_client._get.assert_called_once_with(
            "/api/costs/forecast",
            params={"workspace_id": "ws-fc", "days": 90},
        )

    def test_get_forecast_empty_ci(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {
            "workspace_id": "default",
            "forecast_days": 7,
            "projected_cost": 100.0,
            "confidence_interval": [],
        }
        result = api.get_forecast()
        assert result.confidence_interval == (0.0, 0.0)

    def test_get_forecast_defaults(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_forecast()
        assert result.workspace_id == "default"
        assert result.forecast_days == 30
        assert result.projected_cost == 0.0
        assert result.trend == "stable"
        assert result.daily_projections == []

    @pytest.mark.asyncio
    async def test_get_forecast_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_FORECAST)
        result = await api.get_forecast_async()
        assert result.projected_cost == 450.0


class TestSimulateScenario:
    def test_simulate_scenario(self, api: CostManagementAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FORECAST
        scenario = {"name": "double_usage", "changes": {"scale_factor": 2.0}}
        result = api.simulate_scenario("ws-123", scenario)
        assert isinstance(result, CostForecast)
        assert result.projected_cost == 450.0
        mock_client._post.assert_called_once_with(
            "/api/costs/forecast/simulate",
            {"workspace_id": "ws-123", "scenario": scenario, "days": 30},
        )

    def test_simulate_scenario_custom_days(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_FORECAST
        scenario = {"name": "test"}
        api.simulate_scenario("ws-1", scenario, days=60)
        body = mock_client._post.call_args[0][1]
        assert body["days"] == 60

    @pytest.mark.asyncio
    async def test_simulate_scenario_async(
        self, api: CostManagementAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_FORECAST)
        scenario = {"name": "scale_down"}
        result = await api.simulate_scenario_async("ws-123", scenario, days=14)
        assert isinstance(result, CostForecast)
        assert result.workspace_id == "ws-123"


class TestParseMethods:
    def test_parse_recommendation_with_datetime(self, api: CostManagementAPI) -> None:
        rec = api._parse_recommendation(SAMPLE_RECOMMENDATION)
        assert rec.created_at is not None
        assert rec.created_at.year == 2026
        assert rec.created_at.month == 2

    def test_parse_recommendation_missing_datetime(self, api: CostManagementAPI) -> None:
        data = {**SAMPLE_RECOMMENDATION, "created_at": None}
        rec = api._parse_recommendation(data)
        assert rec.created_at is None

    def test_parse_recommendation_no_datetime_key(self, api: CostManagementAPI) -> None:
        data = {"id": "r1", "type": "caching", "title": "Cache it", "description": "d"}
        rec = api._parse_recommendation(data)
        assert rec.created_at is None

    def test_parse_recommendation_invalid_datetime(self, api: CostManagementAPI) -> None:
        data = {**SAMPLE_RECOMMENDATION, "created_at": "not-a-date"}
        rec = api._parse_recommendation(data)
        assert rec.created_at is None

    def test_parse_recommendation_defaults(self, api: CostManagementAPI) -> None:
        rec = api._parse_recommendation({})
        assert rec.id == ""
        assert rec.type == ""
        assert rec.title == ""
        assert rec.description == ""
        assert rec.estimated_savings == 0.0
        assert rec.effort == "medium"
        assert rec.status == "pending"

    def test_parse_breakdown_item(self, api: CostManagementAPI) -> None:
        item = api._parse_breakdown_item(SAMPLE_BREAKDOWN_ITEM)
        assert item.name == "anthropic"
        assert item.cost == 42.5
        assert item.percentage == 65.0

    def test_parse_breakdown_item_defaults(self, api: CostManagementAPI) -> None:
        item = api._parse_breakdown_item({})
        assert item.name == ""
        assert item.cost == 0.0
        assert item.percentage == 0.0

    def test_parse_daily_cost(self, api: CostManagementAPI) -> None:
        dc = api._parse_daily_cost(SAMPLE_DAILY_COST)
        assert dc.date == "2026-02-10"
        assert dc.cost == 12.3
        assert dc.tokens == 50000

    def test_parse_daily_cost_defaults(self, api: CostManagementAPI) -> None:
        dc = api._parse_daily_cost({})
        assert dc.date == ""
        assert dc.cost == 0.0
        assert dc.tokens == 0

    def test_parse_alert(self, api: CostManagementAPI) -> None:
        alert = api._parse_alert(SAMPLE_ALERT)
        assert alert.id == "alert-001"
        assert alert.type == "budget_warning"
        assert alert.message == "Approaching 80% of monthly budget"
        assert alert.severity == "warning"
        assert alert.acknowledged is False

    def test_parse_alert_defaults(self, api: CostManagementAPI) -> None:
        alert = api._parse_alert({})
        assert alert.id == ""
        assert alert.type == ""
        assert alert.message == ""
        assert alert.severity == "info"
        assert alert.timestamp == ""
        assert alert.acknowledged is False

    def test_parse_efficiency_nested(self, api: CostManagementAPI) -> None:
        result = api._parse_efficiency(SAMPLE_EFFICIENCY)
        assert result.cost_per_1k_tokens == 0.015
        assert len(result.model_utilization) == 1

    def test_parse_forecast_partial_ci(self, api: CostManagementAPI) -> None:
        data = {**SAMPLE_FORECAST, "confidence_interval": [100.0]}
        result = api._parse_forecast(data)
        assert result.confidence_interval == (100.0, 0.0)


class TestDataclasses:
    def test_cost_breakdown_item(self) -> None:
        item = CostBreakdownItem(name="openai", cost=30.0, percentage=45.0)
        assert item.name == "openai"
        assert item.cost == 30.0
        assert item.percentage == 45.0

    def test_daily_cost_defaults(self) -> None:
        dc = DailyCost(date="2026-02-10", cost=10.0)
        assert dc.tokens == 0

    def test_cost_alert_defaults(self) -> None:
        alert = CostAlert(
            id="a1", type="spike_detected", message="Spike", severity="critical", timestamp="t"
        )
        assert alert.acknowledged is False

    def test_cost_summary_defaults(self) -> None:
        summary = CostSummary(
            total_cost=100.0, budget=500.0, tokens_used=1000, api_calls=50, last_updated="now"
        )
        assert summary.cost_by_provider == []
        assert summary.cost_by_feature == []
        assert summary.daily_costs == []
        assert summary.alerts == []

    def test_budget_defaults(self) -> None:
        budget = Budget(workspace_id="ws-1", monthly_limit=500.0)
        assert budget.daily_limit is None
        assert budget.current_spend == 0.0
        assert budget.alert_thresholds == [50, 80, 100]

    def test_budget_custom_thresholds(self) -> None:
        budget = Budget(
            workspace_id="ws-1",
            monthly_limit=1000.0,
            daily_limit=50.0,
            current_spend=200.0,
            alert_thresholds=[25, 50, 75, 100],
        )
        assert budget.daily_limit == 50.0
        assert budget.current_spend == 200.0
        assert budget.alert_thresholds == [25, 50, 75, 100]

    def test_cost_recommendation_defaults(self) -> None:
        rec = CostRecommendation(
            id="r1",
            type="batching",
            title="Batch API calls",
            description="Group calls together",
            estimated_savings=10.0,
            effort="medium",
            status="pending",
        )
        assert rec.created_at is None

    def test_efficiency_metrics_defaults(self) -> None:
        metrics = EfficiencyMetrics(
            cost_per_1k_tokens=0.01,
            tokens_per_call=500.0,
            cost_per_call=0.005,
            total_tokens=100000,
            total_calls=200,
            total_cost=50.0,
        )
        assert metrics.model_utilization == []

    def test_cost_forecast_defaults(self) -> None:
        forecast = CostForecast(workspace_id="ws-1", forecast_days=30, projected_cost=300.0)
        assert forecast.confidence_interval == (0.0, 0.0)
        assert forecast.trend == "stable"
        assert forecast.daily_projections == []
