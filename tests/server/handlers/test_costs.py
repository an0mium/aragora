"""
Tests for the cost visibility API handler.

Tests cover:
- Data models (CostEntry, BudgetAlert, CostSummary)
- Cost summary generation (with and without tracker)
- Cost breakdown endpoints
- Timeline data
- Budget alerts
- Budget setting
- Recommendations (get, apply, dismiss)
- Efficiency metrics
- Forecasting and simulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from aragora.server.handlers.costs import (
    BudgetAlert,
    CostEntry,
    CostHandler,
    CostSummary,
    _generate_mock_summary,
    _get_active_alerts,
    get_cost_summary,
    record_cost,
    register_routes,
)


# =============================================================================
# Data Model Tests
# =============================================================================


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_cost_entry_creation(self):
        """CostEntry can be created with all fields."""
        now = datetime.now(timezone.utc)
        entry = CostEntry(
            timestamp=now,
            provider="anthropic",
            feature="debate",
            tokens_input=1000,
            tokens_output=500,
            cost=0.015,
            model="claude-3-opus",
            workspace_id="ws_123",
            user_id="user_456",
        )

        assert entry.timestamp == now
        assert entry.provider == "anthropic"
        assert entry.feature == "debate"
        assert entry.tokens_input == 1000
        assert entry.tokens_output == 500
        assert entry.cost == 0.015
        assert entry.model == "claude-3-opus"
        assert entry.workspace_id == "ws_123"
        assert entry.user_id == "user_456"

    def test_cost_entry_optional_user_id(self):
        """CostEntry user_id is optional."""
        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            provider="openai",
            feature="code_review",
            tokens_input=500,
            tokens_output=200,
            cost=0.008,
            model="gpt-4",
            workspace_id="ws_123",
        )

        assert entry.user_id is None


class TestBudgetAlert:
    """Tests for BudgetAlert dataclass."""

    def test_budget_alert_creation(self):
        """BudgetAlert can be created with all fields."""
        now = datetime.now(timezone.utc)
        alert = BudgetAlert(
            id="alert_123",
            type="budget_warning",
            message="Budget at 80%",
            severity="warning",
            timestamp=now,
            acknowledged=False,
        )

        assert alert.id == "alert_123"
        assert alert.type == "budget_warning"
        assert alert.message == "Budget at 80%"
        assert alert.severity == "warning"
        assert alert.timestamp == now
        assert alert.acknowledged is False

    def test_budget_alert_default_acknowledged(self):
        """BudgetAlert acknowledged defaults to False."""
        alert = BudgetAlert(
            id="alert_456",
            type="spike_detected",
            message="Cost spike",
            severity="info",
            timestamp=datetime.now(timezone.utc),
        )

        assert alert.acknowledged is False


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_cost_summary_creation(self):
        """CostSummary can be created with all fields."""
        now = datetime.now(timezone.utc)
        summary = CostSummary(
            total_cost=150.50,
            budget=500.00,
            tokens_used=5000000,
            api_calls=15000,
            last_updated=now,
            cost_by_provider=[{"name": "Anthropic", "cost": 100.00, "percentage": 66.4}],
            cost_by_feature=[{"name": "Debates", "cost": 80.00, "percentage": 53.2}],
            daily_costs=[{"date": "2026-01-20", "cost": 20.00, "tokens": 500000}],
            alerts=[{"id": "1", "type": "budget_warning", "severity": "warning"}],
        )

        assert summary.total_cost == 150.50
        assert summary.budget == 500.00
        assert summary.tokens_used == 5000000
        assert summary.api_calls == 15000
        assert len(summary.cost_by_provider) == 1
        assert len(summary.cost_by_feature) == 1
        assert len(summary.daily_costs) == 1
        assert len(summary.alerts) == 1

    def test_cost_summary_default_lists(self):
        """CostSummary lists default to empty."""
        summary = CostSummary(
            total_cost=0.0,
            budget=500.00,
            tokens_used=0,
            api_calls=0,
            last_updated=datetime.now(timezone.utc),
        )

        assert summary.cost_by_provider == []
        assert summary.cost_by_feature == []
        assert summary.daily_costs == []
        assert summary.alerts == []


# =============================================================================
# Mock Summary Generation Tests
# =============================================================================


class TestMockSummaryGeneration:
    """Tests for mock summary generation."""

    def test_generate_mock_summary_7d(self):
        """Mock summary for 7 days."""
        summary = _generate_mock_summary("7d")

        assert summary.budget == 500.00
        assert len(summary.daily_costs) == 7
        assert len(summary.cost_by_provider) == 4
        assert len(summary.cost_by_feature) == 4
        assert len(summary.alerts) == 2

    def test_generate_mock_summary_24h(self):
        """Mock summary for 24 hours."""
        summary = _generate_mock_summary("24h")

        assert len(summary.daily_costs) == 1

    def test_generate_mock_summary_30d(self):
        """Mock summary for 30 days."""
        summary = _generate_mock_summary("30d")

        assert len(summary.daily_costs) == 30

    def test_generate_mock_summary_90d(self):
        """Mock summary for 90 days."""
        summary = _generate_mock_summary("90d")

        assert len(summary.daily_costs) == 90

    def test_generate_mock_summary_invalid_range(self):
        """Invalid range defaults to 7 days."""
        summary = _generate_mock_summary("invalid")

        assert len(summary.daily_costs) == 7

    def test_mock_summary_provider_percentages(self):
        """Provider percentages sum to 100."""
        summary = _generate_mock_summary("7d")

        total_pct = sum(p["percentage"] for p in summary.cost_by_provider)
        assert abs(total_pct - 100.0) < 0.1

    def test_mock_summary_feature_percentages(self):
        """Feature percentages sum to 100."""
        summary = _generate_mock_summary("7d")

        total_pct = sum(f["percentage"] for f in summary.cost_by_feature)
        assert abs(total_pct - 100.0) < 0.1


# =============================================================================
# Record Cost Tests
# =============================================================================


class TestRecordCost:
    """Tests for recording costs."""

    def test_record_cost_no_tracker(self):
        """Record cost when tracker is unavailable."""
        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=None):
            # Should not raise
            record_cost(
                provider="anthropic",
                feature="debate",
                tokens_input=1000,
                tokens_output=500,
                cost=0.015,
                model="claude-3-opus",
                workspace_id="ws_123",
            )

    def test_record_cost_with_tracker(self):
        """Record cost when tracker is available."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            # Just verify no exception is raised
            record_cost(
                provider="openai",
                feature="code_review",
                tokens_input=500,
                tokens_output=200,
                cost=0.008,
                model="gpt-4",
                workspace_id="ws_456",
                user_id="user_789",
            )
            # If we get here without exception, tracking was attempted


# =============================================================================
# Get Cost Summary Tests
# =============================================================================


class TestGetCostSummary:
    """Tests for getting cost summary."""

    @pytest.mark.asyncio
    async def test_get_cost_summary_no_tracker(self):
        """Get summary when tracker unavailable returns mock."""
        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=None):
            summary = await get_cost_summary(workspace_id="test", time_range="7d")

            assert summary.budget == 500.00
            assert len(summary.daily_costs) == 7

    @pytest.mark.asyncio
    async def test_get_cost_summary_with_tracker_empty_data(self):
        """Get summary with tracker but no data returns mock."""
        mock_report = MagicMock()
        mock_report.total_cost_usd = Decimal("0")
        mock_report.cost_over_time = []
        mock_report.cost_by_provider = {}
        mock_report.cost_by_operation = {}
        mock_report.total_tokens_in = 0
        mock_report.total_tokens_out = 0
        mock_report.total_api_calls = 0

        mock_tracker = MagicMock()
        mock_tracker.generate_report = AsyncMock(return_value=mock_report)
        mock_tracker.get_budget = MagicMock(return_value=None)

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            summary = await get_cost_summary(workspace_id="test", time_range="7d")

            # Should fall back to mock when no real data
            assert summary.budget == 500.00

    @pytest.mark.asyncio
    async def test_get_cost_summary_with_tracker_real_data(self):
        """Get summary with tracker and real data."""
        mock_report = MagicMock()
        mock_report.total_cost_usd = Decimal("150.50")
        mock_report.cost_over_time = [{"date": "2026-01-20", "cost": 20.0}]
        mock_report.cost_by_provider = {"anthropic": Decimal("100.00"), "openai": Decimal("50.50")}
        mock_report.cost_by_operation = {"debate": Decimal("80.00"), "review": Decimal("70.50")}
        mock_report.total_tokens_in = 1000000
        mock_report.total_tokens_out = 500000
        mock_report.total_api_calls = 5000

        mock_budget = MagicMock()
        mock_budget.monthly_limit_usd = Decimal("1000.00")

        mock_tracker = MagicMock()
        mock_tracker.generate_report = AsyncMock(return_value=mock_report)
        mock_tracker.get_budget = MagicMock(return_value=mock_budget)

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            summary = await get_cost_summary(workspace_id="test", time_range="7d")

            assert summary.total_cost == 150.50
            assert summary.budget == 1000.0
            assert summary.tokens_used == 1500000
            assert summary.api_calls == 5000


# =============================================================================
# Active Alerts Tests
# =============================================================================


class TestGetActiveAlerts:
    """Tests for getting active alerts."""

    def test_get_active_alerts_no_budget(self):
        """No alerts when no budget configured."""
        mock_tracker = MagicMock()
        mock_tracker.get_budget = MagicMock(return_value=None)

        alerts = _get_active_alerts(mock_tracker, "ws_123")

        assert alerts == []

    def test_get_active_alerts_no_alert_level(self):
        """No alerts when budget is within limits."""
        mock_budget = MagicMock()
        mock_budget.check_alert_level = MagicMock(return_value=None)

        mock_tracker = MagicMock()
        mock_tracker.get_budget = MagicMock(return_value=mock_budget)

        alerts = _get_active_alerts(mock_tracker, "ws_123")

        assert alerts == []

    def test_get_active_alerts_warning_level(self):
        """Alert generated when budget at warning level."""
        # Import the actual enum to use as return value
        try:
            from aragora.billing.cost_tracker import BudgetAlertLevel

            mock_budget = MagicMock()
            mock_budget.check_alert_level = MagicMock(return_value=BudgetAlertLevel.WARNING)
            mock_budget.current_monthly_spend = Decimal("400.00")
            mock_budget.monthly_limit_usd = Decimal("500.00")

            mock_tracker = MagicMock()
            mock_tracker.get_budget = MagicMock(return_value=mock_budget)

            alerts = _get_active_alerts(mock_tracker, "ws_123")

            assert len(alerts) == 1
            assert alerts[0]["type"] == "budget_warning"
            assert "80.0%" in alerts[0]["message"]
        except ImportError:
            pytest.skip("BudgetAlertLevel not available")


# =============================================================================
# HTTP Handler Tests
# =============================================================================


class TestCostHandlerHTTP:
    """Tests for CostHandler HTTP endpoints."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return CostHandler()

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock(spec=web.Request)
        request.query = {}
        request.match_info = {}
        return request

    @pytest.mark.asyncio
    async def test_handle_get_costs(self, handler, mock_request):
        """GET /api/costs returns cost dashboard."""
        mock_request.query = {"range": "7d", "workspace_id": "test"}

        with patch("aragora.server.handlers.costs.get_cost_summary") as mock_summary:
            mock_summary.return_value = CostSummary(
                total_cost=150.50,
                budget=500.00,
                tokens_used=1000000,
                api_calls=5000,
                last_updated=datetime.now(timezone.utc),
                cost_by_provider=[{"name": "Anthropic", "cost": 100.00}],
                cost_by_feature=[{"name": "Debates", "cost": 80.00}],
                daily_costs=[{"date": "2026-01-20", "cost": 20.00}],
                alerts=[],
            )

            response = await handler.handle_get_costs(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["totalCost"] == 150.50
            assert data["budget"] == 500.00

    @pytest.mark.asyncio
    async def test_handle_get_costs_error(self, handler, mock_request):
        """GET /api/costs handles errors."""
        with patch("aragora.server.handlers.costs.get_cost_summary") as mock_summary:
            mock_summary.side_effect = Exception("Database error")

            response = await handler.handle_get_costs(mock_request)

            assert response.status == 500
            data = json.loads(response.body)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_handle_get_breakdown_by_provider(self, handler, mock_request):
        """GET /api/costs/breakdown groups by provider."""
        mock_request.query = {"group_by": "provider"}

        with patch("aragora.server.handlers.costs.get_cost_summary") as mock_summary:
            mock_summary.return_value = CostSummary(
                total_cost=150.50,
                budget=500.00,
                tokens_used=0,
                api_calls=0,
                last_updated=datetime.now(timezone.utc),
                cost_by_provider=[{"name": "Anthropic", "cost": 100.00}],
                cost_by_feature=[{"name": "Debates", "cost": 80.00}],
            )

            response = await handler.handle_get_breakdown(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["groupBy"] == "provider"
            assert len(data["breakdown"]) == 1
            assert data["breakdown"][0]["name"] == "Anthropic"

    @pytest.mark.asyncio
    async def test_handle_get_breakdown_by_feature(self, handler, mock_request):
        """GET /api/costs/breakdown groups by feature."""
        mock_request.query = {"group_by": "feature"}

        with patch("aragora.server.handlers.costs.get_cost_summary") as mock_summary:
            mock_summary.return_value = CostSummary(
                total_cost=150.50,
                budget=500.00,
                tokens_used=0,
                api_calls=0,
                last_updated=datetime.now(timezone.utc),
                cost_by_provider=[{"name": "Anthropic", "cost": 100.00}],
                cost_by_feature=[{"name": "Debates", "cost": 80.00}],
            )

            response = await handler.handle_get_breakdown(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["groupBy"] == "feature"
            assert data["breakdown"][0]["name"] == "Debates"

    @pytest.mark.asyncio
    async def test_handle_get_timeline(self, handler, mock_request):
        """GET /api/costs/timeline returns timeline data."""
        with patch("aragora.server.handlers.costs.get_cost_summary") as mock_summary:
            mock_summary.return_value = CostSummary(
                total_cost=140.00,
                budget=500.00,
                tokens_used=0,
                api_calls=0,
                last_updated=datetime.now(timezone.utc),
                daily_costs=[
                    {"date": "2026-01-20", "cost": 20.00},
                    {"date": "2026-01-21", "cost": 25.00},
                ],
            )

            response = await handler.handle_get_timeline(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert len(data["timeline"]) == 2
            assert data["total"] == 140.00
            assert data["average"] == 70.00

    @pytest.mark.asyncio
    async def test_handle_get_alerts_with_tracker(self, handler, mock_request):
        """GET /api/costs/alerts returns alerts from tracker."""
        mock_tracker = MagicMock()
        mock_tracker.query_km_workspace_alerts = MagicMock(return_value=[])

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            with patch("aragora.server.handlers.costs._get_active_alerts", return_value=[]):
                response = await handler.handle_get_alerts(mock_request)

                assert response.status == 200
                data = json.loads(response.body)
                assert "alerts" in data

    @pytest.mark.asyncio
    async def test_handle_get_alerts_no_tracker(self, handler, mock_request):
        """GET /api/costs/alerts returns empty when no tracker."""
        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=None):
            response = await handler.handle_get_alerts(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["alerts"] == []

    @pytest.mark.asyncio
    async def test_handle_set_budget_success(self, handler, mock_request):
        """POST /api/costs/budget sets budget."""
        mock_request.json = AsyncMock(
            return_value={"budget": 1000.00, "workspace_id": "test", "daily_limit": 50.00}
        )

        mock_tracker = MagicMock()
        mock_tracker.set_budget = MagicMock()

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            response = await handler.handle_set_budget(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["success"] is True
            assert data["budget"] == 1000.00

    @pytest.mark.asyncio
    async def test_handle_set_budget_invalid(self, handler, mock_request):
        """POST /api/costs/budget rejects invalid budget."""
        mock_request.json = AsyncMock(return_value={"budget": -100})

        response = await handler.handle_set_budget(mock_request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_handle_set_budget_missing(self, handler, mock_request):
        """POST /api/costs/budget rejects missing budget."""
        mock_request.json = AsyncMock(return_value={"workspace_id": "test"})

        response = await handler.handle_set_budget(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_dismiss_alert(self, handler, mock_request):
        """POST /api/costs/alerts/{id}/dismiss dismisses alert."""
        mock_request.match_info = {"alert_id": "alert_123"}

        response = await handler.handle_dismiss_alert(mock_request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["alert_id"] == "alert_123"
        assert data["dismissed"] is True


# =============================================================================
# Recommendations Handler Tests
# =============================================================================


class TestRecommendationsHandler:
    """Tests for recommendation endpoints."""

    @pytest.fixture
    def handler(self):
        return CostHandler()

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=web.Request)
        request.query = {}
        request.match_info = {}
        return request

    @pytest.mark.asyncio
    async def test_handle_get_recommendations(self, handler, mock_request):
        """GET /api/costs/recommendations returns recommendations."""
        mock_request.query = {"workspace_id": "test"}

        mock_rec = MagicMock()
        mock_rec.to_dict = MagicMock(return_value={"id": "rec_1", "type": "model_downgrade"})

        mock_summary = MagicMock()
        mock_summary.to_dict = MagicMock(return_value={"total_savings": 100.00})

        mock_optimizer = MagicMock()
        mock_optimizer.get_workspace_recommendations = MagicMock(return_value=[mock_rec])
        mock_optimizer.analyze_workspace = AsyncMock()
        mock_optimizer.get_summary = MagicMock(return_value=mock_summary)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_get_recommendations(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert len(data["recommendations"]) == 1

    @pytest.mark.asyncio
    async def test_handle_get_recommendation_found(self, handler, mock_request):
        """GET /api/costs/recommendations/{id} returns recommendation."""
        mock_request.match_info = {"recommendation_id": "rec_123"}

        mock_rec = MagicMock()
        mock_rec.to_dict = MagicMock(
            return_value={"id": "rec_123", "type": "caching", "savings": 50.00}
        )

        mock_optimizer = MagicMock()
        mock_optimizer.get_recommendation = MagicMock(return_value=mock_rec)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_get_recommendation(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["id"] == "rec_123"

    @pytest.mark.asyncio
    async def test_handle_get_recommendation_not_found(self, handler, mock_request):
        """GET /api/costs/recommendations/{id} returns 404 when not found."""
        mock_request.match_info = {"recommendation_id": "nonexistent"}

        mock_optimizer = MagicMock()
        mock_optimizer.get_recommendation = MagicMock(return_value=None)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_get_recommendation(mock_request)

            assert response.status == 404

    @pytest.mark.asyncio
    async def test_handle_apply_recommendation_success(self, handler, mock_request):
        """POST /api/costs/recommendations/{id}/apply applies recommendation."""
        mock_request.match_info = {"recommendation_id": "rec_123"}
        mock_request.json = AsyncMock(return_value={"user_id": "user_456"})

        mock_rec = MagicMock()
        mock_rec.to_dict = MagicMock(return_value={"id": "rec_123", "status": "applied"})

        mock_optimizer = MagicMock()
        mock_optimizer.apply_recommendation = MagicMock(return_value=True)
        mock_optimizer.get_recommendation = MagicMock(return_value=mock_rec)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_apply_recommendation(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_handle_apply_recommendation_not_found(self, handler, mock_request):
        """POST /api/costs/recommendations/{id}/apply returns 404 when not found."""
        mock_request.match_info = {"recommendation_id": "nonexistent"}
        mock_request.json = AsyncMock(return_value={})

        mock_optimizer = MagicMock()
        mock_optimizer.apply_recommendation = MagicMock(return_value=False)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_apply_recommendation(mock_request)

            assert response.status == 404

    @pytest.mark.asyncio
    async def test_handle_dismiss_recommendation_success(self, handler, mock_request):
        """POST /api/costs/recommendations/{id}/dismiss dismisses recommendation."""
        mock_request.match_info = {"recommendation_id": "rec_123"}

        mock_optimizer = MagicMock()
        mock_optimizer.dismiss_recommendation = MagicMock(return_value=True)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_dismiss_recommendation(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["success"] is True
            assert data["dismissed"] is True

    @pytest.mark.asyncio
    async def test_handle_dismiss_recommendation_not_found(self, handler, mock_request):
        """POST /api/costs/recommendations/{id}/dismiss returns 404 when not found."""
        mock_request.match_info = {"recommendation_id": "nonexistent"}

        mock_optimizer = MagicMock()
        mock_optimizer.dismiss_recommendation = MagicMock(return_value=False)

        with patch("aragora.billing.optimizer.get_cost_optimizer", return_value=mock_optimizer):
            response = await handler.handle_dismiss_recommendation(mock_request)

            assert response.status == 404


# =============================================================================
# Efficiency Handler Tests
# =============================================================================


class TestEfficiencyHandler:
    """Tests for efficiency endpoint."""

    @pytest.fixture
    def handler(self):
        return CostHandler()

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=web.Request)
        request.query = {}
        return request

    @pytest.mark.asyncio
    async def test_handle_get_efficiency_success(self, handler, mock_request):
        """GET /api/costs/efficiency returns metrics."""
        mock_request.query = {"workspace_id": "test", "range": "7d"}

        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats = MagicMock(
            return_value={
                "total_tokens_in": 1000000,
                "total_tokens_out": 500000,
                "total_api_calls": 5000,
                "total_cost_usd": Decimal("150.00"),
                "cost_by_model": {
                    "claude-3-opus": Decimal("100.00"),
                    "gpt-4": Decimal("50.00"),
                },
            }
        )

        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=mock_tracker):
            response = await handler.handle_get_efficiency(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["workspace_id"] == "test"
            assert "metrics" in data
            assert data["metrics"]["total_tokens"] == 1500000
            assert data["metrics"]["total_calls"] == 5000

    @pytest.mark.asyncio
    async def test_handle_get_efficiency_no_tracker(self, handler, mock_request):
        """GET /api/costs/efficiency returns 503 when no tracker."""
        with patch("aragora.server.handlers.costs._get_cost_tracker", return_value=None):
            response = await handler.handle_get_efficiency(mock_request)

            assert response.status == 503
            data = json.loads(response.body)
            assert "error" in data


# =============================================================================
# Forecast Handler Tests
# =============================================================================


class TestForecastHandler:
    """Tests for forecast endpoints."""

    @pytest.fixture
    def handler(self):
        return CostHandler()

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=web.Request)
        request.query = {}
        return request

    @pytest.mark.asyncio
    async def test_handle_get_forecast(self, handler, mock_request):
        """GET /api/costs/forecast returns forecast."""
        mock_request.query = {"workspace_id": "test", "days": "30"}

        mock_report = MagicMock()
        mock_report.to_dict = MagicMock(
            return_value={
                "forecast_days": 30,
                "predicted_total": 450.00,
                "confidence": 0.85,
            }
        )

        mock_forecaster = MagicMock()
        mock_forecaster.generate_forecast = AsyncMock(return_value=mock_report)

        with patch("aragora.billing.forecaster.get_cost_forecaster", return_value=mock_forecaster):
            response = await handler.handle_get_forecast(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["forecast_days"] == 30

    @pytest.mark.asyncio
    async def test_handle_simulate_forecast(self, handler, mock_request):
        """POST /api/costs/forecast/simulate runs simulation."""
        mock_request.json = AsyncMock(
            return_value={
                "workspace_id": "test",
                "scenario": {
                    "name": "Model Downgrade",
                    "description": "Switch to cheaper models",
                    "changes": {"model_cost_multiplier": 0.5},
                },
                "days": 30,
            }
        )

        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(
            return_value={
                "scenario_name": "Model Downgrade",
                "original_cost": 300.00,
                "simulated_cost": 150.00,
                "savings": 150.00,
            }
        )

        mock_forecaster = MagicMock()
        mock_forecaster.simulate_scenario = AsyncMock(return_value=mock_result)

        with patch("aragora.billing.forecaster.get_cost_forecaster", return_value=mock_forecaster):
            response = await handler.handle_simulate_forecast(mock_request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["scenario_name"] == "Model Downgrade"
            assert data["savings"] == 150.00


# =============================================================================
# Route Registration Tests
# =============================================================================


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_routes(self):
        """All routes are registered."""
        app = web.Application()

        register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]

        assert "/api/costs" in routes
        assert "/api/costs/breakdown" in routes
        assert "/api/costs/timeline" in routes
        assert "/api/costs/alerts" in routes
        assert "/api/costs/budget" in routes
        assert "/api/costs/alerts/{alert_id}/dismiss" in routes
        assert "/api/costs/recommendations" in routes
        assert "/api/costs/recommendations/{recommendation_id}" in routes
        assert "/api/costs/recommendations/{recommendation_id}/apply" in routes
        assert "/api/costs/recommendations/{recommendation_id}/dismiss" in routes
        assert "/api/costs/efficiency" in routes
        assert "/api/costs/forecast" in routes
        assert "/api/costs/forecast/simulate" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
