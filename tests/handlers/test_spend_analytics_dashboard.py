"""
Tests for the SpendAnalyticsDashboardHandler.

Tests the five /api/v1/analytics/spend/* endpoints:
- GET /api/v1/analytics/spend/summary
- GET /api/v1/analytics/spend/trends
- GET /api/v1/analytics/spend/by-agent
- GET /api/v1/analytics/spend/by-decision
- GET /api/v1/analytics/spend/budget
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler():
    """Create a SpendAnalyticsDashboardHandler instance."""
    from aragora.server.handlers.spend_analytics_dashboard import (
        SpendAnalyticsDashboardHandler,
    )

    return SpendAnalyticsDashboardHandler(ctx={})


def _parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_workspace_stats(
    total_cost: str = "10.00",
    api_calls: int = 100,
    agent_costs: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "workspace_id": "ws_123",
        "total_cost_usd": total_cost,
        "total_api_calls": api_calls,
        "total_tokens_in": 50000,
        "total_tokens_out": 25000,
        "cost_by_agent": agent_costs
        if agent_costs is not None
        else {
            "claude": "5.00",
            "gpt-4": "3.00",
            "gemini": "2.00",
        },
        "cost_by_model": {},
    }


def _mock_budget(amount: float = 500.0, spent: float = 125.0):
    """Create a mock budget object."""
    b = MagicMock()
    b.amount_usd = amount
    b.spent_usd = spent
    return b


@pytest.fixture()
def handler():
    return _make_handler()


@pytest.fixture()
def mock_http():
    """Mock HTTP handler for rate limiting (get_client_ip)."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    return h


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRouting:
    """Tests for route matching and handler wiring."""

    def test_routes_list_has_five_entries(self, handler):
        assert len(handler.ROUTES) == 5

    def test_can_handle_summary(self, handler):
        assert handler.can_handle("/api/analytics/spend/summary") is True

    def test_can_handle_versioned_summary(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/summary") is True

    def test_can_handle_trends(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/trends") is True

    def test_can_handle_by_agent(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/by-agent") is True

    def test_can_handle_by_decision(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/by-decision") is True

    def test_can_handle_budget(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/budget") is True

    def test_cannot_handle_unknown(self, handler):
        assert handler.can_handle("/api/v1/analytics/spend/unknown") is False

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


# ---------------------------------------------------------------------------
# Summary endpoint tests
# ---------------------------------------------------------------------------


class TestSummaryEndpoint:
    """Tests for GET /api/v1/analytics/spend/summary."""

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_returns_200(self, mock_tracker_fn, mock_budget_fn, handler, mock_http):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats()
        tracker.get_dashboard_summary.return_value = {}
        mock_tracker_fn.return_value = tracker
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/summary", {}, mock_http
        )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_summary_returns_all_keys(
        self, mock_tracker_fn, mock_budget_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats()
        tracker.get_dashboard_summary.return_value = {}
        mock_tracker_fn.return_value = tracker
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/summary", {}, mock_http
        )
        body = _parse_body(result)
        expected_keys = {
            "total_spend_usd",
            "total_api_calls",
            "total_tokens",
            "budget_limit_usd",
            "budget_spent_usd",
            "utilization_pct",
            "trend_direction",
            "avg_cost_per_decision",
        }
        assert expected_keys.issubset(body.keys())

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_summary_total_spend(
        self, mock_tracker_fn, mock_budget_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats(
            total_cost="42.50"
        )
        tracker.get_dashboard_summary.return_value = {}
        mock_tracker_fn.return_value = tracker
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/summary", {}, mock_http
        )
        body = _parse_body(result)
        assert body["total_spend_usd"] == "42.50"

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_summary_budget_utilization(
        self, mock_tracker_fn, mock_budget_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats()
        tracker.get_dashboard_summary.return_value = {}
        mock_tracker_fn.return_value = tracker

        budget_mgr = MagicMock()
        budget_mgr.get_budgets_for_org.return_value = [
            _mock_budget(500.0, 250.0)
        ]
        mock_budget_fn.return_value = budget_mgr

        result = handler.handle(
            "/api/v1/analytics/spend/summary",
            {"org_id": "org_1"},
            mock_http,
        )
        body = _parse_body(result)
        assert body["utilization_pct"] == 50.0
        assert body["budget_limit_usd"] == 500.0
        assert body["budget_spent_usd"] == 250.0

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_summary_no_tracker(
        self, mock_tracker_fn, mock_budget_fn, handler, mock_http
    ):
        """When tracker is unavailable, defaults to zero values."""
        mock_tracker_fn.return_value = None
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/summary", {}, mock_http
        )
        body = _parse_body(result)
        assert body["total_spend_usd"] == "0.00"
        assert body["total_api_calls"] == 0
        assert body["total_tokens"] == 0

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_summary_trend_increasing(
        self, mock_tracker_fn, mock_budget_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats(
            total_cost="10.00"
        )
        tracker.get_dashboard_summary.return_value = {
            "projections": {"projected_monthly_usd": "20.00"}
        }
        mock_tracker_fn.return_value = tracker
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/summary", {}, mock_http
        )
        body = _parse_body(result)
        assert body["trend_direction"] == "increasing"


# ---------------------------------------------------------------------------
# Trends endpoint tests
# ---------------------------------------------------------------------------


class TestTrendsEndpoint:
    """Tests for GET /api/v1/analytics/spend/trends."""

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_trends_returns_200(self, mock_budget_fn, handler, mock_http):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/trends", {}, mock_http
        )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_trends_default_params(self, mock_budget_fn, handler, mock_http):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/trends", {}, mock_http
        )
        body = _parse_body(result)
        assert body["org_id"] == "default"
        assert body["period"] == "daily"
        assert body["days"] == 30

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_trends_custom_period(self, mock_budget_fn, handler, mock_http):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/trends",
            {"period": "weekly", "days": "7"},
            mock_http,
        )
        body = _parse_body(result)
        assert body["period"] == "weekly"
        assert body["days"] == 7

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_trends_invalid_period_defaults(
        self, mock_budget_fn, handler, mock_http
    ):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/trends",
            {"period": "hourly"},
            mock_http,
        )
        body = _parse_body(result)
        assert body["period"] == "daily"

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_trends_days_clamped(self, mock_budget_fn, handler, mock_http):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/trends",
            {"days": "999"},
            mock_http,
        )
        body = _parse_body(result)
        assert body["days"] == 365  # max clamp


# ---------------------------------------------------------------------------
# By-agent endpoint tests
# ---------------------------------------------------------------------------


class TestByAgentEndpoint:
    """Tests for GET /api/v1/analytics/spend/by-agent."""

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_agent_returns_200(self, mock_tracker_fn, handler, mock_http):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-agent", {}, mock_http
        )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_agent_returns_agents(self, mock_tracker_fn, handler, mock_http):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-agent", {}, mock_http
        )
        body = _parse_body(result)
        assert len(body["agents"]) == 3
        # Should be sorted by cost descending
        assert body["agents"][0]["agent_name"] == "claude"
        assert body["agents"][0]["cost_usd"] == "5.00"

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_agent_percentages(self, mock_tracker_fn, handler, mock_http):
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _make_workspace_stats(
            total_cost="10.00",
            agent_costs={"agent_a": "7.50", "agent_b": "2.50"},
        )
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-agent", {}, mock_http
        )
        body = _parse_body(result)
        assert body["agents"][0]["percentage"] == 75.0
        assert body["agents"][1]["percentage"] == 25.0

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_agent_no_tracker(self, mock_tracker_fn, handler, mock_http):
        mock_tracker_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/by-agent", {}, mock_http
        )
        body = _parse_body(result)
        assert body["agents"] == []
        assert body["total_usd"] == "0"


# ---------------------------------------------------------------------------
# By-decision endpoint tests
# ---------------------------------------------------------------------------


class TestByDecisionEndpoint:
    """Tests for GET /api/v1/analytics/spend/by-decision."""

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_decision_returns_200(
        self, mock_tracker_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker._debate_costs = {}
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-decision", {}, mock_http
        )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_decision_returns_costs(
        self, mock_tracker_fn, handler, mock_http
    ):
        tracker = MagicMock()
        tracker._debate_costs = {
            "debate_1": Decimal("5.00"),
            "debate_2": Decimal("3.00"),
            "debate_3": Decimal("1.00"),
        }
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-decision", {}, mock_http
        )
        body = _parse_body(result)
        assert body["count"] == 3
        # Should be sorted by cost descending
        assert body["decisions"][0]["debate_id"] == "debate_1"

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_decision_limit(self, mock_tracker_fn, handler, mock_http):
        tracker = MagicMock()
        tracker._debate_costs = {
            f"debate_{i}": Decimal(f"{i}.00") for i in range(10)
        }
        mock_tracker_fn.return_value = tracker

        result = handler.handle(
            "/api/v1/analytics/spend/by-decision",
            {"limit": "3"},
            mock_http,
        )
        body = _parse_body(result)
        assert body["count"] == 3

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_cost_tracker"
    )
    def test_by_decision_no_tracker(
        self, mock_tracker_fn, handler, mock_http
    ):
        mock_tracker_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/by-decision", {}, mock_http
        )
        body = _parse_body(result)
        assert body["decisions"] == []
        assert body["count"] == 0


# ---------------------------------------------------------------------------
# Budget endpoint tests
# ---------------------------------------------------------------------------


class TestBudgetEndpoint:
    """Tests for GET /api/v1/analytics/spend/budget."""

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_budget_returns_200(self, mock_budget_fn, handler, mock_http):
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/budget", {}, mock_http
        )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_budget_no_manager(self, mock_budget_fn, handler, mock_http):
        """When budget manager is unavailable, returns empty structure."""
        mock_budget_fn.return_value = None

        result = handler.handle(
            "/api/v1/analytics/spend/budget", {}, mock_http
        )
        body = _parse_body(result)
        assert body["budgets"] == []
        assert body["total_budget_usd"] == 0.0
        assert body["total_spent_usd"] == 0.0
        assert body["forecast_exhaustion_days"] is None

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_budget_with_data(self, mock_budget_fn, handler, mock_http):
        budget_mgr = MagicMock()
        budget_mgr.get_summary.return_value = {
            "total_budget_usd": 1000,
            "total_spent_usd": 400,
            "total_remaining_usd": 600,
            "budgets": [
                {
                    "status": "active",
                    "period_start": 1700000000,
                    "spent_usd": 400,
                }
            ],
        }
        mock_budget_fn.return_value = budget_mgr

        result = handler.handle(
            "/api/v1/analytics/spend/budget", {}, mock_http
        )
        body = _parse_body(result)
        assert body["total_budget_usd"] == 1000
        assert body["total_spent_usd"] == 400
        assert body["total_remaining_usd"] == 600
        assert body["utilization_pct"] == 40.0

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_budget_utilization_zero_budget(
        self, mock_budget_fn, handler, mock_http
    ):
        budget_mgr = MagicMock()
        budget_mgr.get_summary.return_value = {
            "total_budget_usd": 0,
            "total_spent_usd": 0,
            "total_remaining_usd": 0,
            "budgets": [],
        }
        mock_budget_fn.return_value = budget_mgr

        result = handler.handle(
            "/api/v1/analytics/spend/budget", {}, mock_http
        )
        body = _parse_body(result)
        assert body["utilization_pct"] == 0.0

    @patch(
        "aragora.server.handlers.spend_analytics_dashboard._get_budget_manager"
    )
    def test_budget_org_id_param(self, mock_budget_fn, handler, mock_http):
        budget_mgr = MagicMock()
        budget_mgr.get_summary.return_value = {
            "total_budget_usd": 0,
            "total_spent_usd": 0,
            "total_remaining_usd": 0,
            "budgets": [],
        }
        mock_budget_fn.return_value = budget_mgr

        handler.handle(
            "/api/v1/analytics/spend/budget",
            {"org_id": "org_abc"},
            mock_http,
        )
        budget_mgr.get_summary.assert_called_once_with("org_abc")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests that the handler is properly registered."""

    def test_handler_in_analytics_registry(self):
        from aragora.server.handler_registry.analytics import (
            ANALYTICS_HANDLER_REGISTRY,
        )

        names = [name for name, _ in ANALYTICS_HANDLER_REGISTRY]
        assert "_spend_analytics_dashboard_handler" in names

    def test_handler_importable(self):
        from aragora.server.handlers.spend_analytics_dashboard import (
            SpendAnalyticsDashboardHandler,
        )

        assert SpendAnalyticsDashboardHandler is not None

    def test_handler_in_analytics_all(self):
        from aragora.server.handler_registry import analytics

        assert "SpendAnalyticsDashboardHandler" in analytics.__all__
