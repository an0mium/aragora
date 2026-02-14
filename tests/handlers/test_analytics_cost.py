"""
Tests for cost breakdown endpoint in analytics dashboard.

Tests the /api/analytics/cost/breakdown endpoint added to UsageAnalyticsMixin
which provides total spend, per-agent cost breakdown, and budget utilization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeBudget:
    monthly_limit_usd: Decimal = Decimal("500")
    current_monthly_spend: Decimal = Decimal("125")


def _make_workspace_stats(
    total_cost="10.00",
    api_calls=100,
    agent_costs=None,
):
    return {
        "workspace_id": "ws_123",
        "total_cost_usd": total_cost,
        "total_api_calls": api_calls,
        "total_tokens_in": 50000,
        "total_tokens_out": 25000,
        "cost_by_agent": agent_costs or {
            "claude": "5.00",
            "gpt-4": "3.00",
            "gemini": "2.00",
        },
        "cost_by_model": {},
    }


def _make_mixin_instance():
    """Create a UsageAnalyticsMixin instance for testing."""
    from aragora.server.handlers.analytics_dashboard.usage import UsageAnalyticsMixin

    class TestHandler(UsageAnalyticsMixin):
        pass

    return TestHandler()


def _mock_handler():
    """Create a mock HTTP handler with headers for auth bypass."""
    h = MagicMock()
    h.headers = {"Authorization": "Bearer test-token"}
    return h


def _parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCostBreakdownEndpoint:
    """Tests for _get_cost_breakdown method."""

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_returns_200(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        assert result.status_code == 200

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_missing_workspace_id(self, mock_get_tracker):
        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({})
        assert result.status_code == 400
        body = _parse_body(result)
        assert "workspace_id" in body.get("error", "")

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_total_spend_returned(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats(total_cost="42.50")
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["total_spend_usd"] == "42.50"

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_agent_costs_returned(self, mock_get_tracker):
        agent_costs = {"claude": "15.00", "gpt-4": "10.00"}
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats(
            agent_costs=agent_costs
        )
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["agent_costs"]["claude"] == "15.00"
        assert body["agent_costs"]["gpt-4"] == "10.00"

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_empty_agent_costs(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats(agent_costs={})
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["agent_costs"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_budget_utilization(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.return_value = FakeBudget()
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        budget = body["budget"]
        assert budget["monthly_limit_usd"] == 500.0
        assert budget["current_spend_usd"] == 125.0
        assert budget["remaining_usd"] == 375.0
        assert budget["utilization_percent"] == 25.0

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_no_budget(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_budget_zero_limit(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        budget = FakeBudget(monthly_limit_usd=Decimal("0"), current_monthly_spend=Decimal("0"))
        mock_tracker.get_budget.return_value = budget
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        # Zero limit means no budget configured effectively
        # The monthly_limit_usd is falsy (0), so budget_info stays empty
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_budget_exception_handled(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.side_effect = AttributeError("no budget table")
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_workspace_id_in_response(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_abc"})
        body = _parse_body(result)
        assert body["workspace_id"] == "ws_abc"

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_response_has_all_keys(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert "workspace_id" in body
        assert "total_spend_usd" in body
        assert "agent_costs" in body
        assert "budget" in body

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_full_budget_utilization(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        budget = FakeBudget(
            monthly_limit_usd=Decimal("100"),
            current_monthly_spend=Decimal("100"),
        )
        mock_tracker.get_budget.return_value = budget
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["budget"]["utilization_percent"] == 100.0
        assert body["budget"]["remaining_usd"] == 0

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_over_budget(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats()
        budget = FakeBudget(
            monthly_limit_usd=Decimal("100"),
            current_monthly_spend=Decimal("150"),
        )
        mock_tracker.get_budget.return_value = budget
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert body["budget"]["utilization_percent"] == 150.0
        assert body["budget"]["remaining_usd"] == 0  # max(0, ...)

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_many_agents(self, mock_get_tracker):
        agents = {f"agent_{i}": f"{i}.00" for i in range(10)}
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = _make_workspace_stats(
            agent_costs=agents
        )
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        handler = _make_mixin_instance()
        result = handler._get_cost_breakdown({"workspace_id": "ws_123"})
        body = _parse_body(result)
        assert len(body["agent_costs"]) == 10


class TestCostBreakdownRouting:
    """Tests that the new route is wired up in the handler."""

    def test_route_in_routes_list(self):
        from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler

        assert "/api/analytics/cost/breakdown" in AnalyticsDashboardHandler.ROUTES

    def test_can_handle_cost_breakdown(self):
        from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler

        handler = AnalyticsDashboardHandler()
        assert handler.can_handle("/api/analytics/cost/breakdown") is True

    def test_can_handle_versioned_path(self):
        from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler

        handler = AnalyticsDashboardHandler()
        assert handler.can_handle("/api/v1/analytics/cost/breakdown") is True

    def test_has_cost_breakdown_method(self):
        from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler

        handler = AnalyticsDashboardHandler()
        assert hasattr(handler, "_get_cost_breakdown")
        assert callable(handler._get_cost_breakdown)
