"""Tests for the CostDashboardHandler and CostTracker.get_dashboard_summary()."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.cost_tracker import Budget, CostTracker
from aragora.server.handlers.billing.cost_dashboard import CostDashboardHandler


# ---------------------------------------------------------------------------
# CostTracker.get_dashboard_summary unit tests
# ---------------------------------------------------------------------------


class TestGetDashboardSummary:
    """Tests for CostTracker.get_dashboard_summary()."""

    def _make_tracker(self) -> CostTracker:
        return CostTracker(usage_tracker=None, km_adapter=None)

    def test_empty_workspace_returns_zeros(self):
        tracker = self._make_tracker()
        result = tracker.get_dashboard_summary(workspace_id="ws-1")

        assert result["workspace_id"] == "ws-1"
        assert result["current_spend"]["total_cost_usd"] == "0"
        assert result["current_spend"]["total_api_calls"] == 0
        assert result["budget"]["configured"] is False
        assert result["top_cost_drivers"]["by_agent"] == []
        assert result["projections"]["projected_monthly_usd"] is None

    def test_with_workspace_stats(self):
        tracker = self._make_tracker()
        # Populate workspace stats directly
        stats = tracker._workspace_stats["ws-1"]
        stats["total_cost"] = Decimal("12.50")
        stats["tokens_in"] = 5000
        stats["tokens_out"] = 2000
        stats["api_calls"] = 10
        stats["by_agent"]["claude"] = Decimal("8.00")
        stats["by_agent"]["gpt"] = Decimal("4.50")
        stats["by_model"]["claude-3"] = Decimal("8.00")
        stats["by_model"]["gpt-4"] = Decimal("4.50")

        result = tracker.get_dashboard_summary(workspace_id="ws-1")

        assert result["current_spend"]["total_cost_usd"] == "12.50"
        assert result["current_spend"]["total_api_calls"] == 10
        assert result["current_spend"]["total_tokens"] == 7000
        assert len(result["top_cost_drivers"]["by_agent"]) == 2
        assert result["top_cost_drivers"]["by_agent"][0]["agent"] == "claude"
        assert result["projections"]["projected_monthly_usd"] is not None

    def test_with_budget(self):
        tracker = self._make_tracker()
        budget = Budget(
            name="test-budget",
            workspace_id="ws-1",
            monthly_limit_usd=Decimal("100.00"),
            current_monthly_spend=Decimal("45.00"),
        )
        tracker.set_budget(budget)

        result = tracker.get_dashboard_summary(workspace_id="ws-1")

        assert result["budget"]["configured"] is True
        assert result["budget"]["monthly_limit_usd"] == "100.00"
        assert result["budget"]["utilization_pct"] == 45.0

    def test_org_id_filter(self):
        tracker = self._make_tracker()
        budget = Budget(
            name="org-budget",
            org_id="org-1",
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("200.00"),
        )
        tracker.set_budget(budget)

        result = tracker.get_dashboard_summary(org_id="org-1")

        assert result["org_id"] == "org-1"
        assert result["budget"]["configured"] is True

    def test_no_filters(self):
        tracker = self._make_tracker()
        result = tracker.get_dashboard_summary()

        assert result["workspace_id"] is None
        assert result["org_id"] is None
        assert result["budget"]["configured"] is False

    def test_top_agents_limited_to_5(self):
        tracker = self._make_tracker()
        stats = tracker._workspace_stats["ws-1"]
        stats["api_calls"] = 10
        stats["total_cost"] = Decimal("10")
        for i in range(8):
            stats["by_agent"][f"agent-{i}"] = Decimal(str(i + 1))

        result = tracker.get_dashboard_summary(workspace_id="ws-1")
        assert len(result["top_cost_drivers"]["by_agent"]) == 5
        # Highest cost first
        assert result["top_cost_drivers"]["by_agent"][0]["agent"] == "agent-7"

    def test_budget_no_monthly_limit(self):
        tracker = self._make_tracker()
        budget = Budget(
            name="no-limit",
            workspace_id="ws-1",
            monthly_limit_usd=None,
        )
        tracker.set_budget(budget)

        result = tracker.get_dashboard_summary(workspace_id="ws-1")
        assert result["budget"]["configured"] is True
        assert result["budget"]["utilization_pct"] == 0


# ---------------------------------------------------------------------------
# CostDashboardHandler unit tests
# ---------------------------------------------------------------------------


class TestCostDashboardHandler:
    """Tests for the CostDashboardHandler HTTP handler."""

    def _make_handler(self, ctx: dict | None = None) -> CostDashboardHandler:
        return CostDashboardHandler(ctx=ctx)

    def _make_http_handler(
        self,
        method: str = "GET",
        client_ip: str = "127.0.0.1",
        query_params: dict[str, Any] | None = None,
    ) -> MagicMock:
        mock = MagicMock()
        mock.command = method
        mock.client_address = (client_ip, 12345)
        mock.headers = {"X-Forwarded-For": client_ip}
        return mock

    def test_can_handle_dashboard_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/v1/billing/dashboard") is True
        assert handler.can_handle("/api/v1/billing/plans") is False
        assert handler.can_handle("/other") is False

    def test_routes(self):
        handler = self._make_handler()
        assert "/api/v1/billing/dashboard" in handler.ROUTES

    def test_resource_type(self):
        handler = self._make_handler()
        assert handler.RESOURCE_TYPE == "cost_dashboard"

    def test_method_not_allowed(self):
        handler = self._make_handler()
        http = self._make_http_handler(method="POST")
        result = handler.handle("/api/v1/billing/dashboard", {}, http, method="POST")
        assert result is not None
        assert result.status_code == 405

    @patch("aragora.server.handlers.billing.cost_dashboard._dashboard_limiter")
    def test_rate_limit_exceeded(self, mock_limiter):
        mock_limiter.is_allowed.return_value = False
        handler = self._make_handler()
        http = self._make_http_handler()
        result = handler.handle("/api/v1/billing/dashboard", {}, http, method="GET")
        assert result is not None
        assert result.status_code == 429

    @patch("aragora.server.handlers.billing.cost_dashboard._dashboard_limiter")
    def test_get_dashboard_success(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True

        mock_summary = {
            "workspace_id": None,
            "org_id": None,
            "current_spend": {"total_cost_usd": "0", "total_api_calls": 0, "total_tokens": 0},
            "budget": {"configured": False},
            "top_cost_drivers": {"by_agent": [], "by_model": []},
            "projections": {"projected_monthly_usd": None},
        }

        handler = self._make_handler()
        http = self._make_http_handler()

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_summary.return_value = mock_summary
            mock_get.return_value = mock_tracker

            # Bypass the require_permission decorator
            result = handler._get_dashboard.__wrapped__(handler, http, {})

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.billing.cost_dashboard._dashboard_limiter")
    def test_get_dashboard_with_query_params(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True

        handler = self._make_handler()
        http = self._make_http_handler()

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_summary.return_value = {
                "workspace_id": "ws-1",
                "org_id": "org-1",
                "current_spend": {"total_cost_usd": "50.00"},
                "budget": {"configured": True},
                "top_cost_drivers": {"by_agent": [], "by_model": []},
                "projections": {"projected_monthly_usd": "150.00"},
            }
            mock_get.return_value = mock_tracker

            query = {"workspace_id": "ws-1", "org_id": "org-1"}
            result = handler._get_dashboard.__wrapped__(handler, http, query)

            mock_tracker.get_dashboard_summary.assert_called_once_with(
                workspace_id="ws-1",
                org_id="org-1",
            )

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.billing.cost_dashboard._dashboard_limiter")
    def test_get_dashboard_error(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True

        handler = self._make_handler()
        http = self._make_http_handler()

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            mock_get.side_effect = RuntimeError("DB connection failed")

            result = handler._get_dashboard.__wrapped__(handler, http, {})

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.billing.cost_dashboard._dashboard_limiter")
    def test_empty_query_params_pass_none(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True

        handler = self._make_handler()
        http = self._make_http_handler()

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_summary.return_value = {"ok": True}
            mock_get.return_value = mock_tracker

            result = handler._get_dashboard.__wrapped__(handler, http, {})

            mock_tracker.get_dashboard_summary.assert_called_once_with(
                workspace_id=None,
                org_id=None,
            )

    def test_handle_uses_command_method(self):
        """Test that handle() reads method from handler.command."""
        handler = self._make_handler()
        http = self._make_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.billing.cost_dashboard._dashboard_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            with patch.object(handler, "_get_dashboard") as mock_dashboard:
                mock_dashboard.return_value = MagicMock(status_code=200)
                result = handler.handle("/api/v1/billing/dashboard", {}, http)

                mock_dashboard.assert_called_once()
