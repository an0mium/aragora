"""Tests for cost analytics handler endpoints in BudgetHandler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.budgets import BudgetHandler

# Patch target for get_cost_tracker (imported locally inside handler methods)
_COST_TRACKER_PATCH = "aragora.billing.cost_tracker.get_cost_tracker"
_RUN_ASYNC_PATCH = "aragora.server.http_utils.run_async"


@pytest.fixture
def handler():
    return BudgetHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with auth context."""
    h = MagicMock()
    h.path = "/api/v1/costs/agents"
    h.command = "GET"
    h.headers = {"Authorization": "Bearer test-token", "Content-Length": "0"}
    h.org_id = "test-org"
    h.user_id = "test-user"
    return h


class TestAgentCosts:
    """Tests for GET /api/v1/costs/agents."""

    def test_agent_costs_returns_breakdown(self, handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "1.50",
            "cost_by_agent": {"claude": "1.00", "gpt4": "0.50"},
            "total_tokens_in": 500,
            "total_tokens_out": 200,
            "total_api_calls": 3,
        }
        with patch(_COST_TRACKER_PATCH, return_value=mock_tracker):
            result = handler._get_agent_costs("test-org", mock_http_handler)

        body = result[0]
        assert body["count"] == 2
        assert body["agents"]["claude"] == "1.00"
        assert body["agents"]["gpt4"] == "0.50"
        assert body["total_cost_usd"] == "1.50"

    def test_agent_costs_empty_state(self, handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "0",
            "cost_by_agent": {},
        }
        with patch(_COST_TRACKER_PATCH, return_value=mock_tracker):
            result = handler._get_agent_costs("test-org", mock_http_handler)

        body = result[0]
        assert body["count"] == 0
        assert body["agents"] == {}

    def test_agent_costs_with_workspace_param(self, handler):
        """Workspace can be specified via query param."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/costs/agents?workspace_id=ws-123"

        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "2.00",
            "cost_by_agent": {"gemini": "2.00"},
        }
        with patch(_COST_TRACKER_PATCH, return_value=mock_tracker):
            result = handler._get_agent_costs("test-org", mock_handler)

        body = result[0]
        assert body["workspace_id"] == "ws-123"
        mock_tracker.get_workspace_stats.assert_called_once_with("ws-123")


class TestCostAnomalies:
    """Tests for GET /api/v1/costs/anomalies."""

    def test_anomalies_returns_list(self, handler, mock_http_handler):
        mock_http_handler.path = "/api/v1/costs/anomalies"
        mock_tracker = MagicMock()
        anomaly_data = [{"type": "spike", "severity": "warning", "actual": 5.0, "expected": 1.0}]
        mock_advisory = MagicMock()
        mock_advisory.to_dict.return_value = {"level": "warning", "message": "Spending spike"}

        with (
            patch(_COST_TRACKER_PATCH, return_value=mock_tracker),
            patch(_RUN_ASYNC_PATCH, return_value=(anomaly_data, mock_advisory)),
        ):
            result = handler._get_cost_anomalies("test-org", mock_http_handler)

        body = result[0]
        assert body["count"] == 1
        assert body["anomalies"][0]["type"] == "spike"
        assert "anomalies detected" in body["advisory"].lower()

    def test_anomalies_empty_state(self, handler, mock_http_handler):
        mock_http_handler.path = "/api/v1/costs/anomalies"
        mock_tracker = MagicMock()

        with (
            patch(_COST_TRACKER_PATCH, return_value=mock_tracker),
            patch(_RUN_ASYNC_PATCH, return_value=([], None)),
        ):
            result = handler._get_cost_anomalies("test-org", mock_http_handler)

        body = result[0]
        assert body["count"] == 0
        assert body["anomalies"] == []
        assert "no anomalies" in body["advisory"].lower()

    def test_anomalies_runtime_error_returns_empty(self, handler, mock_http_handler):
        """RuntimeError from run_async should return empty anomalies."""
        mock_http_handler.path = "/api/v1/costs/anomalies"
        mock_tracker = MagicMock()

        with (
            patch(_COST_TRACKER_PATCH, return_value=mock_tracker),
            patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("no loop")),
        ):
            result = handler._get_cost_anomalies("test-org", mock_http_handler)

        body = result[0]
        assert body["count"] == 0
        assert body["anomalies"] == []


class TestCanHandle:
    """Tests for route matching of cost endpoints."""

    def test_can_handle_costs_agents(self, handler):
        assert handler.can_handle("/api/v1/costs/agents")

    def test_can_handle_costs_anomalies(self, handler):
        assert handler.can_handle("/api/v1/costs/anomalies")

    def test_can_handle_budgets(self, handler):
        assert handler.can_handle("/api/v1/budgets")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates")
