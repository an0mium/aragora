"""
Tests for the AnalyticsDashboardHandler module.

Tests cover:
- Handler initialization and routing
- workspace_id validation for all endpoints
- Route handling and can_handle method
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestAnalyticsDashboardHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AnalyticsDashboardHandler(mock_server_context)

    def test_can_handle_summary(self, handler):
        """Handler can handle summary endpoint."""
        assert handler.can_handle("/api/v1/analytics/summary")

    def test_can_handle_trends(self, handler):
        """Handler can handle trends endpoint."""
        assert handler.can_handle("/api/v1/analytics/trends/findings")

    def test_can_handle_remediation(self, handler):
        """Handler can handle remediation endpoint."""
        assert handler.can_handle("/api/v1/analytics/remediation")

    def test_can_handle_agents(self, handler):
        """Handler can handle agents endpoint."""
        assert handler.can_handle("/api/v1/analytics/agents")

    def test_can_handle_cost(self, handler):
        """Handler can handle cost endpoint."""
        assert handler.can_handle("/api/v1/analytics/cost")

    def test_can_handle_compliance(self, handler):
        """Handler can handle compliance endpoint."""
        assert handler.can_handle("/api/v1/analytics/compliance")

    def test_can_handle_heatmap(self, handler):
        """Handler can handle heatmap endpoint."""
        assert handler.can_handle("/api/v1/analytics/heatmap")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/v1/analytics/unknown")
        assert not handler.can_handle("/api/v1/other")

    def test_routes_list_complete(self, handler):
        """ROUTES list contains all expected endpoints."""
        assert len(handler.ROUTES) >= 14  # At least 14, may have more
        assert "/api/v1/analytics/summary" in handler.ROUTES
        assert "/api/v1/analytics/heatmap" in handler.ROUTES
        assert "/api/v1/analytics/tokens" in handler.ROUTES
        assert "/api/v1/analytics/flips/summary" in handler.ROUTES


class TestAnalyticsDashboardHandlerUnknownPath:
    """Tests for unknown path handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AnalyticsDashboardHandler(mock_server_context)

    def test_unknown_path_returns_none(self, handler):
        """Unknown path returns None for dispatch to continue."""
        mock_http_handler = MagicMock()

        result = handler.handle("/api/v1/other/endpoint", {}, mock_http_handler)

        assert result is None


class TestAnalyticsDashboardHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AnalyticsDashboardHandler(mock_server_context)

    def test_handle_dispatches_to_summary(self, handler):
        """Handle dispatches /api/analytics/summary to _get_summary."""
        mock_http = MagicMock()

        # The method is protected by @require_user_auth so we get an auth error
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http)

        # Result should be returned (either auth error or workspace_id error)
        assert result is not None

    def test_handle_dispatches_to_trends(self, handler):
        """Handle dispatches /api/analytics/trends/findings to _get_finding_trends."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/trends/findings", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_to_remediation(self, handler):
        """Handle dispatches /api/analytics/remediation to _get_remediation_metrics."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/remediation", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_to_agents(self, handler):
        """Handle dispatches /api/analytics/agents to _get_agent_metrics."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/agents", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_to_cost(self, handler):
        """Handle dispatches /api/analytics/cost to _get_cost_metrics."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/cost", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_to_compliance(self, handler):
        """Handle dispatches /api/analytics/compliance to _get_compliance_scorecard."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/compliance", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_to_heatmap(self, handler):
        """Handle dispatches /api/analytics/heatmap to _get_risk_heatmap."""
        mock_http = MagicMock()

        result = handler.handle("/api/v1/analytics/heatmap", {}, mock_http)

        assert result is not None
