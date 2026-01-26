"""
Tests for the AnalyticsDashboardHandler module.

Tests cover:
- Handler initialization and routing
- Route handling and can_handle method
- Stub responses for unauthenticated requests
- Handle method dispatch
- Unknown path handling
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def handler(mock_server_context):
    """Create handler instance for tests."""
    return AnalyticsDashboardHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler for tests."""
    mock = MagicMock()
    mock.headers = {"Authorization": ""}
    return mock


class TestAnalyticsDashboardHandlerRouting:
    """Tests for handler routing."""

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

    def test_can_handle_tokens(self, handler):
        """Handler can handle tokens endpoint."""
        assert handler.can_handle("/api/v1/analytics/tokens")

    def test_can_handle_tokens_trends(self, handler):
        """Handler can handle tokens trends endpoint."""
        assert handler.can_handle("/api/v1/analytics/tokens/trends")

    def test_can_handle_tokens_providers(self, handler):
        """Handler can handle tokens providers endpoint."""
        assert handler.can_handle("/api/v1/analytics/tokens/providers")

    def test_can_handle_flips_summary(self, handler):
        """Handler can handle flips summary endpoint."""
        assert handler.can_handle("/api/v1/analytics/flips/summary")

    def test_can_handle_flips_recent(self, handler):
        """Handler can handle flips recent endpoint."""
        assert handler.can_handle("/api/v1/analytics/flips/recent")

    def test_can_handle_flips_consistency(self, handler):
        """Handler can handle flips consistency endpoint."""
        assert handler.can_handle("/api/v1/analytics/flips/consistency")

    def test_can_handle_flips_trends(self, handler):
        """Handler can handle flips trends endpoint."""
        assert handler.can_handle("/api/v1/analytics/flips/trends")

    def test_can_handle_deliberations(self, handler):
        """Handler can handle deliberations endpoint."""
        assert handler.can_handle("/api/v1/analytics/deliberations")

    def test_can_handle_deliberations_channels(self, handler):
        """Handler can handle deliberations channels endpoint."""
        assert handler.can_handle("/api/v1/analytics/deliberations/channels")

    def test_can_handle_deliberations_consensus(self, handler):
        """Handler can handle deliberations consensus endpoint."""
        assert handler.can_handle("/api/v1/analytics/deliberations/consensus")

    def test_can_handle_deliberations_performance(self, handler):
        """Handler can handle deliberations performance endpoint."""
        assert handler.can_handle("/api/v1/analytics/deliberations/performance")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/v1/analytics/unknown")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/debates")

    def test_routes_list_complete(self, handler):
        """ROUTES list contains all expected endpoints."""
        assert len(handler.ROUTES) >= 18
        assert "/api/analytics/summary" in handler.ROUTES
        assert "/api/analytics/heatmap" in handler.ROUTES
        assert "/api/analytics/tokens" in handler.ROUTES
        assert "/api/analytics/flips/summary" in handler.ROUTES
        assert "/api/analytics/deliberations" in handler.ROUTES


class TestAnalyticsDashboardHandlerUnknownPath:
    """Tests for unknown path handling."""

    def test_unknown_path_returns_none(self, handler, mock_http_handler):
        """Unknown path returns None for dispatch to continue."""
        result = handler.handle("/api/v1/other/endpoint", {}, mock_http_handler)
        assert result is None

    def test_unknown_analytics_path_returns_none(self, handler, mock_http_handler):
        """Unknown analytics path returns None."""
        result = handler.handle("/api/v1/analytics/nonexistent", {}, mock_http_handler)
        assert result is None


class TestStubResponses:
    """Tests for stub responses when no auth/workspace_id."""

    def test_trends_returns_stub(self, handler, mock_http_handler):
        """Trends returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/trends/findings", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_remediation_returns_stub(self, handler, mock_http_handler):
        """Remediation returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/remediation", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_agents_returns_stub(self, handler, mock_http_handler):
        """Agents returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/agents", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_cost_returns_stub(self, handler, mock_http_handler):
        """Cost returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/cost", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_compliance_returns_stub(self, handler, mock_http_handler):
        """Compliance returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/compliance", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_heatmap_returns_stub(self, handler, mock_http_handler):
        """Heatmap returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/heatmap", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_tokens_returns_stub(self, handler, mock_http_handler):
        """Tokens returns stub when no org_id provided."""
        result = handler.handle("/api/v1/analytics/tokens", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_tokens_trends_returns_stub(self, handler, mock_http_handler):
        """Token trends returns stub when no org_id provided."""
        result = handler.handle("/api/v1/analytics/tokens/trends", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_tokens_providers_returns_stub(self, handler, mock_http_handler):
        """Token providers returns stub when no org_id provided."""
        result = handler.handle("/api/v1/analytics/tokens/providers", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_flips_summary_returns_stub(self, handler, mock_http_handler):
        """Flips summary returns stub."""
        result = handler.handle("/api/v1/analytics/flips/summary", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_flips_recent_returns_stub(self, handler, mock_http_handler):
        """Recent flips returns stub."""
        result = handler.handle("/api/v1/analytics/flips/recent", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_deliberations_returns_stub(self, handler, mock_http_handler):
        """Deliberations returns stub when no org_id provided."""
        result = handler.handle("/api/v1/analytics/deliberations", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200


class TestHandleRouteDispatch:
    """Tests for the main handle() method route dispatch."""

    def test_dispatches_to_summary(self, handler, mock_http_handler):
        """Handle dispatches to _get_summary."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_trends(self, handler, mock_http_handler):
        """Handle dispatches to _get_finding_trends."""
        result = handler.handle("/api/v1/analytics/trends/findings", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_remediation(self, handler, mock_http_handler):
        """Handle dispatches to _get_remediation_metrics."""
        result = handler.handle("/api/v1/analytics/remediation", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_agents(self, handler, mock_http_handler):
        """Handle dispatches to _get_agent_metrics."""
        result = handler.handle("/api/v1/analytics/agents", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_cost(self, handler, mock_http_handler):
        """Handle dispatches to _get_cost_metrics."""
        result = handler.handle("/api/v1/analytics/cost", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_compliance(self, handler, mock_http_handler):
        """Handle dispatches to _get_compliance_scorecard."""
        result = handler.handle("/api/v1/analytics/compliance", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_heatmap(self, handler, mock_http_handler):
        """Handle dispatches to _get_risk_heatmap."""
        result = handler.handle("/api/v1/analytics/heatmap", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_tokens(self, handler, mock_http_handler):
        """Handle dispatches to _get_token_usage."""
        result = handler.handle("/api/v1/analytics/tokens", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_tokens_trends(self, handler, mock_http_handler):
        """Handle dispatches to _get_token_trends."""
        result = handler.handle("/api/v1/analytics/tokens/trends", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_tokens_providers(self, handler, mock_http_handler):
        """Handle dispatches to _get_provider_breakdown."""
        result = handler.handle("/api/v1/analytics/tokens/providers", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_flips_summary(self, handler, mock_http_handler):
        """Handle dispatches to _get_flip_summary."""
        result = handler.handle("/api/v1/analytics/flips/summary", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_flips_recent(self, handler, mock_http_handler):
        """Handle dispatches to _get_recent_flips."""
        result = handler.handle("/api/v1/analytics/flips/recent", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_flips_consistency(self, handler, mock_http_handler):
        """Handle dispatches to _get_agent_consistency."""
        result = handler.handle("/api/v1/analytics/flips/consistency", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_flips_trends(self, handler, mock_http_handler):
        """Handle dispatches to _get_flip_trends."""
        result = handler.handle("/api/v1/analytics/flips/trends", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_deliberations(self, handler, mock_http_handler):
        """Handle dispatches to _get_deliberation_summary."""
        result = handler.handle("/api/v1/analytics/deliberations", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_deliberations_channels(self, handler, mock_http_handler):
        """Handle dispatches correctly."""
        result = handler.handle("/api/v1/analytics/deliberations/channels", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_deliberations_consensus(self, handler, mock_http_handler):
        """Handle dispatches correctly."""
        result = handler.handle("/api/v1/analytics/deliberations/consensus", {}, mock_http_handler)
        assert result is not None

    def test_dispatches_to_deliberations_performance(self, handler, mock_http_handler):
        """Handle dispatches correctly."""
        result = handler.handle(
            "/api/v1/analytics/deliberations/performance", {}, mock_http_handler
        )
        assert result is not None
