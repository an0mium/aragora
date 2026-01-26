"""
Tests for the AnalyticsDashboardHandler module.

Tests cover:
- Handler initialization and routing
- workspace_id/org_id validation for all endpoints
- Route handling and can_handle method
- Stub responses for unauthenticated requests
- Public flip detection endpoints (no auth required)
- Query parameter validation
- Error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
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
        # ROUTES uses normalized paths without version prefix
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

    def test_trends_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Trends returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/trends/findings", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_remediation_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Remediation returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/remediation", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_agents_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Agents returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/agents", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_cost_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Cost returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/cost", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_compliance_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Compliance returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/compliance", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_heatmap_returns_stub_without_workspace(self, handler, mock_http_handler):
        """Heatmap returns stub when no workspace_id provided."""
        result = handler.handle("/api/v1/analytics/heatmap", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_tokens_returns_stub_without_org(self, handler, mock_http_handler):
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


class TestSummaryEndpoint:
    """Tests for /api/analytics/summary endpoint."""

    def test_summary_requires_workspace_id(self, handler, mock_http_handler):
        """Summary endpoint requires workspace_id when auth provided."""
        # Call unwrapped to bypass decorator
        result = handler._get_summary.__wrapped__(handler, {}, mock_http_handler, user=MagicMock())
        assert result is not None
        assert result.status_code == 400
        assert "workspace_id" in result.body.get("error", "").lower()

    def test_summary_with_valid_workspace(self, handler, mock_http_handler):
        """Summary endpoint works with valid workspace_id."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dashboard:
            mock_summary = MagicMock()
            mock_summary.to_dict.return_value = {
                "total_findings": 150,
                "open_findings": 45,
                "critical_findings": 5,
            }
            mock_dashboard.return_value.get_summary = AsyncMock(return_value=mock_summary)

            result = handler._get_summary.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None
            assert result.status_code == 200

    def test_summary_invalid_time_range(self, handler, mock_http_handler):
        """Summary endpoint handles invalid time_range gracefully."""
        with patch("aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"):
            with patch("aragora.server.handlers.analytics_dashboard.TimeRange") as mock_tr:
                mock_tr.side_effect = ValueError("Invalid time range")

                result = handler._get_summary.__wrapped__(
                    handler,
                    {"workspace_id": "ws-123", "time_range": "invalid"},
                    mock_http_handler,
                    user=MagicMock(),
                )

                assert result is not None
                assert result.status_code == 400


class TestFindingTrendsEndpoint:
    """Tests for /api/analytics/trends/findings endpoint."""

    def test_trends_requires_workspace_id(self, handler, mock_http_handler):
        """Trends endpoint requires workspace_id."""
        result = handler._get_finding_trends.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400
        assert "workspace_id" in result.body.get("error", "").lower()

    def test_trends_with_valid_params(self, handler, mock_http_handler):
        """Trends endpoint works with valid parameters."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_dash.return_value.get_finding_trends = AsyncMock(return_value=[])

            result = handler._get_finding_trends.__wrapped__(
                handler,
                {"workspace_id": "ws-123", "time_range": "7d", "granularity": "day"},
                mock_http_handler,
                user=MagicMock(),
            )

            assert result is not None


class TestRemediationEndpoint:
    """Tests for /api/analytics/remediation endpoint."""

    def test_remediation_requires_workspace_id(self, handler, mock_http_handler):
        """Remediation endpoint requires workspace_id."""
        result = handler._get_remediation_metrics.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_remediation_with_valid_workspace(self, handler, mock_http_handler):
        """Remediation endpoint works with valid workspace_id."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_metrics = MagicMock()
            mock_metrics.to_dict.return_value = {
                "total_resolved": 120,
                "total_open": 45,
                "mttr_hours": 48.5,
            }
            mock_dash.return_value.get_remediation_metrics = AsyncMock(return_value=mock_metrics)

            result = handler._get_remediation_metrics.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None


class TestAgentMetricsEndpoint:
    """Tests for /api/analytics/agents endpoint."""

    def test_agents_requires_workspace_id(self, handler, mock_http_handler):
        """Agents endpoint requires workspace_id."""
        result = handler._get_agent_metrics.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_agents_with_valid_workspace(self, handler, mock_http_handler):
        """Agents endpoint works with valid workspace_id."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_dash.return_value.get_agent_metrics = AsyncMock(return_value=[])

            result = handler._get_agent_metrics.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None


class TestCostMetricsEndpoint:
    """Tests for /api/analytics/cost endpoint."""

    def test_cost_requires_workspace_id(self, handler, mock_http_handler):
        """Cost endpoint requires workspace_id."""
        result = handler._get_cost_metrics.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_cost_with_valid_workspace(self, handler, mock_http_handler):
        """Cost endpoint works with valid workspace_id."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_metrics = MagicMock()
            mock_metrics.to_dict.return_value = {"total_cost_usd": 125.50}
            mock_dash.return_value.get_cost_metrics = AsyncMock(return_value=mock_metrics)

            result = handler._get_cost_metrics.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None


class TestComplianceEndpoint:
    """Tests for /api/analytics/compliance endpoint."""

    def test_compliance_requires_workspace_id(self, handler, mock_http_handler):
        """Compliance endpoint requires workspace_id."""
        result = handler._get_compliance_scorecard.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_compliance_with_frameworks(self, handler, mock_http_handler):
        """Compliance endpoint accepts frameworks parameter."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_dash.return_value.get_compliance_scorecard = AsyncMock(return_value=[])

            result = handler._get_compliance_scorecard.__wrapped__(
                handler,
                {"workspace_id": "ws-123", "frameworks": "SOC2,GDPR"},
                mock_http_handler,
                user=MagicMock(),
            )

            assert result is not None


class TestRiskHeatmapEndpoint:
    """Tests for /api/analytics/heatmap endpoint."""

    def test_heatmap_requires_workspace_id(self, handler, mock_http_handler):
        """Heatmap endpoint requires workspace_id."""
        result = handler._get_risk_heatmap.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_heatmap_with_valid_workspace(self, handler, mock_http_handler):
        """Heatmap endpoint works with valid workspace_id."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_dash.return_value.get_risk_heatmap = AsyncMock(return_value=[])

            result = handler._get_risk_heatmap.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None


class TestTokenUsageEndpoint:
    """Tests for /api/analytics/tokens endpoint."""

    def test_tokens_requires_org_id(self, handler, mock_http_handler):
        """Tokens endpoint requires org_id."""
        result = handler._get_token_usage.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400
        assert "org_id" in result.body.get("error", "").lower()

    def test_tokens_with_valid_org(self, handler, mock_http_handler):
        """Tokens endpoint works with valid org_id."""
        with patch("aragora.server.handlers.analytics_dashboard.UsageTracker") as mock_tracker:
            mock_summary = MagicMock()
            mock_summary.total_tokens_in = 500000
            mock_summary.total_tokens_out = 100000
            mock_summary.total_cost_usd = 125.50
            mock_summary.total_debates = 50
            mock_summary.total_agent_calls = 200
            mock_summary.cost_by_provider = {"anthropic": 80.00}
            mock_summary.debates_by_day = {}
            mock_tracker.return_value.get_summary.return_value = mock_summary

            result = handler._get_token_usage.__wrapped__(
                handler, {"org_id": "org-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None
            assert result.status_code == 200


class TestTokenTrendsEndpoint:
    """Tests for /api/analytics/tokens/trends endpoint."""

    def test_trends_requires_org_id(self, handler, mock_http_handler):
        """Token trends endpoint requires org_id."""
        result = handler._get_token_trends.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_trends_accepts_granularity(self, handler, mock_http_handler):
        """Token trends endpoint accepts granularity parameter."""
        with patch("aragora.server.handlers.analytics_dashboard.UsageTracker") as mock_tracker:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_tracker.return_value._connection.return_value.__enter__.return_value = mock_conn

            result = handler._get_token_trends.__wrapped__(
                handler,
                {"org_id": "org-123", "granularity": "hour"},
                mock_http_handler,
                user=MagicMock(),
            )

            assert result is not None


class TestProviderBreakdownEndpoint:
    """Tests for /api/analytics/tokens/providers endpoint."""

    def test_providers_requires_org_id(self, handler, mock_http_handler):
        """Provider breakdown endpoint requires org_id."""
        result = handler._get_provider_breakdown.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400

    def test_providers_with_valid_org(self, handler, mock_http_handler):
        """Provider breakdown endpoint works with valid org_id."""
        with patch("aragora.server.handlers.analytics_dashboard.UsageTracker") as mock_tracker:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_tracker.return_value._connection.return_value.__enter__.return_value = mock_conn

            result = handler._get_provider_breakdown.__wrapped__(
                handler, {"org_id": "org-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None
            assert result.status_code == 200


class TestFlipSummaryEndpoint:
    """Tests for /api/analytics/flips/summary endpoint (no auth required)."""

    def test_flip_summary_no_auth_required(self, handler):
        """Flip summary endpoint does not require authentication."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_detector.return_value.get_flip_summary.return_value = {
                "total_flips": 150,
                "by_type": {"contradiction": 45},
                "by_agent": {"claude": 30},
            }

            result = handler._get_flip_summary({})

            assert result is not None
            assert result.status_code == 200

    def test_flip_summary_handles_detector_error(self, handler):
        """Flip summary handles detector errors gracefully."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_detector.return_value.get_flip_summary.side_effect = Exception("DB error")

            result = handler._get_flip_summary({})

            assert result is not None
            assert result.status_code == 500


class TestRecentFlipsEndpoint:
    """Tests for /api/analytics/flips/recent endpoint (no auth required)."""

    def test_recent_flips_default_limit(self, handler):
        """Recent flips uses default limit of 20."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            with patch(
                "aragora.server.handlers.analytics_dashboard.format_flip_for_ui"
            ) as mock_format:
                mock_detector.return_value.get_recent_flips.return_value = []
                mock_format.return_value = {}

                result = handler._get_recent_flips({})

                assert result is not None
                # Fetches limit * 2 for filtering
                mock_detector.return_value.get_recent_flips.assert_called_once_with(limit=40)

    def test_recent_flips_respects_max_limit(self, handler):
        """Recent flips respects max limit of 100."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_detector.return_value.get_recent_flips.return_value = []

            result = handler._get_recent_flips({"limit": "500"})

            assert result is not None
            # Should cap at 100 * 2 = 200
            mock_detector.return_value.get_recent_flips.assert_called_once_with(limit=200)


class TestAgentConsistencyEndpoint:
    """Tests for /api/analytics/flips/consistency endpoint (no auth required)."""

    def test_consistency_returns_all_agents(self, handler):
        """Consistency returns all agents when no filter specified."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            with patch(
                "aragora.server.handlers.analytics_dashboard.format_consistency_for_ui"
            ) as mock_format:
                mock_detector.return_value.get_flip_summary.return_value = {
                    "by_agent": {"claude": 10, "gpt-4": 5}
                }
                mock_detector.return_value.get_agents_consistency_batch.return_value = {}
                mock_format.return_value = {"consistency": "90%"}

                result = handler._get_agent_consistency({})

                assert result is not None
                assert result.status_code == 200


class TestFlipTrendsEndpoint:
    """Tests for /api/analytics/flips/trends endpoint (no auth required)."""

    def test_flip_trends_default_params(self, handler):
        """Flip trends uses default parameters."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_detector.return_value.db.connection.return_value.__enter__.return_value = mock_conn

            result = handler._get_flip_trends({})

            assert result is not None
            assert result.status_code == 200

    def test_flip_trends_calculates_summary(self, handler):
        """Flip trends calculates summary statistics."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                ("2026-01-15", "contradiction", 3),
                ("2026-01-15", "retraction", 2),
                ("2026-01-16", "contradiction", 5),
            ]
            mock_detector.return_value.db.connection.return_value.__enter__.return_value = mock_conn

            result = handler._get_flip_trends({"days": "7"})

            assert result is not None
            data = result.body
            assert "summary" in data
            assert "total_flips" in data["summary"]
            assert "avg_per_day" in data["summary"]
            assert "trend" in data["summary"]


class TestDeliberationSummaryEndpoint:
    """Tests for /api/analytics/deliberations endpoint."""

    def test_deliberation_summary_requires_org_id(self, handler, mock_http_handler):
        """Deliberation summary requires org_id."""
        result = handler._get_deliberation_summary.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400
        assert "org_id" in result.body.get("error", "").lower()

    def test_deliberation_summary_with_valid_org(self, handler, mock_http_handler):
        """Deliberation summary works with valid org_id."""
        with patch("aragora.server.handlers.analytics_dashboard.get_debate_store") as mock_store:
            mock_store.return_value.get_deliberation_stats.return_value = {
                "total": 150,
                "completed": 145,
                "consensus_reached": 120,
                "in_progress": 3,
                "failed": 2,
                "avg_rounds": 3.5,
                "avg_duration_seconds": 45.2,
                "by_template": {},
                "by_priority": {},
            }

            result = handler._get_deliberation_summary.__wrapped__(
                handler, {"org_id": "org-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None
            assert result.status_code == 200


class TestDeliberationByChannelEndpoint:
    """Tests for /api/analytics/deliberations/channels endpoint."""

    def test_channels_requires_org_id(self, handler, mock_http_handler):
        """Deliberation channels requires org_id."""
        result = handler._get_deliberation_by_channel.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400


class TestConsensusRatesEndpoint:
    """Tests for /api/analytics/deliberations/consensus endpoint."""

    def test_consensus_requires_org_id(self, handler, mock_http_handler):
        """Consensus rates requires org_id."""
        result = handler._get_consensus_rates.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400


class TestDeliberationPerformanceEndpoint:
    """Tests for /api/analytics/deliberations/performance endpoint."""

    def test_performance_requires_org_id(self, handler, mock_http_handler):
        """Deliberation performance requires org_id."""
        result = handler._get_deliberation_performance.__wrapped__(
            handler, {}, mock_http_handler, user=MagicMock()
        )
        assert result.status_code == 400


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_summary_handles_import_error(self, handler, mock_http_handler):
        """Summary handles import errors gracefully."""
        with patch(
            "aragora.server.handlers.analytics_dashboard.get_analytics_dashboard"
        ) as mock_dash:
            mock_dash.side_effect = ImportError("Module not found")

            result = handler._get_summary.__wrapped__(
                handler, {"workspace_id": "ws-123"}, mock_http_handler, user=MagicMock()
            )

            assert result is not None
            assert result.status_code == 500

    def test_flip_summary_handles_exception(self, handler):
        """Flip summary handles exceptions gracefully."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_detector.side_effect = Exception("Detector initialization failed")

            result = handler._get_flip_summary({})

            assert result is not None
            assert result.status_code == 500


class TestHandleRouteDispatch:
    """Tests for the main handle() method route dispatch."""

    def test_handle_dispatches_to_summary(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/summary to _get_summary."""
        result = handler.handle("/api/v1/analytics/summary", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_trends(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/trends/findings to _get_finding_trends."""
        result = handler.handle("/api/v1/analytics/trends/findings", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_remediation(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/remediation to _get_remediation_metrics."""
        result = handler.handle("/api/v1/analytics/remediation", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_agents(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/agents to _get_agent_metrics."""
        result = handler.handle("/api/v1/analytics/agents", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_cost(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/cost to _get_cost_metrics."""
        result = handler.handle("/api/v1/analytics/cost", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_compliance(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/compliance to _get_compliance_scorecard."""
        result = handler.handle("/api/v1/analytics/compliance", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_heatmap(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/heatmap to _get_risk_heatmap."""
        result = handler.handle("/api/v1/analytics/heatmap", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_tokens(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/tokens to _get_token_usage."""
        result = handler.handle("/api/v1/analytics/tokens", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_flips_summary(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/flips/summary to _get_flip_summary."""
        with patch("aragora.server.handlers.analytics_dashboard.FlipDetector") as mock_detector:
            mock_detector.return_value.get_flip_summary.return_value = {}
            result = handler.handle("/api/v1/analytics/flips/summary", {}, mock_http_handler)
        assert result is not None

    def test_handle_dispatches_to_deliberations(self, handler, mock_http_handler):
        """Handle dispatches /api/analytics/deliberations to _get_deliberation_summary."""
        result = handler.handle("/api/v1/analytics/deliberations", {}, mock_http_handler)
        assert result is not None
