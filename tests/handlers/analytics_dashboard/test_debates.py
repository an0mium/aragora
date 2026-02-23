"""Comprehensive tests for the DebateAnalyticsMixin handler.

Tests all five endpoints provided by the debate analytics handler:
- GET /api/analytics/summary         - Dashboard summary (cached: 60s)
- GET /api/analytics/trends/findings - Finding trends over time (cached: 300s)
- GET /api/analytics/remediation     - Remediation metrics (cached: 300s)
- GET /api/analytics/compliance      - Compliance scorecard
- GET /api/analytics/heatmap         - Risk heatmap data

Also tests routing, version prefix handling, error handling, and security.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeUserCtx:
    user_id: str = "test_user"
    org_id: str = "org_123"
    role: str = "admin"
    is_authenticated: bool = True
    error_reason: str | None = None


@dataclass
class FakeSummary:
    """Fake summary returned by analytics dashboard."""

    total_debates: int = 47
    total_messages: int = 312
    consensus_rate: float = 72.3
    avg_debate_duration_ms: int = 45200
    active_users_24h: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_debates": self.total_debates,
            "total_messages": self.total_messages,
            "consensus_rate": self.consensus_rate,
            "avg_debate_duration_ms": self.avg_debate_duration_ms,
            "active_users_24h": self.active_users_24h,
        }


@dataclass
class FakeTrend:
    """Fake trend data point."""

    date: str = "2026-02-20"
    findings: int = 5
    resolved: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "findings": self.findings,
            "resolved": self.resolved,
        }


@dataclass
class FakeRemediationMetrics:
    """Fake remediation metrics."""

    total_findings: int = 21
    remediated: int = 18
    pending: int = 3
    avg_remediation_time_hours: float = 2.4
    remediation_rate: float = 85.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_findings": self.total_findings,
            "remediated": self.remediated,
            "pending": self.pending,
            "avg_remediation_time_hours": self.avg_remediation_time_hours,
            "remediation_rate": self.remediation_rate,
        }


@dataclass
class FakeComplianceScore:
    """Fake compliance score for a framework."""

    framework: str = "SOC2"
    score: float = 94.0
    status: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "score": self.score,
            "status": self.status,
        }


@dataclass
class FakeHeatmapCell:
    """Fake heatmap cell."""

    category: str = "security"
    severity: str = "high"
    count: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "count": self.count,
        }


def _body(result) -> dict:
    """Extract decoded JSON body from a HandlerResult (tuple-style unpacking)."""
    body, _status, _headers = result
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    _body_val, status, _headers = result
    return status


def _make_handler() -> AnalyticsDashboardHandler:
    """Create a fresh AnalyticsDashboardHandler instance."""
    return AnalyticsDashboardHandler(ctx={})


def _mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler with minimal auth."""
    h = MagicMock()
    h.headers = {"Authorization": "Bearer test-token", "Content-Length": "0"}
    h.command = "GET"
    h.path = "/api/analytics/summary"
    return h


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_auth():
    """Patch auth to always succeed for all tests."""
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=FakeUserCtx(),
    ):
        yield


@pytest.fixture
def handler():
    return _make_handler()


@pytest.fixture
def mock_http():
    return _mock_http_handler()


# =========================================================================
# 1. SUMMARY ENDPOINT - GET /api/analytics/summary
# =========================================================================


class TestGetSummary:
    """Tests for _get_summary endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_happy_path(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123", "time_range": "7d"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 47
        assert body["total_messages"] == 312
        assert body["consensus_rate"] == 72.3

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_default_time_range(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        # Default time_range is 30d -- we just verify it succeeded
        assert "total_debates" in body

    def test_summary_missing_workspace_id(self, handler, mock_http):
        result = handler._get_summary({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_summary_empty_workspace_id(self, handler, mock_http):
        result = handler._get_summary({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    def test_summary_none_workspace_id(self, handler, mock_http):
        result = handler._get_summary({"workspace_id": None}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad range"))
    def test_summary_invalid_time_range(self, _mock_tr, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123", "time_range": "invalid"},
            handler=mock_http,
        )
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "time_range" in error_text.lower() or "invalid" in error_text.lower()

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_summary_import_error(self, _mock, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch(
        "aragora.server.handlers.analytics_dashboard._run_async",
        side_effect=RuntimeError("event loop"),
    )
    def test_summary_runtime_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async", side_effect=OSError("disk"))
    def test_summary_os_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_key_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = KeyError("missing_key")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_type_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = TypeError("bad type")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_attribute_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = AttributeError("no attr")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_includes_all_fields(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "total_debates" in body
        assert "total_messages" in body
        assert "consensus_rate" in body
        assert "avg_debate_duration_ms" in body
        assert "active_users_24h" in body

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_various_time_ranges(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """Test that different valid time_range values are passed through."""
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        for tr in ("24h", "7d", "30d", "90d", "365d", "all"):
            result = handler._get_summary(
                {"workspace_id": "ws_123", "time_range": tr},
                handler=mock_http,
            )
            assert _status(result) == 200

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_calls_dashboard_correctly(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard
        mock_tr_instance = MagicMock()
        mock_time_range.return_value = mock_tr_instance

        handler._get_summary(
            {"workspace_id": "ws_abc", "time_range": "7d"},
            handler=mock_http,
        )
        mock_time_range.assert_called_with("7d")
        mock_dashboard.get_summary.assert_called_once_with("ws_abc", mock_tr_instance)

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_custom_values(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        custom = FakeSummary(total_debates=100, consensus_rate=95.5)
        mock_run_async.return_value = custom
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_debates"] == 100
        assert body["consensus_rate"] == 95.5


# =========================================================================
# 2. FINDING TRENDS ENDPOINT - GET /api/analytics/trends/findings
# =========================================================================


class TestGetFindingTrends:
    """Tests for _get_finding_trends endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_happy_path(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        trends = [FakeTrend(), FakeTrend(date="2026-02-21", findings=3, resolved=2)]
        mock_arun.return_value = trends
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123", "time_range": "7d", "granularity": "day"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["time_range"] == "7d"
        assert body["granularity"] == "day"
        assert len(body["trends"]) == 2

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_default_params(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_arun.return_value = [FakeTrend()]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["time_range"] == "30d"
        assert body["granularity"] == "day"

    def test_trends_missing_workspace_id(self, handler, mock_http):
        result = handler._get_finding_trends({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_trends_empty_workspace_id(self, handler, mock_http):
        result = handler._get_finding_trends({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    def test_trends_none_workspace_id(self, handler, mock_http):
        result = handler._get_finding_trends({"workspace_id": None}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad range"))
    def test_trends_invalid_time_range(self, _mock_tr, handler, mock_http):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123", "time_range": "invalid"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity", side_effect=ValueError("bad granularity"))
    def test_trends_invalid_granularity(self, _mock_g, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123", "granularity": "invalid"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_trends_import_error(self, _mock, handler, mock_http):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run", side_effect=RuntimeError("event loop"))
    def test_trends_runtime_error(
        self, _mock_arun, _mock_g, _mock_tr, _mock_dash, handler, mock_http
    ):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run", side_effect=OSError("disk error"))
    def test_trends_os_error(self, _mock_arun, _mock_g, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_key_error(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_trend = MagicMock()
        bad_trend.to_dict.side_effect = KeyError("missing_key")
        mock_arun.return_value = [bad_trend]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_type_error(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_trend = MagicMock()
        bad_trend.to_dict.side_effect = TypeError("bad type")
        mock_arun.return_value = [bad_trend]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_attribute_error(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_arun.side_effect = AttributeError("no attr")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_empty_result(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_arun.return_value = []
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["trends"] == []

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_calls_dashboard_correctly(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_arun.return_value = []
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard
        mock_tr_instance = MagicMock()
        mock_time_range.return_value = mock_tr_instance
        mock_g_instance = MagicMock()
        mock_granularity.return_value = mock_g_instance

        handler._get_finding_trends(
            {"workspace_id": "ws_abc", "time_range": "7d", "granularity": "hour"},
            handler=mock_http,
        )
        mock_time_range.assert_called_with("7d")
        mock_granularity.assert_called_with("hour")
        mock_dashboard.get_finding_trends.assert_called_once_with(
            "ws_abc", mock_tr_instance, mock_g_instance
        )

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_various_granularities(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_arun.return_value = []
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        for g in ("hour", "day", "week", "month"):
            result = handler._get_finding_trends(
                {"workspace_id": "ws_123", "granularity": g},
                handler=mock_http,
            )
            assert _status(result) == 200

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_response_structure(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        trends = [FakeTrend(date="2026-02-20", findings=5, resolved=3)]
        mock_arun.return_value = trends
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "time_range" in body
        assert "granularity" in body
        assert "trends" in body
        assert isinstance(body["trends"], list)
        assert body["trends"][0]["date"] == "2026-02-20"
        assert body["trends"][0]["findings"] == 5


# =========================================================================
# 3. REMEDIATION METRICS ENDPOINT - GET /api/analytics/remediation
# =========================================================================


class TestGetRemediationMetrics:
    """Tests for _get_remediation_metrics endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_happy_path(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeRemediationMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123", "time_range": "7d"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["time_range"] == "7d"
        assert body["total_findings"] == 21
        assert body["remediated"] == 18
        assert body["remediation_rate"] == 85.7

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_default_time_range(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeRemediationMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_remediation_missing_workspace_id(self, handler, mock_http):
        result = handler._get_remediation_metrics({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_remediation_empty_workspace_id(self, handler, mock_http):
        result = handler._get_remediation_metrics({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    def test_remediation_none_workspace_id(self, handler, mock_http):
        result = handler._get_remediation_metrics({"workspace_id": None}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad range"))
    def test_remediation_invalid_time_range(self, _mock_tr, handler, mock_http):
        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123", "time_range": "bad"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_remediation_import_error(self, _mock, handler, mock_http):
        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch(
        "aragora.server.handlers.analytics_dashboard._run_async", side_effect=RuntimeError("boom")
    )
    def test_remediation_runtime_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async", side_effect=OSError("disk"))
    def test_remediation_os_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_key_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = KeyError("missing_key")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_type_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = TypeError("bad type")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_includes_all_fields(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeRemediationMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "time_range" in body
        assert "total_findings" in body
        assert "remediated" in body
        assert "pending" in body
        assert "avg_remediation_time_hours" in body
        assert "remediation_rate" in body

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_calls_dashboard_correctly(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeRemediationMetrics()
        mock_run_async.return_value = fake_metrics
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard
        mock_tr_instance = MagicMock()
        mock_time_range.return_value = mock_tr_instance

        handler._get_remediation_metrics(
            {"workspace_id": "ws_xyz", "time_range": "90d"},
            handler=mock_http,
        )
        mock_time_range.assert_called_with("90d")
        mock_dashboard.get_remediation_metrics.assert_called_once_with("ws_xyz", mock_tr_instance)

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_custom_values(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        custom = FakeRemediationMetrics(
            total_findings=100, remediated=50, pending=50, remediation_rate=50.0
        )
        mock_run_async.return_value = custom
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_findings"] == 100
        assert body["remediated"] == 50
        assert body["remediation_rate"] == 50.0

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_remediation_attribute_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = AttributeError("no attr")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_remediation_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400


# =========================================================================
# 4. COMPLIANCE SCORECARD ENDPOINT - GET /api/analytics/compliance
# =========================================================================


class TestGetComplianceScorecard:
    """Tests for _get_compliance_scorecard endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_happy_path(self, mock_run_async, mock_get_dash, handler, mock_http):
        scores = [
            FakeComplianceScore("SOC2", 94.0, "pass"),
            FakeComplianceScore("GDPR", 91.0, "pass"),
        ]
        mock_run_async.return_value = scores
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert len(body["scores"]) == 2
        assert body["scores"][0]["framework"] == "SOC2"
        assert body["scores"][0]["score"] == 94.0

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_default_frameworks(self, mock_run_async, mock_get_dash, handler, mock_http):
        mock_run_async.return_value = []
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard

        handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        # Default frameworks list is SOC2,GDPR,HIPAA,PCI-DSS
        call_args = mock_dashboard.get_compliance_scorecard.call_args
        frameworks_arg = call_args[0][1]
        assert "SOC2" in frameworks_arg
        assert "GDPR" in frameworks_arg
        assert "HIPAA" in frameworks_arg
        assert "PCI-DSS" in frameworks_arg

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_custom_frameworks(self, mock_run_async, mock_get_dash, handler, mock_http):
        mock_run_async.return_value = []
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard

        handler._get_compliance_scorecard(
            {"workspace_id": "ws_123", "frameworks": "SOC2,HIPAA"},
            handler=mock_http,
        )
        call_args = mock_dashboard.get_compliance_scorecard.call_args
        frameworks_arg = call_args[0][1]
        assert frameworks_arg == ["SOC2", "HIPAA"]

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_single_framework(self, mock_run_async, mock_get_dash, handler, mock_http):
        mock_run_async.return_value = [FakeComplianceScore("SOC2", 94.0, "pass")]
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard

        handler._get_compliance_scorecard(
            {"workspace_id": "ws_123", "frameworks": "SOC2"},
            handler=mock_http,
        )
        call_args = mock_dashboard.get_compliance_scorecard.call_args
        frameworks_arg = call_args[0][1]
        assert frameworks_arg == ["SOC2"]

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_frameworks_with_spaces(
        self, mock_run_async, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = []
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard

        handler._get_compliance_scorecard(
            {"workspace_id": "ws_123", "frameworks": "SOC2 , GDPR , HIPAA"},
            handler=mock_http,
        )
        call_args = mock_dashboard.get_compliance_scorecard.call_args
        frameworks_arg = call_args[0][1]
        assert frameworks_arg == ["SOC2", "GDPR", "HIPAA"]

    def test_compliance_missing_workspace_id(self, handler, mock_http):
        result = handler._get_compliance_scorecard({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_compliance_empty_workspace_id(self, handler, mock_http):
        result = handler._get_compliance_scorecard({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    def test_compliance_none_workspace_id(self, handler, mock_http):
        result = handler._get_compliance_scorecard({"workspace_id": None}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_compliance_import_error(self, _mock, handler, mock_http):
        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch(
        "aragora.server.handlers.analytics_dashboard._run_async",
        side_effect=RuntimeError("event loop"),
    )
    def test_compliance_runtime_error(self, _mock_run, _mock_dash, handler, mock_http):
        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async", side_effect=OSError("disk"))
    def test_compliance_os_error(self, _mock_run, _mock_dash, handler, mock_http):
        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch(
        "aragora.server.handlers.analytics_dashboard._run_async", side_effect=ValueError("bad val")
    )
    def test_compliance_value_error(self, _mock_run, _mock_dash, handler, mock_http):
        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_key_error(self, mock_run_async, mock_get_dash, handler, mock_http):
        bad_score = MagicMock()
        bad_score.to_dict.side_effect = KeyError("missing")
        mock_run_async.return_value = [bad_score]
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_type_error(self, mock_run_async, mock_get_dash, handler, mock_http):
        bad_score = MagicMock()
        bad_score.to_dict.side_effect = TypeError("bad type")
        mock_run_async.return_value = [bad_score]
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_attribute_error(self, mock_run_async, mock_get_dash, handler, mock_http):
        bad_score = MagicMock()
        bad_score.to_dict.side_effect = AttributeError("no attr")
        mock_run_async.return_value = [bad_score]
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_empty_scores(self, mock_run_async, mock_get_dash, handler, mock_http):
        mock_run_async.return_value = []
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["scores"] == []

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_compliance_response_structure(self, mock_run_async, mock_get_dash, handler, mock_http):
        scores = [FakeComplianceScore("SOC2", 94.0, "pass")]
        mock_run_async.return_value = scores
        mock_get_dash.return_value = MagicMock()

        result = handler._get_compliance_scorecard(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "scores" in body
        assert isinstance(body["scores"], list)


# =========================================================================
# 5. RISK HEATMAP ENDPOINT - GET /api/analytics/heatmap
# =========================================================================


class TestGetRiskHeatmap:
    """Tests for _get_risk_heatmap endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_happy_path(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        cells = [
            FakeHeatmapCell("security", "high", 5),
            FakeHeatmapCell("reliability", "medium", 3),
        ]
        mock_run_async.return_value = cells
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123", "time_range": "7d"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["time_range"] == "7d"
        assert len(body["cells"]) == 2
        assert body["cells"][0]["category"] == "security"
        assert body["cells"][0]["count"] == 5

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_default_time_range(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = []
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_heatmap_missing_workspace_id(self, handler, mock_http):
        result = handler._get_risk_heatmap({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_heatmap_empty_workspace_id(self, handler, mock_http):
        result = handler._get_risk_heatmap({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    def test_heatmap_none_workspace_id(self, handler, mock_http):
        result = handler._get_risk_heatmap({"workspace_id": None}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad range"))
    def test_heatmap_invalid_time_range(self, _mock_tr, handler, mock_http):
        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123", "time_range": "bad"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_heatmap_import_error(self, _mock, handler, mock_http):
        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch(
        "aragora.server.handlers.analytics_dashboard._run_async",
        side_effect=RuntimeError("event loop"),
    )
    def test_heatmap_runtime_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async", side_effect=OSError("disk"))
    def test_heatmap_os_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_key_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_cell = MagicMock()
        bad_cell.to_dict.side_effect = KeyError("missing")
        mock_run_async.return_value = [bad_cell]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_type_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_cell = MagicMock()
        bad_cell.to_dict.side_effect = TypeError("bad type")
        mock_run_async.return_value = [bad_cell]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_attribute_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_cell = MagicMock()
        bad_cell.to_dict.side_effect = AttributeError("no attr")
        mock_run_async.return_value = [bad_cell]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_empty_cells(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = []
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["cells"] == []

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_response_structure(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        cells = [FakeHeatmapCell()]
        mock_run_async.return_value = cells
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "time_range" in body
        assert "cells" in body
        assert isinstance(body["cells"], list)

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_calls_dashboard_correctly(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = []
        mock_dashboard = MagicMock()
        mock_get_dash.return_value = mock_dashboard
        mock_tr_instance = MagicMock()
        mock_time_range.return_value = mock_tr_instance

        handler._get_risk_heatmap(
            {"workspace_id": "ws_xyz", "time_range": "365d"},
            handler=mock_http,
        )
        mock_time_range.assert_called_with("365d")
        mock_dashboard.get_risk_heatmap.assert_called_once_with("ws_xyz", mock_tr_instance)

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_multiple_cells(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        cells = [
            FakeHeatmapCell("security", "critical", 10),
            FakeHeatmapCell("security", "high", 5),
            FakeHeatmapCell("reliability", "high", 3),
            FakeHeatmapCell("performance", "medium", 8),
        ]
        mock_run_async.return_value = cells
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["cells"]) == 4

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_heatmap_value_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """ValueError from _run_async should return 400."""
        mock_run_async.side_effect = ValueError("bad value")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_risk_heatmap(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400


# =========================================================================
# 6. ROUTING TESTS - handle() method
# =========================================================================


class TestRouting:
    """Tests for the handle() routing in the main handler."""

    def test_can_handle_summary(self, handler):
        assert handler.can_handle("/api/analytics/summary")

    def test_can_handle_trends(self, handler):
        assert handler.can_handle("/api/analytics/trends/findings")

    def test_can_handle_remediation(self, handler):
        assert handler.can_handle("/api/analytics/remediation")

    def test_can_handle_compliance(self, handler):
        assert handler.can_handle("/api/analytics/compliance")

    def test_can_handle_heatmap(self, handler):
        assert handler.can_handle("/api/analytics/heatmap")

    def test_can_handle_versioned_summary(self, handler):
        assert handler.can_handle("/api/v1/analytics/summary")

    def test_can_handle_versioned_trends(self, handler):
        assert handler.can_handle("/api/v1/analytics/trends/findings")

    def test_can_handle_versioned_remediation(self, handler):
        assert handler.can_handle("/api/v1/analytics/remediation")

    def test_can_handle_versioned_compliance(self, handler):
        assert handler.can_handle("/api/v1/analytics/compliance")

    def test_can_handle_versioned_heatmap(self, handler):
        assert handler.can_handle("/api/v1/analytics/heatmap")

    def test_cannot_handle_unknown(self, handler):
        assert not handler.can_handle("/api/analytics/unknown")

    def test_cannot_handle_empty(self, handler):
        assert not handler.can_handle("")

    def test_cannot_handle_root(self, handler):
        assert not handler.can_handle("/")

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_handle_routes_summary(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """handle() routes /api/analytics/summary to _get_summary."""
        fake_summary = FakeSummary()
        mock_run_async.return_value = fake_summary
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler.handle(
            "/api/analytics/summary",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_handle_routes_trends(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """handle() routes /api/analytics/trends/findings to _get_finding_trends."""
        mock_arun.return_value = [FakeTrend()]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler.handle(
            "/api/analytics/trends/findings",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_handle_routes_remediation(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """handle() routes /api/analytics/remediation to _get_remediation_metrics."""
        mock_run_async.return_value = FakeRemediationMetrics()
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler.handle(
            "/api/analytics/remediation",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_handle_routes_heatmap(self, mock_run_async, mock_get_dash, handler, mock_http):
        """handle() routes /api/analytics/heatmap to _get_risk_heatmap."""
        mock_run_async.return_value = []
        mock_get_dash.return_value = MagicMock()

        result = handler.handle(
            "/api/analytics/heatmap",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None

    def test_handle_returns_stub_for_unauthenticated_summary(self, handler):
        """No handler/no auth returns stub response for known routes."""
        result = handler.handle(
            "/api/analytics/summary",
            {},
            None,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        # Stub response has "summary" key
        assert "summary" in body

    def test_handle_returns_stub_for_unauthenticated_trends(self, handler):
        result = handler.handle(
            "/api/analytics/trends/findings",
            {},
            None,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "trends" in body

    def test_handle_returns_stub_for_unauthenticated_remediation(self, handler):
        result = handler.handle(
            "/api/analytics/remediation",
            {},
            None,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "metrics" in body

    def test_handle_returns_stub_for_unauthenticated_compliance(self, handler):
        result = handler.handle(
            "/api/analytics/compliance",
            {},
            None,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "compliance" in body

    def test_handle_returns_stub_for_unauthenticated_heatmap(self, handler):
        result = handler.handle(
            "/api/analytics/heatmap",
            {},
            None,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "heatmap" in body

    def test_handle_returns_stub_without_workspace_id(self, handler, mock_http):
        """Authenticated but no workspace_id returns stub."""
        result = handler.handle(
            "/api/analytics/summary",
            {},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_handle_versioned_routes_summary(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        """Versioned path /api/v1/analytics/summary routes correctly."""
        mock_run_async.return_value = FakeSummary()
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler.handle(
            "/api/v1/analytics/summary",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200


# =========================================================================
# 7. AUTHENTICATION TESTS
# =========================================================================


class TestAuthentication:
    """Tests for authentication behavior."""

    @pytest.mark.no_auto_auth
    def test_summary_no_handler_returns_401(self):
        """Calling _get_summary without handler arg returns 401."""
        h = _make_handler()
        result = h._get_summary({"workspace_id": "ws_123"})
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_trends_no_handler_returns_401(self):
        h = _make_handler()
        result = h._get_finding_trends({"workspace_id": "ws_123"})
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_remediation_no_handler_returns_401(self):
        h = _make_handler()
        result = h._get_remediation_metrics({"workspace_id": "ws_123"})
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_compliance_no_handler_returns_401(self):
        h = _make_handler()
        result = h._get_compliance_scorecard({"workspace_id": "ws_123"})
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_heatmap_no_handler_returns_401(self):
        h = _make_handler()
        result = h._get_risk_heatmap({"workspace_id": "ws_123"})
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_unauthenticated_user_returns_401(self):
        """Unauthenticated user gets 401."""
        h = _make_handler()
        mock_http = _mock_http_handler()
        unauth = FakeUserCtx(is_authenticated=False, error_reason="Token expired")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=unauth,
        ):
            result = h._get_summary({"workspace_id": "ws_123"}, handler=mock_http)
        assert _status(result) == 401


# =========================================================================
# 8. ERROR CODE TESTS
# =========================================================================


class TestErrorCodes:
    """Tests that error responses include proper error codes."""

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad"))
    def test_summary_invalid_time_range_code(self, _mock_tr, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123", "time_range": "invalid"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "INVALID_TIME_RANGE"

    def test_trends_missing_workspace_code(self, handler, mock_http):
        result = handler._get_finding_trends({}, handler=mock_http)
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_WORKSPACE_ID"

    def test_remediation_missing_workspace_code(self, handler, mock_http):
        result = handler._get_remediation_metrics({}, handler=mock_http)
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_WORKSPACE_ID"

    def test_compliance_missing_workspace_code(self, handler, mock_http):
        result = handler._get_compliance_scorecard({}, handler=mock_http)
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_WORKSPACE_ID"

    def test_heatmap_missing_workspace_code(self, handler, mock_http):
        result = handler._get_risk_heatmap({}, handler=mock_http)
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_WORKSPACE_ID"

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_summary_internal_error_code(self, _mock, handler, mock_http):
        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "INTERNAL_ERROR"

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_trends_internal_error_code(self, _mock, handler, mock_http):
        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "INTERNAL_ERROR"

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_summary_data_error_code(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = KeyError("missing")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_summary(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "DATA_ERROR"

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.analytics.Granularity")
    @patch("asyncio.run")
    def test_trends_data_error_code(
        self, mock_arun, mock_granularity, mock_time_range, mock_get_dash, handler, mock_http
    ):
        bad_trend = MagicMock()
        bad_trend.to_dict.side_effect = KeyError("missing")
        mock_arun.return_value = [bad_trend]
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()
        mock_granularity.return_value = MagicMock()

        result = handler._get_finding_trends(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "DATA_ERROR"
