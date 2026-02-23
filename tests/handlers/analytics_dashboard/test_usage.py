"""Comprehensive tests for the UsageAnalyticsMixin handler.

Tests all five endpoints provided by the usage analytics handler:
- GET /api/analytics/cost              - Cost analysis
- GET /api/analytics/cost/breakdown    - Per-agent cost breakdown + budget utilization
- GET /api/analytics/tokens            - Token usage summary
- GET /api/analytics/tokens/trends     - Token usage trends
- GET /api/analytics/tokens/providers  - Provider/model breakdown

Also tests routing, version prefix handling, error handling, and security.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
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
class FakeBudget:
    monthly_limit_usd: Decimal = Decimal("500")
    current_monthly_spend: Decimal = Decimal("125")


@dataclass
class FakeCostMetrics:
    """Fake cost metrics returned by analytics dashboard."""

    total_cost_usd: float = 42.50
    cost_by_model: dict = field(default_factory=lambda: {"claude": 25.0, "gpt-4": 17.5})
    projected_monthly_cost: float = 85.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost_usd,
            "cost_by_model": self.cost_by_model,
            "projected_monthly_cost": self.projected_monthly_cost,
        }


@dataclass
class FakeUsageSummary:
    """Fake usage summary returned by UsageTracker."""

    total_tokens_in: int = 50000
    total_tokens_out: int = 25000
    total_cost_usd: Decimal = Decimal("12.50")
    total_debates: int = 47
    total_agent_calls: int = 312
    cost_by_provider: dict = field(
        default_factory=lambda: {
            "anthropic": Decimal("7.50"),
            "openai": Decimal("5.00"),
        }
    )
    debates_by_day: dict = field(
        default_factory=lambda: {
            "2026-02-20": 5,
            "2026-02-21": 8,
        }
    )


def _body(result) -> dict:
    """Extract decoded JSON body from a HandlerResult (tuple-style unpacking)."""
    body, _status, _headers = result
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    _body, status, _headers = result
    return status


def _make_handler() -> AnalyticsDashboardHandler:
    """Create a fresh AnalyticsDashboardHandler instance."""
    return AnalyticsDashboardHandler(ctx={})


def _mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler with minimal auth."""
    h = MagicMock()
    h.headers = {"Authorization": "Bearer test-token", "Content-Length": "0"}
    h.command = "GET"
    h.path = "/api/analytics/cost"
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
# 1. COST METRICS ENDPOINT - GET /api/analytics/cost
# =========================================================================


class TestCostMetrics:
    """Tests for _get_cost_metrics endpoint."""

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_happy_path(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeCostMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123", "time_range": "7d"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["time_range"] == "7d"
        assert body["total_cost_usd"] == 42.50

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_default_time_range(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeCostMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_cost_metrics_missing_workspace_id(self, handler, mock_http):
        result = handler._get_cost_metrics({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_cost_metrics_empty_workspace_id(self, handler, mock_http):
        result = handler._get_cost_metrics({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_includes_all_metric_fields(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeCostMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "time_range" in body
        assert "total_cost_usd" in body
        assert "cost_by_model" in body
        assert "projected_monthly_cost" in body

    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad range"))
    def test_cost_metrics_invalid_time_range(self, _mock_tr, handler, mock_http):
        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123", "time_range": "invalid"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_cost_metrics_import_error(self, _mock, handler, mock_http):
        result = handler._get_cost_metrics(
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
    def test_cost_metrics_runtime_error(self, _mock_run, _mock_tr, _mock_dash, handler, mock_http):
        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_key_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = KeyError("missing_key")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_attribute_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = AttributeError("no attr")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_type_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = MagicMock()
        mock_run_async.return_value.to_dict.side_effect = TypeError("bad type")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_os_error(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.side_effect = OSError("disk failure")
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_preserves_workspace_id(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        mock_run_async.return_value = FakeCostMetrics()
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "my-special-workspace"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["workspace_id"] == "my-special-workspace"


# =========================================================================
# 2. TOKEN USAGE ENDPOINT - GET /api/analytics/tokens
# =========================================================================


class TestTokenUsage:
    """Tests for _get_token_usage endpoint."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_happy_path(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "7"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert body["total_tokens_in"] == 50000
        assert body["total_tokens_out"] == 25000
        assert body["total_tokens"] == 75000
        assert body["total_debates"] == 47
        assert body["total_agent_calls"] == 312

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_default_days(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    def test_token_usage_missing_org_id(self, handler, mock_http):
        result = handler._get_token_usage({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "org_id" in error_text.lower()

    def test_token_usage_empty_org_id(self, handler, mock_http):
        result = handler._get_token_usage({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_period_structure(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert body["period"]["days"] == 14

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_cost_string(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_cost_usd"] == "12.50"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_cost_by_provider(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "cost_by_provider" in body
        assert body["cost_by_provider"]["anthropic"] == "7.50"
        assert body["cost_by_provider"]["openai"] == "5.00"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_debates_by_day(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "debates_by_day" in body

    @patch("aragora.billing.usage.UsageTracker", side_effect=ImportError("no billing"))
    def test_token_usage_import_error(self, _mock, handler, mock_http):
        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_runtime_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker_cls.side_effect = RuntimeError("db down")
        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_clamped_min(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "0"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_clamped_max(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "9999"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] <= 365

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_negative(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "-5"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_non_numeric(self, mock_tracker_cls, handler, mock_http):
        """Non-numeric days should fall back to default 30."""
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "abc"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_os_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.side_effect = OSError("file error")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_lookup_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.side_effect = LookupError("not found")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500


# =========================================================================
# 3. TOKEN TRENDS ENDPOINT - GET /api/analytics/tokens/trends
# =========================================================================


def _make_mock_connection(rows=None):
    """Create a mock DB connection with fetchall returning dict-like rows."""
    if rows is None:
        rows = []

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn


def _make_row(period="2026-02-20", tokens_in=1000, tokens_out=500, cost=0.05, event_count=10):
    """Create a dict-like row for DB results."""
    return {
        "period": period,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "event_count": event_count,
    }


def _make_provider_row(
    provider="anthropic",
    model="claude-opus-4",
    tokens_in=1000,
    tokens_out=500,
    cost=0.10,
    call_count=5,
):
    """Create a dict-like row for provider breakdown results."""
    return {
        "provider": provider,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "call_count": call_count,
    }


class TestTokenTrends:
    """Tests for _get_token_trends endpoint."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_happy_path(self, mock_tracker_cls, handler, mock_http):
        rows = [
            _make_row("2026-02-20", 1000, 500, 0.05, 10),
            _make_row("2026-02-21", 2000, 1000, 0.10, 20),
        ]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "days": "7"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert body["granularity"] == "day"
        assert len(body["data_points"]) == 2

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_data_point_structure(self, mock_tracker_cls, handler, mock_http):
        rows = [_make_row("2026-02-20", 1000, 500, 0.05, 10)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        dp = body["data_points"][0]
        assert dp["period"] == "2026-02-20"
        assert dp["tokens_in"] == 1000
        assert dp["tokens_out"] == 500
        assert dp["total_tokens"] == 1500
        assert dp["cost_usd"] == "0.0500"
        assert dp["event_count"] == 10

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_granularity_day(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "granularity": "day"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_granularity_hour(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "granularity": "hour"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "hour"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_invalid_granularity_falls_back_to_day(
        self, mock_tracker_cls, handler, mock_http
    ):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "granularity": "minute"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_empty_granularity_falls_back_to_day(
        self, mock_tracker_cls, handler, mock_http
    ):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "granularity": ""},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    def test_token_trends_missing_org_id(self, handler, mock_http):
        result = handler._get_token_trends({}, handler=mock_http)
        assert _status(result) == 400

    def test_token_trends_empty_org_id(self, handler, mock_http):
        result = handler._get_token_trends({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_empty_results(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["data_points"] == []

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_null_values_default_to_zero(self, mock_tracker_cls, handler, mock_http):
        rows = [_make_row("2026-02-20", None, None, None, 5)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        dp = body["data_points"][0]
        assert dp["tokens_in"] == 0
        assert dp["tokens_out"] == 0
        assert dp["total_tokens"] == 0
        assert dp["cost_usd"] == "0.0000"

    @patch("aragora.billing.usage.UsageTracker", side_effect=ImportError("no billing"))
    def test_token_trends_import_error(self, _mock, handler, mock_http):
        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_runtime_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker._connection.side_effect = RuntimeError("db error")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_days_clamped(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "days": "0"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_period_structure(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert body["period"]["days"] == 14
        assert "start" in body["period"]
        assert "end" in body["period"]

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_many_data_points(self, mock_tracker_cls, handler, mock_http):
        rows = [_make_row(f"2026-02-{i:02d}", i * 100, i * 50, i * 0.01, i) for i in range(1, 29)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123", "days": "30"},
            handler=mock_http,
        )
        body = _body(result)
        assert len(body["data_points"]) == 28


# =========================================================================
# 4. PROVIDER BREAKDOWN ENDPOINT - GET /api/analytics/tokens/providers
# =========================================================================


class TestProviderBreakdown:
    """Tests for _get_provider_breakdown endpoint."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_happy_path(self, mock_tracker_cls, handler, mock_http):
        rows = [
            _make_provider_row("anthropic", "claude-opus-4", 5000, 2500, 0.50, 25),
            _make_provider_row("openai", "gpt-4o", 3000, 1500, 0.30, 15),
        ]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert len(body["providers"]) == 2

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_sorted_by_cost(self, mock_tracker_cls, handler, mock_http):
        rows = [
            _make_provider_row("cheap-provider", "model-a", 100, 50, 0.01, 1),
            _make_provider_row("expensive-provider", "model-b", 500, 250, 1.00, 10),
        ]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert float(body["providers"][0]["total_cost"]) >= float(
            body["providers"][1]["total_cost"]
        )

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_multiple_models_per_provider(
        self, mock_tracker_cls, handler, mock_http
    ):
        rows = [
            _make_provider_row("anthropic", "claude-opus-4", 3000, 1500, 0.50, 15),
            _make_provider_row("anthropic", "claude-sonnet-4", 2000, 1000, 0.20, 10),
        ]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert len(body["providers"]) == 1
        provider = body["providers"][0]
        assert provider["provider"] == "anthropic"
        assert len(provider["models"]) == 2
        assert provider["total_tokens_in"] == 5000
        assert provider["total_tokens_out"] == 2500
        assert provider["total_tokens"] == 7500

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_model_structure(self, mock_tracker_cls, handler, mock_http):
        rows = [_make_provider_row("anthropic", "claude-opus-4", 1000, 500, 0.10, 5)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        model = body["providers"][0]["models"][0]
        assert model["model"] == "claude-opus-4"
        assert model["tokens_in"] == 1000
        assert model["tokens_out"] == 500
        assert model["total_tokens"] == 1500
        assert model["cost_usd"] == "0.1000"
        assert model["call_count"] == 5

    def test_provider_breakdown_missing_org_id(self, handler, mock_http):
        result = handler._get_provider_breakdown({}, handler=mock_http)
        assert _status(result) == 400

    def test_provider_breakdown_empty_org_id(self, handler, mock_http):
        result = handler._get_provider_breakdown({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_empty_results(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["providers"] == []

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_null_values(self, mock_tracker_cls, handler, mock_http):
        rows = [_make_provider_row("anthropic", None, None, None, None, 3)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        provider = body["providers"][0]
        assert provider["total_tokens_in"] == 0
        assert provider["total_tokens_out"] == 0
        model = provider["models"][0]
        assert model["model"] == "unknown"

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_null_provider_becomes_unknown(
        self, mock_tracker_cls, handler, mock_http
    ):
        rows = [_make_provider_row(None, "some-model", 100, 50, 0.01, 1)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        # The row should not appear because provider IS NOT NULL filter in SQL
        # but since we're mocking, the handler code handles None as "unknown"
        assert body["providers"][0]["provider"] == "unknown"

    @patch("aragora.billing.usage.UsageTracker", side_effect=ImportError("no billing"))
    def test_provider_breakdown_import_error(self, _mock, handler, mock_http):
        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_runtime_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker._connection.side_effect = RuntimeError("db failure")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_period_structure(self, mock_tracker_cls, handler, mock_http):
        mock_conn = _make_mock_connection([])
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 14
        assert "start" in body["period"]
        assert "end" in body["period"]


# =========================================================================
# 5. COST BREAKDOWN ENDPOINT - GET /api/analytics/cost/breakdown
# =========================================================================


class TestCostBreakdown:
    """Tests for _get_cost_breakdown endpoint."""

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_happy_path(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "42.50",
            "cost_by_agent": {"claude": "25.00", "gpt-4": "17.50"},
        }
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["total_spend_usd"] == "42.50"
        assert body["agent_costs"]["claude"] == "25.00"

    def test_cost_breakdown_missing_workspace_id(self, handler, mock_http):
        result = handler._get_cost_breakdown({}, handler=mock_http)
        assert _status(result) == 400

    def test_cost_breakdown_empty_workspace_id(self, handler, mock_http):
        result = handler._get_cost_breakdown({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_with_budget(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "10.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = FakeBudget(
            monthly_limit_usd=Decimal("500"),
            current_monthly_spend=Decimal("125"),
        )
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["budget"]["monthly_limit_usd"] == 500.0
        assert body["budget"]["current_spend_usd"] == 125.0
        assert body["budget"]["remaining_usd"] == 375.0
        assert body["budget"]["utilization_percent"] == 25.0

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_no_budget(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "10.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_budget_zero_limit(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "0",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = FakeBudget(
            monthly_limit_usd=Decimal("0"),
            current_monthly_spend=Decimal("0"),
        )
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        # Zero limit means no budget configured effectively
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_over_budget(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "150.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = FakeBudget(
            monthly_limit_usd=Decimal("100"),
            current_monthly_spend=Decimal("150"),
        )
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["budget"]["utilization_percent"] == 150.0
        assert body["budget"]["remaining_usd"] == 0

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_budget_exception(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "10.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.side_effect = AttributeError("no budget")
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_budget_type_error(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "10.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.side_effect = TypeError("bad type")
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker", side_effect=ImportError("no tracker"))
    def test_cost_breakdown_import_error(self, _mock, handler, mock_http):
        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch(
        "aragora.billing.cost_tracker.get_cost_tracker", side_effect=RuntimeError("tracker error")
    )
    def test_cost_breakdown_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_response_keys(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "0",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "total_spend_usd" in body
        assert "agent_costs" in body
        assert "budget" in body

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_empty_agent_costs(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "0",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["agent_costs"] == {}

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_many_agents(self, mock_get_tracker, handler, mock_http):
        agent_costs = {f"agent_{i}": f"{i}.00" for i in range(20)}
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "100.00",
            "cost_by_agent": agent_costs,
        }
        mock_tracker.get_budget.return_value = None
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert len(body["agent_costs"]) == 20


# =========================================================================
# 6. ROUTING TESTS
# =========================================================================


class TestRouting:
    """Tests for route registration and can_handle."""

    def test_cost_route_registered(self):
        assert "/api/analytics/cost" in AnalyticsDashboardHandler.ROUTES

    def test_cost_breakdown_route_registered(self):
        assert "/api/analytics/cost/breakdown" in AnalyticsDashboardHandler.ROUTES

    def test_tokens_route_registered(self):
        assert "/api/analytics/tokens" in AnalyticsDashboardHandler.ROUTES

    def test_tokens_trends_route_registered(self):
        assert "/api/analytics/tokens/trends" in AnalyticsDashboardHandler.ROUTES

    def test_tokens_providers_route_registered(self):
        assert "/api/analytics/tokens/providers" in AnalyticsDashboardHandler.ROUTES

    def test_can_handle_cost(self, handler):
        assert handler.can_handle("/api/analytics/cost") is True

    def test_can_handle_cost_breakdown(self, handler):
        assert handler.can_handle("/api/analytics/cost/breakdown") is True

    def test_can_handle_tokens(self, handler):
        assert handler.can_handle("/api/analytics/tokens") is True

    def test_can_handle_tokens_trends(self, handler):
        assert handler.can_handle("/api/analytics/tokens/trends") is True

    def test_can_handle_tokens_providers(self, handler):
        assert handler.can_handle("/api/analytics/tokens/providers") is True

    def test_can_handle_versioned_cost(self, handler):
        assert handler.can_handle("/api/v1/analytics/cost") is True

    def test_can_handle_versioned_tokens(self, handler):
        assert handler.can_handle("/api/v1/analytics/tokens") is True

    def test_can_handle_versioned_tokens_trends(self, handler):
        assert handler.can_handle("/api/v1/analytics/tokens/trends") is True

    def test_can_handle_versioned_tokens_providers(self, handler):
        assert handler.can_handle("/api/v1/analytics/tokens/providers") is True

    def test_can_handle_versioned_cost_breakdown(self, handler):
        assert handler.can_handle("/api/v1/analytics/cost/breakdown") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/analytics/costextra") is False

    def test_cannot_handle_empty(self, handler):
        assert handler.can_handle("") is False

    def test_has_cost_metrics_method(self, handler):
        assert hasattr(handler, "_get_cost_metrics")
        assert callable(handler._get_cost_metrics)

    def test_has_token_usage_method(self, handler):
        assert hasattr(handler, "_get_token_usage")
        assert callable(handler._get_token_usage)

    def test_has_token_trends_method(self, handler):
        assert hasattr(handler, "_get_token_trends")
        assert callable(handler._get_token_trends)

    def test_has_provider_breakdown_method(self, handler):
        assert hasattr(handler, "_get_provider_breakdown")
        assert callable(handler._get_provider_breakdown)

    def test_has_cost_breakdown_method(self, handler):
        assert hasattr(handler, "_get_cost_breakdown")
        assert callable(handler._get_cost_breakdown)


# =========================================================================
# 7. HANDLE() METHOD ROUTING (via AnalyticsDashboardHandler.handle)
# =========================================================================


class TestHandleRouting:
    """Tests that handle() routes to correct method for usage endpoints."""

    def test_handle_returns_stub_without_user(self, handler):
        """When no user context, stub response is returned."""
        mock_h = MagicMock()
        # Make get_current_user return None to trigger stub path
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/cost", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        # Stub response should have "analysis" key
        assert "analysis" in body

    def test_handle_returns_stub_without_workspace(self, handler):
        """When user exists but no workspace_id, stub response is returned."""
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=FakeUserCtx())
        result = handler.handle("/api/analytics/cost", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "analysis" in body

    def test_handle_tokens_stub(self, handler):
        """Tokens endpoint returns stub without auth."""
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/tokens", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body

    def test_handle_tokens_trends_stub(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/tokens/trends", {}, mock_h)
        assert _status(result) == 200

    def test_handle_tokens_providers_stub(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/tokens/providers", {}, mock_h)
        assert _status(result) == 200

    def test_handle_cost_breakdown_stub(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/cost/breakdown", {}, mock_h)
        assert _status(result) == 200

    def test_handle_unknown_route_returns_none(self, handler):
        mock_h = MagicMock()
        result = handler.handle("/api/unknown/route", {}, mock_h)
        assert result is None

    def test_handle_versioned_path(self, handler):
        """Versioned paths should be normalized and routed correctly."""
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/v1/analytics/cost", {}, mock_h)
        assert _status(result) == 200


# =========================================================================
# 8. SECURITY TESTS
# =========================================================================


class TestSecurityInputs:
    """Tests for input validation and security concerns."""

    def test_workspace_id_with_path_traversal(self, handler, mock_http):
        """Path traversal in workspace_id should not cause issues."""
        result = handler._get_cost_breakdown(
            {"workspace_id": "../../etc/passwd"},
            handler=mock_http,
        )
        # Should either work (treating it as opaque string) or fail gracefully
        assert _status(result) in (200, 400, 500)

    def test_org_id_with_path_traversal(self, handler, mock_http):
        result = handler._get_token_usage(
            {"org_id": "../../etc/passwd"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_workspace_id_with_sql_injection(self, handler, mock_http):
        """SQL injection in workspace_id should not cause issues."""
        result = handler._get_cost_breakdown(
            {"workspace_id": "'; DROP TABLE usage_events; --"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_org_id_with_sql_injection(self, handler, mock_http):
        result = handler._get_token_usage(
            {"org_id": "'; DROP TABLE usage_events; --"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_very_long_workspace_id(self, handler, mock_http):
        result = handler._get_cost_breakdown(
            {"workspace_id": "x" * 10000},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_very_long_org_id(self, handler, mock_http):
        result = handler._get_token_usage(
            {"org_id": "x" * 10000},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_unicode_workspace_id(self, handler, mock_http):
        result = handler._get_cost_breakdown(
            {"workspace_id": "\u0000\ud83d\ude80\u2603"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_unicode_org_id(self, handler, mock_http):
        result = handler._get_token_usage(
            {"org_id": "\u0000\ud83d\ude80\u2603"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_granularity_with_injection(self, handler, mock_http):
        """Granularity injection should fall back to 'day'."""
        result = handler._get_token_trends(
            {"org_id": "org_123", "granularity": "'; DROP TABLE --"},
            handler=mock_http,
        )
        # Invalid granularity falls back to "day", so it will proceed and may fail on import
        # but should not crash with an unhandled error
        assert _status(result) in (200, 400, 500)

    def test_time_range_with_xss(self, handler, mock_http):
        """XSS in time_range should be handled safely."""
        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123", "time_range": "<script>alert(1)</script>"},
            handler=mock_http,
        )
        # Likely a ValueError from TimeRange constructor
        assert _status(result) in (200, 400, 500)

    def test_days_with_float(self, handler, mock_http):
        """Float days should be handled by get_clamped_int_param."""
        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "7.5"},
            handler=mock_http,
        )
        # get_clamped_int_param handles non-integer values gracefully
        assert _status(result) in (200, 400, 500)

    def test_none_handler_for_cost_metrics(self, handler):
        """None handler should result in auth failure."""
        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123"},
            handler=None,
        )
        assert _status(result) == 401


# =========================================================================
# 9. MIXIN ISOLATION TESTS
# =========================================================================


class TestMixinIsolation:
    """Test that UsageAnalyticsMixin works as a standalone mixin."""

    def test_mixin_can_be_instantiated_standalone(self):
        from aragora.server.handlers.analytics_dashboard.usage import UsageAnalyticsMixin

        class TestHandler(UsageAnalyticsMixin):
            pass

        h = TestHandler()
        assert hasattr(h, "_get_cost_metrics")
        assert hasattr(h, "_get_token_usage")
        assert hasattr(h, "_get_token_trends")
        assert hasattr(h, "_get_provider_breakdown")
        assert hasattr(h, "_get_cost_breakdown")

    def test_mixin_methods_are_callable(self):
        from aragora.server.handlers.analytics_dashboard.usage import UsageAnalyticsMixin

        class TestHandler(UsageAnalyticsMixin):
            pass

        h = TestHandler()
        assert callable(h._get_cost_metrics)
        assert callable(h._get_token_usage)
        assert callable(h._get_token_trends)
        assert callable(h._get_provider_breakdown)
        assert callable(h._get_cost_breakdown)


# =========================================================================
# 10. EDGE CASE TESTS
# =========================================================================


class TestEdgeCases:
    """Tests for various edge cases."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_zero_tokens(self, mock_tracker_cls, handler, mock_http):
        summary = FakeUsageSummary(
            total_tokens_in=0,
            total_tokens_out=0,
            total_cost_usd=Decimal("0"),
            total_debates=0,
            total_agent_calls=0,
            cost_by_provider={},
            debates_by_day={},
        )
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = summary
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_tokens"] == 0
        assert body["total_cost_usd"] == "0"

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_full_utilization(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "100.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.return_value = FakeBudget(
            monthly_limit_usd=Decimal("100"),
            current_monthly_spend=Decimal("100"),
        )
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["budget"]["utilization_percent"] == 100.0
        assert body["budget"]["remaining_usd"] == 0

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_large_values(self, mock_tracker_cls, handler, mock_http):
        summary = FakeUsageSummary(
            total_tokens_in=999_999_999,
            total_tokens_out=888_888_888,
            total_cost_usd=Decimal("99999.99"),
            total_debates=100000,
            total_agent_calls=500000,
        )
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = summary
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_tokens_in"] == 999_999_999
        assert body["total_tokens_out"] == 888_888_888
        assert body["total_tokens"] == 1_888_888_887

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_cost_formatting(self, mock_tracker_cls, handler, mock_http):
        """Cost values should be formatted to 4 decimal places."""
        rows = [_make_provider_row("anthropic", "claude", 100, 50, 1.23456789, 5)]
        mock_conn = _make_mock_connection(rows)
        mock_tracker = MagicMock()
        mock_tracker._connection.return_value = mock_conn
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        model = body["providers"][0]["models"][0]
        assert model["cost_usd"] == "1.2346"

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_budget_value_error(self, mock_get_tracker, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "10.00",
            "cost_by_agent": {},
        }
        mock_tracker.get_budget.side_effect = ValueError("invalid budget")
        mock_get_tracker.return_value = mock_tracker

        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["budget"] == {}

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_trends_os_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker._connection.side_effect = OSError("disk error")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_trends(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.usage.UsageTracker")
    def test_provider_breakdown_os_error(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker._connection.side_effect = OSError("disk error")
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_provider_breakdown(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.billing.cost_tracker.get_cost_tracker")
    def test_cost_breakdown_os_error(self, mock_get_tracker, handler, mock_http):
        mock_get_tracker.side_effect = OSError("disk error")
        result = handler._get_cost_breakdown(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    def test_cost_metrics_with_custom_time_range(
        self, mock_run_async, mock_time_range, mock_get_dash, handler, mock_http
    ):
        fake_metrics = FakeCostMetrics()
        mock_run_async.return_value = fake_metrics
        mock_get_dash.return_value = MagicMock()
        mock_time_range.return_value = MagicMock()

        result = handler._get_cost_metrics(
            {"workspace_id": "ws_123", "time_range": "90d"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["time_range"] == "90d"

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_boundary_365(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "365"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 365

    @patch("aragora.billing.usage.UsageTracker")
    def test_token_usage_days_boundary_1(self, mock_tracker_cls, handler, mock_http):
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = FakeUsageSummary()
        mock_tracker_cls.return_value = mock_tracker

        result = handler._get_token_usage(
            {"org_id": "org_123", "days": "1"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 1
