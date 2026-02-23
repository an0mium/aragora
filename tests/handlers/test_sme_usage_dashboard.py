"""Tests for SME Usage Dashboard handler.

Covers all routes and behavior of the SMEUsageDashboardHandler class:
- can_handle() routing
- GET /api/v1/usage/summary       - Unified usage metrics
- GET /api/v1/usage/breakdown     - Detailed breakdown by dimension
- GET /api/v1/usage/roi           - ROI analysis
- GET /api/v1/usage/export        - CSV/PDF/JSON export
- GET /api/v1/usage/budget-status - Budget utilization
- GET /api/v1/usage/forecast      - Usage forecast
- GET /api/v1/usage/benchmarks    - Industry benchmark comparison
- Rate limiting
- Period parsing
- Error paths and validation
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme_usage_dashboard import (
    SMEUsageDashboardHandler,
    _dashboard_limiter,
    _get_real_consensus_rate,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult, unwrapping 'data' envelope."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            raw = json.loads(result.body.decode("utf-8"))
        else:
            raw = result.body
    elif isinstance(result, dict):
        raw = result.get("body", result)
    else:
        raw = {}
    # Unwrap { "data": ... } envelope used by GET endpoints
    if isinstance(raw, dict) and "data" in raw and len(raw) == 1:
        return raw["data"]
    return raw


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        query_params: dict[str, str] | None = None,
    ):
        self.rfile = MagicMock()
        self.command = "GET"
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.client_address = ("127.0.0.1", 12345)
        self._body = body
        self._query_params = query_params or {}
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"

    def get(self, key: str, default=None):
        """Support for get_string_param resolution."""
        return self._query_params.get(key, default)


def _make_handler(
    body: dict[str, Any] | None = None,
    method: str = "GET",
    query_params: dict[str, str] | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body, method, and query params."""
    h = MockHTTPHandler(body=body, query_params=query_params)
    h.command = method
    return h


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


@dataclass
class MockUser:
    """Mock user object."""

    user_id: str = "test-user-001"
    org_id: str = "test-org-001"
    email: str = "test@example.com"


@dataclass
class MockOrg:
    """Mock organization object."""

    id: str = "test-org-001"
    name: str = "Test Org"
    slug: str = "test-org"


@dataclass
class MockBudget:
    """Mock budget object."""

    monthly_limit_usd: Decimal = Decimal("100.00")
    current_monthly_spend: Decimal = Decimal("45.50")
    daily_limit_usd: Decimal = Decimal("10.00")
    current_daily_spend: Decimal = Decimal("3.25")

    def check_alert_level(self):
        return None


class MockAlertLevel:
    """Mock alert level enum value."""

    value = "warning"


@dataclass
class MockUsageSummary:
    """Mock usage summary returned by UsageTracker.get_summary()."""

    total_debates: int = 45
    cost_by_provider: dict | None = None

    def __post_init__(self):
        if self.cost_by_provider is None:
            self.cost_by_provider = {"anthropic": Decimal("8.00"), "openai": Decimal("4.50")}


@dataclass
class MockROIMetrics:
    """Mock ROI metrics returned by ROICalculator.calculate_period_roi()."""

    def to_dict(self):
        return {
            "time_savings_hours": 12.5,
            "cost_savings_usd": "250.00",
            "roi_percent": 185.0,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    _dashboard_limiter._buckets.clear()
    yield
    _dashboard_limiter._buckets.clear()


def _make_workspace_stats(
    total_cost="12.50",
    tokens_in=500000,
    tokens_out=250000,
    api_calls=150,
    cost_by_agent=None,
    cost_by_model=None,
):
    """Create standard workspace stats dict."""
    return {
        "total_cost_usd": total_cost,
        "total_tokens_in": tokens_in,
        "total_tokens_out": tokens_out,
        "total_api_calls": api_calls,
        "cost_by_agent": cost_by_agent or {"claude": "8.00", "gpt-4": "4.50"},
        "cost_by_model": cost_by_model or {"claude-3-opus": "8.00", "gpt-4-turbo": "4.50"},
    }


def _make_mock_user_store(user=None, org=None):
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id.return_value = user or MockUser()
    store.get_organization_by_id.return_value = org or MockOrg()
    return store


def _make_mock_cost_tracker(workspace_stats=None, budget=None):
    """Create a mock cost tracker."""
    tracker = MagicMock()
    tracker.get_workspace_stats.return_value = workspace_stats or _make_workspace_stats()
    tracker.get_budget.return_value = budget
    return tracker


def _make_mock_usage_tracker(summary=None):
    """Create a mock usage tracker."""
    tracker = MagicMock()
    tracker.get_summary.return_value = summary or MockUsageSummary()
    return tracker


def _make_mock_roi_calculator(metrics=None, benchmarks=None, projections=None):
    """Create a mock ROI calculator."""
    calc = MagicMock()
    calc.calculate_period_roi.return_value = metrics or MockROIMetrics()
    calc.get_benchmark_comparison.return_value = benchmarks or {
        "benchmarks": {
            "sme": {"avg_decision_cost_usd": "50.00", "avg_hours_per_decision": 4, "avg_participants": 3},
            "enterprise": {"avg_decision_cost_usd": "200.00", "avg_hours_per_decision": 12, "avg_participants": 8},
        }
    }
    calc.estimate_future_savings.return_value = {
        "projected_monthly_savings": "500.00",
        "projected_annual_savings": "6000.00",
    }
    return calc


@pytest.fixture
def handler():
    """Create an SMEUsageDashboardHandler with standard mocks."""
    user_store = _make_mock_user_store()
    ctx = {"user_store": user_store}
    h = SMEUsageDashboardHandler(ctx=ctx)
    return h


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_summary_path(self, handler):
        assert handler.can_handle("/api/v1/usage/summary")

    def test_breakdown_path(self, handler):
        assert handler.can_handle("/api/v1/usage/breakdown")

    def test_roi_path(self, handler):
        assert handler.can_handle("/api/v1/usage/roi")

    def test_export_path(self, handler):
        assert handler.can_handle("/api/v1/usage/export")

    def test_budget_status_path(self, handler):
        assert handler.can_handle("/api/v1/usage/budget-status")

    def test_forecast_path(self, handler):
        assert handler.can_handle("/api/v1/usage/forecast")

    def test_benchmarks_path(self, handler):
        assert handler.can_handle("/api/v1/usage/benchmarks")

    def test_rejects_unknown_path(self, handler):
        assert not handler.can_handle("/api/v1/usage/unknown")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/usage/summary")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/v1/usage")

    def test_rejects_different_prefix(self, handler):
        assert not handler.can_handle("/api/v1/billing/summary")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = SMEUsageDashboardHandler()
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"user_store": MagicMock()}
        h = SMEUsageDashboardHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "usage_dashboard"

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 7


# ============================================================================
# handle() dispatch
# ============================================================================


class TestHandleDispatch:
    """Test the main handle() dispatch routing."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    def test_dispatch_to_summary(self, mock_consensus, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "period" in body  # summary data unwrapped

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_dispatch_to_breakdown(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "dimension" in body  # breakdown data unwrapped

    def test_dispatch_non_get_returns_405(self, handler):
        h = _make_handler(method="POST")
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 405

    def test_dispatch_delete_returns_405(self, handler):
        h = _make_handler(method="DELETE")
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 405

    def test_dispatch_unknown_path_returns_405(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/usage/nonexistent", {}, h)
        assert _status(result) == 405

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=90.0)
    def test_dispatch_uses_handler_command_attribute(self, mock_consensus, mock_ct, handler):
        """When handler has 'command' attribute, it overrides method param."""
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(method="GET")
        h.command = "GET"
        result = handler.handle("/api/v1/usage/summary", {}, h, method="POST")
        # handler.command is GET, so should still succeed
        assert _status(result) == 200


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limiting on the dashboard handler."""

    def test_rate_limit_allows_normal_requests(self, handler):
        h = _make_handler()
        # Should not hit rate limit for a single request
        with patch.object(handler, "_get_summary", return_value=HandlerResult(
            status_code=200, content_type="application/json", body=b'{"summary": {}}'
        )):
            result = handler.handle("/api/v1/usage/summary", {}, h)
            assert _status(result) == 200

    def test_rate_limit_exceeded_returns_429(self, handler):
        # Fill up the rate limiter
        import time
        now = time.time()
        _dashboard_limiter._buckets["127.0.0.1"] = [now] * 61
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in body.get("error", "").lower()

    def test_rate_limit_different_ips_independent(self, handler):
        import time
        now = time.time()
        # Exhaust limiter for IP 1
        _dashboard_limiter._buckets["10.0.0.1"] = [now] * 61
        # Different IP should be fine
        h = _make_handler()
        h.client_address = ("10.0.0.2", 12345)
        with patch.object(handler, "_get_summary", return_value=HandlerResult(
            status_code=200, content_type="application/json", body=b'{"summary": {}}'
        )):
            result = handler.handle("/api/v1/usage/summary", {}, h)
            assert _status(result) == 200


# ============================================================================
# _get_user_and_org helper
# ============================================================================


class TestGetUserAndOrg:
    """Test the _get_user_and_org internal method."""

    def test_no_user_store_returns_503(self):
        h = SMEUsageDashboardHandler(ctx={})
        mock_user = MockUser()
        _, _, error = h._get_user_and_org(None, mock_user)
        assert _status(error) == 503

    def test_user_not_found_returns_404(self):
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = None
        h = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        mock_user = MockUser()
        _, _, error = h._get_user_and_org(None, mock_user)
        assert _status(error) == 404
        assert "user not found" in _body(error).get("error", "").lower()

    def test_no_org_id_returns_404(self):
        user = MockUser(org_id=None)
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        h = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        _, _, error = h._get_user_and_org(None, MockUser())
        assert _status(error) == 404

    def test_org_not_found_returns_404(self):
        user = MockUser(org_id="missing-org")
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.get_organization_by_id.return_value = None
        h = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        _, _, error = h._get_user_and_org(None, MockUser())
        assert _status(error) == 404
        assert "no organization" in _body(error).get("error", "").lower()

    def test_success_returns_user_and_org(self):
        user = MockUser()
        org = MockOrg()
        user_store = _make_mock_user_store(user=user, org=org)
        h = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        db_user, db_org, error = h._get_user_and_org(None, MockUser())
        assert error is None
        assert db_user is user
        assert db_org is org


# ============================================================================
# _parse_period
# ============================================================================


class TestParsePeriod:
    """Test period parsing from query parameters."""

    def test_default_period_is_month(self, handler):
        h = _make_handler()
        start, end, period = handler._parse_period(h)
        assert period == "month"
        # Roughly 30 days
        diff = (end - start).days
        assert 29 <= diff <= 31

    def test_period_hour(self, handler):
        h = _make_handler(query_params={"period": "hour"})
        start, end, period = handler._parse_period(h)
        assert period == "hour"
        diff = (end - start).total_seconds()
        assert 3500 <= diff <= 3700

    def test_period_day(self, handler):
        h = _make_handler(query_params={"period": "day"})
        start, end, period = handler._parse_period(h)
        assert period == "day"
        diff = (end - start).days
        assert diff == 1

    def test_period_week(self, handler):
        h = _make_handler(query_params={"period": "week"})
        start, end, period = handler._parse_period(h)
        assert period == "week"
        diff = (end - start).days
        assert diff == 7

    def test_period_quarter(self, handler):
        h = _make_handler(query_params={"period": "quarter"})
        start, end, period = handler._parse_period(h)
        assert period == "quarter"
        diff = (end - start).days
        assert diff == 90

    def test_period_year(self, handler):
        h = _make_handler(query_params={"period": "year"})
        start, end, period = handler._parse_period(h)
        assert period == "year"
        diff = (end - start).days
        assert diff == 365

    def test_unknown_period_defaults_to_30_days(self, handler):
        h = _make_handler(query_params={"period": "unknown"})
        start, end, period = handler._parse_period(h)
        assert period == "unknown"
        diff = (end - start).days
        assert 29 <= diff <= 31

    def test_custom_start_date(self, handler):
        h = _make_handler(query_params={"start": "2025-06-01T00:00:00+00:00"})
        start, end, period = handler._parse_period(h)
        assert start.year == 2025
        assert start.month == 6
        assert start.day == 1

    def test_custom_end_date(self, handler):
        h = _make_handler(query_params={"end": "2025-12-31T23:59:59+00:00"})
        start, end, period = handler._parse_period(h)
        assert end.year == 2025
        assert end.month == 12
        assert end.day == 31

    def test_custom_start_with_z_suffix(self, handler):
        h = _make_handler(query_params={"start": "2025-01-15T12:00:00Z"})
        start, end, period = handler._parse_period(h)
        assert start.year == 2025
        assert start.month == 1

    def test_invalid_start_date_ignored(self, handler):
        h = _make_handler(query_params={"start": "not-a-date"})
        start, end, period = handler._parse_period(h)
        # Should fall back to default period calculation
        diff = (end - start).days
        assert 29 <= diff <= 31

    def test_invalid_end_date_ignored(self, handler):
        h = _make_handler(query_params={"end": "not-a-date"})
        start, end, period = handler._parse_period(h)
        # end should be approximately now
        diff = (datetime.now(timezone.utc) - end).total_seconds()
        assert abs(diff) < 5


# ============================================================================
# GET /api/v1/usage/summary
# ============================================================================


class TestGetSummary:
    """Test the usage summary endpoint."""

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_basic_summary(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        summary = body
        assert "period" in summary
        assert "debates" in summary
        assert "costs" in summary
        assert "tokens" in summary
        assert "activity" in summary

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=92.5)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_period_info(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker()
        h = _make_handler(query_params={"period": "week"})
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        period = body["period"]
        assert period["type"] == "week"
        assert "start" in period
        assert "end" in period
        assert period["days"] == 7

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_debate_counts(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=45))
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        debates = body["debates"]
        assert debates["total"] == 45
        assert debates["completed"] == 45
        assert debates["consensus_rate"] == 85.0

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_cost_data(self, mock_ct, mock_ut, mock_consensus, handler):
        stats = _make_workspace_stats(total_cost="12.50")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=45))
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        costs = body["costs"]
        assert costs["total_usd"] == "12.50"
        assert "avg_per_debate_usd" in costs
        assert "by_provider" in costs

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_token_data(self, mock_ct, mock_ut, mock_consensus, handler):
        stats = _make_workspace_stats(tokens_in=500000, tokens_out=250000)
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_ut.return_value = _make_mock_usage_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        tokens = body["tokens"]
        assert tokens["total"] == 750000
        assert tokens["input"] == 500000
        assert tokens["output"] == 250000

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_activity_data(self, mock_ct, mock_ut, mock_consensus, handler):
        stats = _make_workspace_stats(api_calls=150)
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=15))
        h = _make_handler(query_params={"period": "month"})
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        activity = body["activity"]
        assert "active_days" in activity
        assert "debates_per_day" in activity
        assert activity["api_calls"] == 150

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_zero_debates(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=0))
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        debates = body["debates"]
        assert debates["total"] == 0
        assert debates["completed"] == 0
        costs = body["costs"]
        # Decimal("0").quantize(Decimal("0.01")) produces "0.00"
        assert costs["avg_per_debate_usd"] == "0.00"
        activity = body["activity"]
        assert activity["active_days"] == 0

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_no_usage_tracker(self, mock_ct, mock_ut, mock_consensus, handler):
        """When usage tracker is None, debates default to 0."""
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = None
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["debates"]["total"] == 0

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_usage_tracker_exception(self, mock_ct, mock_ut, mock_consensus, handler):
        """When usage tracker raises, handle gracefully."""
        mock_ct.return_value = _make_mock_cost_tracker()
        tracker = MagicMock()
        tracker.get_summary.side_effect = RuntimeError("connection failed")
        mock_ut.return_value = tracker
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["debates"]["total"] == 0

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_cost_by_provider(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker(
            summary=MockUsageSummary(
                total_debates=10,
                cost_by_provider={"anthropic": Decimal("5.00"), "openai": Decimal("3.00")},
            )
        )
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        by_provider = body["costs"]["by_provider"]
        assert "anthropic" in by_provider
        assert "openai" in by_provider

    def test_summary_no_user_store(self):
        """Handler with no user_store returns 503."""
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_summary(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/breakdown
# ============================================================================


class TestGetBreakdown:
    """Test the usage breakdown endpoint."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_by_agent(self, mock_ct, handler):
        stats = _make_workspace_stats(
            cost_by_agent={"claude": "8.00", "gpt-4": "4.50"},
            total_cost="12.50",
        )
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"dimension": "agent"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        assert _status(result) == 200
        body = _body(result)
        bd = body
        assert bd["dimension"] == "agent"
        assert len(bd["items"]) == 2
        # Sorted by cost descending
        assert Decimal(bd["items"][0]["cost_usd"]) >= Decimal(bd["items"][1]["cost_usd"])

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_by_model(self, mock_ct, handler):
        stats = _make_workspace_stats(
            cost_by_model={"claude-3-opus": "6.00", "gpt-4-turbo": "4.00"},
            total_cost="10.00",
        )
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"dimension": "model"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        assert _status(result) == 200
        body = _body(result)
        bd = body
        assert bd["dimension"] == "model"
        assert len(bd["items"]) == 2

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_default_dimension_is_agent(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        assert body["dimension"] == "agent"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_unknown_dimension_returns_empty_items(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"dimension": "unknown"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        assert body["dimension"] == "unknown"
        assert body["items"] == []

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_includes_period(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_percentage_calculation(self, mock_ct, handler):
        stats = _make_workspace_stats(
            cost_by_agent={"claude": "7.50", "gpt-4": "2.50"},
            total_cost="10.00",
        )
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"dimension": "agent"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        items = body["items"]
        percentages = [item["percentage"] for item in items]
        assert 75.0 in percentages
        assert 25.0 in percentages

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_zero_total_cost(self, mock_ct, handler):
        stats = _make_workspace_stats(
            cost_by_agent={"claude": "0"},
            total_cost="0",
        )
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"dimension": "agent"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        assert body["items"][0]["percentage"] == 0

    def test_breakdown_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_breakdown(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/roi
# ============================================================================


class TestGetROI:
    """Test the ROI analysis endpoint."""

    @patch("aragora.billing.roi_calculator.ROICalculator")
    @patch("aragora.billing.roi_calculator.IndustryBenchmark")
    @patch("aragora.billing.roi_calculator.DebateROIInput")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_roi_basic(self, mock_ct, mock_input, mock_benchmark, mock_roi_calc, handler):
        stats = _make_workspace_stats(api_calls=100, total_cost="10.00")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_benchmark.side_effect = lambda v: v
        calc_instance = _make_mock_roi_calculator()
        mock_roi_calc.return_value = calc_instance
        h = _make_handler()
        result = handler.handle("/api/v1/usage/roi", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "roi_percent" in body  # ROI data unwrapped

    @patch("aragora.billing.roi_calculator.ROICalculator")
    @patch("aragora.billing.roi_calculator.IndustryBenchmark")
    @patch("aragora.billing.roi_calculator.DebateROIInput")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_roi_with_hourly_rate(self, mock_ct, mock_input, mock_benchmark, mock_roi_calc, handler):
        stats = _make_workspace_stats(api_calls=100, total_cost="10.00")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_benchmark.side_effect = lambda v: v
        calc_instance = _make_mock_roi_calculator()
        mock_roi_calc.return_value = calc_instance
        h = _make_handler(query_params={"hourly_rate": "75"})
        result = handler.handle("/api/v1/usage/roi", {}, h)
        assert _status(result) == 200
        # ROICalculator should have been called with hourly_rate_override
        mock_roi_calc.assert_called()

    @patch("aragora.billing.roi_calculator.ROICalculator")
    @patch("aragora.billing.roi_calculator.IndustryBenchmark")
    @patch("aragora.billing.roi_calculator.DebateROIInput")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_roi_zero_api_calls(self, mock_ct, mock_input, mock_benchmark, mock_roi_calc, handler):
        stats = _make_workspace_stats(api_calls=0, total_cost="0")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        mock_benchmark.side_effect = lambda v: v
        calc_instance = _make_mock_roi_calculator()
        mock_roi_calc.return_value = calc_instance
        h = _make_handler()
        result = handler.handle("/api/v1/usage/roi", {}, h)
        assert _status(result) == 200
        # With 0 API calls, no debates are constructed
        calc_instance.calculate_period_roi.assert_called_once()
        call_args = calc_instance.calculate_period_roi.call_args
        # debates is passed as keyword argument
        debates = call_args.kwargs.get("debates", call_args[1].get("debates", None))
        assert debates == []

    def test_roi_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_roi(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/budget-status
# ============================================================================


class TestGetBudgetStatus:
    """Test the budget status endpoint."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_with_limits(self, mock_ct, handler):
        budget = MockBudget()
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "monthly" in body  # budget data unwrapped
        b = body
        assert "monthly" in b
        assert "daily" in b
        assert b["monthly"]["limit_usd"] == "100.00"
        assert b["monthly"]["spent_usd"] == "45.50"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_remaining_calculation(self, mock_ct, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("100.00"),
            current_monthly_spend=Decimal("45.50"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        remaining = Decimal(body["monthly"]["remaining_usd"])
        assert remaining == Decimal("54.50")

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_percent_used(self, mock_ct, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("200.00"),
            current_monthly_spend=Decimal("100.00"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        assert body["monthly"]["percent_used"] == 50.0

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_no_budget_set(self, mock_ct, handler):
        """When no budget is configured, returns unlimited values."""
        stats = _make_workspace_stats(total_cost="25.00")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats, budget=None)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        assert _status(result) == 200
        body = _body(result)
        b = body
        assert b["monthly"]["limit_usd"] == "unlimited"
        assert b["monthly"]["remaining_usd"] == "unlimited"
        assert b["monthly"]["percent_used"] == 0
        assert b["alert_level"] is None

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_daily_data(self, mock_ct, handler):
        budget = MockBudget(
            daily_limit_usd=Decimal("10.00"),
            current_daily_spend=Decimal("3.25"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        daily = body["daily"]
        assert daily["limit_usd"] == "10.00"
        assert daily["spent_usd"] == "3.25"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_no_daily_limit(self, mock_ct, handler):
        budget = MockBudget(daily_limit_usd=None)
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        assert body["daily"]["limit_usd"] == "unlimited"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_with_alert_level(self, mock_ct, handler):
        budget = MockBudget()
        budget.check_alert_level = lambda: MockAlertLevel()
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        assert body["alert_level"] == "warning"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_projected_spend(self, mock_ct, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("100.00"),
            current_monthly_spend=Decimal("50.00"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        # Projected spend should be a valid decimal string
        projected = body["monthly"]["projected_end_spend_usd"]
        assert Decimal(projected) > 0

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_days_remaining(self, mock_ct, handler):
        budget = MockBudget()
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        days_remaining = body["monthly"]["days_remaining"]
        assert isinstance(days_remaining, int)
        assert 0 <= days_remaining <= 30

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_overspend_remaining_is_zero(self, mock_ct, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("50.00"),
            current_monthly_spend=Decimal("75.00"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        remaining = Decimal(body["monthly"]["remaining_usd"])
        assert remaining == Decimal("0")

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_zero_monthly_limit(self, mock_ct, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("0"),
            current_monthly_spend=Decimal("0"),
        )
        mock_ct.return_value = _make_mock_cost_tracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        assert body["monthly"]["percent_used"] == 0

    def test_budget_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_budget_status(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/forecast
# ============================================================================


class TestGetForecast:
    """Test the usage forecast endpoint."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_forecast_basic(self, mock_ct, mock_get_roi, handler):
        stats = _make_workspace_stats(api_calls=100, total_cost="10.00")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler()
        result = handler.handle("/api/v1/usage/forecast", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert isinstance(body, (dict, list))  # forecast data unwrapped

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_forecast_zero_api_calls(self, mock_ct, mock_get_roi, handler):
        stats = _make_workspace_stats(api_calls=0, total_cost="0")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler()
        result = handler.handle("/api/v1/usage/forecast", {}, h)
        assert _status(result) == 200
        # With 0 API calls, estimated_debates defaults to 5
        calc.estimate_future_savings.assert_called_once()
        call_kwargs = calc.estimate_future_savings.call_args
        assert call_kwargs[1]["projected_debates_per_month"] == 5

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_forecast_with_benchmark(self, mock_ct, mock_get_roi, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler(query_params={"benchmark": "enterprise"})
        result = handler.handle("/api/v1/usage/forecast", {}, h)
        assert _status(result) == 200
        mock_get_roi.assert_called_with("enterprise")

    def test_forecast_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_forecast(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/benchmarks
# ============================================================================


class TestGetBenchmarks:
    """Test the benchmarks comparison endpoint."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    def test_benchmarks_basic(self, mock_get_roi, handler):
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler()
        result = handler.handle("/api/v1/usage/benchmarks", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "benchmarks" in body  # benchmarks data has benchmarks key

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    def test_benchmarks_contains_data(self, mock_get_roi, handler):
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler()
        result = handler.handle("/api/v1/usage/benchmarks", {}, h)
        body = _body(result)
        benchmarks = body["benchmarks"]
        assert "sme" in benchmarks
        assert "enterprise" in benchmarks

    def test_benchmarks_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler()
        result = h_instance._get_benchmarks(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/export - JSON format
# ============================================================================


class TestExportJSON:
    """Test usage data export in JSON format."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_basic(self, mock_ct, handler):
        stats = _make_workspace_stats(total_cost="12.50", api_calls=100)
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "json"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["organization"] == "Test Org"
        assert "period" in body
        assert "totals" in body
        assert body["totals"]["cost_usd"] == "12.50"
        assert body["totals"]["api_calls"] == 100

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_includes_by_agent(self, mock_ct, handler):
        stats = _make_workspace_stats(cost_by_agent={"claude": "8.00"})
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "json"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        body = _body(result)
        assert "by_agent" in body
        assert "by_model" in body

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_with_roi(self, mock_ct, mock_get_roi, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler(query_params={"format": "json", "include_roi": "true"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        body = _body(result)
        assert "roi" in body

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_without_roi(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "json", "include_roi": "false"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        body = _body(result)
        assert "roi" not in body

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_token_count(self, mock_ct, handler):
        stats = _make_workspace_stats(tokens_in=300, tokens_out=200)
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "json"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        body = _body(result)
        assert body["totals"]["tokens"] == 500

    def test_export_no_user_store(self):
        h_instance = SMEUsageDashboardHandler(ctx={})
        http = _make_handler(query_params={"format": "json"})
        result = h_instance._export_usage(http, {}, user=MockUser())
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/usage/export - CSV format
# ============================================================================


class TestExportCSV:
    """Test usage data export in CSV format."""

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_returns_csv_content_type(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        assert _status(result) == 200
        assert result.content_type == "text/csv"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_has_content_disposition(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        assert "Content-Disposition" in result.headers
        assert "attachment" in result.headers["Content-Disposition"]
        assert "sme_usage_" in result.headers["Content-Disposition"]

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_contains_headers(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "SME Usage Dashboard Export" in csv_content
        assert "Organization" in csv_content
        assert "Test Org" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_contains_summary(self, mock_ct, handler):
        stats = _make_workspace_stats(total_cost="12.50", api_calls=100)
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "Summary" in csv_content
        assert "Total Cost (USD)" in csv_content
        assert "12.50" in csv_content
        assert "Total API Calls" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_contains_agent_breakdown(self, mock_ct, handler):
        stats = _make_workspace_stats(cost_by_agent={"claude": "8.00", "gpt-4": "4.50"})
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "Cost by Agent" in csv_content
        assert "claude" in csv_content
        assert "gpt-4" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_contains_model_breakdown(self, mock_ct, handler):
        stats = _make_workspace_stats(cost_by_model={"claude-3-opus": "8.00"})
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "Cost by Model" in csv_content
        assert "claude-3-opus" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_roi_calculator")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_with_roi(self, mock_ct, mock_get_roi, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        calc = _make_mock_roi_calculator()
        mock_get_roi.return_value = calc
        h = _make_handler(query_params={"format": "csv", "include_roi": "true"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "ROI Comparison" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_without_roi(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "ROI Comparison" not in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_default_format_is_csv(self, mock_ct, handler):
        """When no format is specified, default to CSV."""
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/export", {}, h)
        assert result.content_type == "text/csv"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_uses_org_slug_in_filename(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        assert "test-org" in result.headers["Content-Disposition"]

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_org_without_slug(self, mock_ct):
        """When org has no slug attribute, use org.id in filename."""
        org = MagicMock(spec=[])
        org.id = "org-123"
        org.name = "Test Org"
        # No slug attribute
        user_store = _make_mock_user_store(org=org)
        h_instance = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        mock_ct.return_value = _make_mock_cost_tracker()
        http = _make_handler(query_params={"format": "csv"})
        result = h_instance._export_usage(http, {}, user=MockUser())
        assert _status(result) == 200
        assert "org-123" in result.headers["Content-Disposition"]


# ============================================================================
# _get_real_consensus_rate
# ============================================================================


class TestGetRealConsensusRate:
    """Test the _get_real_consensus_rate helper function."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_rate_from_store(self, mock_get_store):
        store = MagicMock()
        store.get_consensus_stats.return_value = {"overall_consensus_rate": "92%"}
        mock_get_store.return_value = store
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 92.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_default_on_empty_rate(self, mock_get_store):
        store = MagicMock()
        store.get_consensus_stats.return_value = {"overall_consensus_rate": ""}
        mock_get_store.return_value = store
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 85.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_default_on_zero_rate(self, mock_get_store):
        store = MagicMock()
        store.get_consensus_stats.return_value = {"overall_consensus_rate": "0%"}
        mock_get_store.return_value = store
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 85.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_default_on_exception(self, mock_get_store):
        mock_get_store.side_effect = RuntimeError("no store")
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 85.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_custom_default(self, mock_get_store):
        mock_get_store.side_effect = RuntimeError("no store")
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now, default=75.0)
        assert rate == 75.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_default_on_missing_key(self, mock_get_store):
        store = MagicMock()
        store.get_consensus_stats.return_value = {}
        mock_get_store.return_value = store
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 85.0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_returns_default_on_value_error(self, mock_get_store):
        store = MagicMock()
        store.get_consensus_stats.return_value = {"overall_consensus_rate": "not-a-number"}
        mock_get_store.return_value = store
        now = datetime.now(timezone.utc)
        rate = _get_real_consensus_rate("org-1", now - timedelta(days=30), now)
        assert rate == 85.0


# ============================================================================
# _get_usage_tracker
# ============================================================================


class TestGetUsageTracker:
    """Test the _get_usage_tracker helper method."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_returns_tracker_on_success(self, mock_cls, handler):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = handler._get_usage_tracker()
        assert result is mock_instance

    @patch("aragora.billing.usage.UsageTracker")
    def test_returns_none_on_error(self, mock_cls, handler):
        mock_cls.side_effect = RuntimeError("init failed")
        result = handler._get_usage_tracker()
        assert result is None

    @patch("aragora.billing.usage.UsageTracker")
    def test_returns_none_on_value_error(self, mock_cls, handler):
        mock_cls.side_effect = ValueError("bad config")
        result = handler._get_usage_tracker()
        assert result is None


# ============================================================================
# _get_roi_calculator
# ============================================================================


class TestGetROICalculator:
    """Test the _get_roi_calculator helper method."""

    @patch("aragora.billing.roi_calculator.ROICalculator")
    @patch("aragora.billing.roi_calculator.IndustryBenchmark")
    def test_returns_calculator_with_sme_benchmark(self, mock_benchmark, mock_calc, handler):
        mock_benchmark.return_value = "sme_benchmark"
        result = handler._get_roi_calculator("sme")
        mock_benchmark.assert_called_with("sme")
        mock_calc.assert_called_with(benchmark="sme_benchmark")

    @patch("aragora.billing.roi_calculator.ROICalculator")
    @patch("aragora.billing.roi_calculator.IndustryBenchmark")
    def test_falls_back_to_sme_on_invalid_benchmark(self, mock_benchmark, mock_calc, handler):
        mock_benchmark.side_effect = ValueError("invalid")
        mock_benchmark.SME = "sme_fallback"
        result = handler._get_roi_calculator("invalid_benchmark")
        # Should have tried "invalid_benchmark" first (raised ValueError)
        mock_benchmark.assert_called_once_with("invalid_benchmark")
        # Then fell back to IndustryBenchmark.SME (attribute access)
        mock_calc.assert_called_once_with(benchmark="sme_fallback")


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Test various edge cases."""

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_with_custom_period(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=10))
        h = _make_handler(query_params={
            "start": "2025-01-01T00:00:00+00:00",
            "end": "2025-01-31T23:59:59+00:00",
        })
        result = handler.handle("/api/v1/usage/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        period = body["period"]
        assert "2025-01-01" in period["start"]
        assert "2025-01-31" in period["end"]

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_usage_summary_no_cost_by_provider_attr(self, mock_ct, mock_ut, mock_consensus, handler):
        """Usage summary without cost_by_provider returns empty dict."""
        mock_ct.return_value = _make_mock_cost_tracker()

        @dataclass
        class SummaryNoCosts:
            total_debates: int = 10

        mock_ut.return_value = SummaryNoCosts()
        h = _make_handler()
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        assert body["costs"]["by_provider"] == {}

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_breakdown_agent_sorted_descending(self, mock_ct, handler):
        stats = _make_workspace_stats(
            cost_by_agent={"cheap": "1.00", "mid": "5.00", "expensive": "10.00"},
            total_cost="16.00",
        )
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats)
        h = _make_handler(query_params={"dimension": "agent"})
        result = handler.handle("/api/v1/usage/breakdown", {}, h)
        body = _body(result)
        items = body["items"]
        costs = [Decimal(item["cost_usd"]) for item in items]
        assert costs == sorted(costs, reverse=True)

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_csv_period_params(self, mock_ct, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        h = _make_handler(query_params={"format": "csv", "period": "week"})
        result = handler.handle("/api/v1/usage/export", {}, h)
        csv_content = result.body.decode("utf-8")
        assert "Period Start" in csv_content
        assert "Period End" in csv_content

    @patch("aragora.server.handlers.sme_usage_dashboard._get_real_consensus_rate", return_value=85.0)
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_usage_tracker")
    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_summary_debates_per_day_rounded(self, mock_ct, mock_ut, mock_consensus, handler):
        mock_ct.return_value = _make_mock_cost_tracker()
        mock_ut.return_value = _make_mock_usage_tracker(summary=MockUsageSummary(total_debates=7))
        h = _make_handler(query_params={"period": "week"})
        result = handler.handle("/api/v1/usage/summary", {}, h)
        body = _body(result)
        dpd = body["activity"]["debates_per_day"]
        assert dpd == 1.0

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_export_json_org_without_name(self, mock_ct):
        """When org has no name attribute, use str(org.id)."""
        org = MagicMock(spec=[])
        org.id = "org-456"
        # No name attribute
        user_store = _make_mock_user_store(org=org)
        h_instance = SMEUsageDashboardHandler(ctx={"user_store": user_store})
        mock_ct.return_value = _make_mock_cost_tracker()
        http = _make_handler(query_params={"format": "json"})
        result = h_instance._export_usage(http, {}, user=MockUser())
        body = _body(result)
        assert body["organization"] == "org-456"

    @patch("aragora.server.handlers.sme_usage_dashboard.SMEUsageDashboardHandler._get_cost_tracker")
    def test_budget_no_budget_unlimited_daily(self, mock_ct, handler):
        stats = _make_workspace_stats(total_cost="5.00")
        mock_ct.return_value = _make_mock_cost_tracker(workspace_stats=stats, budget=None)
        h = _make_handler()
        result = handler.handle("/api/v1/usage/budget-status", {}, h)
        body = _body(result)
        assert body["daily"]["limit_usd"] == "unlimited"
        assert body["daily"]["spent_usd"] == "5.00"
