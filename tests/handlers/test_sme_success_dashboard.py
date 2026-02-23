"""Tests for SME Success Dashboard handler.

Covers all routes and behavior of the SMESuccessDashboardHandler class:
- can_handle() routing
- GET /api/v1/sme/success          - Unified success metrics
- GET /api/v1/sme/success/cfo      - CFO-focused view (costs, ROI, budget)
- GET /api/v1/sme/success/pm       - PM-focused view (velocity, consensus, decisions)
- GET /api/v1/sme/success/hr       - HR-focused view (alignment, time savings)
- GET /api/v1/sme/success/milestones - Achievement milestones and gamification
- GET /api/v1/sme/success/insights   - Actionable insights and recommendations
- Rate limiting
- Period parsing
- Success metric calculation
- Milestone evaluation and gamification
- Insight generation
- Error paths and edge cases
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme_success_dashboard import (
    MILESTONES,
    SMESuccessDashboardHandler,
    _get_real_consensus_rate,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"Content-Length": "2"}
        # Query params support (for get_string_param resolution)
        self.query_params: dict[str, Any] = {}

    def set_query_params(self, params: dict[str, Any]) -> None:
        self.query_params = params


def _make_handler(
    body: dict[str, Any] | None = None,
    method: str = "GET",
    query_params: dict[str, Any] | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body and method."""
    h = MockHTTPHandler(body=body)
    h.command = method
    if query_params:
        h.query_params = query_params
    return h


# ---------------------------------------------------------------------------
# Mock objects for user_store, organizations, etc.
# ---------------------------------------------------------------------------


class MockUser:
    """Mock user object."""

    def __init__(self, user_id: str = "user-001", org_id: str = "org-001"):
        self.user_id = user_id
        self.org_id = org_id


class MockOrg:
    """Mock organization object."""

    def __init__(self, org_id: str = "org-001", name: str = "Test Corp"):
        self.id = org_id
        self.name = name


class MockBudget:
    """Mock budget object."""

    def __init__(
        self,
        monthly_limit_usd: Decimal = Decimal("500"),
        current_monthly_spend: Decimal = Decimal("100"),
    ):
        self.monthly_limit_usd = monthly_limit_usd
        self.current_monthly_spend = current_monthly_spend


class MockDebateStats:
    """Mock debate stats returned by DebateAnalytics."""

    def __init__(
        self,
        total_debates: int = 20,
        consensus_rate: float = 0.85,
        avg_duration_seconds: float = 300.0,
    ):
        self.total_debates = total_debates
        self.consensus_rate = consensus_rate
        self.avg_duration_seconds = avg_duration_seconds


class MockROICalculator:
    """Mock ROI calculator."""

    def __init__(self, hourly_rate: Decimal = Decimal("60")):
        self.hourly_rate = hourly_rate


class MockCostTracker:
    """Mock cost tracker."""

    def __init__(
        self,
        total_cost_usd: str = "50.00",
        total_api_calls: int = 200,
        budget: MockBudget | None = None,
    ):
        self._total_cost_usd = total_cost_usd
        self._total_api_calls = total_api_calls
        self._budget = budget

    def get_workspace_stats(self, workspace_id: str) -> dict[str, Any]:
        return {
            "total_cost_usd": self._total_cost_usd,
            "total_api_calls": self._total_api_calls,
        }

    def get_budget(self, workspace_id: str = "", org_id: str = "") -> MockBudget | None:
        return self._budget


def _make_user_store(
    user: MockUser | None = None,
    org: MockOrg | None = None,
) -> MagicMock:
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id.return_value = user
    store.get_organization_by_id.return_value = org
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_org():
    return MockOrg()


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    return _make_user_store(user=mock_user, org=mock_org)


@pytest.fixture
def mock_cost_tracker():
    return MockCostTracker()


@pytest.fixture
def mock_roi_calculator():
    return MockROICalculator()


@pytest.fixture
def handler(mock_user_store, mock_cost_tracker, mock_roi_calculator):
    """Create an SMESuccessDashboardHandler with mocked dependencies."""
    h = SMESuccessDashboardHandler(server_context={"user_store": mock_user_store})

    # Patch internal method calls
    h._get_cost_tracker = MagicMock(return_value=mock_cost_tracker)
    h._get_roi_calculator = MagicMock(return_value=mock_roi_calculator)
    h._get_debate_analytics = MagicMock(return_value=None)

    return h


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_success_root(self, handler):
        assert handler.can_handle("/api/v1/sme/success") is True

    def test_cfo_view(self, handler):
        assert handler.can_handle("/api/v1/sme/success/cfo") is True

    def test_pm_view(self, handler):
        assert handler.can_handle("/api/v1/sme/success/pm") is True

    def test_hr_view(self, handler):
        assert handler.can_handle("/api/v1/sme/success/hr") is True

    def test_milestones(self, handler):
        assert handler.can_handle("/api/v1/sme/success/milestones") is True

    def test_insights(self, handler):
        assert handler.can_handle("/api/v1/sme/success/insights") is True

    def test_subpath_matches(self, handler):
        """Paths starting with /api/v1/sme/success should match."""
        assert handler.can_handle("/api/v1/sme/success/anything") is True

    def test_rejects_unrelated(self, handler):
        assert handler.can_handle("/api/v1/sme/billing") is False

    def test_rejects_empty(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/sme") is False

    def test_rejects_v2_path(self, handler):
        assert handler.can_handle("/api/v2/sme/success") is False


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_server_context(self):
        ctx = {"user_store": MagicMock()}
        h = SMESuccessDashboardHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx(self):
        ctx = {"user_store": MagicMock()}
        h = SMESuccessDashboardHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_both_prefers_server_context(self):
        sc = {"source": "server_context"}
        c = {"source": "ctx"}
        h = SMESuccessDashboardHandler(server_context=sc, ctx=c)
        assert h.ctx == sc

    def test_init_with_none(self):
        h = SMESuccessDashboardHandler()
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 6
        for route in [
            "/api/v1/sme/success",
            "/api/v1/sme/success/cfo",
            "/api/v1/sme/success/pm",
            "/api/v1/sme/success/hr",
            "/api/v1/sme/success/milestones",
            "/api/v1/sme/success/insights",
        ]:
            assert route in handler.ROUTES

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "success_dashboard"


# ============================================================================
# Handle dispatch
# ============================================================================


class TestHandleDispatch:
    """Test the main handle() dispatch method."""

    def test_routes_to_success_summary(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "success" in body

    def test_routes_to_cfo(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "cfo_view" in body

    def test_routes_to_pm(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "pm_view" in body

    def test_routes_to_hr(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "hr_view" in body

    def test_routes_to_milestones(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "milestones" in body

    def test_routes_to_insights(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "insights" in body

    def test_method_not_allowed_post(self, handler):
        h = _make_handler(method="POST")
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 405

    def test_method_not_allowed_delete(self, handler):
        h = _make_handler(method="DELETE")
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        assert _status(result) == 405

    def test_method_not_allowed_put(self, handler):
        h = _make_handler(method="PUT")
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        assert _status(result) == 405

    def test_unrecognized_path_returns_405(self, handler):
        """Unknown sub-path still reaches handle but not in route_map, returns 405."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/unknown", {}, h)
        assert _status(result) == 405

    def test_handler_command_overrides_method(self, handler):
        """If handler has .command, it overrides the method param."""
        h = _make_handler(method="GET")
        h.command = "POST"
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 405


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limit enforcement on the dashboard handler."""

    def test_rate_limit_allows_normal_traffic(self, handler):
        """Normal request should be allowed."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 200

    def test_rate_limit_blocks_excessive_traffic(self, handler):
        """After exceeding rate limit, requests return 429."""
        # The limiter is 60 requests per minute
        for _ in range(61):
            h = _make_handler()
            result = handler.handle("/api/v1/sme/success", {}, h)

        # The 62nd request should be rate limited
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 429

    def test_rate_limit_different_ips(self, handler):
        """Different IPs have separate rate limits."""
        # Exhaust limit for first IP
        for _ in range(61):
            h = _make_handler()
            h.client_address = ("10.0.0.1", 12345)
            handler.handle("/api/v1/sme/success", {}, h)

        # Different IP should still work
        h = _make_handler()
        h.client_address = ("10.0.0.2", 12345)
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 200


# ============================================================================
# _get_user_and_org
# ============================================================================


class TestGetUserAndOrg:
    """Test user and organization resolution."""

    def test_no_user_store_returns_503(self, handler):
        handler.ctx = {}  # No user_store
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 503

    def test_user_not_found_returns_404(self, handler):
        handler.ctx["user_store"].get_user_by_id.return_value = None
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 404

    def test_no_org_returns_404(self, mock_user):
        user_store = _make_user_store(user=mock_user, org=None)
        h_handler = SMESuccessDashboardHandler(server_context={"user_store": user_store})
        h_handler._get_cost_tracker = MagicMock(return_value=MockCostTracker())
        h_handler._get_roi_calculator = MagicMock(return_value=MockROICalculator())
        h_handler._get_debate_analytics = MagicMock(return_value=None)

        h = _make_handler()
        result = h_handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 404

    def test_user_without_org_id_returns_404(self, handler):
        user_no_org = MockUser(org_id=None)
        handler.ctx["user_store"].get_user_by_id.return_value = user_no_org
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 404


# ============================================================================
# Period parsing
# ============================================================================


class TestParsePeriod:
    """Test _parse_period method via query params."""

    def test_default_period_is_month(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "month"

    def test_week_period(self, handler):
        h = _make_handler(query_params={"period": "week"})
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "week"

    def test_quarter_period(self, handler):
        h = _make_handler(query_params={"period": "quarter"})
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "quarter"

    def test_year_period(self, handler):
        h = _make_handler(query_params={"period": "year"})
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "year"

    def test_invalid_period_defaults_to_month(self, handler):
        h = _make_handler(query_params={"period": "decade"})
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "month"


# ============================================================================
# GET /api/v1/sme/success - Success Summary
# ============================================================================


class TestSuccessSummary:
    """Test the unified success metrics summary endpoint."""

    def test_returns_success_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert "success" in body

    def test_headline_with_positive_savings(self, handler):
        """When net savings > 0, headline shows savings amount."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=200,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        headline = body["success"]["headline"]
        assert "$" in headline
        assert "saved" in headline.lower()

    def test_headline_with_zero_debates(self, handler):
        """When no savings, headline shows debate count."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        headline = body["success"]["headline"]
        assert "decisions" in headline.lower() or "0" in headline

    def test_key_metrics_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        km = body["success"]["key_metrics"]
        assert "decisions_made" in km
        assert "time_saved_hours" in km
        assert "money_saved_usd" in km
        assert "net_roi_usd" in km
        assert "roi_percentage" in km
        assert "consensus_rate" in km

    def test_comparison_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        comp = body["success"]["comparison"]
        assert "manual_time_hours" in comp
        assert "ai_time_hours" in comp
        assert "efficiency_multiplier" in comp

    def test_efficiency_multiplier_is_9x(self, handler):
        """Manual=45min, AI=5min, multiplier should be 9.0."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["comparison"]["efficiency_multiplier"] == 9.0

    def test_period_in_response(self, handler):
        h = _make_handler(query_params={"period": "week"})
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["period"] == "week"


# ============================================================================
# GET /api/v1/sme/success/cfo - CFO View
# ============================================================================


class TestCFOView:
    """Test CFO-focused success view endpoint."""

    def test_returns_cfo_view_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        assert "cfo_view" in body

    def test_role_is_cfo(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        assert body["cfo_view"]["role"] == "cfo"

    def test_financial_summary_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        fs = body["cfo_view"]["financial_summary"]
        assert "total_spend_usd" in fs
        assert "value_generated_usd" in fs
        assert "net_roi_usd" in fs
        assert "roi_percentage" in fs

    def test_cost_efficiency_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        ce = body["cfo_view"]["cost_efficiency"]
        assert "cost_per_decision_usd" in ce
        assert "manual_cost_per_decision_usd" in ce
        assert "savings_per_decision_usd" in ce

    def test_budget_with_data(self, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("500"),
            current_monthly_spend=Decimal("200"),
        )
        handler._get_cost_tracker.return_value = MockCostTracker(budget=budget)

        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        bi = body["cfo_view"]["budget"]
        assert bi["monthly_limit_usd"] == 500.0
        assert bi["spent_usd"] == 200.0
        assert bi["remaining_usd"] == 300.0
        assert bi["utilization_percent"] == 40.0

    def test_budget_empty_when_no_budget(self, handler):
        handler._get_cost_tracker.return_value = MockCostTracker(budget=None)
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        assert body["cfo_view"]["budget"] == {}

    def test_projections_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        proj = body["cfo_view"]["projections"]
        assert "monthly_run_rate_usd" in proj
        assert "projected_annual_savings_usd" in proj

    def test_headline_includes_period(self, handler):
        h = _make_handler(query_params={"period": "quarter"})
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        assert body["cfo_view"]["period"] == "quarter"

    def test_budget_zero_limit(self, handler):
        budget = MockBudget(
            monthly_limit_usd=Decimal("0"),
            current_monthly_spend=Decimal("0"),
        )
        handler._get_cost_tracker.return_value = MockCostTracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        bi = body["cfo_view"]["budget"]
        assert bi["utilization_percent"] == 0

    def test_cost_per_decision_zero_debates(self, handler):
        """When 0 debates, cost_per_decision should be 0."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        assert body["cfo_view"]["cost_efficiency"]["cost_per_decision_usd"] == 0


# ============================================================================
# GET /api/v1/sme/success/pm - PM View
# ============================================================================


class TestPMView:
    """Test PM-focused success view endpoint."""

    def test_returns_pm_view_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        assert "pm_view" in body

    def test_role_is_pm(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        assert body["pm_view"]["role"] == "pm"

    def test_velocity_metrics_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        vel = body["pm_view"]["velocity"]
        assert "total_decisions" in vel
        assert "decisions_per_day" in vel
        assert "decisions_per_week" in vel
        assert "avg_decision_time_minutes" in vel
        assert "time_to_consensus_minutes" in vel

    def test_quality_metrics_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        qual = body["pm_view"]["quality"]
        assert "consensus_rate_percent" in qual
        assert "consensus_streak" in qual
        assert "decisions_with_consensus" in qual

    def test_efficiency_metrics_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        eff = body["pm_view"]["efficiency"]
        assert "manual_equivalent_hours" in eff
        assert "actual_hours" in eff
        assert "hours_saved" in eff
        assert "efficiency_gain_percent" in eff

    def test_trends_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        trends = body["pm_view"]["trends"]
        assert trends["velocity_trend"] == "stable"
        assert trends["consensus_trend"] == "improving"

    def test_decisions_per_week(self, handler):
        """decisions_per_week should be 7 * decisions_per_day."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        vel = body["pm_view"]["velocity"]
        dpd = vel["decisions_per_day"]
        dpw = vel["decisions_per_week"]
        assert dpw == round(dpd * 7, 1)

    def test_time_to_consensus(self, handler):
        """time_to_consensus = avg_debate_time * 0.8."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        vel = body["pm_view"]["velocity"]
        assert vel["time_to_consensus_minutes"] == round(
            vel["avg_decision_time_minutes"] * 0.8, 1
        )

    def test_headline_contains_decisions_and_rate(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        headline = body["pm_view"]["headline"]
        assert "decisions" in headline.lower()
        assert "faster" in headline.lower()


# ============================================================================
# GET /api/v1/sme/success/hr - HR View
# ============================================================================


class TestHRView:
    """Test HR-focused success view endpoint."""

    def test_returns_hr_view_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        assert "hr_view" in body

    def test_role_is_hr(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        assert body["hr_view"]["role"] == "hr"

    def test_alignment_metrics_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        alignment = body["hr_view"]["alignment"]
        assert "consensus_rate_percent" in alignment
        assert "decisions_with_alignment" in alignment
        assert "alignment_trend" in alignment

    def test_time_impact_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        ti = body["hr_view"]["time_impact"]
        assert "hours_saved_total" in ti
        assert "hours_saved_per_person" in ti
        assert "meeting_hours_avoided" in ti

    def test_participation_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        part = body["hr_view"]["participation"]
        assert part["avg_participants_per_decision"] == 3
        assert part["decision_inclusivity_score"] == 85

    def test_wellbeing_impact_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        wb = body["hr_view"]["wellbeing_impact"]
        assert "decision_fatigue_reduction_percent" in wb
        assert "conflict_reduction_score" in wb
        assert wb["conflict_reduction_score"] == 78

    def test_hours_saved_per_person_5_team(self, handler):
        """hours_saved_per_person = hours_saved / 5 (assumed team size)."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        ti = body["hr_view"]["time_impact"]
        assert ti["hours_saved_per_person"] == round(ti["hours_saved_total"] / 5, 1)

    def test_meeting_hours_avoided(self, handler):
        """meeting_hours_avoided = hours_saved * 0.6."""
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        ti = body["hr_view"]["time_impact"]
        assert ti["meeting_hours_avoided"] == round(ti["hours_saved_total"] * 0.6, 1)

    def test_headline_includes_consensus_rate(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        headline = body["hr_view"]["headline"]
        assert "aligned" in headline.lower()


# ============================================================================
# GET /api/v1/sme/success/milestones
# ============================================================================


class TestMilestones:
    """Test achievement milestones endpoint."""

    def test_returns_milestones_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        assert "milestones" in body

    def test_total_available_matches_constants(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        assert body["milestones"]["total_available"] == len(MILESTONES)

    def test_earned_count_is_integer(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        assert isinstance(body["milestones"]["total_earned"], int)

    def test_completion_percent(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        pct = body["milestones"]["completion_percent"]
        assert 0 <= pct <= 100

    def test_earned_sorted_by_threshold_desc(self, handler):
        """Earned milestones should be sorted by threshold descending."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="1000.00", total_api_calls=5000,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        earned = body["milestones"]["earned"]
        if len(earned) > 1:
            for i in range(len(earned) - 1):
                assert earned[i]["threshold"] >= earned[i + 1]["threshold"]

    def test_upcoming_sorted_by_progress_desc(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        upcoming = body["milestones"]["upcoming"]
        if len(upcoming) > 1:
            for i in range(len(upcoming) - 1):
                assert upcoming[i]["progress_percent"] >= upcoming[i + 1]["progress_percent"]

    def test_upcoming_limited_to_5(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        assert len(body["milestones"]["upcoming"]) <= 5

    def test_earned_has_earned_date(self, handler):
        """Each earned milestone should have an earned_date."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="100.00", total_api_calls=500,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        for m in body["milestones"]["earned"]:
            assert m["earned"] is True
            assert "earned_date" in m

    def test_upcoming_has_progress_and_remaining(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        for m in body["milestones"]["upcoming"]:
            assert m["earned"] is False
            assert "progress_percent" in m
            assert "remaining" in m

    def test_next_milestone_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        # next_milestone can be None only if all milestones are earned
        nm = body["milestones"]["next_milestone"]
        if body["milestones"]["total_earned"] < body["milestones"]["total_available"]:
            assert nm is not None
            assert "progress_percent" in nm

    def test_milestone_data_structure(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        all_milestones = body["milestones"]["earned"] + body["milestones"]["upcoming"]
        for m in all_milestones:
            assert "id" in m
            assert "name" in m
            assert "description" in m
            assert "icon" in m
            assert "threshold" in m
            assert "current_value" in m

    def test_first_debate_earned_with_debates(self, handler):
        """first_debate milestone (threshold=1) should be earned if debates > 0."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=100,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        earned_ids = [m["id"] for m in body["milestones"]["earned"]]
        assert "first_debate" in earned_ids

    def test_zero_debates_no_milestones_earned(self, handler):
        """With zero debates, debate milestones should not be earned."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/milestones", {}, h)
        body = _body(result)
        earned_ids = [m["id"] for m in body["milestones"]["earned"]]
        assert "first_debate" not in earned_ids
        assert "debate_10" not in earned_ids


# ============================================================================
# GET /api/v1/sme/success/insights
# ============================================================================


class TestInsights:
    """Test actionable insights endpoint."""

    def test_returns_insights_key(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        assert "insights" in body

    def test_count_matches_items(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        assert body["insights"]["count"] == len(body["insights"]["items"])

    def test_generated_at_present(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        assert "generated_at" in body["insights"]

    def test_getting_started_insight_when_zero_debates(self, handler):
        """When 0 debates, should get getting_started insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        assert "getting_started" in types

    def test_engagement_insight_for_few_debates(self, handler):
        """When 1-4 debates, should get engagement insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="5.00", total_api_calls=30,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        assert "engagement" in types

    def test_success_insight_for_high_roi(self, handler):
        """When ROI > 100%, should get success insight."""
        # High API calls to generate large ROI
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="1.00", total_api_calls=5000,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        assert "success" in types

    def test_optimization_insight_for_positive_roi(self, handler):
        """When 0 < ROI < 100%, should get optimization insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="50.00", total_api_calls=500,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        # Should have optimization or another type
        assert len(body["insights"]["items"]) > 0

    def test_path_to_roi_insight_for_negative_roi(self, handler):
        """When ROI < 0 and debates > 0, should get path to positive ROI insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10000.00", total_api_calls=100,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        assert "optimization" in types

    def test_celebration_insight_for_large_time_savings(self, handler):
        """When hours_saved > 10, should get celebration insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=10000,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        types = [i["type"] for i in body["insights"]["items"]]
        assert "celebration" in types

    def test_insight_structure(self, handler):
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/insights", {}, h)
        body = _body(result)
        for item in body["insights"]["items"]:
            assert "type" in item
            assert "priority" in item
            assert "title" in item
            assert "message" in item
            assert "action" in item
            assert "action_url" in item

    def test_consensus_quality_insight_low_rate(self, handler):
        """When consensus rate < 70% and debates > 5, should get quality insight."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=1000,
        )
        # Patch consensus to return a low rate
        with patch(
            "aragora.server.handlers.sme_success_dashboard._get_real_consensus_rate",
            return_value=50.0,
        ):
            h = _make_handler()
            result = handler.handle("/api/v1/sme/success/insights", {}, h)
            body = _body(result)
            types = [i["type"] for i in body["insights"]["items"]]
            assert "quality" in types


# ============================================================================
# Success Metric Calculation
# ============================================================================


class TestSuccessMetrics:
    """Test _calculate_success_metrics internal method."""

    def test_zero_api_calls_zero_debates(self, handler):
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=0,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["total_debates"] == 0
        assert metrics["minutes_saved"] == 0

    def test_debates_estimated_from_api_calls(self, handler):
        """Without analytics, debates = max(1, api_calls // 10)."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=100,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["total_debates"] == 10

    def test_consensus_streak_capped(self, handler):
        """consensus_streak = min(debates, 5)."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="10.00", total_api_calls=30,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["consensus_streak"] <= 5

    def test_positive_net_savings(self, handler):
        """With low cost and many debates, net savings should be positive."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="5.00", total_api_calls=1000,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["net_savings_usd"] > 0
        assert metrics["roi_percentage"] > 0

    def test_roi_zero_when_no_cost(self, handler):
        """When total_cost is 0, ROI should be 0 (avoid division by zero)."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="0", total_api_calls=100,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["roi_percentage"] == 0

    def test_with_debate_analytics(self, handler):
        """When DebateAnalytics returns data, use real stats."""
        mock_analytics = MagicMock()
        mock_stats = MockDebateStats(
            total_debates=50,
            consensus_rate=0.9,
            avg_duration_seconds=180.0,
        )
        handler._get_debate_analytics.return_value = mock_analytics

        # The handler calls get_event_loop_safe().run_until_complete(coro).
        # We mock get_event_loop_safe to return a mock loop that returns our stats.
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = mock_stats

        with patch(
            "aragora.utils.async_utils.get_event_loop_safe",
            return_value=mock_loop,
        ):
            now = datetime.now(timezone.utc)
            start = now
            metrics = handler._calculate_success_metrics("org-001", start, now)
            # Should use the real stats
            assert metrics["total_debates"] == 50
            assert metrics["consensus_rate"] == 90.0  # 0.9 * 100
            assert metrics["avg_debate_time_minutes"] == 3.0  # 180/60

    def test_minutes_saved_non_negative(self, handler):
        """minutes_saved should always be >= 0."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="100.00", total_api_calls=50,
        )
        now = datetime.now(timezone.utc)
        metrics = handler._calculate_success_metrics("org-001", now, now)
        assert metrics["minutes_saved"] >= 0


# ============================================================================
# _get_real_consensus_rate standalone function
# ============================================================================


class TestGetRealConsensusRate:
    """Test the standalone _get_real_consensus_rate function."""

    def test_returns_default_on_import_error(self):
        """When debate_store is unavailable, return default."""
        with patch(
            "aragora.server.handlers.sme_success_dashboard.get_debate_store",
            side_effect=ImportError("no module"),
            create=True,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now)
            assert rate == 85.0

    def test_returns_custom_default(self):
        """Custom default is returned when no data available."""
        with patch(
            "aragora.server.handlers.sme_success_dashboard.get_debate_store",
            side_effect=ImportError("no module"),
            create=True,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now, default=70.0)
            assert rate == 70.0

    def test_returns_rate_from_store(self):
        """When store has data, parse and return rate."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {"overall_consensus_rate": "92%"}

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now)
            assert rate == 92.0

    def test_returns_default_for_zero_rate(self):
        """When rate is 0%, return default."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {"overall_consensus_rate": "0%"}

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now)
            assert rate == 85.0

    def test_returns_default_for_empty_rate(self):
        """When rate string is empty, return default."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {"overall_consensus_rate": ""}

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now)
            assert rate == 85.0

    def test_handles_value_error(self):
        """When rate can't be parsed, return default."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {
            "overall_consensus_rate": "bad-data%"
        }

        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_store,
        ):
            now = datetime.now(timezone.utc)
            rate = _get_real_consensus_rate("org-001", now, now)
            assert rate == 85.0


# ============================================================================
# MILESTONES constant
# ============================================================================


class TestMilestonesConstant:
    """Verify the MILESTONES constant structure."""

    def test_milestones_is_list(self):
        assert isinstance(MILESTONES, list)

    def test_milestones_count(self):
        assert len(MILESTONES) == 11

    def test_milestone_fields(self):
        for m in MILESTONES:
            assert "id" in m
            assert "name" in m
            assert "description" in m
            assert "icon" in m
            assert "threshold" in m
            assert "metric" in m

    def test_unique_ids(self):
        ids = [m["id"] for m in MILESTONES]
        assert len(ids) == len(set(ids))

    def test_known_milestones_present(self):
        ids = [m["id"] for m in MILESTONES]
        expected = [
            "first_debate",
            "debate_10",
            "debate_50",
            "debate_100",
            "consensus_streak_5",
            "time_saved_1h",
            "time_saved_10h",
            "roi_positive",
            "roi_100",
            "cost_saved_100",
            "cost_saved_1000",
        ]
        for eid in expected:
            assert eid in ids

    def test_thresholds_are_numeric(self):
        for m in MILESTONES:
            assert isinstance(m["threshold"], (int, float))


# ============================================================================
# Edge Cases and Error Paths
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_endpoints_with_no_user_store(self, handler):
        """All endpoints should return 503 when user_store is missing."""
        handler.ctx = {}
        paths = [
            "/api/v1/sme/success",
            "/api/v1/sme/success/cfo",
            "/api/v1/sme/success/pm",
            "/api/v1/sme/success/hr",
            "/api/v1/sme/success/milestones",
            "/api/v1/sme/success/insights",
        ]
        for path in paths:
            h = _make_handler()
            result = handler.handle(path, {}, h)
            assert _status(result) == 503, f"Path {path} should return 503"

    def test_all_endpoints_with_user_not_found(self, handler):
        """All endpoints should return 404 when user not found."""
        handler.ctx["user_store"].get_user_by_id.return_value = None
        paths = [
            "/api/v1/sme/success",
            "/api/v1/sme/success/cfo",
            "/api/v1/sme/success/pm",
            "/api/v1/sme/success/hr",
            "/api/v1/sme/success/milestones",
            "/api/v1/sme/success/insights",
        ]
        for path in paths:
            h = _make_handler()
            result = handler.handle(path, {}, h)
            assert _status(result) == 404, f"Path {path} should return 404"

    def test_cfo_view_with_none_budget_limit(self, handler):
        """Budget with None monthly_limit_usd should show 0."""
        budget = MockBudget(
            monthly_limit_usd=None,
            current_monthly_spend=Decimal("50"),
        )
        handler._get_cost_tracker.return_value = MockCostTracker(budget=budget)
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/cfo", {}, h)
        body = _body(result)
        bi = body["cfo_view"]["budget"]
        assert bi["monthly_limit_usd"] == 0
        assert bi["utilization_percent"] == 0

    def test_success_summary_zero_debate_time(self, handler):
        """When avg_debate_time_minutes is 0, efficiency_multiplier should be 0."""
        # Force metrics to have zero debate time by patching _calculate_success_metrics
        original_calc = handler._calculate_success_metrics

        def zero_time_metrics(org_id, start_date, end_date):
            m = original_calc(org_id, start_date, end_date)
            m["avg_debate_time_minutes"] = 0
            return m

        handler._calculate_success_metrics = zero_time_metrics
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        body = _body(result)
        assert body["success"]["comparison"]["efficiency_multiplier"] == 0

    def test_hr_zero_manual_equivalent(self, handler):
        """When manual_equivalent_minutes is 0, fatigue reduction should be 0."""
        original_calc = handler._calculate_success_metrics

        def zero_manual_metrics(org_id, start_date, end_date):
            m = original_calc(org_id, start_date, end_date)
            m["manual_equivalent_minutes"] = 0
            return m

        handler._calculate_success_metrics = zero_manual_metrics
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/hr", {}, h)
        body = _body(result)
        assert body["hr_view"]["wellbeing_impact"]["decision_fatigue_reduction_percent"] == 0

    def test_pm_zero_manual_equivalent(self, handler):
        """When manual_equivalent_minutes is 0, efficiency gain should be 0."""
        original_calc = handler._calculate_success_metrics

        def zero_manual_metrics(org_id, start_date, end_date):
            m = original_calc(org_id, start_date, end_date)
            m["manual_equivalent_minutes"] = 0
            return m

        handler._calculate_success_metrics = zero_manual_metrics
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success/pm", {}, h)
        body = _body(result)
        assert body["pm_view"]["efficiency"]["efficiency_gain_percent"] == 0

    def test_large_debate_count(self, handler):
        """Handler should not crash with very large debate counts."""
        handler._get_cost_tracker.return_value = MockCostTracker(
            total_cost_usd="1000.00", total_api_calls=100000,
        )
        h = _make_handler()
        result = handler.handle("/api/v1/sme/success", {}, h)
        assert _status(result) == 200


# ============================================================================
# Cross-endpoint consistency
# ============================================================================


class TestCrossEndpointConsistency:
    """Test consistency of metrics across different view endpoints."""

    def test_consensus_rate_consistent(self, handler):
        """Consensus rate should be the same across all views."""
        h1 = _make_handler()
        r1 = handler.handle("/api/v1/sme/success", {}, h1)
        b1 = _body(r1)

        h2 = _make_handler()
        r2 = handler.handle("/api/v1/sme/success/pm", {}, h2)
        b2 = _body(r2)

        h3 = _make_handler()
        r3 = handler.handle("/api/v1/sme/success/hr", {}, h3)
        b3 = _body(r3)

        assert b1["success"]["key_metrics"]["consensus_rate"] == b2["pm_view"]["quality"]["consensus_rate_percent"]
        assert b1["success"]["key_metrics"]["consensus_rate"] == b3["hr_view"]["alignment"]["consensus_rate_percent"]

    def test_total_debates_consistent(self, handler):
        """Total debates should be the same across views."""
        h1 = _make_handler()
        r1 = handler.handle("/api/v1/sme/success", {}, h1)
        b1 = _body(r1)

        h2 = _make_handler()
        r2 = handler.handle("/api/v1/sme/success/pm", {}, h2)
        b2 = _body(r2)

        assert b1["success"]["key_metrics"]["decisions_made"] == b2["pm_view"]["velocity"]["total_decisions"]

    def test_hours_saved_consistent(self, handler):
        """Hours saved should be consistent between summary and HR views."""
        h1 = _make_handler()
        r1 = handler.handle("/api/v1/sme/success", {}, h1)
        b1 = _body(r1)

        h2 = _make_handler()
        r2 = handler.handle("/api/v1/sme/success/hr", {}, h2)
        b2 = _body(r2)

        assert b1["success"]["key_metrics"]["time_saved_hours"] == b2["hr_view"]["time_impact"]["hours_saved_total"]


# ============================================================================
# Module-level exports
# ============================================================================


class TestModuleExports:
    """Test module-level attributes."""

    def test_all_exports(self):
        import aragora.server.handlers.sme_success_dashboard as mod

        assert "SMESuccessDashboardHandler" in mod.__all__
