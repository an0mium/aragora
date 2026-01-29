"""
Tests for SME Success Dashboard API Handler.

Tests coverage for:
- GET /api/v1/sme/success - Unified success metrics
- GET /api/v1/sme/success/cfo - CFO-focused view (costs, ROI, budget)
- GET /api/v1/sme/success/pm - PM-focused view (velocity, consensus, decisions)
- GET /api/v1/sme/success/hr - HR-focused view (alignment, time savings)
- GET /api/v1/sme/success/milestones - Achievement milestones and gamification
- GET /api/v1/sme/success/insights - Actionable insights and recommendations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers & Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeUser:
    user_id: str = "test_user"
    org_id: str = "org_123"
    role: str = "admin"
    is_authenticated: bool = True
    error_reason: Optional[str] = None


@dataclass
class FakeOrg:
    id: str = "org_123"
    name: str = "Test Corp"


@dataclass
class FakeBudget:
    monthly_limit_usd: Decimal = Decimal("500")
    current_monthly_spend: Decimal = Decimal("125")


def _make_server_context(user_store=None):
    """Build a minimal ServerContext dict with a mock user_store."""
    ctx: dict = {
        "user_store": user_store or MagicMock(),
        "storage": MagicMock(),
    }
    return ctx


def _make_handler(method: str = "GET", client_ip: str = "127.0.0.1", period: str = "month"):
    """Create a mock HTTP handler with common attributes."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = (client_ip, 12345)
    handler.headers = {}
    # Support get_string_param(handler, "period", ...) which calls handler.get(key, default)
    handler.get = lambda key, default=None: {"period": period}.get(key, default)
    return handler


def _make_workspace_stats(total_cost="10.00", api_calls=100):
    return {
        "total_cost_usd": total_cost,
        "total_api_calls": api_calls,
    }


def _parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Autouse fixture to clear rate limiter state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the module-level rate limiter between tests."""
    from aragora.server.handlers.sme_success_dashboard import _dashboard_limiter

    # Reset internal state by clearing the requests dict
    if hasattr(_dashboard_limiter, "_requests"):
        _dashboard_limiter._requests.clear()
    if hasattr(_dashboard_limiter, "_buckets"):
        _dashboard_limiter._buckets.clear()
    if hasattr(_dashboard_limiter, "_timestamps"):
        _dashboard_limiter._timestamps.clear()
    yield


# ---------------------------------------------------------------------------
# Handler construction helper
# ---------------------------------------------------------------------------


def _build_handler_with_mocks(
    workspace_stats=None,
    budget=None,
    user=None,
    org=None,
):
    """Build SMESuccessDashboardHandler with all dependencies mocked."""
    from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

    fake_user = user or FakeUser()
    fake_org = org or FakeOrg()
    ws = workspace_stats or _make_workspace_stats()

    user_store = MagicMock()
    user_store.get_user_by_id.return_value = fake_user
    user_store.get_organization_by_id.return_value = fake_org

    ctx = _make_server_context(user_store=user_store)
    dashboard = SMESuccessDashboardHandler(ctx)

    mock_cost_tracker = MagicMock()
    mock_cost_tracker.get_workspace_stats.return_value = ws
    mock_cost_tracker.get_budget.return_value = budget

    mock_roi_calc = MagicMock()
    mock_roi_calc.hourly_rate = Decimal("75")

    dashboard._get_cost_tracker = MagicMock(return_value=mock_cost_tracker)
    dashboard._get_roi_calculator = MagicMock(return_value=mock_roi_calc)

    return dashboard, mock_cost_tracker, mock_roi_calc


# =========================================================================
# Route handling
# =========================================================================


class TestCanHandle:
    def test_valid_routes(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        h = SMESuccessDashboardHandler(_make_server_context())
        for route in [
            "/api/v1/sme/success",
            "/api/v1/sme/success/cfo",
            "/api/v1/sme/success/pm",
            "/api/v1/sme/success/hr",
            "/api/v1/sme/success/milestones",
            "/api/v1/sme/success/insights",
        ]:
            assert h.can_handle(route) is True, f"Expected True for {route}"

    def test_prefix_match(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        h = SMESuccessDashboardHandler(_make_server_context())
        assert h.can_handle("/api/v1/sme/success/anything") is True

    def test_invalid_routes(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        h = SMESuccessDashboardHandler(_make_server_context())
        assert h.can_handle("/api/v1/sme/other") is False
        assert h.can_handle("/api/v1/debates") is False
        assert h.can_handle("/sme/success") is False


class TestRouting:
    def test_routes_to_success_summary(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(
            dashboard, "_get_success_summary", return_value=MagicMock(status_code=200)
        ) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_routes_to_cfo(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(dashboard, "_get_cfo_view", return_value=MagicMock(status_code=200)) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success/cfo", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_routes_to_pm(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(dashboard, "_get_pm_view", return_value=MagicMock(status_code=200)) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success/pm", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_routes_to_hr(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(dashboard, "_get_hr_view", return_value=MagicMock(status_code=200)) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success/hr", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_routes_to_milestones(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(
            dashboard, "_get_milestones", return_value=MagicMock(status_code=200)
        ) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success/milestones", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_routes_to_insights(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch.object(dashboard, "_get_insights", return_value=MagicMock(status_code=200)) as m:
            with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
                lim.is_allowed.return_value = True
                dashboard.handle("/api/v1/sme/success/insights", {}, _make_handler(), "GET")
                m.assert_called_once()

    def test_method_not_allowed(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(method="DELETE")
        with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            result = dashboard.handle("/api/v1/sme/success", {}, handler, "DELETE")
            assert result.status_code == 405

    def test_rate_limit_exceeded(self):
        dashboard, _, _ = _build_handler_with_mocks()
        with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = False
            result = dashboard.handle("/api/v1/sme/success", {}, _make_handler(), "GET")
            assert result.status_code == 429

    def test_command_attribute_overrides_method(self):
        """Handler.command attribute overrides the method parameter."""
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(method="POST")
        handler.command = "POST"
        with patch("aragora.server.handlers.sme_success_dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            result = dashboard.handle("/api/v1/sme/success", {}, handler, "POST")
            # POST not in route_map so returns 405
            assert result.status_code == 405


# =========================================================================
# Period parsing
# =========================================================================


class TestParsePeriod:
    def test_default_period_is_month(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(period="month")
        start, end, period = dashboard._parse_period(handler)
        assert period == "month"
        assert (end - start).days == pytest.approx(30, abs=1)

    def test_week_period(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(period="week")
        start, end, period = dashboard._parse_period(handler)
        assert period == "week"
        assert (end - start).days == pytest.approx(7, abs=1)

    def test_quarter_period(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(period="quarter")
        start, end, period = dashboard._parse_period(handler)
        assert period == "quarter"
        assert (end - start).days == pytest.approx(90, abs=1)

    def test_year_period(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(period="year")
        start, end, period = dashboard._parse_period(handler)
        assert period == "year"
        assert (end - start).days == pytest.approx(365, abs=1)

    def test_invalid_period_defaults_to_month(self):
        dashboard, _, _ = _build_handler_with_mocks()
        handler = _make_handler(period="invalid_period")
        start, end, period = dashboard._parse_period(handler)
        assert period == "month"


# =========================================================================
# Success Metrics Calculation
# =========================================================================


class TestCalculateSuccessMetrics:
    def test_basic_metrics_with_usage(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        now = datetime.now(timezone.utc)
        metrics = dashboard._calculate_success_metrics("org_123", now - timedelta(days=30), now)

        assert metrics["total_debates"] == 10  # 100 // 10
        assert metrics["total_cost_usd"] == 10.0
        assert metrics["minutes_saved"] > 0
        assert metrics["hours_saved"] > 0
        assert metrics["consensus_rate"] == 85.0

    def test_zero_api_calls(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="0", api_calls=0),
        )
        now = datetime.now(timezone.utc)
        metrics = dashboard._calculate_success_metrics("org_123", now - timedelta(days=30), now)

        assert metrics["total_debates"] == 0
        assert metrics["minutes_saved"] == 0
        assert metrics["roi_percentage"] == 0

    def test_roi_is_positive_for_cheap_usage(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="1.00", api_calls=200),
        )
        now = datetime.now(timezone.utc)
        metrics = dashboard._calculate_success_metrics("org_123", now - timedelta(days=30), now)

        # 20 debates, 5 min each = 100 min AI; manual = 20 * 45 = 900 min
        # savings = 800 min = 13.33 hours * $75/hr = $1000 - cost $1 = $999 ROI
        assert metrics["roi_percentage"] > 0
        assert metrics["net_savings_usd"] > 0

    def test_consensus_streak_capped(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=20),
        )
        now = datetime.now(timezone.utc)
        metrics = dashboard._calculate_success_metrics("org_123", now - timedelta(days=30), now)

        # 20 // 10 = 2 debates, consensus_streak = min(2, 5) = 2
        assert metrics["consensus_streak"] == 2


# =========================================================================
# GET /api/v1/sme/success - Unified success summary
# =========================================================================


class TestSuccessSummary:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        result = dashboard._get_success_summary(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        result = dashboard._get_success_summary(_make_handler(), {})
        body = _parse_body(result)

        assert "success" in body
        s = body["success"]
        assert "headline" in s
        assert "subheadline" in s
        assert "period" in s
        assert "key_metrics" in s
        assert "comparison" in s

    def test_key_metrics_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_success_summary(_make_handler(), {}))
        km = body["success"]["key_metrics"]

        for key in [
            "decisions_made",
            "time_saved_hours",
            "money_saved_usd",
            "net_roi_usd",
            "roi_percentage",
            "consensus_rate",
        ]:
            assert key in km, f"Missing key_metrics field: {key}"

    def test_positive_savings_headline(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="1.00", api_calls=200),
        )
        body = _parse_body(dashboard._get_success_summary(_make_handler(), {}))
        assert "saved" in body["success"]["headline"].lower()

    def test_zero_debates_headline(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="0", api_calls=0),
        )
        body = _parse_body(dashboard._get_success_summary(_make_handler(), {}))
        # 0 debates, net_savings <= 0, so headline says "0 decisions"
        assert "0 decisions" in body["success"]["headline"]

    def test_comparison_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_success_summary(_make_handler(), {}))
        comp = body["success"]["comparison"]
        assert "manual_time_hours" in comp
        assert "ai_time_hours" in comp
        assert "efficiency_multiplier" in comp
        assert comp["efficiency_multiplier"] == 9.0  # 45 / 5

    def test_user_not_found(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        user_store = MagicMock()
        user_store.get_user_by_id.return_value = None
        ctx = _make_server_context(user_store=user_store)
        dashboard = SMESuccessDashboardHandler(ctx)
        result = dashboard._get_success_summary(_make_handler(), {})
        assert result.status_code == 404

    def test_no_user_store(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        ctx: dict = {"storage": MagicMock()}  # no user_store
        dashboard = SMESuccessDashboardHandler(ctx)
        result = dashboard._get_success_summary(_make_handler(), {})
        assert result.status_code == 503

    def test_no_organization(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        user_store = MagicMock()
        user = FakeUser(org_id=None)
        user_store.get_user_by_id.return_value = user
        user_store.get_organization_by_id.return_value = None
        ctx = _make_server_context(user_store=user_store)
        dashboard = SMESuccessDashboardHandler(ctx)
        result = dashboard._get_success_summary(_make_handler(), {})
        assert result.status_code == 404


# =========================================================================
# GET /api/v1/sme/success/cfo - CFO view
# =========================================================================


class TestCFOView:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        result = dashboard._get_cfo_view(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        cfo = body["cfo_view"]

        assert cfo["role"] == "cfo"
        assert "headline" in cfo
        assert "period" in cfo
        assert "financial_summary" in cfo
        assert "cost_efficiency" in cfo
        assert "budget" in cfo
        assert "projections" in cfo

    def test_financial_summary_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        fs = body["cfo_view"]["financial_summary"]

        assert "total_spend_usd" in fs
        assert "value_generated_usd" in fs
        assert "net_roi_usd" in fs
        assert "roi_percentage" in fs

    def test_cost_efficiency_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        ce = body["cfo_view"]["cost_efficiency"]

        assert "cost_per_decision_usd" in ce
        assert "manual_cost_per_decision_usd" in ce
        assert "savings_per_decision_usd" in ce

    def test_budget_info_with_budget(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
            budget=FakeBudget(),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        b = body["cfo_view"]["budget"]

        assert b["monthly_limit_usd"] == 500.0
        assert b["spent_usd"] == 125.0
        assert b["remaining_usd"] == 375.0
        assert b["utilization_percent"] == 25.0

    def test_budget_empty_when_no_budget(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
            budget=None,
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        assert body["cfo_view"]["budget"] == {}

    def test_projections(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        proj = body["cfo_view"]["projections"]

        assert "monthly_run_rate_usd" in proj
        assert "projected_annual_savings_usd" in proj
        assert proj["monthly_run_rate_usd"] == 10.0  # 10 * 30 / 30

    def test_cfo_headline_includes_savings(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_cfo_view(_make_handler(), {}))
        assert "net savings" in body["cfo_view"]["headline"]


# =========================================================================
# GET /api/v1/sme/success/pm - PM view
# =========================================================================


class TestPMView:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        result = dashboard._get_pm_view(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        pm = body["pm_view"]

        assert pm["role"] == "pm"
        assert "headline" in pm
        assert "period" in pm
        assert "velocity" in pm
        assert "quality" in pm
        assert "efficiency" in pm
        assert "trends" in pm

    def test_velocity_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        v = body["pm_view"]["velocity"]

        assert "total_decisions" in v
        assert "decisions_per_day" in v
        assert "decisions_per_week" in v
        assert "avg_decision_time_minutes" in v
        assert "time_to_consensus_minutes" in v

    def test_quality_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        q = body["pm_view"]["quality"]

        assert "consensus_rate_percent" in q
        assert "consensus_streak" in q
        assert "decisions_with_consensus" in q
        assert q["consensus_rate_percent"] == 85.0

    def test_efficiency_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        eff = body["pm_view"]["efficiency"]

        assert "manual_equivalent_hours" in eff
        assert "actual_hours" in eff
        assert "hours_saved" in eff
        assert "efficiency_gain_percent" in eff

    def test_trends_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        t = body["pm_view"]["trends"]

        assert t["velocity_trend"] == "stable"
        assert t["consensus_trend"] == "improving"

    def test_pm_headline(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_pm_view(_make_handler(), {}))
        assert "decisions made" in body["pm_view"]["headline"]


# =========================================================================
# GET /api/v1/sme/success/hr - HR view
# =========================================================================


class TestHRView:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        result = dashboard._get_hr_view(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        hr = body["hr_view"]

        assert hr["role"] == "hr"
        assert "headline" in hr
        assert "period" in hr
        assert "alignment" in hr
        assert "time_impact" in hr
        assert "participation" in hr
        assert "wellbeing_impact" in hr

    def test_alignment_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        a = body["hr_view"]["alignment"]

        assert "consensus_rate_percent" in a
        assert "decisions_with_alignment" in a
        assert "alignment_trend" in a
        assert a["alignment_trend"] == "improving"

    def test_time_impact_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        ti = body["hr_view"]["time_impact"]

        assert "hours_saved_total" in ti
        assert "hours_saved_per_person" in ti
        assert "meeting_hours_avoided" in ti

    def test_participation_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        p = body["hr_view"]["participation"]

        assert p["total_decisions"] == 10
        assert p["avg_participants_per_decision"] == 3
        assert p["decision_inclusivity_score"] == 85

    def test_wellbeing_impact_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        wb = body["hr_view"]["wellbeing_impact"]

        assert "decision_fatigue_reduction_percent" in wb
        assert "conflict_reduction_score" in wb
        assert wb["conflict_reduction_score"] == 78

    def test_hr_headline(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        assert "aligned" in body["hr_view"]["headline"].lower()

    def test_hours_saved_per_person_assumes_five(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_hr_view(_make_handler(), {}))
        ti = body["hr_view"]["time_impact"]
        # hours_saved_per_person = hours_saved / 5
        assert ti["hours_saved_per_person"] == round(ti["hours_saved_total"] / 5, 1)


# =========================================================================
# GET /api/v1/sme/success/milestones
# =========================================================================


class TestMilestones:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        result = dashboard._get_milestones(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        m = body["milestones"]

        assert "total_earned" in m
        assert "total_available" in m
        assert "completion_percent" in m
        assert "earned" in m
        assert "upcoming" in m
        assert "next_milestone" in m

    def test_total_available_matches_milestones(self):
        from aragora.server.handlers.sme_success_dashboard import MILESTONES

        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        assert body["milestones"]["total_available"] == len(MILESTONES)

    def test_earned_milestones_have_required_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        for earned in body["milestones"]["earned"]:
            assert "id" in earned
            assert "name" in earned
            assert "description" in earned
            assert "icon" in earned
            assert earned["earned"] is True
            assert "earned_date" in earned

    def test_upcoming_milestones_have_progress(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        for upcoming in body["milestones"]["upcoming"]:
            assert upcoming["earned"] is False
            assert "progress_percent" in upcoming
            assert "remaining" in upcoming

    def test_upcoming_limited_to_5(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        assert len(body["milestones"]["upcoming"]) <= 5

    def test_first_debate_milestone_earned(self):
        """With 10 debates, the first_debate milestone should be earned."""
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        earned_ids = [m["id"] for m in body["milestones"]["earned"]]
        assert "first_debate" in earned_ids

    def test_debate_10_milestone_earned(self):
        """With 100 api_calls => 10 debates, debate_10 should be earned."""
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        earned_ids = [m["id"] for m in body["milestones"]["earned"]]
        assert "debate_10" in earned_ids

    def test_no_milestones_with_zero_usage(self):
        """With 0 api calls, only ROI milestones with threshold=0 may be earned."""
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="0", api_calls=0),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        # 0 debates, 0 minutes saved, 0 roi => only roi_positive (threshold=0) earned
        earned_ids = [m["id"] for m in body["milestones"]["earned"]]
        # roi_percentage is 0 which >= threshold 0
        assert "roi_positive" in earned_ids

    def test_earned_sorted_by_threshold_desc(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="10.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_milestones(_make_handler(), {}))
        thresholds = [m["threshold"] for m in body["milestones"]["earned"]]
        assert thresholds == sorted(thresholds, reverse=True)


# =========================================================================
# GET /api/v1/sme/success/insights
# =========================================================================


class TestInsights:
    def test_returns_200(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        result = dashboard._get_insights(_make_handler(), {})
        assert result.status_code == 200

    def test_response_structure(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        ins = body["insights"]

        assert "count" in ins
        assert "items" in ins
        assert "generated_at" in ins
        assert ins["count"] == len(ins["items"])

    def test_zero_debates_getting_started_insight(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="0", api_calls=0),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        types = [i["type"] for i in body["insights"]["items"]]
        assert "getting_started" in types

    def test_few_debates_engagement_insight(self):
        # 30 api_calls => 3 debates (< 5)
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="2.00", api_calls=30),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        types = [i["type"] for i in body["insights"]["items"]]
        assert "engagement" in types

    def test_high_roi_success_insight(self):
        # Cheap usage -> high ROI
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="1.00", api_calls=200),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        types = [i["type"] for i in body["insights"]["items"]]
        assert "success" in types

    def test_insights_have_required_fields(self):
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=100),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        for item in body["insights"]["items"]:
            assert "type" in item
            assert "priority" in item
            assert "title" in item
            assert "message" in item
            assert "action" in item
            assert "action_url" in item

    def test_hours_saved_celebration_insight(self):
        # Many debates => lots of hours saved (> 10)
        dashboard, _, _ = _build_handler_with_mocks(
            workspace_stats=_make_workspace_stats(total_cost="5.00", api_calls=500),
        )
        body = _parse_body(dashboard._get_insights(_make_handler(), {}))
        types = [i["type"] for i in body["insights"]["items"]]
        assert "celebration" in types


# =========================================================================
# MILESTONES constant
# =========================================================================


class TestMilestonesConstant:
    def test_all_milestones_have_required_keys(self):
        from aragora.server.handlers.sme_success_dashboard import MILESTONES

        for m in MILESTONES:
            assert "id" in m
            assert "name" in m
            assert "description" in m
            assert "icon" in m
            assert "threshold" in m
            assert "metric" in m

    def test_milestone_ids_unique(self):
        from aragora.server.handlers.sme_success_dashboard import MILESTONES

        ids = [m["id"] for m in MILESTONES]
        assert len(ids) == len(set(ids))

    def test_milestones_count(self):
        from aragora.server.handlers.sme_success_dashboard import MILESTONES

        assert len(MILESTONES) == 11


# =========================================================================
# Module exports
# =========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.server.handlers.sme_success_dashboard import __all__

        assert "SMESuccessDashboardHandler" in __all__

    def test_handler_resource_type(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        assert SMESuccessDashboardHandler.RESOURCE_TYPE == "success_dashboard"

    def test_handler_routes_list(self):
        from aragora.server.handlers.sme_success_dashboard import SMESuccessDashboardHandler

        assert len(SMESuccessDashboardHandler.ROUTES) == 6
