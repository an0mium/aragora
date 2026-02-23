"""Comprehensive tests for UsageAnalyticsMixin endpoints.

Tests the three usage analytics mixin methods from
aragora/server/handlers/_analytics_metrics_usage.py:

- _get_usage_tokens: Token consumption trends by agent and model
- _get_usage_costs: Cost breakdown by provider and model
- _get_active_users: Active user counts with growth data

Also tests routing via the AnalyticsMetricsHandler.handle() async method.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics import AnalyticsMetricsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Minimal mock HTTP handler for handle() routing tests."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 54321)
        self.headers: dict[str, str] = {"User-Agent": "test"}
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Auth context helpers
# ---------------------------------------------------------------------------


class AdminAuth:
    """Mock admin auth context."""

    org_id = "test-org-001"
    roles = {"admin"}


class UserAuth:
    """Mock non-admin auth context."""

    org_id = "user-org-001"
    roles = set()


class PlatformAdminAuth:
    """Mock platform admin auth context."""

    org_id = "platform-org"
    roles = {"platform_admin"}


# ---------------------------------------------------------------------------
# Cost tracker factory helpers
# ---------------------------------------------------------------------------


def _make_cost_tracker(
    total_tokens_in: int = 500000,
    total_tokens_out: int = 100000,
    total_cost_usd: str = "125.50",
    total_api_calls: int = 150,
    cost_by_agent: dict[str, Any] | None = None,
    cost_by_model: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock cost tracker."""
    tracker = MagicMock()
    stats = {
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_cost_usd": total_cost_usd,
        "total_api_calls": total_api_calls,
        "cost_by_agent": cost_by_agent or {},
        "cost_by_model": cost_by_model or {},
    }
    tracker.get_workspace_stats.return_value = stats
    return tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AnalyticsMetricsHandler with empty context."""
    return AnalyticsMetricsHandler({})


@pytest.fixture
def handler_with_user_store():
    """Create an AnalyticsMetricsHandler with a user_store in context."""
    user_store = MagicMock()
    user_store.get_active_user_counts.return_value = {
        "daily": 25,
        "weekly": 85,
        "monthly": 150,
    }
    user_store.get_user_growth.return_value = {
        "new_users": 15,
        "churned_users": 5,
        "net_growth": 10,
    }
    return AnalyticsMetricsHandler({"user_store": user_store})


@pytest.fixture
def http_handler():
    """Mock HTTP handler for async handle() tests."""
    return MockHTTPHandler()


@pytest.fixture
def mock_tracker():
    """Create a default mock cost tracker."""
    return _make_cost_tracker()


# ============================================================================
# _get_usage_tokens
# ============================================================================


class TestUsageTokens:
    """Tests for _get_usage_tokens."""

    def test_success_with_cost_tracker(self, handler, mock_tracker):
        """Returns token consumption data from cost tracker."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "time_range": "30d", "granularity": "daily"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["org_id"] == "test-org-001"
        assert body["time_range"] == "30d"
        assert body["granularity"] == "daily"
        assert body["summary"]["total_tokens_in"] == 500000
        assert body["summary"]["total_tokens_out"] == 100000
        assert body["summary"]["total_tokens"] == 600000
        assert "avg_tokens_per_day" in body["summary"]
        assert "generated_at" in body

    def test_missing_org_id_returns_400(self, handler):
        """org_id is required for token usage."""
        result = handler._get_usage_tokens({}, auth_context=AdminAuth())

        assert _status(result) == 400
        body = _body(result)
        assert "org_id" in body.get("error", body.get("message", "")).lower()

    def test_org_access_denied_for_non_admin(self, handler):
        """Non-admin users cannot access another org's tokens."""
        result = handler._get_usage_tokens(
            {"org_id": "other-org"},
            auth_context=UserAuth(),
        )

        assert _status(result) == 403

    def test_admin_can_access_any_org(self, handler, mock_tracker):
        """Admin role bypasses org access check."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "any-org"},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "any-org"

    def test_platform_admin_can_access_any_org(self, handler, mock_tracker):
        """Platform admin role bypasses org access check."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "other-org"},
                auth_context=PlatformAdminAuth(),
            )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "other-org"

    def test_invalid_time_range_defaults_to_30d(self, handler, mock_tracker):
        """Invalid time_range silently defaults to 30d."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "time_range": "garbage"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_invalid_granularity_defaults_to_daily(self, handler, mock_tracker):
        """Invalid granularity silently defaults to daily."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "granularity": "hourly"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["granularity"] == "daily"

    def test_all_valid_time_ranges(self, handler, mock_tracker):
        """Every valid time range returns 200."""
        for tr in ("7d", "14d", "30d", "90d", "180d", "365d", "all"):
            with patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
                create=True,
            ):
                result = handler._get_usage_tokens(
                    {"org_id": "test-org-001", "time_range": tr},
                    auth_context=AdminAuth(),
                )
            assert _status(result) == 200, f"Failed for time_range={tr}"
            assert _body(result)["time_range"] == tr

    def test_all_valid_granularities(self, handler, mock_tracker):
        """Every valid granularity returns 200."""
        for g in ("daily", "weekly", "monthly"):
            with patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
                create=True,
            ):
                result = handler._get_usage_tokens(
                    {"org_id": "test-org-001", "granularity": g},
                    auth_context=AdminAuth(),
                )
            assert _status(result) == 200, f"Failed for granularity={g}"
            assert _body(result)["granularity"] == g

    def test_total_tokens_is_sum(self, handler):
        """total_tokens = total_tokens_in + total_tokens_out."""
        tracker = _make_cost_tracker(total_tokens_in=300, total_tokens_out=200)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["total_tokens"] == 500

    def test_avg_tokens_per_day_calculation(self, handler):
        """avg_tokens_per_day is total / days."""
        tracker = _make_cost_tracker(total_tokens_in=600, total_tokens_out=400)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "time_range": "30d"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        # avg_tokens_per_day = (600 + 400) / 30 = 33.333... rounded to 0 decimals
        assert body["summary"]["avg_tokens_per_day"] == round(1000 / 30, 0)

    def test_time_range_all_no_days_division(self, handler):
        """time_range=all means _parse_time_range returns None, so days=30 (default)."""
        tracker = _make_cost_tracker(total_tokens_in=300, total_tokens_out=0)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "time_range": "all"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "all"
        # When start_time is None (all), days stays 30 (default)
        assert body["summary"]["avg_tokens_per_day"] == round(300 / 30, 0)

    def test_by_agent_data_included(self, handler):
        """by_agent data from cost tracker is passed through."""
        tracker = _make_cost_tracker(
            cost_by_agent={"claude": "80.00", "gpt-4": "45.00"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_agent"] == {"claude": "80.00", "gpt-4": "45.00"}

    def test_by_model_data_included(self, handler):
        """by_model data from cost tracker is passed through."""
        tracker = _make_cost_tracker(
            cost_by_model={"claude-opus-4": "60.00"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_model"] == {"claude-opus-4": "60.00"}

    def test_import_error_fallback(self, handler):
        """When cost_tracker import fails, return zeros with message."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no cost tracker"),
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["summary"]["total_tokens_in"] == 0
        assert body["summary"]["total_tokens_out"] == 0
        assert body["summary"]["total_tokens"] == 0
        assert body["summary"]["avg_tokens_per_day"] == 0
        assert "message" in body

    def test_zero_tokens(self, handler):
        """Zero tokens should be returned correctly."""
        tracker = _make_cost_tracker(total_tokens_in=0, total_tokens_out=0)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["total_tokens"] == 0
        assert body["summary"]["avg_tokens_per_day"] == 0

    def test_generated_at_is_present(self, handler, mock_tracker):
        """Response always includes generated_at timestamp."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert "generated_at" in body
        # Validate it's an ISO format datetime
        datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00"))

    def test_missing_stats_keys_default_to_zero(self, handler):
        """Missing keys in stats dict default to 0."""
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = {}  # Empty stats
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["total_tokens_in"] == 0
        assert body["summary"]["total_tokens_out"] == 0
        assert body["summary"]["total_tokens"] == 0

    def test_user_can_access_own_org(self, handler, mock_tracker):
        """Non-admin user can access their own org."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "user-org-001"},
                auth_context=UserAuth(),
            )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "user-org-001"


# ============================================================================
# _get_usage_costs
# ============================================================================


class TestUsageCosts:
    """Tests for _get_usage_costs."""

    def test_success_with_cost_tracker(self, handler):
        """Returns cost breakdown data from cost tracker."""
        tracker = _make_cost_tracker(
            total_cost_usd="125.50",
            total_api_calls=150,
            cost_by_agent={"anthropic": "80.00", "openai": "45.50"},
            cost_by_model={"claude-opus-4": "60.00", "gpt-4": "45.50"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001", "time_range": "30d"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["org_id"] == "test-org-001"
        assert body["time_range"] == "30d"
        assert body["summary"]["total_cost_usd"] == "125.50"
        assert body["summary"]["total_api_calls"] == 150
        assert "avg_cost_per_day" in body["summary"]
        assert "avg_cost_per_debate" in body["summary"]
        assert "by_provider" in body
        assert "by_model" in body
        assert "generated_at" in body

    def test_missing_org_id_returns_400(self, handler):
        """org_id is required for cost usage."""
        result = handler._get_usage_costs({}, auth_context=AdminAuth())

        assert _status(result) == 400
        body = _body(result)
        assert "org_id" in body.get("error", body.get("message", "")).lower()

    def test_org_access_denied_for_non_admin(self, handler):
        """Non-admin users cannot access another org's costs."""
        result = handler._get_usage_costs(
            {"org_id": "other-org"},
            auth_context=UserAuth(),
        )

        assert _status(result) == 403

    def test_admin_can_access_any_org(self, handler):
        """Admin role bypasses org access check for costs."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "any-org"},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "any-org"

    def test_invalid_time_range_defaults_to_30d(self, handler):
        """Invalid time_range silently defaults to 30d."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001", "time_range": "invalid"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_all_valid_time_ranges(self, handler):
        """Every valid time range returns 200."""
        tracker = _make_cost_tracker()
        for tr in ("7d", "14d", "30d", "90d", "180d", "365d", "all"):
            with patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=tracker,
                create=True,
            ):
                result = handler._get_usage_costs(
                    {"org_id": "test-org-001", "time_range": tr},
                    auth_context=AdminAuth(),
                )
            assert _status(result) == 200, f"Failed for time_range={tr}"
            assert _body(result)["time_range"] == tr

    def test_avg_cost_per_day_calculation(self, handler):
        """avg_cost_per_day = total_cost / days."""
        tracker = _make_cost_tracker(total_cost_usd="300.00")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001", "time_range": "30d"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        # avg_cost_per_day = 300 / 30 = 10.00
        assert body["summary"]["avg_cost_per_day"] == f"{300.0 / 30:.2f}"

    def test_avg_cost_per_debate_calculation(self, handler):
        """avg_cost_per_debate = total_cost / total_api_calls."""
        tracker = _make_cost_tracker(total_cost_usd="100.00", total_api_calls=50)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["avg_cost_per_debate"] == "2.00"

    def test_zero_api_calls_avg_per_debate_is_zero(self, handler):
        """avg_cost_per_debate = 0 when no api calls."""
        tracker = _make_cost_tracker(total_cost_usd="100.00", total_api_calls=0)
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["avg_cost_per_debate"] == "0.00"

    def test_provider_percentages(self, handler):
        """Provider breakdown includes cost and percentage."""
        tracker = _make_cost_tracker(
            total_cost_usd="100.00",
            cost_by_agent={"anthropic": "60.00", "openai": "40.00"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        providers = body["by_provider"]
        assert providers["anthropic"]["cost"] == "60.00"
        assert providers["anthropic"]["percentage"] == 60.0
        assert providers["openai"]["cost"] == "40.00"
        assert providers["openai"]["percentage"] == 40.0

    def test_zero_total_cost_provider_percentages(self, handler):
        """When total cost is 0, all provider percentages are 0."""
        tracker = _make_cost_tracker(
            total_cost_usd="0",
            cost_by_agent={"anthropic": "0"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_provider"]["anthropic"]["percentage"] == 0

    def test_import_error_fallback(self, handler):
        """When cost_tracker import fails, return zeros with message."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no cost tracker"),
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["summary"]["total_cost_usd"] == "0.00"
        assert body["summary"]["avg_cost_per_day"] == "0.00"
        assert body["summary"]["avg_cost_per_debate"] == "0.00"
        assert body["by_provider"] == {}
        assert body["by_model"] == {}
        assert "message" in body

    def test_generated_at_is_present(self, handler):
        """Response always includes generated_at timestamp."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert "generated_at" in body
        datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00"))

    def test_by_model_passed_through(self, handler):
        """by_model data from cost tracker is passed through directly."""
        tracker = _make_cost_tracker(
            cost_by_model={"claude-opus-4": {"cost": "60.00", "tokens": 400000}},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_model"]["claude-opus-4"]["cost"] == "60.00"
        assert body["by_model"]["claude-opus-4"]["tokens"] == 400000

    def test_empty_cost_by_agent(self, handler):
        """Empty cost_by_agent returns empty by_provider."""
        tracker = _make_cost_tracker(cost_by_agent={})
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_provider"] == {}

    def test_user_can_access_own_org(self, handler):
        """Non-admin user can access their own org's costs."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "user-org-001"},
                auth_context=UserAuth(),
            )

        assert _status(result) == 200

    def test_cost_formatting_precision(self, handler):
        """Costs are formatted to 2 decimal places."""
        tracker = _make_cost_tracker(
            total_cost_usd="99.999",
            cost_by_agent={"provider_a": "33.333"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        # total_cost should be formatted
        assert body["summary"]["total_cost_usd"] == "100.00"
        # provider cost should also be formatted
        assert body["by_provider"]["provider_a"]["cost"] == "33.33"

    def test_percentage_rounding(self, handler):
        """Provider percentages are rounded to 1 decimal place."""
        tracker = _make_cost_tracker(
            total_cost_usd="99.00",
            cost_by_agent={"a": "33.00", "b": "33.00", "c": "33.00"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        for provider in body["by_provider"].values():
            assert provider["percentage"] == pytest.approx(33.3, abs=0.1)


# ============================================================================
# _get_active_users
# ============================================================================


class TestActiveUsers:
    """Tests for _get_active_users."""

    def test_success_with_user_store(self, handler_with_user_store):
        """Returns active user counts from user store."""
        result = handler_with_user_store._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["active_users"]["daily"] == 25
        assert body["active_users"]["weekly"] == 85
        assert body["active_users"]["monthly"] == 150
        assert body["user_growth"]["new_users"] == 15
        assert body["user_growth"]["churned_users"] == 5
        assert body["user_growth"]["net_growth"] == 10
        assert "generated_at" in body

    def test_no_user_store_returns_zeros(self, handler):
        """When user store is not available, return zeros with message."""
        result = handler._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["active_users"]["daily"] == 0
        assert body["active_users"]["weekly"] == 0
        assert body["active_users"]["monthly"] == 0
        assert body["user_growth"]["new_users"] == 0
        assert body["user_growth"]["churned_users"] == 0
        assert body["user_growth"]["net_growth"] == 0
        assert body["activity_distribution"]["power_users"] == 0
        assert body["activity_distribution"]["regular_users"] == 0
        assert body["activity_distribution"]["occasional_users"] == 0
        assert "message" in body

    def test_org_access_denied_for_non_admin(self, handler):
        """Non-admin users cannot access another org's active users."""
        result = handler._get_active_users(
            {"org_id": "other-org"},
            auth_context=UserAuth(),
        )

        assert _status(result) == 403

    def test_admin_can_access_any_org(self, handler_with_user_store):
        """Admin role bypasses org access check."""
        result = handler_with_user_store._get_active_users(
            {"org_id": "any-org"},
            auth_context=AdminAuth(),
        )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "any-org"

    def test_invalid_time_range_defaults_to_30d(self, handler):
        """Invalid time_range defaults to 30d (only 7d, 30d, 90d accepted)."""
        result = handler._get_active_users(
            {"time_range": "365d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_valid_time_ranges(self, handler):
        """Only 7d, 30d, 90d are valid for active users."""
        for tr in ("7d", "30d", "90d"):
            result = handler._get_active_users(
                {"time_range": tr},
                auth_context=AdminAuth(),
            )
            body = _body(result)
            assert body["time_range"] == tr, f"Failed for time_range={tr}"

    def test_time_ranges_180d_365d_all_default_to_30d(self, handler):
        """180d, 365d, all are not valid for active_users - default to 30d."""
        for tr in ("180d", "365d", "all", "bogus"):
            result = handler._get_active_users(
                {"time_range": tr},
                auth_context=AdminAuth(),
            )
            body = _body(result)
            assert body["time_range"] == "30d", f"Expected 30d for time_range={tr}"

    def test_no_org_id_admin_gets_none_org(self, handler):
        """Admin with no org_id gets org_id=None (admin check returns requested_org_id)."""
        result = handler._get_active_users(
            {},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        # Admin path returns requested_org_id directly (None when not provided)
        assert body["org_id"] is None

    def test_no_org_id_regular_user_gets_own_org(self, handler):
        """Non-admin with no org_id gets their own org_id."""
        result = handler._get_active_users(
            {},
            auth_context=UserAuth(),
        )

        body = _body(result)
        assert body["org_id"] == "user-org-001"

    def test_user_store_without_get_active_user_counts(self):
        """When user_store lacks get_active_user_counts, return zeros."""
        user_store = MagicMock(spec=[])  # no methods
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["active_users"]["daily"] == 0

    def test_user_store_without_get_user_growth(self):
        """When user_store lacks get_user_growth, growth defaults to zeros."""
        user_store = MagicMock()
        user_store.get_active_user_counts.return_value = {
            "daily": 10,
            "weekly": 20,
            "monthly": 30,
        }
        # Remove get_user_growth so hasattr returns False
        del user_store.get_user_growth
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert body["active_users"]["daily"] == 10
        assert body["user_growth"]["new_users"] == 0

    def test_user_store_raises_value_error(self):
        """ValueError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = ValueError("bad data")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["active_users"]["daily"] == 0
        assert "error" in body

    def test_user_store_raises_type_error(self):
        """TypeError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = TypeError("type error")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert "error" in body

    def test_user_store_raises_key_error(self):
        """KeyError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = KeyError("missing")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert "error" in body

    def test_user_store_raises_attribute_error(self):
        """AttributeError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = AttributeError("no attr")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert "error" in body

    def test_user_store_raises_os_error(self):
        """OSError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = OSError("disk error")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert "error" in body

    def test_user_store_raises_runtime_error(self):
        """RuntimeError from user_store is caught gracefully."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = RuntimeError("runtime issue")
        h = AnalyticsMetricsHandler({"user_store": user_store})

        result = h._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert _status(result) == 200
        assert "error" in body

    def test_generated_at_is_present(self, handler):
        """Response always includes generated_at timestamp."""
        result = handler._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert "generated_at" in body
        datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00"))

    def test_activity_distribution_zeros_with_user_store(self, handler_with_user_store):
        """activity_distribution is always zeros (placeholder)."""
        result = handler_with_user_store._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert body["activity_distribution"]["power_users"] == 0
        assert body["activity_distribution"]["regular_users"] == 0
        assert body["activity_distribution"]["occasional_users"] == 0

    def test_platform_admin_access(self, handler):
        """Platform admin can access active users for any org."""
        result = handler._get_active_users(
            {"org_id": "some-other-org"},
            auth_context=PlatformAdminAuth(),
        )

        assert _status(result) == 200
        assert _body(result)["org_id"] == "some-other-org"

    def test_user_can_access_own_org(self, handler):
        """Non-admin can access active users for their own org."""
        result = handler._get_active_users(
            {"org_id": "user-org-001"},
            auth_context=UserAuth(),
        )

        assert _status(result) == 200


# ============================================================================
# Async handle() routing tests
# ============================================================================


class TestHandleRouting:
    """Tests for routing usage endpoints through the async handle() method."""

    @pytest.mark.asyncio
    async def test_route_usage_tokens(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/tokens."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = await handler.handle(
                "/api/v1/analytics/usage/tokens",
                {"org_id": "test-org-001"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        assert "total_tokens_in" in body["summary"]

    @pytest.mark.asyncio
    async def test_route_usage_costs(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/costs."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = await handler.handle(
                "/api/v1/analytics/usage/costs",
                {"org_id": "test-org-001"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        assert "total_cost_usd" in body["summary"]

    @pytest.mark.asyncio
    async def test_route_usage_active_users(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/active_users."""
        result = await handler.handle(
            "/api/v1/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "active_users" in body

    @pytest.mark.asyncio
    async def test_route_unversioned_tokens(self, handler, http_handler):
        """handle() accepts unversioned /api/analytics/usage/tokens."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = await handler.handle(
                "/api/analytics/usage/tokens",
                {"org_id": "test-org-001"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_unversioned_costs(self, handler, http_handler):
        """handle() accepts unversioned /api/analytics/usage/costs."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = await handler.handle(
                "/api/analytics/usage/costs",
                {"org_id": "test-org-001"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_unversioned_active_users(self, handler, http_handler):
        """handle() accepts unversioned /api/analytics/usage/active_users."""
        result = await handler.handle(
            "/api/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_usage_route_returns_none(self, handler, http_handler):
        """handle() returns None for unrecognized usage routes."""
        result = await handler.handle(
            "/api/v1/analytics/usage/unknown_endpoint",
            {},
            http_handler,
        )

        assert result is None


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() route matching for usage endpoints."""

    def test_all_usage_routes_versioned(self, handler):
        """All versioned usage analytics routes are recognized."""
        routes = [
            "/api/v1/analytics/usage/tokens",
            "/api/v1/analytics/usage/costs",
            "/api/v1/analytics/usage/active_users",
        ]
        for route in routes:
            assert handler.can_handle(route), f"can_handle failed for {route}"

    def test_all_usage_routes_unversioned(self, handler):
        """All unversioned usage analytics routes are recognized."""
        routes = [
            "/api/analytics/usage/tokens",
            "/api/analytics/usage/costs",
            "/api/analytics/usage/active_users",
        ]
        for route in routes:
            assert handler.can_handle(route), f"can_handle failed for {route}"

    def test_unknown_usage_routes_not_handled(self, handler):
        """Unrecognized usage routes return False."""
        assert not handler.can_handle("/api/v1/analytics/usage/unknown")
        assert not handler.can_handle("/api/v1/analytics/usage")
        assert not handler.can_handle("/api/v1/usage/tokens")

    def test_non_analytics_routes_not_handled(self, handler):
        """Non-analytics routes return False."""
        assert not handler.can_handle("/api/v1/debates/list")
        assert not handler.can_handle("/api/v1/health")
        assert not handler.can_handle("/random/path")


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityEdgeCases:
    """Security and edge case tests for usage analytics endpoints."""

    def test_path_traversal_org_id_tokens(self, handler, mock_tracker):
        """Path traversal in org_id should not cause issues."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "../../etc/passwd"},
                auth_context=AdminAuth(),
            )

        # Should still return 200 - org_id is just passed to the tracker
        assert _status(result) == 200

    def test_sql_injection_org_id_costs(self, handler, mock_tracker):
        """SQL injection in org_id should not cause issues."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "'; DROP TABLE costs; --"},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200

    def test_xss_in_org_id(self, handler, mock_tracker):
        """XSS in org_id should be harmlessly passed through."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "<script>alert(1)</script>"},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200

    def test_very_long_org_id(self, handler, mock_tracker):
        """Very long org_id does not crash."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "a" * 10000},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200

    def test_empty_string_org_id_returns_400(self, handler):
        """Empty string org_id is falsy and returns 400."""
        result = handler._get_usage_tokens(
            {"org_id": ""},
            auth_context=AdminAuth(),
        )

        assert _status(result) == 400

    def test_none_auth_context_active_users(self, handler):
        """None auth context should still be handled (org_id from params)."""
        # The handler uses getattr on auth_context, which handles None
        result = handler._get_active_users(
            {"org_id": "some-org"},
            auth_context=None,
        )

        # With None auth_context, user_org_id is None, roles is []
        # Since no org_id mismatch (user_org_id is None), it returns the org_id
        assert _status(result) == 200

    def test_unicode_org_id(self, handler, mock_tracker):
        """Unicode in org_id should be handled gracefully."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "\u00e9\u00e8\u00ea-org"},
                auth_context=AdminAuth(),
            )

        assert _status(result) == 200

    def test_null_byte_in_time_range(self, handler, mock_tracker):
        """Null byte in time_range defaults to 30d."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001", "time_range": "30d\x00DROP"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "30d"  # Invalid, defaults to 30d


# ============================================================================
# _validate_org_access tests
# ============================================================================


class TestValidateOrgAccess:
    """Tests for _validate_org_access method used by all usage endpoints."""

    def test_admin_role_grants_access(self, handler):
        """Admin role can access any org."""
        org_id, err = handler._validate_org_access(AdminAuth(), "any-org")
        assert org_id == "any-org"
        assert err is None

    def test_platform_admin_grants_access(self, handler):
        """Platform admin role can access any org."""
        org_id, err = handler._validate_org_access(PlatformAdminAuth(), "any-org")
        assert org_id == "any-org"
        assert err is None

    def test_user_own_org_allowed(self, handler):
        """Non-admin can access their own org."""
        org_id, err = handler._validate_org_access(UserAuth(), "user-org-001")
        assert org_id == "user-org-001"
        assert err is None

    def test_user_other_org_denied(self, handler):
        """Non-admin denied access to other org."""
        org_id, err = handler._validate_org_access(UserAuth(), "other-org")
        assert org_id is None
        assert err is not None
        assert _status(err) == 403

    def test_no_requested_org_uses_user_org(self, handler):
        """When no org requested, returns user's org."""
        org_id, err = handler._validate_org_access(UserAuth(), None)
        assert org_id == "user-org-001"
        assert err is None

    def test_none_auth_context_no_org_requested(self, handler):
        """None auth context with no requested org returns None org_id."""
        org_id, err = handler._validate_org_access(None, None)
        assert org_id is None
        assert err is None

    def test_admin_in_list_roles(self, handler):
        """Admin role works when roles is a list too."""

        class ListRolesAuth:
            org_id = "some-org"
            roles = ["admin", "viewer"]

        org_id, err = handler._validate_org_access(ListRolesAuth(), "other-org")
        assert org_id == "other-org"
        assert err is None


# ============================================================================
# Integration-style tests
# ============================================================================


class TestIntegration:
    """Integration-style tests combining multiple aspects."""

    def test_tokens_and_costs_use_same_org(self, handler):
        """Both tokens and costs endpoints validate org_id the same way."""
        tracker = _make_cost_tracker()

        for method_name in ("_get_usage_tokens", "_get_usage_costs"):
            method = getattr(handler, method_name)
            # Missing org_id
            result = method({}, auth_context=AdminAuth())
            assert _status(result) == 400

            # Access denied
            result = method(
                {"org_id": "other-org"},
                auth_context=UserAuth(),
            )
            assert _status(result) == 403

    def test_all_endpoints_return_generated_at(self, handler):
        """All three usage endpoints include generated_at."""
        tracker = _make_cost_tracker()

        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            r1 = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )
            r2 = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        r3 = handler._get_active_users(
            {"time_range": "30d"},
            auth_context=AdminAuth(),
        )

        assert "generated_at" in _body(r1)
        assert "generated_at" in _body(r2)
        assert "generated_at" in _body(r3)

    def test_import_error_tokens_and_costs_both_degrade(self, handler):
        """Both tokens and costs degrade gracefully on import error."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no module"),
        ):
            tokens_result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )
            costs_result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        assert _status(tokens_result) == 200
        assert "message" in _body(tokens_result)
        assert _status(costs_result) == 200
        assert "message" in _body(costs_result)

    def test_large_token_values(self, handler):
        """Large token values are handled correctly."""
        tracker = _make_cost_tracker(
            total_tokens_in=999_999_999,
            total_tokens_out=888_888_888,
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["total_tokens_in"] == 999_999_999
        assert body["summary"]["total_tokens_out"] == 888_888_888
        assert body["summary"]["total_tokens"] == 1_888_888_887

    def test_high_cost_values(self, handler):
        """High cost values are formatted correctly."""
        tracker = _make_cost_tracker(
            total_cost_usd="99999.99",
            total_api_calls=1,
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["summary"]["total_cost_usd"] == "99999.99"
        assert body["summary"]["avg_cost_per_debate"] == "99999.99"

    def test_tokens_default_query_params(self, handler):
        """Tokens endpoint uses defaults when no time_range or granularity."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_tokens(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "30d"
        assert body["granularity"] == "daily"

    def test_costs_default_query_params(self, handler):
        """Costs endpoint uses 30d default when no time_range."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_active_users_default_query_params(self, handler):
        """Active users endpoint uses 30d default when no time_range."""
        result = handler._get_active_users(
            {},
            auth_context=AdminAuth(),
        )

        body = _body(result)
        assert body["time_range"] == "30d"

    def test_tokens_tracker_get_workspace_stats_called_with_org_id(self, handler):
        """Tokens tracker.get_workspace_stats receives the correct org_id."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            handler._get_usage_tokens(
                {"org_id": "specific-org"},
                auth_context=AdminAuth(),
            )

        tracker.get_workspace_stats.assert_called_once_with("specific-org")

    def test_costs_tracker_get_workspace_stats_called_with_org_id(self, handler):
        """Costs tracker.get_workspace_stats receives the correct org_id."""
        tracker = _make_cost_tracker()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            handler._get_usage_costs(
                {"org_id": "specific-org"},
                auth_context=AdminAuth(),
            )

        tracker.get_workspace_stats.assert_called_once_with("specific-org")

    def test_user_store_get_active_user_counts_called(self):
        """user_store.get_active_user_counts is called with org_id."""
        user_store = MagicMock()
        user_store.get_active_user_counts.return_value = {"daily": 0, "weekly": 0, "monthly": 0}
        user_store.get_user_growth.return_value = {
            "new_users": 0,
            "churned_users": 0,
            "net_growth": 0,
        }
        h = AnalyticsMetricsHandler({"user_store": user_store})

        h._get_active_users(
            {"org_id": "target-org"},
            auth_context=AdminAuth(),
        )

        user_store.get_active_user_counts.assert_called_once_with(org_id="target-org")

    def test_user_store_get_user_growth_called(self):
        """user_store.get_user_growth is called with org_id and days."""
        user_store = MagicMock()
        user_store.get_active_user_counts.return_value = {"daily": 0, "weekly": 0, "monthly": 0}
        user_store.get_user_growth.return_value = {
            "new_users": 0,
            "churned_users": 0,
            "net_growth": 0,
        }
        h = AnalyticsMetricsHandler({"user_store": user_store})

        h._get_active_users(
            {"org_id": "target-org"},
            auth_context=AdminAuth(),
        )

        user_store.get_user_growth.assert_called_once_with(org_id="target-org", days=30)

    def test_tokens_import_error_preserves_org_and_time_range(self, handler):
        """ImportError fallback for tokens preserves org_id and time_range."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("not available"),
        ):
            result = handler._get_usage_tokens(
                {"org_id": "my-org", "time_range": "7d", "granularity": "weekly"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["org_id"] == "my-org"
        assert body["time_range"] == "7d"
        assert body["granularity"] == "weekly"

    def test_costs_import_error_preserves_org_and_time_range(self, handler):
        """ImportError fallback for costs preserves org_id and time_range."""
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("not available"),
        ):
            result = handler._get_usage_costs(
                {"org_id": "my-org", "time_range": "90d"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["org_id"] == "my-org"
        assert body["time_range"] == "90d"

    def test_single_provider_percentage_is_100(self, handler):
        """Single provider with all cost has 100% percentage."""
        tracker = _make_cost_tracker(
            total_cost_usd="50.00",
            cost_by_agent={"anthropic": "50.00"},
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        assert body["by_provider"]["anthropic"]["percentage"] == 100.0

    def test_many_providers_sum_near_100(self, handler):
        """Multiple providers' percentages sum to approximately 100."""
        tracker = _make_cost_tracker(
            total_cost_usd="100.00",
            cost_by_agent={
                "anthropic": "40.00",
                "openai": "30.00",
                "google": "20.00",
                "mistral": "10.00",
            },
        )
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            result = handler._get_usage_costs(
                {"org_id": "test-org-001"},
                auth_context=AdminAuth(),
            )

        body = _body(result)
        total_pct = sum(p["percentage"] for p in body["by_provider"].values())
        assert total_pct == pytest.approx(100.0, abs=0.5)
