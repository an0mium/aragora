"""
Tests for BillingHandler export and forecast endpoints.

Covers:
- GET /api/v1/billing/usage/export - Export usage as CSV
- GET /api/v1/billing/usage/forecast - Usage forecast and cost projection
- Invoice payment recovery (paid + failed)
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.billing.core import BillingHandler


# ---------------------------------------------------------------------------
# Mock classes (reuse pattern from test_billing_core)
# ---------------------------------------------------------------------------


class FakeTier(Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class FakeUser:
    user_id: str = "user-123"
    email: str = "test@example.com"
    role: str = "owner"
    org_id: str = "org-123"


@dataclass
class FakeDbUser:
    id: str = "user-123"
    email: str = "test@example.com"
    org_id: str = "org-123"


@dataclass
class FakeTierLimits:
    debates_per_month: int = 100
    users_per_org: int = 10
    api_access: bool = True
    all_agents: bool = True
    custom_agents: bool = False
    sso_enabled: bool = False
    audit_logs: bool = False
    priority_support: bool = False
    price_monthly_cents: int = 2900

    def to_dict(self) -> dict[str, Any]:
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
        }


@dataclass
class FakeOrganization:
    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: FakeTier = field(default_factory=lambda: FakeTier.STARTER)
    limits: FakeTierLimits = field(default_factory=FakeTierLimits)
    stripe_customer_id: str | None = "cus_test123"
    stripe_subscription_id: str | None = "sub_test123"
    debates_used_this_month: int = 42
    debates_remaining: int = 58
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, tzinfo=timezone.utc)
    )


class FakeHandler:
    """Mock HTTP handler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        headers: dict | None = None,
        query_params: dict | None = None,
    ):
        self.command = method
        self._body = json.dumps(body).encode() if body else b"{}"
        self.headers = headers or {}
        self.client_address = ("127.0.0.1", 12345)
        self._query_params = query_params or {}

    @property
    def rfile(self):
        return io.BytesIO(self._body)

    def get(self, key: str, default: Any = None) -> Any:
        return self._query_params.get(key, default)


class FakeCursor:
    """Mock database cursor that returns rows."""

    def __init__(self, rows: list[tuple] | None = None, error: Exception | None = None):
        self._rows = rows or []
        self._error = error

    def execute(self, query: str, params: list[Any] | None = None) -> None:
        if self._error:
            raise self._error

    def fetchall(self) -> list[tuple]:
        return self._rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    store = MagicMock()
    store.get_user_by_id = MagicMock(return_value=FakeDbUser())
    store.get_organization_by_id = MagicMock(return_value=FakeOrganization())
    store.get_organization_by_stripe_customer = MagicMock(return_value=FakeOrganization())
    store.get_organization_owner = MagicMock(return_value=MagicMock(email="owner@example.com"))
    store.reset_org_usage = MagicMock()
    return store


@pytest.fixture
def billing_handler(mock_user_store):
    return BillingHandler(ctx={"user_store": mock_user_store})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    from aragora.server.handlers.billing.core import _billing_limiter

    _billing_limiter._requests.clear()
    yield
    _billing_limiter._requests.clear()


# ---------------------------------------------------------------------------
# Test _export_usage_csv
# ---------------------------------------------------------------------------


class TestExportUsageCsv:
    """Tests for GET /api/v1/billing/usage/export."""

    def _call(self, billing_handler, handler, user=None):
        """Unwrap decorators and call directly."""
        fn = billing_handler._export_usage_csv.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user or FakeUser())

    def test_returns_csv_content_type(self, billing_handler, mock_user_store):
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        assert result.status_code == 200
        assert result.content_type == "text/csv"
        assert "Content-Disposition" in result.headers
        assert "usage_export_test-org_" in result.headers["Content-Disposition"]

    def test_csv_has_header_row(self, billing_handler, mock_user_store):
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        content = result.body.decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert rows[0] == ["Date", "Event Type", "Count", "Metadata"]

    def test_csv_includes_usage_rows(self, billing_handler, mock_user_store):
        """CSV includes data from database."""
        fake_rows = [
            (1, "org-123", "debate", 5, '{"model":"claude"}', "2026-01-20T10:00:00Z"),
            (2, "org-123", "api_call", 100, "{}", "2026-01-21T10:00:00Z"),
        ]

        @contextmanager
        def fake_transaction():
            yield FakeCursor(rows=fake_rows)

        mock_user_store._transaction = fake_transaction

        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        content = result.body.decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        # Header + 2 data rows + blank + summary header + 5 summary rows
        assert len(rows) >= 4
        assert rows[1][1] == "debate"
        assert rows[2][1] == "api_call"

    def test_csv_includes_summary(self, billing_handler, mock_user_store):
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        content = result.body.decode("utf-8")
        assert "Summary" in content
        assert "Test Org" in content
        assert "starter" in content

    def test_csv_with_date_filtering(self, billing_handler, mock_user_store):
        """Date params are passed to database query."""
        query_captured = {}

        class CapturingCursor:
            def execute(self, query, params=None):
                query_captured["query"] = query
                query_captured["params"] = params

            def fetchall(self):
                return []

        @contextmanager
        def fake_transaction():
            yield CapturingCursor()

        mock_user_store._transaction = fake_transaction

        handler = FakeHandler(query_params={"start": "2026-01-01", "end": "2026-01-31"})
        result = self._call(billing_handler, handler)

        assert result.status_code == 200
        assert "AND created_at >= ?" in query_captured["query"]
        assert "AND created_at <= ?" in query_captured["query"]
        assert "2026-01-01" in query_captured["params"]
        assert "2026-01-31" in query_captured["params"]

    def test_csv_ignores_invalid_dates(self, billing_handler, mock_user_store):
        """Invalid date params are silently ignored."""
        query_captured = {}

        class CapturingCursor:
            def execute(self, query, params=None):
                query_captured["query"] = query
                query_captured["params"] = params

            def fetchall(self):
                return []

        @contextmanager
        def fake_transaction():
            yield CapturingCursor()

        mock_user_store._transaction = fake_transaction

        handler = FakeHandler(query_params={"start": "not-a-date", "end": "2026-13-45"})
        result = self._call(billing_handler, handler)

        assert result.status_code == 200
        assert "AND created_at >= ?" not in query_captured["query"]
        assert "AND created_at <= ?" not in query_captured["query"]

    def test_csv_database_error_returns_500(self, billing_handler, mock_user_store):
        """Database errors return 500 with generic message."""

        @contextmanager
        def failing_transaction():
            yield FakeCursor(error=sqlite3.Error("disk full"))

        mock_user_store._transaction = failing_transaction

        handler = FakeHandler()
        result = self._call(billing_handler, handler)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "disk" not in data.get("error", "").lower()

    def test_csv_empty_results(self, billing_handler, mock_user_store):
        """Empty database results produce CSV with header and summary only."""

        @contextmanager
        def empty_transaction():
            yield FakeCursor(rows=[])

        mock_user_store._transaction = empty_transaction

        handler = FakeHandler()
        result = self._call(billing_handler, handler)

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        # Header row + blank + "Summary" + summary data
        assert rows[0] == ["Date", "Event Type", "Count", "Metadata"]
        assert ["Summary"] in rows

    def test_returns_503_without_user_store(self):
        handler_obj = BillingHandler(ctx={})
        handler = FakeHandler()
        fn = handler_obj._export_usage_csv.__wrapped__.__wrapped__
        result = fn(handler_obj, handler, user=FakeUser())
        assert result.status_code == 503

    def test_returns_404_for_unknown_user(self, billing_handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        assert result.status_code == 404

    def test_returns_404_for_missing_org(self, billing_handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _get_usage_forecast
# ---------------------------------------------------------------------------


class TestGetUsageForecast:
    """Tests for GET /api/v1/billing/usage/forecast."""

    def _call(self, billing_handler, handler, user=None):
        fn = billing_handler._get_usage_forecast.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user or FakeUser())

    def test_returns_forecast_data(self, billing_handler, mock_user_store):
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        assert result.status_code == 200

        data = json.loads(result.body)
        assert "forecast" in data
        forecast = data["forecast"]
        assert "current_usage" in forecast
        assert "projection" in forecast
        assert "days_remaining" in forecast
        assert "will_hit_limit" in forecast

    def test_forecast_projects_debates(self, billing_handler, mock_user_store):
        """Forecast calculates correct debate projection."""
        # 42 debates in ~19 days => ~2.2/day => ~23 more in 11 remaining days
        handler = FakeHandler()
        result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        projection = data["forecast"]["projection"]
        assert projection["debates_per_day"] > 0
        assert projection["debates_end_of_cycle"] >= 42

    def test_forecast_day_zero_avoids_division_by_zero(self, billing_handler, mock_user_store):
        """First day of billing cycle doesn't crash."""
        org = FakeOrganization(
            billing_cycle_start=datetime.now(timezone.utc),
            debates_used_this_month=0,
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()
        result = self._call(billing_handler, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["forecast"]["days_elapsed"] >= 1

    def test_forecast_will_hit_limit(self, billing_handler, mock_user_store):
        """Forecast detects when projected usage exceeds limit."""
        org = FakeOrganization(
            debates_used_this_month=80,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=10),
            limits=FakeTierLimits(debates_per_month=100),
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()

        with (
            patch(
                "aragora.billing.models.SubscriptionTier",
                FakeTier,
            ),
            patch(
                "aragora.billing.models.TIER_LIMITS",
                {
                    FakeTier.STARTER: org.limits,
                    FakeTier.PROFESSIONAL: FakeTierLimits(
                        debates_per_month=500, price_monthly_cents=9900
                    ),
                },
            ),
        ):
            result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        # 80 debates in 10 days = 8/day, 20 days remaining => 160 more
        assert data["forecast"]["will_hit_limit"] is True
        assert data["forecast"]["debates_overage"] > 0

    def test_forecast_under_limit(self, billing_handler, mock_user_store):
        """Forecast shows no limit breach for low usage."""
        org = FakeOrganization(
            debates_used_this_month=5,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=25),
            limits=FakeTierLimits(debates_per_month=100),
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()
        result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        assert data["forecast"]["will_hit_limit"] is False
        assert data["forecast"]["debates_overage"] == 0

    def test_forecast_tier_recommendation(self, billing_handler, mock_user_store):
        """Forecast recommends tier upgrade when hitting limits."""
        org = FakeOrganization(
            tier=FakeTier.FREE,
            debates_used_this_month=90,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=10),
            limits=FakeTierLimits(debates_per_month=50),
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()

        with (
            patch(
                "aragora.billing.models.TIER_LIMITS",
                {
                    org.tier: org.limits,
                    FakeTier.STARTER: FakeTierLimits(
                        debates_per_month=500,
                        price_monthly_cents=2900,
                    ),
                },
            ),
            patch(
                "aragora.billing.models.SubscriptionTier",
                FakeTier,
            ),
        ):
            result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        rec = data["forecast"].get("tier_recommendation")
        assert rec is not None
        assert "recommended_tier" in rec
        assert rec["debates_limit"] > 50

    def test_forecast_no_recommendation_for_enterprise(self, billing_handler, mock_user_store):
        """No upgrade recommendation for enterprise tier."""
        org = FakeOrganization(
            tier=FakeTier.ENTERPRISE,
            debates_used_this_month=900,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=10),
            limits=FakeTierLimits(debates_per_month=100),
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()

        with (
            patch(
                "aragora.billing.models.TIER_LIMITS",
                {org.tier: org.limits},
            ),
            patch(
                "aragora.billing.models.SubscriptionTier",
                FakeTier,
            ),
        ):
            result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        assert data["forecast"]["tier_recommendation"] is None

    def test_forecast_with_usage_tracker(self, billing_handler, mock_user_store):
        """Forecast includes token and cost projections when tracker available."""
        mock_tracker = MagicMock()
        mock_summary = MagicMock()
        mock_summary.total_tokens = 100_000
        mock_summary.total_cost = 5.0
        mock_tracker.get_summary.return_value = mock_summary

        with patch.object(billing_handler, "_get_usage_tracker", return_value=mock_tracker):
            handler = FakeHandler()
            result = self._call(billing_handler, handler)

        data = json.loads(result.body)
        projection = data["forecast"]["projection"]
        assert projection["tokens_per_day"] > 0
        assert projection["cost_end_of_cycle_usd"] > 0

    def test_returns_503_without_user_store(self):
        handler_obj = BillingHandler(ctx={})
        handler = FakeHandler()
        fn = handler_obj._get_usage_forecast.__wrapped__.__wrapped__
        result = fn(handler_obj, handler, user=FakeUser())
        assert result.status_code == 503

    def test_returns_404_for_unknown_user(self, billing_handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        handler = FakeHandler()
        result = self._call(billing_handler, handler)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _handle_invoice_paid (payment recovery)
# ---------------------------------------------------------------------------


class TestHandleInvoicePaid:
    """Tests for invoice.payment_succeeded webhook handler."""

    def test_resets_usage_on_payment(self, billing_handler, mock_user_store):
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs:
            mock_rs.return_value.mark_recovered.return_value = False
            result = billing_handler._handle_invoice_paid(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.reset_org_usage.assert_called_once_with("org-123")

    def test_marks_recovery_on_payment(self, billing_handler, mock_user_store):
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs:
            mock_rs.return_value.mark_recovered.return_value = True
            result = billing_handler._handle_invoice_paid(event, mock_user_store)

        assert result.status_code == 200
        mock_rs.return_value.mark_recovered.assert_called_once_with("org-123")

    def test_handles_usage_reset_failure(self, billing_handler, mock_user_store):
        """Usage reset failure doesn't crash the handler."""
        mock_user_store.reset_org_usage.side_effect = AttributeError("no method")

        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs:
            mock_rs.return_value.mark_recovered.return_value = False
            result = billing_handler._handle_invoice_paid(event, mock_user_store)

        assert result.status_code == 200

    def test_handles_recovery_store_failure(self, billing_handler, mock_user_store):
        """Recovery store failure doesn't crash the handler."""
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs:
            mock_rs.return_value.mark_recovered.side_effect = OSError("store down")
            result = billing_handler._handle_invoice_paid(event, mock_user_store)

        assert result.status_code == 200

    def test_handles_unknown_customer(self, billing_handler, mock_user_store):
        """Unknown Stripe customer doesn't crash."""
        mock_user_store.get_organization_by_stripe_customer.return_value = None

        event = MagicMock()
        event.object = {
            "customer": "cus_unknown",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }

        result = billing_handler._handle_invoice_paid(event, mock_user_store)
        assert result.status_code == 200
        mock_user_store.reset_org_usage.assert_not_called()

    def test_handles_no_user_store(self, billing_handler):
        """Null user_store doesn't crash."""
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 0,
        }

        result = billing_handler._handle_invoice_paid(event, None)
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# Test _handle_invoice_failed (payment failure recovery)
# ---------------------------------------------------------------------------


class TestHandleInvoiceFailed:
    """Tests for invoice.payment_failed webhook handler."""

    def _make_event(self, attempt_count=1, customer="cus_test123"):
        event = MagicMock()
        event.object = {
            "customer": customer,
            "subscription": "sub_test",
            "attempt_count": attempt_count,
            "id": "inv_test",
            "hosted_invoice_url": "https://invoice.stripe.com/pay/test",
        }
        return event

    def test_records_failure(self, billing_handler, mock_user_store):
        event = self._make_event()

        mock_failure = MagicMock()
        mock_failure.attempt_count = 1
        mock_failure.days_failing = 0
        mock_failure.days_until_downgrade = 14

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
        ):
            mock_rs.return_value.record_failure.return_value = mock_failure
            mock_notifier.return_value.notify_payment_failed.return_value = MagicMock(
                method="email", success=True
            )
            result = billing_handler._handle_invoice_failed(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["failure_tracked"] is True
        mock_rs.return_value.record_failure.assert_called_once()

    def test_sends_notification_to_owner(self, billing_handler, mock_user_store):
        event = self._make_event(attempt_count=2)

        mock_failure = MagicMock()
        mock_failure.attempt_count = 2
        mock_failure.days_failing = 3
        mock_failure.days_until_downgrade = 11

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
        ):
            mock_rs.return_value.record_failure.return_value = mock_failure
            mock_notifier.return_value.notify_payment_failed.return_value = MagicMock(
                method="email", success=True
            )
            billing_handler._handle_invoice_failed(event, mock_user_store)

        mock_notifier.return_value.notify_payment_failed.assert_called_once()
        call_kwargs = mock_notifier.return_value.notify_payment_failed.call_args
        assert call_kwargs.kwargs["attempt_count"] == 2
        assert call_kwargs.kwargs["email"] == "owner@example.com"

    def test_grace_period_warning(self, billing_handler, mock_user_store):
        """Logs warning when grace period is almost over."""
        event = self._make_event(attempt_count=4)

        mock_failure = MagicMock()
        mock_failure.attempt_count = 4
        mock_failure.days_failing = 12
        mock_failure.days_until_downgrade = 2

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
            patch("aragora.server.handlers.billing.core.logger") as mock_logger,
        ):
            mock_rs.return_value.record_failure.return_value = mock_failure
            mock_notifier.return_value.notify_payment_failed.return_value = MagicMock(
                method="email", success=True
            )
            billing_handler._handle_invoice_failed(event, mock_user_store)

        # Should log a warning about approaching grace period end
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("grace period" in w.lower() for w in warning_calls)

    def test_recovery_store_failure_graceful(self, billing_handler, mock_user_store):
        """Recovery store failure still returns 200 with failure_tracked=False."""
        event = self._make_event()

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
        ):
            mock_rs.return_value.record_failure.side_effect = OSError("store down")
            mock_notifier.return_value.notify_payment_failed.return_value = MagicMock(
                method="email", success=True
            )
            result = billing_handler._handle_invoice_failed(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["failure_tracked"] is False

    def test_notification_failure_graceful(self, billing_handler, mock_user_store):
        """Notification failure doesn't crash handler."""
        event = self._make_event()

        mock_failure = MagicMock()
        mock_failure.attempt_count = 1
        mock_failure.days_failing = 0
        mock_failure.days_until_downgrade = 14

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
        ):
            mock_rs.return_value.record_failure.return_value = mock_failure
            mock_notifier.return_value.notify_payment_failed.side_effect = OSError("email down")
            result = billing_handler._handle_invoice_failed(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["failure_tracked"] is True

    def test_unknown_customer_returns_success(self, billing_handler, mock_user_store):
        """Unknown customer ID still returns 200."""
        mock_user_store.get_organization_by_stripe_customer.return_value = None

        event = self._make_event(customer="cus_unknown")
        result = billing_handler._handle_invoice_failed(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["failure_tracked"] is False

    def test_no_user_store_returns_success(self, billing_handler):
        event = self._make_event()
        result = billing_handler._handle_invoice_failed(event, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["failure_tracked"] is False

    def test_no_owner_email_skips_notification(self, billing_handler, mock_user_store):
        """If org owner has no email, notification is skipped."""
        mock_user_store.get_organization_owner.return_value = MagicMock(email=None)

        event = self._make_event()

        mock_failure = MagicMock()
        mock_failure.attempt_count = 1
        mock_failure.days_failing = 0
        mock_failure.days_until_downgrade = 14

        with (
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_rs,
            patch("aragora.billing.notifications.get_billing_notifier") as mock_notifier,
        ):
            mock_rs.return_value.record_failure.return_value = mock_failure
            result = billing_handler._handle_invoice_failed(event, mock_user_store)

        assert result.status_code == 200
        mock_notifier.return_value.notify_payment_failed.assert_not_called()


# ---------------------------------------------------------------------------
# Test handler routing for export/forecast
# ---------------------------------------------------------------------------


class TestExportForecastRouting:
    """Verify the router dispatches export/forecast correctly."""

    def test_routes_export_csv(self, billing_handler, mock_user_store):
        handler = FakeHandler(method="GET")
        result = billing_handler.handle("/api/v1/billing/usage/export", {}, handler, method="GET")
        # Will hit auth check (no real user) but shouldn't be 404 or 405
        assert result.status_code != 404
        assert result.status_code != 405

    def test_routes_forecast(self, billing_handler, mock_user_store):
        handler = FakeHandler(method="GET")
        result = billing_handler.handle("/api/v1/billing/usage/forecast", {}, handler, method="GET")
        assert result.status_code != 404
        assert result.status_code != 405

    def test_rejects_post_to_export(self, billing_handler, mock_user_store):
        handler = FakeHandler(method="POST")
        result = billing_handler.handle("/api/v1/billing/usage/export", {}, handler, method="POST")
        assert result.status_code == 405
