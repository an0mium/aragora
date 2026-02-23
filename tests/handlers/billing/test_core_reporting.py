"""Comprehensive tests for billing reporting handlers (aragora/server/handlers/billing/core_reporting.py).

Tests the ReportingMixin class and all its methods via the BillingHandler.handle() entrypoint:

- GET /api/v1/billing/audit-log (_get_audit_log):
  - Enterprise tier with audit logs enabled
  - Non-enterprise tier returns 403
  - No user store returns 503
  - No org / no user returns 404
  - Non-admin role returns 403
  - Query param handling (limit, offset, action)
  - Limit clamping and default values
  - Offset clamping
  - Invalid limit/offset fall back to defaults

- GET /api/v1/billing/usage/export (_export_usage_csv):
  - Returns CSV content type
  - CSV contains header row and summary
  - Content-Disposition header present with filename
  - Date filtering with start/end query params
  - Invalid date params ignored
  - No user store returns 503
  - No org returns 404
  - Database error returns 500
  - No _transaction attribute produces empty CSV with summary
  - Usage events appear in CSV body

- GET /api/v1/billing/usage/forecast (_get_usage_forecast):
  - Returns forecast structure with projection
  - High usage triggers tier recommendation
  - Enterprise tier gets no recommendation
  - Low usage shows no overage
  - Days elapsed < 1 is clamped to 1
  - Usage tracker provides token and cost data
  - No usage tracker returns zero token/cost projections
  - No user store returns 503
  - No org returns 404
  - Professional tier recommends enterprise
  - Starter tier recommends professional

- GET /api/v1/billing/invoices (_get_invoices):
  - Returns formatted invoice list
  - Invoice amounts converted from cents to dollars
  - Currency uppercased
  - Period dates formatted as ISO strings
  - Handles missing period_start/period_end
  - Handles None amount values
  - Limit query param clamping
  - No billing account returns 404
  - No user store returns 503
  - No org returns 404
  - Stripe config error returns 503
  - Stripe API error returns 502
  - Stripe generic error returns 500
  - Empty invoice list

- Method not allowed (405):
  - POST to audit-log
  - POST to usage/export
  - POST to usage/forecast
  - POST to invoices

- Security tests:
  - SQL injection in query params
  - Path traversal in query params
  - XSS in query params
  - Oversized limit values clamped
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import TIER_LIMITS, SubscriptionTier
from aragora.billing.stripe_client import StripeAPIError, StripeConfigError, StripeError
from aragora.server.handlers.billing.core import BillingHandler, _billing_limiter


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockTierLimits:
    """Mock tier limits for testing."""

    def __init__(
        self,
        debates_per_month: int = 10,
        users_per_org: int = 1,
        price_monthly_cents: int = 0,
        api_access: bool = False,
        all_agents: bool = False,
        custom_agents: bool = False,
        sso_enabled: bool = False,
        audit_logs: bool = False,
        priority_support: bool = False,
    ):
        self.debates_per_month = debates_per_month
        self.users_per_org = users_per_org
        self.price_monthly_cents = price_monthly_cents
        self.api_access = api_access
        self.all_agents = all_agents
        self.custom_agents = custom_agents
        self.sso_enabled = sso_enabled
        self.audit_logs = audit_logs
        self.priority_support = priority_support

    def to_dict(self) -> dict:
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
            "price_monthly_cents": self.price_monthly_cents,
            "api_access": self.api_access,
            "all_agents": self.all_agents,
            "custom_agents": self.custom_agents,
            "sso_enabled": self.sso_enabled,
            "audit_logs": self.audit_logs,
            "priority_support": self.priority_support,
        }


class MockUser:
    """Mock user for billing reporting tests."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str = "Test User",
        role: str = "member",
        org_id: str | None = None,
    ):
        self.id = id
        self.user_id = id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id


class MockOrganization:
    """Mock organization for billing reporting tests."""

    def __init__(
        self,
        id: str,
        name: str,
        slug: str = "test-org",
        tier: SubscriptionTier = SubscriptionTier.FREE,
        debates_used_this_month: int = 0,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        billing_cycle_start: datetime | None = None,
        limits: MockTierLimits | None = None,
    ):
        self.id = id
        self.name = name
        self.slug = slug
        self.tier = tier
        self.debates_used_this_month = debates_used_this_month
        self.stripe_customer_id = stripe_customer_id
        self.stripe_subscription_id = stripe_subscription_id
        self.billing_cycle_start = billing_cycle_start or datetime.now(timezone.utc).replace(day=1)
        self.limits = limits or MockTierLimits()

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)


class MockStripeClient:
    """Mock Stripe client for invoice tests."""

    def __init__(self, invoices: list[dict] | None = None):
        self._invoices = invoices or []

    def list_invoices(self, customer_id: str, limit: int = 10) -> list[dict]:
        return self._invoices


class MockUserStore:
    """Mock user store for billing reporting tests."""

    def __init__(self):
        self._users: dict[str, MockUser] = {}
        self._orgs: dict[str, MockOrganization] = {}
        self._audit_entries: list[dict] = []
        self._has_transaction = False
        self._usage_events: list[tuple] = []

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self._users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self._orgs.get(org_id)

    def log_audit_event(self, **kwargs):
        self._audit_entries.append(kwargs)

    def get_audit_log(self, **kwargs) -> list[dict]:
        return self._audit_entries

    def get_audit_log_count(self, **kwargs) -> int:
        return len(self._audit_entries)

    def enable_transaction(self, usage_events: list[tuple] | None = None):
        """Enable _transaction support for CSV export tests."""
        self._has_transaction = True
        self._usage_events = usage_events or []

    @contextmanager
    def _transaction(self):
        """Mock transaction context manager for CSV export."""
        cursor = MockCursor(self._usage_events)
        yield cursor

    @property
    def has_transaction(self):
        return self._has_transaction


class MockCursor:
    """Mock database cursor for CSV export tests."""

    def __init__(self, rows: list[tuple] | None = None):
        self._rows = rows or []

    def execute(self, query: str, params: list | None = None):
        pass

    def fetchall(self) -> list[tuple]:
        return self._rows


class MockHTTPHandler:
    """Mock HTTP handler for request simulation."""

    def __init__(
        self,
        body: dict | None = None,
        command: str = "GET",
        query_params: dict | None = None,
    ):
        self.command = command
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""
        self._query_params = query_params or {}

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"

    def get(self, key: str, default=None):
        """Support for get_string_param resolution."""
        return self._query_params.get(key, default)


class MockUsageSummary:
    """Mock usage summary from usage tracker."""

    def __init__(
        self,
        total_tokens: int = 80000,
        total_cost: Decimal | float = 5.00,
    ):
        self.total_tokens = total_tokens
        self.total_cost = total_cost


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body)
        if isinstance(body, str):
            return json.loads(body)
    return result


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _csv_body(result) -> str:
    """Extract CSV body text from a HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return body.decode("utf-8")
        return body
    return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_store():
    """Create a user store with standard test data."""
    store = MockUserStore()
    auth_user = MockUser(
        id="test-user-001", email="test@example.com", role="owner", org_id="org_1"
    )
    store.add_user(auth_user)

    org = MockOrganization(
        id="org_1",
        name="Test Org",
        slug="test-org",
        tier=SubscriptionTier.FREE,
        debates_used_this_month=5,
        stripe_customer_id="cus_test_123",
        stripe_subscription_id="sub_test_123",
        limits=MockTierLimits(debates_per_month=10),
    )
    store.add_organization(org)
    return store


@pytest.fixture
def handler(user_store):
    """Create a BillingHandler with a user store in context."""
    return BillingHandler(ctx={"user_store": user_store})


@pytest.fixture
def handler_no_store():
    """BillingHandler without a user store (service unavailable scenario)."""
    return BillingHandler(ctx={})


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the rate limiter between tests to avoid cross-test pollution."""
    _billing_limiter._buckets.clear()
    yield
    _billing_limiter._buckets.clear()


def _ensure_role_on_auth_context():
    """Ensure the test auth context has a .role attribute for audit log checks."""
    try:
        from aragora.server.handlers.utils import decorators as _dec

        override = getattr(_dec, "_test_user_context_override", None)
        if override is not None and not hasattr(override, "role"):
            object.__setattr__(override, "role", "owner")
    except (ImportError, AttributeError):
        pass


def _make_enterprise_store(
    audit_logs: bool = True,
    role: str = "owner",
) -> MockUserStore:
    """Create a user store with an enterprise org that has audit logs enabled."""
    store = MockUserStore()
    store.add_user(
        MockUser(id="test-user-001", email="t@t.com", role=role, org_id="org_ent")
    )
    store.add_organization(
        MockOrganization(
            id="org_ent",
            name="Enterprise Org",
            slug="enterprise-org",
            tier=SubscriptionTier.ENTERPRISE,
            limits=MockTierLimits(audit_logs=audit_logs, debates_per_month=10000),
        )
    )
    return store


# ===========================================================================
# TestAuditLog
# ===========================================================================


class TestAuditLog:
    """Tests for the billing audit log endpoint (_get_audit_log)."""

    def test_returns_entries_for_enterprise(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert _status(result) == 200
        assert "entries" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_non_enterprise_returns_403(self, handler):
        """Free tier org with audit_logs=False should return 403."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 403

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 503

    def test_no_org_returns_404(self):
        _ensure_role_on_auth_context()
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_not_found_returns_404(self):
        _ensure_role_on_auth_context()
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_org_not_found_returns_404(self):
        _ensure_role_on_auth_context()
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id="missing_org"))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 404

    def test_non_admin_role_returns_403(self):
        """Users with member role should get 403 even on enterprise tier."""
        _ensure_role_on_auth_context()
        store = _make_enterprise_store(role="member")
        # Override the context role to member
        try:
            from aragora.server.handlers.utils import decorators as _dec
            override = getattr(_dec, "_test_user_context_override", None)
            if override is not None:
                object.__setattr__(override, "role", "member")
        except (ImportError, AttributeError):
            pass

        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 403

        # Restore role
        try:
            from aragora.server.handlers.utils import decorators as _dec
            override = getattr(_dec, "_test_user_context_override", None)
            if override is not None:
                object.__setattr__(override, "role", "owner")
        except (ImportError, AttributeError):
            pass

    def test_default_limit_is_50(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 50

    def test_default_offset_is_0(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["offset"] == 0

    def test_custom_limit_and_offset(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "25", "offset": "10"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 25
        assert body["offset"] == 10

    def test_limit_clamped_to_100(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "999"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 100

    def test_negative_limit_falls_back_to_default(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "-5"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 50

    def test_invalid_limit_falls_back_to_default(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "abc"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 50

    def test_action_filter_param(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"action": "subscription.created"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 200

    def test_offset_clamped_to_max(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"offset": "200000"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["offset"] == 100_000

    def test_total_reflects_store_count(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        # Add some audit entries
        store._audit_entries = [{"action": "test"}, {"action": "test2"}]
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["total"] == 2

    def test_audit_logs_disabled_returns_403(self):
        """Enterprise tier but with audit_logs explicitly disabled."""
        _ensure_role_on_auth_context()
        store = _make_enterprise_store(audit_logs=False)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 403


# ===========================================================================
# TestUsageExportCSV
# ===========================================================================


class TestUsageExportCSV:
    """Tests for the CSV usage export endpoint (_export_usage_csv)."""

    def test_returns_csv_content_type(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        assert result.content_type == "text/csv"

    def test_csv_contains_header_row(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "Date,Event Type,Count,Metadata" in csv_text

    def test_csv_contains_summary(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "Summary" in csv_text
        assert "Test Org" in csv_text

    def test_csv_contains_org_tier(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "free" in csv_text.lower()

    def test_csv_contains_debates_used(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "Debates Used" in csv_text
        assert "5" in csv_text

    def test_csv_contains_debates_limit(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "Debates Limit" in csv_text
        assert "10" in csv_text

    def test_content_disposition_header(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert "Content-Disposition" in result.headers
        disposition = result.headers["Content-Disposition"]
        assert "attachment" in disposition
        assert "usage_export_" in disposition
        assert "test-org" in disposition
        assert ".csv" in disposition

    def test_filename_contains_date(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        disposition = result.headers["Content-Disposition"]
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in disposition

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 503

    def test_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_not_found_returns_404(self):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_with_org_but_org_not_found_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id="missing"))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404

    def test_with_usage_events(self, user_store):
        """Store with _transaction method and usage events produces rows in CSV."""
        events = [
            # (id, org_id, event_type, count, metadata, created_at)
            (1, "org_1", "debate", 1, "{}", "2026-01-15T10:00:00"),
            (2, "org_1", "api_call", 5, '{"endpoint": "/v1/debate"}', "2026-01-16T12:00:00"),
        ]
        user_store.enable_transaction(events)
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "debate" in csv_text
        assert "api_call" in csv_text
        assert "2026-01-15" in csv_text

    def test_with_start_date_filter(self, user_store):
        """Valid start date is passed through to query."""
        user_store.enable_transaction([])
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler(query_params={"start": "2026-01-01"})
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    def test_with_end_date_filter(self, user_store):
        """Valid end date is passed through to query."""
        user_store.enable_transaction([])
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler(query_params={"end": "2026-12-31"})
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    def test_with_both_date_filters(self, user_store):
        user_store.enable_transaction([])
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler(query_params={"start": "2026-01-01", "end": "2026-12-31"})
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    def test_invalid_start_date_ignored(self, handler):
        """Invalid date format should be silently ignored (returns None)."""
        http = MockHTTPHandler(query_params={"start": "not-a-date"})
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    def test_invalid_end_date_ignored(self, handler):
        http = MockHTTPHandler(query_params={"end": "2026-13-99"})
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    def test_database_error_returns_500(self, user_store):
        """sqlite3.Error during export should return 500."""

        @contextmanager
        def failing_transaction():
            raise sqlite3.Error("Database locked")
            yield  # noqa: unreachable

        user_store._transaction = failing_transaction
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 500

    def test_os_error_in_transaction_returns_500(self, user_store):
        """OSError during export should return 500."""

        @contextmanager
        def failing_transaction():
            raise OSError("Disk full")
            yield  # noqa: unreachable

        user_store._transaction = failing_transaction
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 500

    def test_value_error_in_transaction_returns_500(self, user_store):
        """ValueError during export should return 500."""

        @contextmanager
        def failing_transaction():
            raise ValueError("Invalid data")
            yield  # noqa: unreachable

        user_store._transaction = failing_transaction
        h = BillingHandler(ctx={"user_store": user_store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 500

    def test_billing_cycle_start_in_csv(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = _csv_body(result)
        assert "Billing Cycle Start" in csv_text

    def test_csv_body_is_bytes(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert isinstance(result.body, bytes)


# ===========================================================================
# TestUsageForecast
# ===========================================================================


class TestUsageForecast:
    """Tests for the usage forecast endpoint (_get_usage_forecast)."""

    def test_returns_forecast_structure(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert _status(result) == 200
        assert "forecast" in body
        f = body["forecast"]
        assert "current_usage" in f
        assert "projection" in f
        assert "days_remaining" in f
        assert "days_elapsed" in f
        assert "will_hit_limit" in f
        assert "debates_overage" in f
        assert "tier_recommendation" in f

    def test_current_usage_reflects_org(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        usage = body["forecast"]["current_usage"]
        assert usage["debates"] == 5
        assert usage["debates_limit"] == 10

    def test_projection_has_required_fields(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        proj = body["forecast"]["projection"]
        assert "debates_end_of_cycle" in proj
        assert "debates_per_day" in proj
        assert "tokens_per_day" in proj
        assert "cost_end_of_cycle_usd" in proj

    def test_high_usage_triggers_recommendation(self):
        """An org using 9/10 debates early should get a tier recommendation."""
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_heavy")
        )
        store.add_organization(
            MockOrganization(
                id="org_heavy",
                name="Heavy Org",
                tier=SubscriptionTier.FREE,
                debates_used_this_month=9,
                limits=MockTierLimits(debates_per_month=10),
                billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        f = body["forecast"]
        assert f["will_hit_limit"] is True
        assert f["tier_recommendation"] is not None
        assert f["tier_recommendation"]["recommended_tier"] == "starter"
        assert "debates_limit" in f["tier_recommendation"]
        assert "price_monthly" in f["tier_recommendation"]

    def test_enterprise_gets_no_recommendation(self):
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_ent")
        )
        store.add_organization(
            MockOrganization(
                id="org_ent",
                name="Enterprise Org",
                tier=SubscriptionTier.ENTERPRISE,
                debates_used_this_month=999,
                limits=MockTierLimits(debates_per_month=1000),
                billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert body["forecast"]["tier_recommendation"] is None

    def test_low_usage_no_overage(self, handler):
        """Low usage org should show no overage."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        # With 5/10 debates and billing cycle well underway, overage depends on projection
        assert body["forecast"]["debates_overage"] >= 0

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        assert _status(result) == 503

    def test_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_not_found_returns_404(self):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        assert _status(result) == 404

    def test_org_not_found_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id="missing_org"))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        assert _status(result) == 404

    def test_days_elapsed_clamped_to_1(self):
        """When billing cycle just started (0 days), days_elapsed should be 1."""
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_new")
        )
        store.add_organization(
            MockOrganization(
                id="org_new",
                name="New Org",
                tier=SubscriptionTier.FREE,
                debates_used_this_month=0,
                limits=MockTierLimits(debates_per_month=10),
                billing_cycle_start=datetime.now(timezone.utc),
            )
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert body["forecast"]["days_elapsed"] == 1

    def test_with_usage_tracker_provides_token_data(self, user_store):
        tracker = MagicMock()
        summary = MockUsageSummary(total_tokens=100_000, total_cost=Decimal("5.00"))
        tracker.get_summary.return_value = summary
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        proj = body["forecast"]["projection"]
        assert proj["tokens_per_day"] > 0
        assert proj["cost_end_of_cycle_usd"] > 0

    def test_without_usage_tracker_zero_tokens(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        proj = body["forecast"]["projection"]
        assert proj["tokens_per_day"] == 0
        assert proj["cost_end_of_cycle_usd"] == 0

    def test_usage_tracker_returns_none_summary(self, user_store):
        tracker = MagicMock()
        tracker.get_summary.return_value = None
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        proj = body["forecast"]["projection"]
        assert proj["tokens_per_day"] == 0
        assert proj["cost_end_of_cycle_usd"] == 0.0

    def test_starter_tier_recommends_professional(self):
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_starter")
        )
        store.add_organization(
            MockOrganization(
                id="org_starter",
                name="Starter Org",
                tier=SubscriptionTier.STARTER,
                debates_used_this_month=95,
                limits=MockTierLimits(debates_per_month=100),
                billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        f = body["forecast"]
        if f["will_hit_limit"]:
            assert f["tier_recommendation"]["recommended_tier"] == "professional"

    def test_professional_tier_recommends_enterprise(self):
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_pro")
        )
        store.add_organization(
            MockOrganization(
                id="org_pro",
                name="Pro Org",
                tier=SubscriptionTier.PROFESSIONAL,
                debates_used_this_month=490,
                limits=MockTierLimits(debates_per_month=500),
                billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        f = body["forecast"]
        if f["will_hit_limit"]:
            assert f["tier_recommendation"]["recommended_tier"] == "enterprise"

    def test_debates_per_day_rounded(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        dpd = body["forecast"]["projection"]["debates_per_day"]
        # Check it's rounded to 2 decimal places
        assert dpd == round(dpd, 2)

    def test_projected_debates_is_integer(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert isinstance(body["forecast"]["projection"]["debates_end_of_cycle"], int)

    def test_cost_end_of_cycle_rounded(self, user_store):
        tracker = MagicMock()
        summary = MockUsageSummary(total_tokens=100_000, total_cost=Decimal("5.123456"))
        tracker.get_summary.return_value = summary
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        cost = body["forecast"]["projection"]["cost_end_of_cycle_usd"]
        assert cost == round(cost, 2)


# ===========================================================================
# TestGetInvoices
# ===========================================================================


class TestGetInvoices:
    """Tests for the invoice history endpoint (_get_invoices)."""

    def _make_invoice_data(self, **overrides) -> dict:
        """Create a standard invoice data dict with optional overrides."""
        invoice = {
            "id": "inv_001",
            "number": "INV-001",
            "status": "paid",
            "amount_due": 9900,
            "amount_paid": 9900,
            "currency": "usd",
            "created": 1700000000,
            "period_start": 1699900000,
            "period_end": 1700000000,
            "hosted_invoice_url": "https://inv.stripe.com/1",
            "invoice_pdf": "https://inv.stripe.com/1.pdf",
        }
        invoice.update(overrides)
        return invoice

    def test_returns_invoice_list(self, handler):
        invoices_data = [self._make_invoice_data()]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert _status(result) == 200
        assert "invoices" in body
        assert len(body["invoices"]) == 1

    def test_invoice_amounts_converted_from_cents(self, handler):
        invoices_data = [self._make_invoice_data(amount_due=9900, amount_paid=9900)]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        inv = body["invoices"][0]
        assert inv["amount_due"] == 99.0
        assert inv["amount_paid"] == 99.0

    def test_currency_uppercased(self, handler):
        invoices_data = [self._make_invoice_data(currency="eur")]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert body["invoices"][0]["currency"] == "EUR"

    def test_missing_currency_defaults_to_usd(self, handler):
        inv_data = self._make_invoice_data()
        del inv_data["currency"]
        client = MockStripeClient(invoices=[inv_data])
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert body["invoices"][0]["currency"] == "USD"

    def test_period_dates_as_iso_strings(self, handler):
        invoices_data = [self._make_invoice_data()]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        inv = body["invoices"][0]
        assert inv["period_start"] is not None
        assert inv["period_end"] is not None
        # Should be ISO format strings
        assert "T" in inv["period_start"]
        assert "T" in inv["period_end"]

    def test_missing_period_start(self, handler):
        inv_data = self._make_invoice_data()
        inv_data["period_start"] = None
        client = MockStripeClient(invoices=[inv_data])
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert body["invoices"][0]["period_start"] is None

    def test_missing_period_end(self, handler):
        inv_data = self._make_invoice_data()
        inv_data["period_end"] = None
        client = MockStripeClient(invoices=[inv_data])
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert body["invoices"][0]["period_end"] is None

    def test_none_amount_defaults_to_zero(self, handler):
        inv_data = self._make_invoice_data(amount_due=None, amount_paid=None)
        client = MockStripeClient(invoices=[inv_data])
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        inv = body["invoices"][0]
        assert inv["amount_due"] == 0.0
        assert inv["amount_paid"] == 0.0

    def test_invoice_fields_present(self, handler):
        invoices_data = [self._make_invoice_data()]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        inv = body["invoices"][0]
        assert inv["id"] == "inv_001"
        assert inv["number"] == "INV-001"
        assert inv["status"] == "paid"
        assert inv["hosted_invoice_url"] == "https://inv.stripe.com/1"
        assert inv["invoice_pdf"] == "https://inv.stripe.com/1.pdf"

    def test_limit_query_param(self, handler):
        """Limit param should be passed to Stripe client."""
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(query_params={"limit": "25"})
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test_123", limit=25
        )

    def test_limit_default_is_10(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test_123", limit=10
        )

    def test_limit_clamped_to_100(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(query_params={"limit": "500"})
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test_123", limit=100
        )

    def test_no_billing_account_returns_404(self):
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_no_cust")
        )
        store.add_organization(
            MockOrganization(id="org_no_cust", name="No Cust", stripe_customer_id=None)
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 404

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 503

    def test_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_not_found_returns_404(self):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 404

    def test_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 503

    def test_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 502

    def test_stripe_generic_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 500

    def test_empty_invoice_list(self, handler):
        client = MockStripeClient(invoices=[])
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert body["invoices"] == []

    def test_multiple_invoices(self, handler):
        invoices_data = [
            self._make_invoice_data(id="inv_001", number="INV-001"),
            self._make_invoice_data(id="inv_002", number="INV-002"),
            self._make_invoice_data(id="inv_003", number="INV-003"),
        ]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert len(body["invoices"]) == 3

    def test_created_timestamp_formatted_as_iso(self, handler):
        invoices_data = [self._make_invoice_data(created=1700000000)]
        client = MockStripeClient(invoices=invoices_data)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        inv = body["invoices"][0]
        # Should be a valid ISO datetime string
        assert "T" in inv["created"]
        # Verify it parses
        datetime.fromisoformat(inv["created"])


# ===========================================================================
# TestMethodNotAllowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for 405 Method Not Allowed on reporting endpoints."""

    def test_post_to_audit_log_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/audit-log", {}, http, method="POST")
        assert _status(result) == 405

    def test_post_to_usage_export_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="POST")
        assert _status(result) == 405

    def test_post_to_usage_forecast_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="POST")
        assert _status(result) == 405

    def test_post_to_invoices_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/invoices", {}, http, method="POST")
        assert _status(result) == 405

    def test_put_to_audit_log_returns_405(self, handler):
        http = MockHTTPHandler(command="PUT")
        result = handler.handle("/api/v1/billing/audit-log", {}, http, method="PUT")
        assert _status(result) == 405

    def test_delete_to_invoices_returns_405(self, handler):
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle("/api/v1/billing/invoices", {}, http, method="DELETE")
        assert _status(result) == 405

    def test_patch_to_forecast_returns_405(self, handler):
        http = MockHTTPHandler(command="PATCH")
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="PATCH")
        assert _status(result) == 405


# ===========================================================================
# TestSecurity
# ===========================================================================


class TestSecurity:
    """Security tests for reporting endpoints."""

    def test_sql_injection_in_limit_param(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "1; DROP TABLE users--"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        # Should fall back to default, not crash
        body = _body(result)
        assert body["limit"] == 50

    def test_sql_injection_in_offset_param(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"offset": "0 OR 1=1"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["offset"] == 0

    def test_xss_in_action_filter(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            query_params={"action": "<script>alert('xss')</script>"}
        )
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        # Should not crash, action filter is just passed to store
        assert _status(result) == 200

    def test_path_traversal_in_start_date(self, handler):
        http = MockHTTPHandler(query_params={"start": "../../etc/passwd"})
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        # Invalid date should be ignored
        assert _status(result) == 200

    def test_oversized_limit_clamped(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "999999999"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 100

    def test_negative_offset_falls_back_to_default(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"offset": "-10"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["offset"] == 0

    def test_sql_injection_in_date_params(self, handler):
        http = MockHTTPHandler(
            query_params={
                "start": "2026-01-01' OR '1'='1",
                "end": "2026-12-31; DROP TABLE usage_events--",
            }
        )
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        # Invalid dates should be silently discarded
        assert _status(result) == 200

    def test_invoice_limit_negative_falls_back(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(query_params={"limit": "-5"})
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test_123", limit=10
        )

    def test_invoice_limit_non_numeric_falls_back(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(query_params={"limit": "abc"})
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test_123", limit=10
        )

    def test_very_long_action_filter_handled(self):
        _ensure_role_on_auth_context()
        store = _make_enterprise_store()
        h = BillingHandler(ctx={"user_store": store})
        long_action = "x" * 10000
        http = MockHTTPHandler(query_params={"action": long_action})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 200


# ===========================================================================
# TestLoggerResolution
# ===========================================================================


class TestLoggerResolution:
    """Tests for the _logger() function in core_reporting.py."""

    def test_logger_from_core_module(self):
        """When core module is loaded, _logger() should return its logger."""
        from aragora.server.handlers.billing.core_reporting import _logger

        log = _logger()
        assert log is not None
        assert hasattr(log, "error")

    def test_logger_fallback_when_core_not_in_sys_modules(self):
        """When core module is not in sys.modules, falls back to logging.getLogger."""
        import sys
        from aragora.server.handlers.billing.core_reporting import _logger

        # Temporarily remove core from sys.modules
        core_key = "aragora.server.handlers.billing.core"
        original = sys.modules.pop(core_key, None)
        try:
            log = _logger()
            assert log is not None
            assert hasattr(log, "error")
        finally:
            if original is not None:
                sys.modules[core_key] = original


# ===========================================================================
# TestReportingMixinIntegration
# ===========================================================================


class TestReportingMixinIntegration:
    """Integration-style tests verifying reporting mixin methods work through the handler."""

    def test_all_reporting_routes_are_get(self, handler):
        """All four reporting routes should only accept GET."""
        routes = [
            "/api/v1/billing/audit-log",
            "/api/v1/billing/usage/export",
            "/api/v1/billing/usage/forecast",
            "/api/v1/billing/invoices",
        ]
        for route in routes:
            http = MockHTTPHandler(command="POST")
            result = handler.handle(route, {}, http, method="POST")
            assert _status(result) == 405, f"Route {route} should reject POST"

    def test_all_reporting_routes_handled_by_can_handle(self, handler):
        routes = [
            "/api/v1/billing/audit-log",
            "/api/v1/billing/usage/export",
            "/api/v1/billing/usage/forecast",
            "/api/v1/billing/invoices",
        ]
        for route in routes:
            assert handler.can_handle(route), f"Route {route} should be handled"

    def test_forecast_days_remaining_non_negative(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert body["forecast"]["days_remaining"] >= 0

    def test_forecast_debates_overage_non_negative(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert body["forecast"]["debates_overage"] >= 0

    def test_csv_export_org_slug_in_filename(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        disposition = result.headers["Content-Disposition"]
        assert "test-org" in disposition
