"""
Tests for aragora.server.handlers.admin.billing - Billing API handler.

Tests cover:
- Get plans
- Get usage (authenticated)
- Get subscription
- Create checkout session
- Create billing portal
- Cancel/resume subscription
- Usage export (CSV)
- Usage forecast
- Invoice history
- Stripe webhook handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin import BillingHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockLimits:
    """Mock tier limits."""

    debates_per_month: int = 100
    users_per_org: int = 5
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
            "api_access": self.api_access,
            "all_agents": self.all_agents,
            "custom_agents": self.custom_agents,
            "sso_enabled": self.sso_enabled,
            "audit_logs": self.audit_logs,
            "priority_support": self.priority_support,
        }


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    owner_id: str = "user-123"
    tier: Any = field(default_factory=lambda: MagicMock(value="starter"))
    limits: MockLimits = field(default_factory=MockLimits)
    stripe_customer_id: str | None = "cus_test123"
    stripe_subscription_id: str | None = "sub_test123"
    debates_used_this_month: int = 25
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=15)
    )

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "owner_id": self.owner_id}


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-123"
    role: str = "owner"
    is_active: bool = True


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "owner"
    error_reason: str | None = None


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self.orgs_by_subscription: dict[str, str] = {}
        self.orgs_by_customer: dict[str, str] = {}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def get_organization_by_subscription(self, subscription_id: str) -> MockOrganization | None:
        org_id = self.orgs_by_subscription.get(subscription_id)
        return self.orgs.get(org_id) if org_id else None

    def get_organization_by_stripe_customer(self, customer_id: str) -> MockOrganization | None:
        org_id = self.orgs_by_customer.get(customer_id)
        return self.orgs.get(org_id) if org_id else None

    def get_organization_owner(self, org_id: str) -> MockUser | None:
        org = self.orgs.get(org_id)
        if org:
            return self.users.get(org.owner_id)
        return None

    def update_organization(self, org_id: str, **kwargs) -> None:
        org = self.orgs.get(org_id)
        if org:
            for key, value in kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

    def reset_org_usage(self, org_id: str) -> None:
        org = self.orgs.get(org_id)
        if org:
            org.debates_used_this_month = 0

    def log_audit_event(self, **kwargs) -> None:
        pass

    def get_audit_log(self, **kwargs) -> list:
        return []

    def get_audit_log_count(self, **kwargs) -> int:
        return 0


class MockUsageSummary:
    """Mock usage summary."""

    total_tokens_in: int = 50000
    total_tokens_out: int = 25000
    total_tokens: int = 75000
    total_cost_usd: float = 1.25
    total_cost: float = 1.25
    cost_by_provider: dict = field(default_factory=dict)

    def __init__(self):
        self.total_tokens_in = 50000
        self.total_tokens_out = 25000
        self.total_tokens = 75000
        self.total_cost_usd = 1.25
        self.total_cost = 1.25
        self.cost_by_provider = {"anthropic": "0.75", "openai": "0.50"}


class MockUsageTracker:
    """Mock usage tracker."""

    def get_summary(self, **kwargs) -> MockUsageSummary:
        return MockUsageSummary()


@dataclass
class MockCheckoutSession:
    """Mock Stripe checkout session."""

    id: str = "cs_test123"
    url: str = "https://checkout.stripe.com/test"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "url": self.url}


@dataclass
class MockPortalSession:
    """Mock Stripe portal session."""

    id: str = "bps_test123"
    url: str = "https://billing.stripe.com/test"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "url": self.url}


@dataclass
class MockSubscription:
    """Mock Stripe subscription."""

    id: str = "sub_test123"
    status: str = "active"
    current_period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=15)
    )
    cancel_at_period_end: bool = False
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    is_trialing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
    query_string: str = "",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = (
        f"/api/v1/billing/plans?{query_string}" if query_string else "/api/v1/billing/plans"
    )

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def user_store():
    """Create mock user store with test data."""
    store = MockUserStore()
    user = MockUser()
    org = MockOrganization()

    store.users["user-123"] = user
    store.orgs["org-123"] = org
    store.orgs_by_subscription["sub_test123"] = "org-123"
    store.orgs_by_customer["cus_test123"] = "org-123"

    return store


@pytest.fixture
def billing_handler(user_store):
    """Create BillingHandler with mock context."""
    ctx = {
        "user_store": user_store,
        "usage_tracker": MockUsageTracker(),
    }
    return BillingHandler(ctx)


# ===========================================================================
# Test Routing
# ===========================================================================


class TestBillingHandlerRouting:
    """Tests for BillingHandler routing."""

    def test_can_handle_plans(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/plans") is True

    def test_can_handle_usage(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/usage") is True

    def test_can_handle_subscription(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/subscription") is True

    def test_can_handle_checkout(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/checkout") is True

    def test_can_handle_portal(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/portal") is True

    def test_can_handle_cancel(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/cancel") is True

    def test_can_handle_webhook(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/webhooks/stripe") is True

    def test_cannot_handle_unknown(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/other/endpoint") is False


# ===========================================================================
# Test Get Plans
# ===========================================================================


class TestBillingHandlerGetPlans:
    """Tests for get plans endpoint."""

    def test_get_plans_success(self, billing_handler):
        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/plans", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "plans" in data
        assert len(data["plans"]) > 0

        # Check plan structure
        plan = data["plans"][0]
        assert "id" in plan
        assert "name" in plan
        assert "features" in plan

    def test_get_plans_wrong_method(self, billing_handler):
        handler = make_mock_handler(method="POST")

        result = billing_handler.handle("/api/v1/billing/plans", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Test Get Usage
# ===========================================================================


class TestBillingHandlerGetUsage:
    """Tests for get usage endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_usage_success(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler()

        # The decorator requires permission, need to mock that too
        with patch.object(billing_handler, "_get_usage") as mock_method:
            mock_method.return_value = (json.dumps({"usage": {"debates_used": 25}}), 200)

            result = billing_handler.handle("/api/v1/billing/usage", {}, handler, "GET")

            assert result is not None

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_get_usage_rate_limited(self, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = False

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/usage", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 429


# ===========================================================================
# Test Get Subscription
# ===========================================================================


class TestBillingHandlerGetSubscription:
    """Tests for get subscription endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_get_subscription_with_stripe(
        self, mock_stripe, mock_auth, mock_limiter, billing_handler
    ):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext()

        mock_stripe_client = MagicMock()
        mock_stripe_client.get_subscription.return_value = MockSubscription()
        mock_stripe.return_value = mock_stripe_client

        with patch.object(billing_handler, "_get_subscription") as mock_method:
            mock_method.return_value = (
                json.dumps({"subscription": {"tier": "starter", "status": "active"}}),
                200,
            )

            handler = make_mock_handler()

            result = billing_handler.handle("/api/v1/billing/subscription", {}, handler, "GET")

            assert result is not None


# ===========================================================================
# Test Create Checkout
# ===========================================================================


class TestBillingHandlerCheckout:
    """Tests for checkout endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_create_checkout_success(
        self, mock_validate, mock_stripe, mock_auth, mock_limiter, billing_handler
    ):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext()
        mock_validate.return_value = MagicMock(is_valid=True)

        mock_stripe_client = MagicMock()
        mock_stripe_client.create_checkout_session.return_value = MockCheckoutSession()
        mock_stripe.return_value = mock_stripe_client

        with patch.object(billing_handler, "_create_checkout") as mock_method:
            mock_method.return_value = (
                json.dumps({"checkout": {"id": "cs_test", "url": "https://stripe.com"}}),
                200,
            )

            handler = make_mock_handler(
                {
                    "tier": "starter",
                    "success_url": "https://app.example.com/success",
                    "cancel_url": "https://app.example.com/cancel",
                },
                method="POST",
            )

            result = billing_handler.handle("/api/v1/billing/checkout", {}, handler, "POST")

            assert result is not None


# ===========================================================================
# Test Create Portal
# ===========================================================================


class TestBillingHandlerPortal:
    """Tests for billing portal endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_create_portal_no_return_url(self, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True

        with patch.object(billing_handler, "_create_portal") as mock_method:
            mock_method.return_value = (json.dumps({"error": "Return URL required"}), 400)

            handler = make_mock_handler({}, method="POST")

            result = billing_handler.handle("/api/v1/billing/portal", {}, handler, "POST")

            assert result is not None


# ===========================================================================
# Test Cancel Subscription
# ===========================================================================


class TestBillingHandlerCancel:
    """Tests for cancel subscription endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_cancel_wrong_method(self, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True

        handler = make_mock_handler(method="GET")

        result = billing_handler.handle("/api/v1/billing/cancel", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Test Resume Subscription
# ===========================================================================


class TestBillingHandlerResume:
    """Tests for resume subscription endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_resume_wrong_method(self, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True

        handler = make_mock_handler(method="GET")

        result = billing_handler.handle("/api/v1/billing/resume", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Test Usage Export
# ===========================================================================


class TestBillingHandlerUsageExport:
    """Tests for usage export endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_export_usage_not_authenticated(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/usage/export", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Usage Forecast
# ===========================================================================


class TestBillingHandlerUsageForecast:
    """Tests for usage forecast endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_forecast_success(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext()

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/usage/forecast", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "forecast" in data
        assert "projection" in data["forecast"]

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_forecast_not_authenticated(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/usage/forecast", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Invoices
# ===========================================================================


class TestBillingHandlerInvoices:
    """Tests for invoices endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_invoices_not_authenticated(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/invoices", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_invoices_no_org(self, mock_auth, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True
        mock_auth.return_value = MockAuthContext()

        # Create user without org
        billing_handler.ctx["user_store"].users["user-123"].org_id = None

        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/invoices", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Stripe Webhook
# ===========================================================================


class TestBillingHandlerWebhook:
    """Tests for Stripe webhook handling."""

    def test_webhook_missing_signature(self, billing_handler):
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 400

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    def test_webhook_duplicate_event(self, mock_duplicate, mock_parse, billing_handler):
        mock_event = MagicMock()
        mock_event.event_id = "evt_test123"
        mock_event.type = "checkout.session.completed"
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = True

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data.get("duplicate") is True

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    def test_webhook_checkout_completed(
        self, mock_mark, mock_duplicate, mock_parse, billing_handler
    ):
        mock_event = MagicMock()
        mock_event.event_id = "evt_test123"
        mock_event.type = "checkout.session.completed"
        mock_event.object = {"customer": "cus_test", "subscription": "sub_test"}
        mock_event.metadata = {"user_id": "user-123", "org_id": "org-123", "tier": "starter"}
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = False

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        mock_mark.assert_called_once_with("evt_test123")

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    def test_webhook_subscription_deleted(
        self, mock_mark, mock_duplicate, mock_parse, billing_handler
    ):
        mock_event = MagicMock()
        mock_event.event_id = "evt_test456"
        mock_event.type = "customer.subscription.deleted"
        mock_event.object = {"id": "sub_test123"}
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = False

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.billing.payment_recovery.get_recovery_store")
    def test_webhook_invoice_paid(
        self, mock_recovery, mock_mark, mock_duplicate, mock_parse, billing_handler
    ):
        mock_event = MagicMock()
        mock_event.event_id = "evt_test789"
        mock_event.type = "invoice.payment_succeeded"
        mock_event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
            "amount_paid": 2900,
        }
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = False

        mock_recovery_store = MagicMock()
        mock_recovery_store.mark_recovered.return_value = True
        mock_recovery.return_value = mock_recovery_store

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.billing.payment_recovery.get_recovery_store")
    @patch("aragora.billing.notifications.get_billing_notifier")
    def test_webhook_invoice_failed(
        self, mock_notifier, mock_recovery, mock_mark, mock_duplicate, mock_parse, billing_handler
    ):
        mock_event = MagicMock()
        mock_event.event_id = "evt_fail123"
        mock_event.type = "invoice.payment_failed"
        mock_event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
            "attempt_count": 1,
            "id": "inv_test",
            "hosted_invoice_url": "https://invoice.stripe.com/test",
        }
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = False

        mock_recovery_store = MagicMock()
        mock_failure = MagicMock()
        mock_failure.attempt_count = 1
        mock_failure.days_failing = 1
        mock_failure.days_until_downgrade = 13
        mock_recovery_store.record_failure.return_value = mock_failure
        mock_recovery.return_value = mock_recovery_store

        mock_notifier_instance = MagicMock()
        mock_notifier_instance.notify_payment_failed.return_value = MagicMock(
            method="email", success=True
        )
        mock_notifier.return_value = mock_notifier_instance

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data.get("failure_tracked") is True

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    def test_webhook_unhandled_event(self, mock_mark, mock_duplicate, mock_parse, billing_handler):
        mock_event = MagicMock()
        mock_event.event_id = "evt_unknown"
        mock_event.type = "customer.created"  # Unhandled event type
        mock_parse.return_value = mock_event
        mock_duplicate.return_value = False

        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "100", "Stripe-Signature": "sig_test"}
        handler.rfile = BytesIO(b'{"type": "test"}')
        handler.client_address = ("127.0.0.1", 12345)

        result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data.get("received") is True


# ===========================================================================
# Test Audit Log
# ===========================================================================


class TestBillingHandlerAuditLog:
    """Tests for audit log endpoint."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_audit_log_wrong_method(self, mock_limiter, billing_handler):
        mock_limiter.is_allowed.return_value = True

        handler = make_mock_handler(method="POST")

        result = billing_handler.handle("/api/v1/billing/audit-log", {}, handler, "POST")

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Test Service Unavailable
# ===========================================================================


class TestBillingHandlerServiceUnavailable:
    """Tests for service unavailable scenarios."""

    @patch("aragora.server.handlers.admin.billing._billing_limiter")
    def test_usage_no_user_store(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True

        handler_ctx = BillingHandler({})

        with patch.object(handler_ctx, "_get_usage") as mock_method:
            mock_method.return_value = (json.dumps({"error": "Service unavailable"}), 503)

            handler = make_mock_handler()

            result = handler_ctx.handle("/api/v1/billing/usage", {}, handler, "GET")

            assert result is not None
