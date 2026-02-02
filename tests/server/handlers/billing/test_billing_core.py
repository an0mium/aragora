"""
Tests for BillingHandler (core billing operations).

Covers:
- GET /api/v1/billing/plans - List subscription plans
- GET /api/v1/billing/usage - Get usage
- GET /api/v1/billing/subscription - Get subscription
- POST /api/v1/billing/checkout - Create checkout session
- POST /api/v1/billing/portal - Create billing portal
- POST /api/v1/billing/cancel - Cancel subscription
- POST /api/v1/billing/resume - Resume subscription
- GET /api/v1/billing/audit-log - Get audit log
- GET /api/v1/billing/invoices - Get invoices
- Stripe webhook handling
- Rate limiting
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.billing.core import BillingHandler


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


class FakeTier(Enum):
    """Fake tier enum that mimics SubscriptionTier."""

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
            "api_access": self.api_access,
            "all_agents": self.all_agents,
            "custom_agents": self.custom_agents,
            "sso_enabled": self.sso_enabled,
            "audit_logs": self.audit_logs,
            "priority_support": self.priority_support,
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
    debates_used_this_month: int = 10
    debates_remaining: int = 90
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )


@dataclass
class FakeCheckoutSession:
    id: str = "cs_test123"
    url: str = "https://checkout.stripe.com/test"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "url": self.url}


@dataclass
class FakePortalSession:
    id: str = "bps_test123"
    url: str = "https://billing.stripe.com/portal"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "url": self.url}


@dataclass
class FakeSubscription:
    id: str = "sub_test123"
    status: str = "active"
    current_period_end: datetime = field(
        default_factory=lambda: datetime(2025, 2, 1, tzinfo=timezone.utc)
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


class FakeHandler:
    """Mock HTTP handler for testing."""

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
        import io

        return io.BytesIO(self._body)

    def get(self, key: str, default: Any = None) -> Any:
        """Support get_string_param() calls on the handler."""
        return self._query_params.get(key, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id = MagicMock(return_value=FakeDbUser())
    store.get_organization_by_id = MagicMock(return_value=FakeOrganization())
    store.get_organization_by_subscription = MagicMock(return_value=FakeOrganization())
    store.get_organization_by_stripe_customer = MagicMock(return_value=FakeOrganization())
    store.update_organization = MagicMock()
    store.reset_org_usage = MagicMock()
    store.get_audit_log = MagicMock(return_value=[])
    store.get_audit_log_count = MagicMock(return_value=0)
    store.log_audit_event = MagicMock()
    return store


@pytest.fixture
def mock_stripe_client():
    """Create a mock Stripe client."""
    client = MagicMock()
    client.create_checkout_session = MagicMock(return_value=FakeCheckoutSession())
    client.create_portal_session = MagicMock(return_value=FakePortalSession())
    client.get_subscription = MagicMock(return_value=FakeSubscription())
    client.cancel_subscription = MagicMock(return_value=FakeSubscription(cancel_at_period_end=True))
    client.resume_subscription = MagicMock(return_value=FakeSubscription())
    client.list_invoices = MagicMock(
        return_value=[
            {
                "id": "inv_test123",
                "number": "INV-0001",
                "status": "paid",
                "amount_due": 2900,
                "amount_paid": 2900,
                "currency": "usd",
                "created": 1704067200,
                "period_start": 1704067200,
                "period_end": 1706745600,
                "hosted_invoice_url": "https://invoice.stripe.com/test",
                "invoice_pdf": "https://files.stripe.com/test.pdf",
            }
        ]
    )
    return client


@pytest.fixture
def billing_handler(mock_user_store):
    """Create a BillingHandler with mocked dependencies."""
    handler = BillingHandler(ctx={"user_store": mock_user_store})
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    from aragora.server.handlers.billing.core import _billing_limiter

    _billing_limiter._requests.clear()
    yield
    _billing_limiter._requests.clear()


# ---------------------------------------------------------------------------
# Test can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_billing_routes(self, billing_handler):
        """Handler accepts billing routes."""
        assert billing_handler.can_handle("/api/v1/billing/plans") is True
        assert billing_handler.can_handle("/api/v1/billing/usage") is True
        assert billing_handler.can_handle("/api/v1/billing/subscription") is True
        assert billing_handler.can_handle("/api/v1/billing/checkout") is True
        assert billing_handler.can_handle("/api/v1/billing/portal") is True
        assert billing_handler.can_handle("/api/v1/billing/cancel") is True
        assert billing_handler.can_handle("/api/v1/billing/resume") is True
        assert billing_handler.can_handle("/api/v1/webhooks/stripe") is True

    def test_rejects_unknown_routes(self, billing_handler):
        """Handler rejects non-billing routes."""
        assert billing_handler.can_handle("/api/v1/debates") is False
        assert billing_handler.can_handle("/api/v1/users") is False
        assert billing_handler.can_handle("/api/v1/billing/unknown") is False


# ---------------------------------------------------------------------------
# Test _get_plans
# ---------------------------------------------------------------------------


class TestGetPlans:
    def test_returns_all_plans(self, billing_handler):
        """Plans endpoint returns all subscription tiers."""
        result = billing_handler._get_plans()
        assert result.status_code == 200

        data = json.loads(result.body)
        assert "plans" in data
        assert len(data["plans"]) >= 3  # At least free, starter, professional

    def test_plan_structure(self, billing_handler):
        """Each plan has required fields."""
        result = billing_handler._get_plans()
        data = json.loads(result.body)

        for plan in data["plans"]:
            assert "id" in plan
            assert "name" in plan
            assert "price_monthly_cents" in plan
            assert "features" in plan
            assert "debates_per_month" in plan["features"]


# ---------------------------------------------------------------------------
# Test _get_usage
# ---------------------------------------------------------------------------


class TestGetUsage:
    def test_returns_usage_data(self, billing_handler, mock_user_store):
        """Usage endpoint returns usage data for authenticated user."""
        handler = FakeHandler()
        user = FakeUser()

        # Access the underlying function
        fn = billing_handler._get_usage.__wrapped__.__wrapped__
        result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "usage" in data
        assert "debates_used" in data["usage"]
        assert "debates_limit" in data["usage"]

    def test_returns_503_without_user_store(self, mock_user_store):
        """Returns 503 when user store unavailable."""
        handler_obj = BillingHandler(ctx={})  # No user_store
        handler = FakeHandler()
        user = FakeUser()

        fn = handler_obj._get_usage.__wrapped__.__wrapped__
        result = fn(handler_obj, handler, user=user)

        assert result.status_code == 503

    def test_returns_404_for_unknown_user(self, billing_handler, mock_user_store):
        """Returns 404 when user not found."""
        mock_user_store.get_user_by_id.return_value = None
        handler = FakeHandler()
        user = FakeUser()

        fn = billing_handler._get_usage.__wrapped__.__wrapped__
        result = fn(billing_handler, handler, user=user)

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _get_subscription
# ---------------------------------------------------------------------------


class TestGetSubscription:
    def test_returns_subscription_data(self, billing_handler, mock_user_store, mock_stripe_client):
        """Subscription endpoint returns subscription data."""
        handler = FakeHandler()
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            # 2 decorators: @handle_errors, @require_permission
            fn = billing_handler._get_subscription.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "subscription" in data
        assert "tier" in data["subscription"]
        assert "status" in data["subscription"]

    def test_handles_stripe_errors_gracefully(self, billing_handler, mock_user_store):
        """Subscription endpoint degrades gracefully on Stripe errors."""
        # Import from the module where it's caught
        from aragora.billing.stripe_client import StripeError

        handler = FakeHandler()
        user = FakeUser()

        failing_client = MagicMock()
        failing_client.get_subscription.side_effect = StripeError("API error")

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=failing_client,
        ):
            fn = billing_handler._get_subscription.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        # Should still return 200 with partial data (Stripe errors are caught and logged)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "subscription" in data


# ---------------------------------------------------------------------------
# Test _create_checkout
# ---------------------------------------------------------------------------


class TestCreateCheckout:
    def test_creates_checkout_session(self, billing_handler, mock_user_store, mock_stripe_client):
        """Checkout endpoint creates Stripe session."""
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "starter",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel",
        }

        with (
            patch.object(billing_handler, "read_json_body", return_value=body),
            patch(
                "aragora.server.handlers.billing.core.get_stripe_client",
                return_value=mock_stripe_client,
            ),
        ):
            fn = billing_handler._create_checkout.__wrapped__.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "checkout" in data
        assert "id" in data["checkout"]

    def test_rejects_invalid_tier(self, billing_handler, mock_user_store):
        """Checkout rejects invalid tier."""
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "invalid_tier",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel",
        }

        with patch.object(billing_handler, "read_json_body", return_value=body):
            fn = billing_handler._create_checkout.__wrapped__.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 400

    def test_rejects_free_tier_checkout(self, billing_handler, mock_user_store):
        """Checkout rejects free tier."""
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "free",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel",
        }

        with patch.object(billing_handler, "read_json_body", return_value=body):
            fn = billing_handler._create_checkout.__wrapped__.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 400


# ---------------------------------------------------------------------------
# Test _create_portal
# ---------------------------------------------------------------------------


class TestCreatePortal:
    def test_creates_portal_session(self, billing_handler, mock_user_store, mock_stripe_client):
        """Portal endpoint creates Stripe portal session."""
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {"return_url": "https://example.com/billing"}

        with (
            patch.object(billing_handler, "read_json_body", return_value=body),
            patch(
                "aragora.server.handlers.billing.core.get_stripe_client",
                return_value=mock_stripe_client,
            ),
        ):
            fn = billing_handler._create_portal.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "portal" in data
        assert "url" in data["portal"]

    def test_requires_return_url(self, billing_handler, mock_user_store):
        """Portal endpoint requires return URL."""
        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch.object(billing_handler, "read_json_body", return_value={}):
            fn = billing_handler._create_portal.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 400

    def test_requires_billing_account(self, billing_handler, mock_user_store):
        """Portal endpoint requires existing Stripe customer."""
        mock_user_store.get_organization_by_id.return_value = FakeOrganization(
            stripe_customer_id=None
        )
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {"return_url": "https://example.com/billing"}

        with patch.object(billing_handler, "read_json_body", return_value=body):
            fn = billing_handler._create_portal.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _cancel_subscription
# ---------------------------------------------------------------------------


class TestCancelSubscription:
    def test_cancels_subscription(self, billing_handler, mock_user_store, mock_stripe_client):
        """Cancel endpoint cancels subscription at period end."""
        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            fn = billing_handler._cancel_subscription.__wrapped__.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "message" in data
        assert "subscription" in data
        mock_stripe_client.cancel_subscription.assert_called_once()

    def test_requires_active_subscription(self, billing_handler, mock_user_store):
        """Cancel endpoint requires active subscription."""
        mock_user_store.get_organization_by_id.return_value = FakeOrganization(
            stripe_subscription_id=None
        )
        handler = FakeHandler(method="POST")
        user = FakeUser()

        fn = billing_handler._cancel_subscription.__wrapped__.__wrapped__.__wrapped__
        result = fn(billing_handler, handler, user=user)

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _resume_subscription
# ---------------------------------------------------------------------------


class TestResumeSubscription:
    def test_resumes_subscription(self, billing_handler, mock_user_store, mock_stripe_client):
        """Resume endpoint resumes canceled subscription."""
        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            fn = billing_handler._resume_subscription.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "message" in data
        mock_stripe_client.resume_subscription.assert_called_once()


# ---------------------------------------------------------------------------
# Test _get_invoices
# ---------------------------------------------------------------------------


class TestGetInvoices:
    def test_returns_invoices(self, billing_handler, mock_user_store, mock_stripe_client):
        """Invoices endpoint returns invoice history."""
        handler = FakeHandler()
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            fn = billing_handler._get_invoices.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "invoices" in data
        assert len(data["invoices"]) == 1
        assert data["invoices"][0]["id"] == "inv_test123"


# ---------------------------------------------------------------------------
# Test _get_audit_log
# ---------------------------------------------------------------------------


class TestGetAuditLog:
    def test_returns_audit_log(self, billing_handler, mock_user_store):
        """Audit log endpoint returns billing audit entries."""
        # Enable audit logs for org
        org = FakeOrganization()
        org.limits = FakeTierLimits(audit_logs=True)
        mock_user_store.get_organization_by_id.return_value = org

        handler = FakeHandler()
        user = FakeUser(role="owner")

        fn = billing_handler._get_audit_log.__wrapped__.__wrapped__
        result = fn(billing_handler, handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "entries" in data
        assert "total" in data

    def test_requires_enterprise_tier(self, billing_handler, mock_user_store):
        """Audit log requires enterprise tier with audit_logs enabled."""
        # Default org has audit_logs=False
        handler = FakeHandler()
        user = FakeUser(role="owner")

        fn = billing_handler._get_audit_log.__wrapped__.__wrapped__
        result = fn(billing_handler, handler, user=user)

        assert result.status_code == 403


# ---------------------------------------------------------------------------
# Test Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limits_billing_endpoints(self, billing_handler, mock_user_store):
        """Billing endpoints are rate limited."""
        from aragora.server.handlers.billing.core import _billing_limiter

        handler = FakeHandler()

        # Disable test name suffix so we use consistent key "127.0.0.1"
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}, clear=False):
            # Exhaust rate limit
            for _ in range(25):
                _billing_limiter.is_allowed("127.0.0.1")

            # Next request should be rejected
            result = billing_handler.handle("/api/v1/billing/plans", {}, handler, method="GET")

        assert result.status_code == 429

    def test_webhooks_bypass_rate_limit(self, billing_handler, mock_user_store):
        """Stripe webhooks are not rate limited."""
        from aragora.server.handlers.billing.core import _billing_limiter

        # Disable test name suffix
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}, clear=False):
            # Exhaust rate limit
            for _ in range(25):
                _billing_limiter.is_allowed("127.0.0.1")

            # Webhook handler should be accessed (though it will fail for other reasons)
            handler = FakeHandler(method="POST", headers={})
            result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, method="POST")

        # Should not be 429 (rate limited)
        assert result.status_code != 429


# ---------------------------------------------------------------------------
# Test Webhook Handling
# ---------------------------------------------------------------------------


class TestStripeWebhooks:
    def test_rejects_missing_signature(self, billing_handler):
        """Webhook rejects requests without signature."""
        handler = FakeHandler(method="POST", headers={})

        with patch.object(billing_handler, "validate_content_length", return_value=10):
            fn = billing_handler._handle_stripe_webhook.__wrapped__
            result = fn(billing_handler, handler)

        assert result.status_code == 400

    def test_rejects_invalid_signature(self, billing_handler):
        """Webhook rejects invalid signature."""
        handler = FakeHandler(
            method="POST",
            headers={"Stripe-Signature": "invalid"},
            body={"type": "test"},
        )

        with (
            patch.object(billing_handler, "validate_content_length", return_value=10),
            patch(
                "aragora.billing.stripe_client.parse_webhook_event",
                return_value=None,
            ),
        ):
            fn = billing_handler._handle_stripe_webhook.__wrapped__
            result = fn(billing_handler, handler)

        assert result.status_code == 400

    def test_handles_checkout_completed(self, billing_handler, mock_user_store):
        """Webhook handles checkout.session.completed event."""
        event = MagicMock()
        event.type = "checkout.session.completed"
        event.event_id = "evt_test123"
        event.object = {"customer": "cus_test", "subscription": "sub_test"}
        event.metadata = {
            "user_id": "user-123",
            "org_id": "org-123",
            "tier": "starter",
        }

        result = billing_handler._handle_checkout_completed(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.update_organization.assert_called()

    def test_handles_subscription_deleted(self, billing_handler, mock_user_store):
        """Webhook handles subscription deletion by downgrading to free."""
        event = MagicMock()
        event.type = "customer.subscription.deleted"
        event.event_id = "evt_test123"
        event.object = {"id": "sub_test123"}
        event.subscription_id = "sub_test123"
        event.metadata = {}

        result = billing_handler._handle_subscription_deleted(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.update_organization.assert_called()

    def test_handles_invoice_paid(self, billing_handler, mock_user_store):
        """Webhook handles invoice payment by resetting usage."""
        event = MagicMock()
        event.type = "invoice.payment_succeeded"
        event.event_id = "evt_test123"
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
            "amount_paid": 2900,
        }
        event.metadata = {}

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery:
            mock_recovery.return_value.mark_recovered.return_value = True
            result = billing_handler._handle_invoice_paid(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.reset_org_usage.assert_called()


# ---------------------------------------------------------------------------
# Test Handler Routing
# ---------------------------------------------------------------------------


class TestHandlerRouting:
    def test_routes_to_correct_method(self, billing_handler, mock_user_store):
        """Handler routes requests to correct methods."""
        handler = FakeHandler(method="GET")

        # Plans endpoint doesn't need auth
        result = billing_handler.handle("/api/v1/billing/plans", {}, handler, method="GET")
        assert result.status_code == 200

    def test_rejects_invalid_method(self, billing_handler, mock_user_store):
        """Handler rejects unsupported methods."""
        handler = FakeHandler(method="DELETE")

        result = billing_handler.handle("/api/v1/billing/plans", {}, handler, method="DELETE")
        assert result.status_code == 405
