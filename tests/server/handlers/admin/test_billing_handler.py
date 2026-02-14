"""
Tests for aragora.server.handlers.admin.billing - Billing API Handler.

Tests cover:
- Route registration and can_handle
- GET /api/v1/billing/plans - List subscription plans
- GET /api/v1/billing/usage - Get current usage
- GET /api/v1/billing/subscription - Get current subscription
- POST /api/v1/billing/checkout - Create checkout session
- POST /api/v1/billing/portal - Create billing portal session
- POST /api/v1/billing/cancel - Cancel subscription
- POST /api/v1/billing/resume - Resume subscription
- GET /api/v1/billing/audit-log - Get billing audit log
- GET /api/v1/billing/usage/export - Export usage CSV
- GET /api/v1/billing/usage/forecast - Get usage forecast
- GET /api/v1/billing/invoices - Get invoices
- POST /api/v1/webhooks/stripe - Handle Stripe webhooks
- Rate limiting
- RBAC permission checks
- Error handling
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _import_billing_module():
    """Import billing module, working around broken sibling imports."""
    try:
        import aragora.server.handlers.admin.billing as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Clear partially loaded modules and stub broken imports
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.admin.billing as mod

    return mod


billing_module = _import_billing_module()
BillingHandler = billing_module.BillingHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockUserContext:
    """Mock user authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "owner"


@dataclass
class MockLimits:
    """Mock subscription limits."""

    debates_per_month: int = 100
    users_per_org: int = 10
    api_access: bool = True
    all_agents: bool = True
    custom_agents: bool = False
    sso_enabled: bool = False
    audit_logs: bool = True
    priority_support: bool = False

    def to_dict(self) -> dict:
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
class MockTier:
    """Mock subscription tier."""

    value: str = "starter"


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: MockTier = field(default_factory=MockTier)
    limits: MockLimits = field(default_factory=MockLimits)
    debates_used_this_month: int = 25
    debates_remaining: int = 75
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=15)
    )
    stripe_customer_id: str | None = "cus_test123"
    stripe_subscription_id: str | None = "sub_test123"


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"


@dataclass
class MockStripeSubscription:
    """Mock Stripe subscription."""

    status: str = "active"
    current_period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=15)
    )
    cancel_at_period_end: bool = False
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    is_trialing: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }


@dataclass
class MockCheckoutSession:
    """Mock Stripe checkout session."""

    id: str = "cs_test123"
    url: str = "https://checkout.stripe.com/test"

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


@dataclass
class MockPortalSession:
    """Mock Stripe billing portal session."""

    id: str = "bps_test123"
    url: str = "https://billing.stripe.com/test"

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self._transaction = MagicMock

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def update_organization(self, org_id: str, **kwargs) -> None:
        pass

    def get_audit_log(self, **kwargs) -> list:
        return []

    def get_audit_log_count(self, **kwargs) -> int:
        return 0

    def log_audit_event(self, **kwargs) -> None:
        pass


class MockUsageTracker:
    """Mock usage tracker."""

    def get_summary(self, **kwargs):
        mock_summary = MagicMock()
        mock_summary.total_tokens_in = 10000
        mock_summary.total_tokens_out = 5000
        mock_summary.total_tokens = 15000
        mock_summary.total_cost_usd = 0.50
        mock_summary.total_cost = 0.50
        mock_summary.cost_by_provider = {"anthropic": "0.30", "openai": "0.20"}
        return mock_summary


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/billing/plans",
    query_params: dict | None = None,
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    # Support dict-like .get() for query params (used by get_string_param)
    _query_params = query_params or {}
    handler.get = lambda key, default=None: _query_params.get(key, default)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def billing_handler():
    """Create BillingHandler with mock context."""
    user_store = MockUserStore()
    user_store.users["user-123"] = MockUser()
    # Add user for conftest's mock_auth_for_handler_tests (user_id="test-user-001")
    user_store.users["test-user-001"] = MockUser(id="test-user-001", org_id="org-123")
    # Also add test_user for @require_permission decorator's auto-generated test user
    user_store.users["test_user"] = MockUser(id="test_user", org_id="org-123")
    user_store.orgs["org-123"] = MockOrganization()

    ctx = {
        "user_store": user_store,
        "usage_tracker": MockUsageTracker(),
    }
    handler = BillingHandler(ctx)
    return handler


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters before each test.

    The module-level ``_billing_limiter`` in billing.py is a standalone
    RateLimiter instance that is NOT registered in the global ``_limiters``
    dict, so ``clear_all_limiters()`` does not reach it.  We must clear it
    explicitly to prevent rate-limit exhaustion from prior tests.
    """
    from aragora.server.handlers.utils.rate_limit import _limiters

    # Clear the billing handler's own module-level limiter
    billing_module._billing_limiter.clear()

    for limiter in _limiters.values():
        limiter.clear()
    yield
    billing_module._billing_limiter.clear()
    for limiter in _limiters.values():
        limiter.clear()


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestBillingHandlerRouting:
    """Tests for BillingHandler.can_handle."""

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

    def test_can_handle_resume(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/resume") is True

    def test_can_handle_audit_log(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/audit-log") is True

    def test_can_handle_usage_export(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/usage/export") is True

    def test_can_handle_usage_forecast(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/usage/forecast") is True

    def test_can_handle_invoices(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/billing/invoices") is True

    def test_can_handle_stripe_webhook(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/webhooks/stripe") is True

    def test_cannot_handle_other_paths(self, billing_handler):
        assert billing_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Test Get Plans (GET /api/v1/billing/plans)
# ===========================================================================


class TestBillingGetPlans:
    """Tests for GET /api/v1/billing/plans endpoint."""

    def test_get_plans_success(self, billing_handler):
        """Happy path: list subscription plans."""
        handler = make_mock_handler()
        result = billing_handler.handle("/api/v1/billing/plans", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "plans" in data
        assert len(data["plans"]) > 0
        # Check plan structure
        plan = data["plans"][0]
        assert "id" in plan
        assert "name" in plan
        assert "price_monthly" in plan
        assert "features" in plan


# ===========================================================================
# Test Get Usage (GET /api/v1/billing/usage)
# ===========================================================================


class TestBillingGetUsage:
    """Tests for GET /api/v1/billing/usage endpoint."""

    def test_get_usage_success(self, billing_handler):
        """Happy path: get usage with auth."""
        handler = make_mock_handler()

        result = billing_handler.handle("/api/v1/billing/usage", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "usage" in data
        assert "debates_used" in data["usage"]
        assert "debates_limit" in data["usage"]

    def test_get_usage_no_user_store(self, billing_handler):
        """Service unavailable when no user store."""
        billing_handler.ctx["user_store"] = None
        handler = make_mock_handler()

        result = billing_handler._get_usage(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Test Get Subscription (GET /api/v1/billing/subscription)
# ===========================================================================


class TestBillingGetSubscription:
    """Tests for GET /api/v1/billing/subscription endpoint."""

    def test_get_subscription_success(self, billing_handler):
        """Happy path: get subscription with auth."""
        handler = make_mock_handler()

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.get_subscription.return_value = MockStripeSubscription()
            mock_stripe.return_value = mock_client

            result = billing_handler._get_subscription(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "subscription" in data
        assert data["subscription"]["tier"] == "starter"

    def test_get_subscription_no_stripe(self, billing_handler):
        """Subscription without Stripe details still works."""
        # Remove Stripe IDs
        org = billing_handler._get_user_store().orgs["org-123"]
        org.stripe_subscription_id = None

        handler = make_mock_handler()
        result = billing_handler._get_subscription(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["subscription"]["is_active"] is True


# ===========================================================================
# Test Create Checkout (POST /api/v1/billing/checkout)
# ===========================================================================


class TestBillingCreateCheckout:
    """Tests for POST /api/v1/billing/checkout endpoint."""

    def test_create_checkout_success(self, billing_handler):
        """Happy path: create checkout session."""
        handler = make_mock_handler(
            body={
                "tier": "professional",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            method="POST",
        )

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.create_checkout_session.return_value = MockCheckoutSession()
            mock_stripe.return_value = mock_client

            result = billing_handler._create_checkout(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "checkout" in data

    def test_create_checkout_invalid_tier(self, billing_handler):
        """Invalid tier returns 400."""
        handler = make_mock_handler(
            body={
                "tier": "invalid_tier",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            method="POST",
        )

        result = billing_handler._create_checkout(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 400

    def test_create_checkout_free_tier(self, billing_handler):
        """Cannot checkout free tier."""
        handler = make_mock_handler(
            body={
                "tier": "free",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            method="POST",
        )

        result = billing_handler._create_checkout(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Create Portal (POST /api/v1/billing/portal)
# ===========================================================================


class TestBillingCreatePortal:
    """Tests for POST /api/v1/billing/portal endpoint."""

    def test_create_portal_success(self, billing_handler):
        """Happy path: create billing portal session."""
        handler = make_mock_handler(
            body={"return_url": "https://example.com/dashboard"},
            method="POST",
        )

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.create_portal_session.return_value = MockPortalSession()
            mock_stripe.return_value = mock_client

            result = billing_handler._create_portal(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "portal" in data

    def test_create_portal_missing_return_url(self, billing_handler):
        """Missing return_url returns 400."""
        handler = make_mock_handler(body={}, method="POST")

        result = billing_handler._create_portal(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 400

    def test_create_portal_no_billing_account(self, billing_handler):
        """No billing account returns 404."""
        org = billing_handler._get_user_store().orgs["org-123"]
        org.stripe_customer_id = None

        handler = make_mock_handler(
            body={"return_url": "https://example.com/dashboard"},
            method="POST",
        )

        result = billing_handler._create_portal(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Cancel Subscription (POST /api/v1/billing/cancel)
# ===========================================================================


class TestBillingCancelSubscription:
    """Tests for POST /api/v1/billing/cancel endpoint."""

    def test_cancel_subscription_success(self, billing_handler):
        """Happy path: cancel subscription."""
        handler = make_mock_handler(method="POST")

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.cancel_subscription.return_value = MockStripeSubscription(
                cancel_at_period_end=True
            )
            mock_stripe.return_value = mock_client

            result = billing_handler._cancel_subscription(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "canceled at period end" in data["message"].lower()

    def test_cancel_subscription_no_subscription(self, billing_handler):
        """No subscription returns 404."""
        org = billing_handler._get_user_store().orgs["org-123"]
        org.stripe_subscription_id = None

        handler = make_mock_handler(method="POST")
        result = billing_handler._cancel_subscription(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Resume Subscription (POST /api/v1/billing/resume)
# ===========================================================================


class TestBillingResumeSubscription:
    """Tests for POST /api/v1/billing/resume endpoint."""

    def test_resume_subscription_success(self, billing_handler):
        """Happy path: resume subscription."""
        handler = make_mock_handler(method="POST")

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.resume_subscription.return_value = MockStripeSubscription()
            mock_stripe.return_value = mock_client

            result = billing_handler._resume_subscription(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "resumed" in data["message"].lower()


# ===========================================================================
# Test Get Audit Log (GET /api/v1/billing/audit-log)
# ===========================================================================


class TestBillingGetAuditLog:
    """Tests for GET /api/v1/billing/audit-log endpoint."""

    def test_get_audit_log_success(self, billing_handler):
        """Happy path: get audit log (Enterprise feature)."""
        handler = make_mock_handler()

        result = billing_handler._get_audit_log(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "entries" in data
        assert "total" in data

    def test_get_audit_log_no_enterprise(self, billing_handler):
        """Audit log requires Enterprise tier."""
        org = billing_handler._get_user_store().orgs["org-123"]
        org.limits.audit_logs = False

        handler = make_mock_handler()
        result = billing_handler._get_audit_log(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 403


# ===========================================================================
# Test Export Usage CSV (GET /api/v1/billing/usage/export)
# ===========================================================================


class TestBillingExportUsage:
    """Tests for GET /api/v1/billing/usage/export endpoint."""

    def test_export_usage_csv_success(self, billing_handler):
        """Happy path: export usage CSV."""
        handler = make_mock_handler()

        result = billing_handler._export_usage_csv(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/csv"
        assert "Content-Disposition" in result.headers


# ===========================================================================
# Test Get Usage Forecast (GET /api/v1/billing/usage/forecast)
# ===========================================================================


class TestBillingGetForecast:
    """Tests for GET /api/v1/billing/usage/forecast endpoint."""

    def test_get_forecast_success(self, billing_handler):
        """Happy path: get usage forecast."""
        handler = make_mock_handler()

        result = billing_handler._get_usage_forecast(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "forecast" in data
        assert "projection" in data["forecast"]
        assert "days_remaining" in data["forecast"]


# ===========================================================================
# Test Get Invoices (GET /api/v1/billing/invoices)
# ===========================================================================


class TestBillingGetInvoices:
    """Tests for GET /api/v1/billing/invoices endpoint."""

    def test_get_invoices_success(self, billing_handler):
        """Happy path: get invoices."""
        handler = make_mock_handler()

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_client = MagicMock()
            mock_client.list_invoices.return_value = [
                {
                    "id": "inv_123",
                    "number": "INV-001",
                    "status": "paid",
                    "amount_due": 4900,
                    "amount_paid": 4900,
                    "currency": "usd",
                    "created": 1704067200,
                    "period_start": 1704067200,
                    "period_end": 1706745600,
                    "hosted_invoice_url": "https://stripe.com/invoice",
                }
            ]
            mock_stripe.return_value = mock_client

            result = billing_handler._get_invoices(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "invoices" in data
        assert len(data["invoices"]) == 1

    def test_get_invoices_no_billing_account(self, billing_handler):
        """No billing account returns 404."""
        org = billing_handler._get_user_store().orgs["org-123"]
        org.stripe_customer_id = None

        handler = make_mock_handler()
        result = billing_handler._get_invoices(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Stripe Webhook (POST /api/v1/webhooks/stripe)
# ===========================================================================


class TestBillingStripeWebhook:
    """Tests for POST /api/v1/webhooks/stripe endpoint."""

    def test_webhook_missing_signature(self, billing_handler):
        """Missing signature returns 400."""
        handler = make_mock_handler(body={}, method="POST")
        handler.headers["Content-Length"] = "2"
        handler.rfile = BytesIO(b"{}")

        result = billing_handler._handle_stripe_webhook(handler)

        assert result is not None
        assert result.status_code == 400

    def test_webhook_invalid_signature(self, billing_handler):
        """Invalid signature returns 400."""
        handler = make_mock_handler(body={}, method="POST")
        handler.headers["Stripe-Signature"] = "invalid_sig"
        handler.headers["Content-Length"] = "2"
        handler.rfile = BytesIO(b"{}")

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            mock_parse.return_value = None

            result = billing_handler._handle_stripe_webhook(handler)

        assert result is not None
        assert result.status_code == 400

    def test_webhook_checkout_completed(self, billing_handler):
        """Webhook handles checkout.session.completed."""
        handler = make_mock_handler(method="POST")
        handler.headers["Stripe-Signature"] = "valid_sig"
        handler.headers["Content-Length"] = "2"
        handler.rfile = BytesIO(b"{}")

        mock_event = MagicMock()
        mock_event.type = "checkout.session.completed"
        mock_event.event_id = "evt_123"
        mock_event.object = {"customer": "cus_123", "subscription": "sub_123"}
        mock_event.metadata = {"user_id": "user-123", "org_id": "org-123", "tier": "starter"}

        with (
            patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse,
            patch.object(billing_module, "_is_duplicate_webhook", return_value=False),
            patch.object(billing_module, "_mark_webhook_processed"),
        ):
            mock_parse.return_value = mock_event

            result = billing_handler._handle_stripe_webhook(handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestBillingRateLimiting:
    """Tests for rate limiting on billing endpoints."""

    def test_rate_limit_exceeded(self, billing_handler):
        """Rate limit exceeded returns 429."""
        with patch.object(billing_module, "_billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            handler = make_mock_handler()
            result = billing_handler.handle("/api/v1/billing/plans", {}, handler)

            assert result is not None
            assert result.status_code == 429

    def test_webhook_bypasses_rate_limit(self, billing_handler):
        """Stripe webhooks bypass rate limiting."""
        with patch.object(billing_module, "_billing_limiter") as mock_limiter:
            # Even with rate limit "exceeded", webhook should not be blocked
            mock_limiter.is_allowed.return_value = False

            handler = make_mock_handler(method="POST")
            handler.headers["Stripe-Signature"] = "test_sig"
            handler.headers["Content-Length"] = "2"
            handler.rfile = BytesIO(b"{}")

            # Webhook will fail on signature validation, but not on rate limit
            result = billing_handler.handle("/api/v1/webhooks/stripe", {}, handler, method="POST")

            # Should fail on signature, not rate limit
            assert result is not None
            assert result.status_code == 400  # Bad signature, not 429


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestBillingErrorHandling:
    """Tests for error handling in billing handler."""

    def test_method_not_allowed(self, billing_handler):
        """Unsupported method returns 405."""
        handler = make_mock_handler(method="PUT")

        result = billing_handler.handle("/api/v1/billing/plans", {}, handler, method="PUT")

        assert result is not None
        assert result.status_code == 405

    def test_stripe_config_error(self, billing_handler):
        """Stripe config error returns 503."""
        handler = make_mock_handler(
            body={
                "tier": "professional",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            method="POST",
        )

        from aragora.billing.stripe_client import StripeConfigError

        with patch("aragora.server.handlers.admin.billing.get_stripe_client") as mock_stripe:
            mock_stripe.side_effect = StripeConfigError("Not configured")

            result = billing_handler._create_checkout(handler, user=MockUserContext())

        assert result is not None
        assert result.status_code == 503
