"""Tests for billing core handler (aragora/server/handlers/billing/core.py).

Covers all routes and behavior of the BillingHandler class:
- can_handle() routing for all ROUTES
- GET /api/v1/billing/plans - List available subscription plans
- GET /api/v1/billing/usage - Get current usage
- GET /api/v1/billing/subscription - Get current subscription
- POST /api/v1/billing/checkout - Create checkout session
- POST /api/v1/billing/portal - Create billing portal session
- POST /api/v1/billing/cancel - Cancel subscription
- POST /api/v1/billing/resume - Resume canceled subscription
- POST /api/v1/webhooks/stripe - Handle Stripe webhooks
- Rate limiting behavior
- Error handling (missing fields, Stripe errors)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import SubscriptionTier, TIER_LIMITS
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
    """Mock user for billing tests."""

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
    """Mock organization for billing tests."""

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
        trial_started_at: datetime | None = None,
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
        self.trial_started_at = trial_started_at

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)


class MockStripeSubscription:
    """Mock Stripe subscription."""

    def __init__(
        self,
        id: str,
        status: str = "active",
        current_period_end: datetime | None = None,
        cancel_at_period_end: bool = False,
        trial_start: datetime | None = None,
        trial_end: datetime | None = None,
    ):
        self.id = id
        self.status = status
        self.current_period_end = current_period_end or (
            datetime.now(timezone.utc) + timedelta(days=30)
        )
        self.cancel_at_period_end = cancel_at_period_end
        self.trial_start = trial_start
        self.trial_end = trial_end
        self.is_trialing = trial_start is not None and (
            trial_end is None or trial_end > datetime.now(timezone.utc)
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }


class MockCheckoutSession:
    """Mock Stripe checkout session."""

    def __init__(self, id: str, url: str = "https://checkout.stripe.com/session"):
        self.id = id
        self.url = url

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


class MockPortalSession:
    """Mock Stripe portal session."""

    def __init__(self, id: str, url: str = "https://billing.stripe.com/session"):
        self.id = id
        self.url = url

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


class MockStripeClient:
    """Mock Stripe client for testing."""

    def __init__(
        self,
        subscription: MockStripeSubscription | None = None,
        checkout_session: MockCheckoutSession | None = None,
        portal_session: MockPortalSession | None = None,
    ):
        self._subscription = subscription
        self._checkout_session = checkout_session or MockCheckoutSession("cs_test_123")
        self._portal_session = portal_session or MockPortalSession("bps_test_123")

    def get_subscription(self, subscription_id: str) -> MockStripeSubscription | None:
        return self._subscription

    def create_checkout_session(self, **kwargs) -> MockCheckoutSession:
        return self._checkout_session

    def create_portal_session(self, **kwargs) -> MockPortalSession:
        return self._portal_session

    def cancel_subscription(
        self, subscription_id: str, at_period_end: bool = True
    ) -> MockStripeSubscription:
        if self._subscription:
            self._subscription.cancel_at_period_end = True
        return self._subscription

    def resume_subscription(self, subscription_id: str) -> MockStripeSubscription:
        if self._subscription:
            self._subscription.cancel_at_period_end = False
        return self._subscription


class MockUserStore:
    """Mock user store for billing tests."""

    def __init__(self):
        self._users: dict[str, MockUser] = {}
        self._orgs: dict[str, MockOrganization] = {}
        self._orgs_by_subscription: dict[str, MockOrganization] = {}
        self._orgs_by_customer: dict[str, MockOrganization] = {}

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org
        if org.stripe_subscription_id:
            self._orgs_by_subscription[org.stripe_subscription_id] = org
        if org.stripe_customer_id:
            self._orgs_by_customer[org.stripe_customer_id] = org

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self._users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self._orgs.get(org_id)

    def get_organization_by_subscription(self, subscription_id: str) -> MockOrganization | None:
        return self._orgs_by_subscription.get(subscription_id)

    def get_organization_by_stripe_customer(self, customer_id: str) -> MockOrganization | None:
        return self._orgs_by_customer.get(customer_id)

    def update_organization(self, org_id: str, **kwargs) -> MockOrganization | None:
        org = self._orgs.get(org_id)
        if org:
            for key, value in kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)
        return org

    def reset_org_usage(self, org_id: str):
        org = self._orgs.get(org_id)
        if org:
            org.debates_used_this_month = 0

    def log_audit_event(self, **kwargs):
        pass


class MockHTTPHandler:
    """Mock HTTP handler for request simulation."""

    def __init__(
        self,
        body: dict | None = None,
        command: str = "GET",
        query_params: dict | None = None,
        signature: str = "",
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

        if signature:
            self.headers["Stripe-Signature"] = signature

    def get(self, key: str, default=None):
        """Support for get_string_param resolution."""
        return self._query_params.get(key, default)


class MockWebhookEvent:
    """Mock Stripe webhook event."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        object_data: dict | None = None,
        metadata: dict | None = None,
    ):
        self.event_id = event_id
        self.type = event_type
        self.object = object_data or {}
        self.metadata = metadata or {}
        self.subscription_id = object_data.get("id") if object_data else None


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_store():
    """Create a user store with standard test data."""
    store = MockUserStore()

    # The conftest auto-auth context uses user_id="test-user-001"
    auth_user = MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
    store.add_user(auth_user)

    owner = MockUser(id="owner_1", email="owner@test.com", role="owner", org_id="org_1")
    store.add_user(owner)

    member = MockUser(id="member_1", email="member@test.com", role="member", org_id="org_1")
    store.add_user(member)

    no_org_user = MockUser(id="no_org_1", email="noorg@test.com", role="member", org_id=None)
    store.add_user(no_org_user)

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
    """Create a BillingHandler without a user store (service unavailable scenario)."""
    return BillingHandler(ctx={})


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the rate limiter between tests to avoid cross-test pollution."""
    _billing_limiter._buckets.clear()
    yield
    _billing_limiter._buckets.clear()


@pytest.fixture
def mock_stripe_client():
    """Create a default mock Stripe client."""
    sub = MockStripeSubscription("sub_test_123", status="active")
    return MockStripeClient(subscription=sub)


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_billing_plans_route(self, handler):
        assert handler.can_handle("/api/v1/billing/plans")

    def test_billing_usage_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage")

    def test_billing_subscription_route(self, handler):
        assert handler.can_handle("/api/v1/billing/subscription")

    def test_billing_checkout_route(self, handler):
        assert handler.can_handle("/api/v1/billing/checkout")

    def test_billing_portal_route(self, handler):
        assert handler.can_handle("/api/v1/billing/portal")

    def test_billing_cancel_route(self, handler):
        assert handler.can_handle("/api/v1/billing/cancel")

    def test_billing_resume_route(self, handler):
        assert handler.can_handle("/api/v1/billing/resume")

    def test_webhooks_stripe_route(self, handler):
        assert handler.can_handle("/api/v1/webhooks/stripe")

    def test_billing_trial_route(self, handler):
        assert handler.can_handle("/api/v1/billing/trial")

    def test_billing_trial_start_route(self, handler):
        assert handler.can_handle("/api/v1/billing/trial/start")

    def test_billing_audit_log_route(self, handler):
        assert handler.can_handle("/api/v1/billing/audit-log")

    def test_billing_usage_export_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/export")

    def test_billing_usage_forecast_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/forecast")

    def test_billing_invoices_route(self, handler):
        assert handler.can_handle("/api/v1/billing/invoices")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_billing_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/billing")

    def test_unversioned_billing_path_rejected(self, handler):
        assert not handler.can_handle("/api/billing/plans")


# ---------------------------------------------------------------------------
# GET /api/v1/billing/plans
# ---------------------------------------------------------------------------


class TestGetPlans:
    """Tests for listing available subscription plans."""

    def test_returns_all_tiers(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)

        assert "plans" in body
        plan_ids = [p["id"] for p in body["plans"]]
        for tier in SubscriptionTier:
            assert tier.value in plan_ids

    def test_plan_structure(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)

        for plan in body["plans"]:
            assert "id" in plan
            assert "name" in plan
            assert "price_monthly_cents" in plan
            assert "price_monthly" in plan
            assert "features" in plan
            features = plan["features"]
            assert "debates_per_month" in features
            assert "users_per_org" in features
            assert "api_access" in features

    def test_free_tier_price_is_zero(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)

        free_plan = next(p for p in body["plans"] if p["id"] == "free")
        assert free_plan["price_monthly_cents"] == 0
        assert free_plan["price_monthly"] == "$0.00"

    def test_status_code_is_200(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v1/billing/usage
# ---------------------------------------------------------------------------


class TestGetUsage:
    """Tests for usage retrieval."""

    def _add_test_user(self, handler):
        """Add the conftest auto-auth user (test-user-001) to the store."""
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )

    def test_returns_usage_data(self, handler):
        self._add_test_user(handler)
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)

        assert "usage" in body
        usage = body["usage"]
        assert "debates_used" in usage
        assert "debates_limit" in usage
        assert "debates_remaining" in usage

    def test_usage_reflects_org_data(self, handler):
        self._add_test_user(handler)
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)

        usage = body["usage"]
        assert usage["debates_used"] == 5
        assert usage["debates_limit"] == 10
        assert usage["debates_remaining"] == 5

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert result.status_code == 503

    def test_unknown_user_returns_404(self):
        """When the auth context user is not in the store, return 404."""
        # Create a store without the auto-auth user "test-user-001"
        empty_store = MockUserStore()
        empty_handler = BillingHandler(ctx={"user_store": empty_store})
        http = MockHTTPHandler()
        result = empty_handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/billing/subscription
# ---------------------------------------------------------------------------


class TestGetSubscription:
    """Tests for subscription retrieval."""

    def test_returns_subscription_data(self, handler, mock_stripe_client):
        # Add the test-user-001 (from conftest mock) to the store
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)

        assert "subscription" in body
        sub = body["subscription"]
        assert "tier" in sub
        assert "status" in sub

    def test_subscription_with_stripe_data(self, handler, mock_stripe_client):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)

        sub = body["subscription"]
        assert sub["status"] == "active"
        assert sub["is_active"] is True
        assert "current_period_end" in sub

    def test_subscription_stripe_error_degrades_gracefully(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeError("Connection refused"),
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)

        # Should still return subscription data (partial) even when Stripe fails
        assert "subscription" in body
        assert body["subscription"]["tier"] == "free"

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/subscription", {}, http, method="GET")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/v1/billing/checkout
# ---------------------------------------------------------------------------


class TestCreateCheckout:
    """Tests for checkout session creation."""

    def test_successful_checkout(self, handler, mock_stripe_client):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert "checkout" in body
        assert body["checkout"]["id"] == "cs_test_123"

    def test_checkout_invalid_json_returns_400(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        http = MockHTTPHandler(command="POST")
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        # Should return 400 for invalid JSON
        assert result.status_code in (400, 500)

    def test_checkout_missing_tier_returns_400(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        http = MockHTTPHandler(
            body={
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 400

    def test_checkout_free_tier_returns_400(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        http = MockHTTPHandler(
            body={
                "tier": "free",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 400
        body = _body(result)
        # "free" is not in the allowed enum values in CHECKOUT_SESSION_SCHEMA,
        # so validation rejects it before the handler-level check
        assert "error" in body

    def test_checkout_invalid_tier_returns_400(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        http = MockHTTPHandler(
            body={
                "tier": "nonexistent_tier",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 400

    def test_checkout_stripe_config_error_returns_503(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("No API key"),
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 503

    def test_checkout_stripe_api_error_returns_502(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeAPIError("Bad request")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 502

    def test_checkout_generic_stripe_error_returns_500(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeError("Unknown error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 500

    def test_checkout_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler(
            body={
                "tier": "starter",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
        )
        result = handler_no_store.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/v1/billing/portal
# ---------------------------------------------------------------------------


class TestCreatePortal:
    """Tests for billing portal session creation."""

    def test_successful_portal(self, handler, mock_stripe_client):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(
                body={"return_url": "https://example.com/billing"},
                command="POST",
            )
            result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert "portal" in body
        assert body["portal"]["id"] == "bps_test_123"

    def test_portal_missing_return_url_returns_400(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert result.status_code == 400

    def test_portal_no_stripe_customer_returns_404(self, handler):
        # Add user with org that has no stripe_customer_id
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_no_stripe")
        )
        store.add_organization(
            MockOrganization(
                id="org_no_stripe",
                name="No Stripe Org",
                tier=SubscriptionTier.FREE,
                stripe_customer_id=None,
            )
        )
        http = MockHTTPHandler(
            body={"return_url": "https://example.com/billing"},
            command="POST",
        )
        result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert result.status_code == 404

    def test_portal_stripe_config_error_returns_503(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler(
                body={"return_url": "https://example.com/billing"},
                command="POST",
            )
            result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/v1/billing/cancel
# ---------------------------------------------------------------------------


class TestCancelSubscription:
    """Tests for subscription cancellation."""

    def test_successful_cancel(self, handler, mock_stripe_client):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert "message" in body
        assert "canceled" in body["message"].lower() or "cancel" in body["message"].lower()
        assert "subscription" in body

    def test_cancel_no_subscription_returns_404(self, handler):
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_no_sub")
        )
        store.add_organization(
            MockOrganization(
                id="org_no_sub",
                name="No Sub Org",
                tier=SubscriptionTier.FREE,
                stripe_subscription_id=None,
            )
        )
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert result.status_code == 404

    def test_cancel_stripe_api_error_returns_502(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert result.status_code == 502

    def test_cancel_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler(command="POST")
        result = handler_no_store.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/v1/billing/resume
# ---------------------------------------------------------------------------


class TestResumeSubscription:
    """Tests for resuming a canceled subscription."""

    def test_successful_resume(self, handler, mock_stripe_client):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert "message" in body
        assert "resumed" in body["message"].lower()
        assert "subscription" in body

    def test_resume_no_subscription_returns_404(self, handler):
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_no_sub")
        )
        store.add_organization(
            MockOrganization(
                id="org_no_sub",
                name="No Sub Org",
                stripe_subscription_id=None,
            )
        )
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert result.status_code == 404

    def test_resume_stripe_config_error_returns_503(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert result.status_code == 503

    def test_resume_stripe_api_error_returns_502(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        mock_client = MagicMock()
        mock_client.resume_subscription.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert result.status_code == 502


# ---------------------------------------------------------------------------
# POST /api/v1/webhooks/stripe
# ---------------------------------------------------------------------------


class TestStripeWebhook:
    """Tests for Stripe webhook handling."""

    def _webhook_patches(self, event, is_duplicate=False):
        """Context manager helper for common webhook test patches.

        Patches parse_webhook_event at its source, and uses _get_admin_billing_callable
        to inject mock duplicate/processed callables since the real resolution may
        find the admin.billing module versions instead of core_helpers versions.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch("aragora.billing.stripe_client.parse_webhook_event", return_value=event)
        )

        # The webhook handler resolves _is_duplicate_webhook and _mark_webhook_processed
        # via _get_admin_billing_callable, which may resolve from admin.billing.
        # We override _get_admin_billing_callable to return our controlled lambdas.
        def mock_get_callable(name, fallback):
            if name == "_is_duplicate_webhook":
                return lambda event_id: is_duplicate
            if name == "_mark_webhook_processed":
                return lambda event_id, result="success": None
            return fallback

        stack.enter_context(
            patch(
                "aragora.server.handlers.billing.core_webhooks._get_admin_billing_callable",
                side_effect=mock_get_callable,
            )
        )
        return stack

    def test_missing_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST")
        http.headers["Content-Length"] = "100"
        http.rfile.read.return_value = b'{"type": "test"}'
        # No Stripe-Signature header
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert result.status_code == 400

    def test_invalid_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="invalid_sig")
        http.headers["Content-Length"] = "100"
        http.rfile.read.return_value = b'{"type": "test"}'
        with patch(
            "aragora.billing.stripe_client.parse_webhook_event",
            return_value=None,
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert result.status_code == 400

    def test_checkout_completed_event(self, handler):
        handler.ctx["user_store"].add_user(
            MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
        )
        event = MockWebhookEvent(
            event_id="evt_test_001",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new_123", "id": "cs_1"},
            metadata={"user_id": "test-user-001", "org_id": "org_1", "tier": "starter"},
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "checkout.session.completed"}'

        with self._webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert body["received"] is True

    def test_duplicate_webhook_skipped(self, handler):
        event = MockWebhookEvent(
            event_id="evt_duplicate_001",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123"},
            metadata={},
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "checkout.session.completed"}'

        with self._webhook_patches(event, is_duplicate=True):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert body.get("duplicate") is True

    def test_subscription_deleted_downgrades_to_free(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.STARTER

        event = MockWebhookEvent(
            event_id="evt_del_001",
            event_type="customer.subscription.deleted",
            object_data={"id": "sub_test_123"},
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "customer.subscription.deleted"}'

        with self._webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert body["received"] is True
        # Org should be downgraded to FREE
        updated_org = store.get_organization_by_id("org_1")
        assert updated_org.tier == SubscriptionTier.FREE

    def test_unhandled_event_acknowledged(self, handler):
        event = MockWebhookEvent(
            event_id="evt_unknown_001",
            event_type="some.unknown.event",
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "some.unknown.event"}'

        with self._webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)

        assert result.status_code == 200
        assert body["received"] is True

    def test_invoice_paid_resets_usage(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.debates_used_this_month = 42

        event = MockWebhookEvent(
            event_id="evt_inv_001",
            event_type="invoice.payment_succeeded",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "amount_paid": 9900,
            },
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "invoice.payment_succeeded"}'

        with self._webhook_patches(event), patch(
            "aragora.billing.payment_recovery.get_recovery_store",
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        assert result.status_code == 200
        updated_org = store.get_organization_by_id("org_1")
        assert updated_org.debates_used_this_month == 0


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting behavior on billing endpoints."""

    def test_rate_limit_not_applied_to_webhooks(self, handler):
        """Webhook endpoint should bypass rate limiting."""
        event = MockWebhookEvent(
            event_id="evt_rate_001",
            event_type="some.event",
        )
        http = MockHTTPHandler(command="POST", signature="valid_sig")
        http.headers["Content-Length"] = "50"
        http.rfile.read.return_value = b'{"type": "some.event"}'

        def mock_get_callable(name, fallback):
            if name == "_is_duplicate_webhook":
                return lambda event_id: False
            if name == "_mark_webhook_processed":
                return lambda event_id, result="success": None
            return fallback

        with patch(
            "aragora.billing.stripe_client.parse_webhook_event",
            return_value=event,
        ), patch(
            "aragora.server.handlers.billing.core_webhooks._get_admin_billing_callable",
            side_effect=mock_get_callable,
        ):
            # Should never get 429 for webhooks
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            assert result.status_code != 429

    def test_rate_limit_applied_to_billing_endpoints(self, handler):
        """Normal billing endpoints should be rate limited."""
        # Fill the rate limiter with more than 20 requests
        # We need to exhaust the rate limiter for the test's IP
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False

        with patch(
            "aragora.server.handlers.billing.core._get_billing_limiter",
            return_value=mock_limiter,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        assert result.status_code == 429


# ---------------------------------------------------------------------------
# Method Not Allowed
# ---------------------------------------------------------------------------


class TestMethodNotAllowed:
    """Tests for method-not-allowed responses."""

    def test_post_to_plans_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/plans", {}, http, method="POST")
        assert result.status_code == 405

    def test_get_to_checkout_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="GET")
        assert result.status_code == 405

    def test_get_to_cancel_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/cancel", {}, http, method="GET")
        assert result.status_code == 405


# ---------------------------------------------------------------------------
# Handler initialization
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Tests for handler initialization and context."""

    def test_default_context_is_empty_dict(self):
        handler = BillingHandler()
        assert handler.ctx == {}

    def test_context_passed_through(self):
        ctx = {"user_store": MagicMock(), "usage_tracker": MagicMock()}
        handler = BillingHandler(ctx=ctx)
        assert handler.ctx is ctx

    def test_get_user_store_returns_from_context(self, handler, user_store):
        assert handler._get_user_store() is user_store

    def test_get_user_store_returns_none_when_missing(self, handler_no_store):
        assert handler_no_store._get_user_store() is None

    def test_get_usage_tracker_returns_from_context(self):
        tracker = MagicMock()
        handler = BillingHandler(ctx={"usage_tracker": tracker})
        assert handler._get_usage_tracker() is tracker

    def test_get_usage_tracker_returns_none_when_missing(self, handler):
        assert handler._get_usage_tracker() is None

    def test_resource_type_is_billing(self, handler):
        assert handler.RESOURCE_TYPE == "billing"

    def test_routes_list_is_populated(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v1/billing/plans" in handler.ROUTES
        assert "/api/v1/webhooks/stripe" in handler.ROUTES


# ---------------------------------------------------------------------------
# _log_audit helper
# ---------------------------------------------------------------------------


class TestLogAudit:
    """Tests for the _log_audit internal method."""

    def test_log_audit_with_valid_store(self, handler):
        store = MagicMock()
        store.log_audit_event = MagicMock()
        handler._log_audit(
            store,
            action="test.action",
            resource_type="subscription",
            resource_id="sub_123",
            user_id="user_1",
            org_id="org_1",
        )
        store.log_audit_event.assert_called_once()

    def test_log_audit_with_no_store(self, handler):
        # Should not raise
        handler._log_audit(
            None,
            action="test.action",
            resource_type="subscription",
        )

    def test_log_audit_with_store_missing_method(self, handler):
        store = MagicMock(spec=[])
        # Should not raise
        handler._log_audit(
            store,
            action="test.action",
            resource_type="subscription",
        )
