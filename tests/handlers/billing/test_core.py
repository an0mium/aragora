"""Comprehensive tests for billing core handler (aragora/server/handlers/billing/core.py).

Covers every route and code path of the BillingHandler class and its mixins:
- can_handle() route matching for all 14 ROUTES
- GET /api/v1/billing/plans
- GET /api/v1/billing/usage (with and without usage tracker)
- GET /api/v1/billing/subscription (with Stripe, without Stripe, errors)
- GET /api/v1/billing/trial (trial status, warnings, upgrade options)
- POST /api/v1/billing/trial/start (new trial, active trial, expired, paid org)
- POST /api/v1/billing/checkout (validation, Stripe errors)
- POST /api/v1/billing/portal (validation, Stripe errors)
- POST /api/v1/billing/cancel (Stripe errors)
- POST /api/v1/billing/resume (Stripe errors)
- GET /api/v1/billing/audit-log (Enterprise tier check, role check)
- GET /api/v1/billing/usage/export (CSV export with dates)
- GET /api/v1/billing/usage/forecast (projections, tier recommendations)
- GET /api/v1/billing/invoices (Stripe invoices, errors)
- POST /api/v1/webhooks/stripe (all event types, idempotency)
- Rate limiting
- Method not allowed (405)
- _log_audit helper
- _get_billing_limiter compatibility
"""

from __future__ import annotations

import json
from contextlib import ExitStack
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
        is_in_trial: bool = False,
        is_trial_expired: bool = False,
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
        self.is_in_trial = is_in_trial
        self.is_trial_expired = is_trial_expired

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
        invoices: list[dict] | None = None,
    ):
        self._subscription = subscription
        self._checkout_session = checkout_session or MockCheckoutSession("cs_test_123")
        self._portal_session = portal_session or MockPortalSession("bps_test_123")
        self._invoices = invoices or []

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

    def list_invoices(self, customer_id: str, limit: int = 10) -> list[dict]:
        return self._invoices


class MockUserStore:
    """Mock user store for billing tests."""

    def __init__(self):
        self._users: dict[str, MockUser] = {}
        self._orgs: dict[str, MockOrganization] = {}
        self._orgs_by_subscription: dict[str, MockOrganization] = {}
        self._orgs_by_customer: dict[str, MockOrganization] = {}
        self._audit_entries: list[dict] = []

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

    def get_organization_owner(self, org_id: str) -> MockUser | None:
        for u in self._users.values():
            if u.org_id == org_id and u.role == "owner":
                return u
        return None

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
        self._audit_entries.append(kwargs)

    def get_audit_log(self, **kwargs) -> list[dict]:
        return self._audit_entries

    def get_audit_log_count(self, **kwargs) -> int:
        return len(self._audit_entries)


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


class MockTrialStatus:
    """Mock trial status returned by TrialManager."""

    def __init__(
        self,
        is_active: bool = True,
        is_expired: bool = False,
        days_remaining: int = 7,
        debates_remaining: int = 10,
        debates_used: int = 0,
    ):
        self.is_active = is_active
        self.is_expired = is_expired
        self.days_remaining = days_remaining
        self.debates_remaining = debates_remaining
        self.debates_used = debates_used

    def to_dict(self) -> dict:
        return {
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "days_remaining": self.days_remaining,
            "debates_remaining": self.debates_remaining,
            "debates_used": self.debates_used,
        }


class MockUsageSummary:
    """Mock usage summary from usage tracker."""

    def __init__(
        self,
        total_tokens_in: int = 5000,
        total_tokens_out: int = 3000,
        total_cost_usd: Decimal | float = 0.25,
        cost_by_provider: dict | None = None,
        total_tokens: int = 8000,
        total_cost: Decimal | float = 0.25,
    ):
        self.total_tokens_in = total_tokens_in
        self.total_tokens_out = total_tokens_out
        self.total_cost_usd = total_cost_usd
        self.cost_by_provider = cost_by_provider or {"anthropic": Decimal("0.20"), "openai": Decimal("0.05")}
        self.total_tokens = total_tokens
        self.total_cost = total_cost


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_store():
    """Create a user store with standard test data."""
    store = MockUserStore()

    # The conftest auto-auth context uses user_id="test-user-001"
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


@pytest.fixture
def mock_stripe_client():
    """Create a default mock Stripe client."""
    sub = MockStripeSubscription("sub_test_123", status="active")
    return MockStripeClient(subscription=sub)


def _webhook_patches(event, is_duplicate=False):
    """Context manager for common webhook test patches."""
    stack = ExitStack()
    stack.enter_context(
        patch("aragora.billing.stripe_client.parse_webhook_event", return_value=event)
    )

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


# ===========================================================================
# TestCanHandle - route matching
# ===========================================================================


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_billing_plans_route(self, handler):
        assert handler.can_handle("/api/v1/billing/plans")

    def test_billing_usage_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage")

    def test_billing_subscription_route(self, handler):
        assert handler.can_handle("/api/v1/billing/subscription")

    def test_billing_trial_route(self, handler):
        assert handler.can_handle("/api/v1/billing/trial")

    def test_billing_trial_start_route(self, handler):
        assert handler.can_handle("/api/v1/billing/trial/start")

    def test_billing_checkout_route(self, handler):
        assert handler.can_handle("/api/v1/billing/checkout")

    def test_billing_portal_route(self, handler):
        assert handler.can_handle("/api/v1/billing/portal")

    def test_billing_cancel_route(self, handler):
        assert handler.can_handle("/api/v1/billing/cancel")

    def test_billing_resume_route(self, handler):
        assert handler.can_handle("/api/v1/billing/resume")

    def test_billing_audit_log_route(self, handler):
        assert handler.can_handle("/api/v1/billing/audit-log")

    def test_billing_usage_export_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/export")

    def test_billing_usage_forecast_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/forecast")

    def test_billing_invoices_route(self, handler):
        assert handler.can_handle("/api/v1/billing/invoices")

    def test_webhooks_stripe_route(self, handler):
        assert handler.can_handle("/api/v1/webhooks/stripe")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_billing_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/billing")

    def test_unversioned_billing_path_rejected(self, handler):
        assert not handler.can_handle("/api/billing/plans")

    def test_empty_path_rejected(self, handler):
        assert not handler.can_handle("")


# ===========================================================================
# TestGetPlans
# ===========================================================================


class TestGetPlans:
    """Tests for listing available subscription plans."""

    def test_returns_all_tiers(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)
        plan_ids = [p["id"] for p in body["plans"]]
        for tier in SubscriptionTier:
            assert tier.value in plan_ids

    def test_plan_structure_has_required_fields(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)
        for plan in body["plans"]:
            assert "id" in plan
            assert "name" in plan
            assert "price_monthly_cents" in plan
            assert "price_monthly" in plan
            features = plan["features"]
            assert "debates_per_month" in features
            assert "users_per_org" in features
            assert "api_access" in features
            assert "all_agents" in features
            assert "custom_agents" in features
            assert "sso_enabled" in features
            assert "audit_logs" in features
            assert "priority_support" in features

    def test_free_tier_price_is_zero(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)
        free_plan = next(p for p in body["plans"] if p["id"] == "free")
        assert free_plan["price_monthly_cents"] == 0
        assert free_plan["price_monthly"] == "$0.00"

    def test_starter_tier_has_correct_price(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)
        starter = next(p for p in body["plans"] if p["id"] == "starter")
        assert starter["price_monthly_cents"] == TIER_LIMITS[SubscriptionTier.STARTER].price_monthly_cents

    def test_status_code_is_200(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        assert _status(result) == 200

    def test_plan_name_is_title_case(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        body = _body(result)
        for plan in body["plans"]:
            # Name should be the enum name in title case
            tier = SubscriptionTier(plan["id"])
            assert plan["name"] == tier.name.title()


# ===========================================================================
# TestGetUsage
# ===========================================================================


class TestGetUsage:
    """Tests for usage retrieval."""

    def test_returns_usage_data(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        assert "usage" in body
        usage = body["usage"]
        assert "debates_used" in usage
        assert "debates_limit" in usage
        assert "debates_remaining" in usage

    def test_usage_reflects_org_data(self, handler):
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
        assert _status(result) == 503

    def test_unknown_user_returns_404(self):
        empty_store = MockUserStore()
        h = BillingHandler(ctx={"user_store": empty_store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 404

    def test_user_without_org_returns_defaults(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        # Should return default usage data with zeros
        assert body["usage"]["debates_used"] == 0
        assert body["usage"]["debates_limit"] == 10

    def test_usage_with_tracker(self, user_store):
        tracker = MagicMock()
        summary = MockUsageSummary()
        tracker.get_summary.return_value = summary
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        usage = body["usage"]
        assert usage["tokens_in"] == 5000
        assert usage["tokens_out"] == 3000
        assert usage["tokens_used"] == 8000
        assert usage["cost_breakdown"] is not None
        assert "cost_by_provider" in usage

    def test_usage_tracker_returns_none_summary(self, user_store):
        tracker = MagicMock()
        tracker.get_summary.return_value = None
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        # Tokens should remain at default 0
        assert body["usage"]["tokens_used"] == 0


# ===========================================================================
# TestGetSubscription
# ===========================================================================


class TestGetSubscription:
    """Tests for subscription retrieval."""

    def test_returns_subscription_data(self, handler, mock_stripe_client):
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

    def test_subscription_with_stripe_active(self, handler, mock_stripe_client):
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

    def test_subscription_with_trial_info(self, handler):
        trial_start = datetime.now(timezone.utc) - timedelta(days=3)
        trial_end = datetime.now(timezone.utc) + timedelta(days=4)
        stripe_sub = MockStripeSubscription(
            "sub_test_123",
            status="trialing",
            trial_start=trial_start,
            trial_end=trial_end,
        )
        client = MockStripeClient(subscription=stripe_sub)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)
        sub = body["subscription"]
        assert sub["is_active"] is True
        assert sub["is_trialing"] is True
        assert "trial_start" in sub
        assert "trial_end" in sub

    def test_subscription_past_due_marks_payment_failed(self, handler):
        stripe_sub = MockStripeSubscription("sub_test_123", status="past_due")
        client = MockStripeClient(subscription=stripe_sub)
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)
        sub = body["subscription"]
        assert sub["payment_failed"] is True
        assert sub["is_active"] is False

    def test_subscription_stripe_error_degrades_gracefully(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeError("Connection refused"),
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)
        assert "subscription" in body
        assert body["subscription"]["tier"] == "free"

    def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/subscription", {}, http, method="GET")
        assert _status(result) == 503

    def test_subscription_includes_org_info(self, handler, mock_stripe_client):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/subscription", {}, http, method="GET")
            body = _body(result)
        sub = body["subscription"]
        assert "organization" in sub
        assert sub["organization"]["id"] == "org_1"
        assert sub["organization"]["name"] == "Test Org"


# ===========================================================================
# TestGetTrialStatus
# ===========================================================================


class TestGetTrialStatus:
    """Tests for trial status retrieval."""

    def test_trial_status_active(self, handler):
        mock_status = MockTrialStatus(is_active=True, days_remaining=5)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager"
        ) as mock_mgr:
            mock_mgr.return_value.get_trial_status.return_value = mock_status
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/trial", {}, http, method="GET")
            body = _body(result)
        assert "trial" in body
        trial = body["trial"]
        assert trial["is_active"] is True
        assert trial["days_remaining"] == 5
        # Active trial should have upgrade options
        assert "upgrade_options" in trial

    def test_trial_status_expired_has_upgrade_options(self, handler):
        mock_status = MockTrialStatus(is_active=False, is_expired=True, days_remaining=0)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager"
        ) as mock_mgr:
            mock_mgr.return_value.get_trial_status.return_value = mock_status
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/trial", {}, http, method="GET")
            body = _body(result)
        trial = body["trial"]
        assert "upgrade_options" in trial
        assert len(trial["upgrade_options"]) == 3

    def test_trial_status_warning_when_expiring_soon(self, handler):
        mock_status = MockTrialStatus(is_active=True, days_remaining=2)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager"
        ) as mock_mgr:
            mock_mgr.return_value.get_trial_status.return_value = mock_status
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/trial", {}, http, method="GET")
            body = _body(result)
        trial = body["trial"]
        assert "warning" in trial
        assert "2 day(s)" in trial["warning"]

    def test_trial_status_no_warning_when_days_above_3(self, handler):
        mock_status = MockTrialStatus(is_active=True, days_remaining=5)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager"
        ) as mock_mgr:
            mock_mgr.return_value.get_trial_status.return_value = mock_status
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/trial", {}, http, method="GET")
            body = _body(result)
        assert "warning" not in body["trial"]

    def test_trial_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/trial", {}, http, method="GET")
        assert _status(result) == 404

    def test_trial_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/trial", {}, http, method="GET")
        assert _status(result) == 503


# ===========================================================================
# TestStartTrial
# ===========================================================================


class TestStartTrial:
    """Tests for starting a free trial."""

    def _make_trial_mgr(self, status):
        mgr = MagicMock()
        mgr.start_trial.return_value = status
        mgr.get_trial_status.return_value = status
        return mgr

    def test_start_trial_success(self, handler):
        # Ensure org has no trial and is free tier
        org = handler.ctx["user_store"].get_organization_by_id("org_1")
        org.trial_started_at = None
        org.tier = SubscriptionTier.FREE

        status = MockTrialStatus(is_active=True, days_remaining=7, debates_remaining=10)
        with patch(
            "aragora.billing.trial_manager.TrialManager", return_value=self._make_trial_mgr(status)
        ), patch(
            "aragora.billing.trial_manager.get_trial_manager",
            return_value=self._make_trial_mgr(status),
        ):
            # _start_trial has no @require_permission, so user=None; provide user_id in body
            http = MockHTTPHandler(body={"user_id": "test-user-001"}, command="POST")
            result = handler.handle("/api/v1/billing/trial/start", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert "trial" in body
        assert "7 days" in body["message"]

    def test_start_trial_already_active(self, handler):
        org = handler.ctx["user_store"].get_organization_by_id("org_1")
        org.trial_started_at = datetime.now(timezone.utc) - timedelta(days=2)
        org.is_in_trial = True
        org.is_trial_expired = False

        status = MockTrialStatus(is_active=True, days_remaining=5)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager",
            return_value=MagicMock(get_trial_status=MagicMock(return_value=status)),
        ):
            http = MockHTTPHandler(body={"user_id": "test-user-001"}, command="POST")
            result = handler.handle("/api/v1/billing/trial/start", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert "already active" in body["message"].lower()

    def test_start_trial_expired(self, handler):
        org = handler.ctx["user_store"].get_organization_by_id("org_1")
        org.trial_started_at = datetime.now(timezone.utc) - timedelta(days=10)
        org.is_in_trial = False
        org.is_trial_expired = True

        status = MockTrialStatus(is_active=False, is_expired=True, days_remaining=0)
        with patch(
            "aragora.billing.trial_manager.get_trial_manager",
            return_value=MagicMock(get_trial_status=MagicMock(return_value=status)),
        ):
            http = MockHTTPHandler(body={"user_id": "test-user-001"}, command="POST")
            result = handler.handle("/api/v1/billing/trial/start", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 403
        assert "expired" in body["message"].lower()

    def test_start_trial_paid_org_returns_400(self, handler):
        org = handler.ctx["user_store"].get_organization_by_id("org_1")
        org.trial_started_at = None
        org.tier = SubscriptionTier.STARTER
        http = MockHTTPHandler(body={"user_id": "test-user-001"}, command="POST")
        result = handler.handle("/api/v1/billing/trial/start", {}, http, method="POST")
        assert _status(result) == 400

    def test_start_trial_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(command="POST")
        result = h.handle("/api/v1/billing/trial/start", {}, http, method="POST")
        assert _status(result) == 404

    def test_start_trial_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler(command="POST")
        result = handler_no_store.handle("/api/v1/billing/trial/start", {}, http, method="POST")
        assert _status(result) == 503

    def test_start_trial_unknown_user_with_body_user_id(self, user_store):
        """When no JWT user, falls back to body user_id."""
        # Remove the auto-auth user so the handler tries the body path
        # The conftest injects user context, so the handler will find test-user-001
        # We test the path where user_store has the user referenced in body
        store = MockUserStore()
        user = MockUser(id="body-user-1", email="body@t.com", role="member", org_id=None)
        store.add_user(user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"user_id": "body-user-1"}, command="POST")
        result = h.handle("/api/v1/billing/trial/start", {}, http, method="POST")
        # User has no org, so should get 404
        assert _status(result) == 404


# ===========================================================================
# TestCreateCheckout
# ===========================================================================


class TestCreateCheckout:
    """Tests for checkout session creation."""

    def test_successful_checkout(self, handler, mock_stripe_client):
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
        assert _status(result) == 200
        assert "checkout" in body
        assert body["checkout"]["id"] == "cs_test_123"

    def test_checkout_invalid_json_returns_400(self, handler):
        http = MockHTTPHandler(command="POST")
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) in (400, 500)

    def test_checkout_missing_tier_returns_400(self, handler):
        http = MockHTTPHandler(
            body={"success_url": "https://a.com/s", "cancel_url": "https://a.com/c"},
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 400

    def test_checkout_free_tier_returns_400(self, handler):
        http = MockHTTPHandler(
            body={"tier": "free", "success_url": "https://a.com/s", "cancel_url": "https://a.com/c"},
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 400

    def test_checkout_invalid_tier_returns_400(self, handler):
        http = MockHTTPHandler(
            body={
                "tier": "nonexistent",
                "success_url": "https://a.com/s",
                "cancel_url": "https://a.com/c",
            },
            command="POST",
        )
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 400

    def test_checkout_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("No API key"),
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://a.com/s",
                    "cancel_url": "https://a.com/c",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 503

    def test_checkout_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeAPIError("Bad request")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://a.com/s",
                    "cancel_url": "https://a.com/c",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 502

    def test_checkout_generic_stripe_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "starter",
                    "success_url": "https://a.com/s",
                    "cancel_url": "https://a.com/c",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 500

    def test_checkout_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler(
            body={
                "tier": "starter",
                "success_url": "https://a.com/s",
                "cancel_url": "https://a.com/c",
            },
            command="POST",
        )
        result = handler_no_store.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 503

    def test_checkout_professional_tier(self, handler, mock_stripe_client):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(
                body={
                    "tier": "professional",
                    "success_url": "https://a.com/s",
                    "cancel_url": "https://a.com/c",
                },
                command="POST",
            )
            result = handler.handle("/api/v1/billing/checkout", {}, http, method="POST")
        assert _status(result) == 200


# ===========================================================================
# TestCreatePortal
# ===========================================================================


class TestCreatePortal:
    """Tests for billing portal session creation."""

    def test_successful_portal(self, handler, mock_stripe_client):
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
        assert _status(result) == 200
        assert "portal" in body
        assert body["portal"]["id"] == "bps_test_123"

    def test_portal_missing_return_url_returns_400(self, handler):
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 400

    def test_portal_no_stripe_customer_returns_404(self, handler):
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_no_stripe")
        )
        store.add_organization(
            MockOrganization(id="org_no_stripe", name="No Stripe", stripe_customer_id=None)
        )
        http = MockHTTPHandler(
            body={"return_url": "https://example.com/billing"}, command="POST"
        )
        result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 404

    def test_portal_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"return_url": "https://example.com/billing"}, command="POST"
        )
        result = h.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 404

    def test_portal_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler(
                body={"return_url": "https://example.com/billing"}, command="POST"
            )
            result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 503

    def test_portal_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.create_portal_session.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={"return_url": "https://example.com/billing"}, command="POST"
            )
            result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 502

    def test_portal_generic_stripe_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.create_portal_session.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(
                body={"return_url": "https://example.com/billing"}, command="POST"
            )
            result = handler.handle("/api/v1/billing/portal", {}, http, method="POST")
        assert _status(result) == 500


# ===========================================================================
# TestCancelSubscription
# ===========================================================================


class TestCancelSubscription:
    """Tests for subscription cancellation."""

    def test_successful_cancel(self, handler, mock_stripe_client):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert "message" in body
        assert "subscription" in body

    def test_cancel_no_subscription_returns_404(self, handler):
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_no_sub")
        )
        store.add_organization(
            MockOrganization(id="org_no_sub", name="No Sub", stripe_subscription_id=None)
        )
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 404

    def test_cancel_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler(command="POST")
        result = handler_no_store.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 503

    def test_cancel_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 502

    def test_cancel_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 503

    def test_cancel_generic_stripe_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 500

    def test_cancel_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(command="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, method="POST")
        assert _status(result) == 404


# ===========================================================================
# TestResumeSubscription
# ===========================================================================


class TestResumeSubscription:
    """Tests for resuming a canceled subscription."""

    def test_successful_resume(self, handler, mock_stripe_client):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_stripe_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert "message" in body
        assert "resumed" in body["message"].lower()

    def test_resume_no_subscription_returns_404(self, handler):
        store = handler.ctx["user_store"]
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_no_sub")
        )
        store.add_organization(
            MockOrganization(id="org_no_sub", name="No Sub", stripe_subscription_id=None)
        )
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert _status(result) == 404

    def test_resume_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert _status(result) == 503

    def test_resume_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.resume_subscription.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert _status(result) == 502

    def test_resume_generic_stripe_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.resume_subscription.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler(command="POST")
            result = handler.handle("/api/v1/billing/resume", {}, http, method="POST")
        assert _status(result) == 500


# ===========================================================================
# TestAuditLog (ReportingMixin)
# ===========================================================================


class TestAuditLog:
    """Tests for the billing audit log endpoint.

    The _get_audit_log method checks ``user.role`` (singular string), but
    the conftest injects an AuthorizationContext whose role info is in the
    ``roles`` *set*.  We monkeypatch a ``role`` attribute onto the override
    context so the handler sees it.
    """

    def _make_enterprise_handler(self, monkeypatch):
        store = MockUserStore()
        store.add_user(
            MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_ent")
        )
        store.add_organization(
            MockOrganization(
                id="org_ent",
                name="Enterprise Org",
                tier=SubscriptionTier.ENTERPRISE,
                limits=MockTierLimits(audit_logs=True),
            )
        )
        # Ensure the _test_user_context_override has a .role attribute
        try:
            from aragora.server.handlers.utils import decorators as _dec

            override = getattr(_dec, "_test_user_context_override", None)
            if override is not None and not hasattr(override, "role"):
                object.__setattr__(override, "role", "owner")
        except (ImportError, AttributeError):
            pass
        return BillingHandler(ctx={"user_store": store})

    def test_audit_log_returns_entries(self, monkeypatch):
        h = self._make_enterprise_handler(monkeypatch)
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert "entries" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_audit_log_non_enterprise_returns_403(self, handler):
        # Default org is FREE tier with audit_logs=False
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 403

    def test_audit_log_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        assert _status(result) == 503

    def test_audit_log_with_query_params(self, monkeypatch):
        h = self._make_enterprise_handler(monkeypatch)
        http = MockHTTPHandler(query_params={"limit": "25", "offset": "10", "action": "subscription.created"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, method="GET")
        body = _body(result)
        assert body["limit"] == 25
        assert body["offset"] == 10


# ===========================================================================
# TestUsageExport (ReportingMixin)
# ===========================================================================


class TestUsageExport:
    """Tests for CSV usage export."""

    def test_export_returns_csv(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        assert result.content_type == "text/csv"
        csv_text = result.body.decode("utf-8")
        assert "Date,Event Type,Count,Metadata" in csv_text
        assert "Summary" in csv_text
        assert "Test Org" in csv_text

    def test_export_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 503

    def test_export_has_content_disposition(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert "Content-Disposition" in result.headers
        assert "usage_export_" in result.headers["Content-Disposition"]

    def test_export_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404


# ===========================================================================
# TestUsageForecast (ReportingMixin)
# ===========================================================================


class TestUsageForecast:
    """Tests for usage forecast endpoint."""

    def test_forecast_returns_projection(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        assert "forecast" in body
        f = body["forecast"]
        assert "current_usage" in f
        assert "projection" in f
        assert "days_remaining" in f
        assert "will_hit_limit" in f

    def test_forecast_with_high_usage_recommends_upgrade(self):
        """An org using lots of debates should get a tier recommendation."""
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_heavy"))
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

    def test_forecast_enterprise_no_recommendation(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_ent"))
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
        # Enterprise tier should not get an upgrade recommendation
        assert body["forecast"]["tier_recommendation"] is None

    def test_forecast_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        assert _status(result) == 503

    def test_forecast_with_usage_tracker(self, user_store):
        tracker = MagicMock()
        summary = MagicMock()
        summary.total_tokens = 100_000
        summary.total_cost = Decimal("5.00")
        tracker.get_summary.return_value = summary
        h = BillingHandler(ctx={"user_store": user_store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, method="GET")
        body = _body(result)
        proj = body["forecast"]["projection"]
        assert proj["tokens_per_day"] > 0
        assert proj["cost_end_of_cycle_usd"] > 0


# ===========================================================================
# TestGetInvoices (ReportingMixin)
# ===========================================================================


class TestGetInvoices:
    """Tests for invoice history endpoint."""

    def test_invoices_returns_list(self, handler):
        invoices_data = [
            {
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
        ]
        client = MockStripeClient(
            subscription=MockStripeSubscription("sub_test_123"),
            invoices=invoices_data,
        )
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
            body = _body(result)
        assert "invoices" in body
        assert len(body["invoices"]) == 1
        inv = body["invoices"][0]
        assert inv["id"] == "inv_001"
        assert inv["amount_due"] == 99.0
        assert inv["currency"] == "USD"

    def test_invoices_no_billing_account_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", role="owner", org_id="org_no_cust"))
        store.add_organization(
            MockOrganization(id="org_no_cust", name="No Cust", stripe_customer_id=None)
        )
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 404

    def test_invoices_stripe_config_error_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("Not configured"),
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 503

    def test_invoices_stripe_api_error_returns_502(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeAPIError("API error")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 502

    def test_invoices_generic_stripe_error_returns_500(self, handler):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeError("Unknown")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 500

    def test_invoices_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = handler_no_store.handle("/api/v1/billing/invoices", {}, http, method="GET")
        assert _status(result) == 503


# ===========================================================================
# TestStripeWebhook (WebhookMixin)
# ===========================================================================


class TestStripeWebhook:
    """Tests for Stripe webhook handling."""

    def test_missing_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST")
        http.headers["Content-Length"] = "100"
        http.rfile.read.return_value = b'{"type": "test"}'
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_invalid_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="invalid_sig")
        http.headers["Content-Length"] = "100"
        http.rfile.read.return_value = b'{"type": "test"}'
        with patch(
            "aragora.billing.stripe_client.parse_webhook_event", return_value=None
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_checkout_completed_event(self, handler):
        event = MockWebhookEvent(
            event_id="evt_001",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new", "id": "cs_1"},
            metadata={"user_id": "test-user-001", "org_id": "org_1", "tier": "starter"},
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b'{"type": "checkout.session.completed"}'
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True

    def test_duplicate_webhook_skipped(self, handler):
        event = MockWebhookEvent(
            event_id="evt_dup",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123"},
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event, is_duplicate=True):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body.get("duplicate") is True

    def test_subscription_deleted_downgrades_to_free(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.STARTER

        event = MockWebhookEvent(
            event_id="evt_del",
            event_type="customer.subscription.deleted",
            object_data={"id": "sub_test_123"},
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert store.get_organization_by_id("org_1").tier == SubscriptionTier.FREE

    def test_unhandled_event_acknowledged(self, handler):
        event = MockWebhookEvent(event_id="evt_unknown", event_type="some.unknown.event")
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "50"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True

    def test_invoice_paid_resets_usage(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.debates_used_this_month = 42

        event = MockWebhookEvent(
            event_id="evt_inv",
            event_type="invoice.payment_succeeded",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "amount_paid": 9900,
            },
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event), patch(
            "aragora.billing.payment_recovery.get_recovery_store"
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert store.get_organization_by_id("org_1").debates_used_this_month == 0

    def test_invoice_failed_tracks_failure(self, handler):
        event = MockWebhookEvent(
            event_id="evt_fail",
            event_type="invoice.payment_failed",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "attempt_count": 2,
                "id": "inv_fail_1",
                "hosted_invoice_url": "https://inv.stripe.com/fail",
            },
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"

        mock_failure = MagicMock()
        mock_failure.attempt_count = 2
        mock_failure.days_failing = 5
        mock_failure.days_until_downgrade = 9

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = mock_failure
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with _webhook_patches(event), patch(
            "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
        ), patch(
            "aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["failure_tracked"] is True

    def test_invoice_finalized_flushes_usage(self, handler):
        event = MockWebhookEvent(
            event_id="evt_fin",
            event_type="invoice.finalized",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
            },
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"

        mock_sync = MagicMock()
        mock_sync.flush_period.return_value = [{"id": "rec_1"}]

        with _webhook_patches(event), patch(
            "aragora.billing.usage_sync.get_usage_sync_service", return_value=mock_sync
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["usage_flushed"] == 1

    def test_subscription_created_acknowledged(self, handler):
        event = MockWebhookEvent(
            event_id="evt_sub_created",
            event_type="customer.subscription.created",
            object_data={"id": "sub_new"},
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "100"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True

    def test_subscription_updated_syncs_tier(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = MockWebhookEvent(
            event_id="evt_sub_upd",
            event_type="customer.subscription.updated",
            object_data={
                "id": "sub_test_123",
                "status": "active",
                "cancel_at_period_end": False,
                "items": {
                    "data": [{"price": {"id": "price_professional"}}]
                },
            },
        )
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "200"
        http.rfile.read.return_value = b"{}"

        with _webhook_patches(event), patch(
            "aragora.billing.stripe_client.get_tier_from_price_id",
            return_value=SubscriptionTier.PROFESSIONAL,
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert store.get_organization_by_id("org_1").tier == SubscriptionTier.PROFESSIONAL


# ===========================================================================
# TestRateLimiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_not_applied_to_webhooks(self, handler):
        event = MockWebhookEvent(event_id="evt_rate", event_type="some.event")
        http = MockHTTPHandler(command="POST", signature="valid")
        http.headers["Content-Length"] = "50"
        http.rfile.read.return_value = b"{}"
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) != 429

    def test_rate_limit_applied_to_billing_endpoints(self, handler):
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        with patch(
            "aragora.server.handlers.billing.core._get_billing_limiter",
            return_value=mock_limiter,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
        assert _status(result) == 429

    def test_rate_limit_returns_429_body(self, handler):
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        with patch(
            "aragora.server.handlers.billing.core._get_billing_limiter",
            return_value=mock_limiter,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/billing/plans", {}, http, method="GET")
            body = _body(result)
        assert "rate limit" in body.get("error", "").lower()


# ===========================================================================
# TestMethodNotAllowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method-not-allowed responses."""

    def test_post_to_plans_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/plans", {}, http, method="POST")
        assert _status(result) == 405

    def test_get_to_checkout_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/checkout", {}, http, method="GET")
        assert _status(result) == 405

    def test_get_to_cancel_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/cancel", {}, http, method="GET")
        assert _status(result) == 405

    def test_get_to_resume_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/resume", {}, http, method="GET")
        assert _status(result) == 405

    def test_get_to_portal_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/portal", {}, http, method="GET")
        assert _status(result) == 405

    def test_post_to_usage_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/usage", {}, http, method="POST")
        assert _status(result) == 405

    def test_get_to_webhook_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="GET")
        assert _status(result) == 405

    def test_get_to_trial_start_returns_405(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/billing/trial/start", {}, http, method="GET")
        assert _status(result) == 405

    def test_post_to_trial_get_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/trial", {}, http, method="POST")
        assert _status(result) == 405

    def test_post_to_subscription_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = handler.handle("/api/v1/billing/subscription", {}, http, method="POST")
        assert _status(result) == 405


# ===========================================================================
# TestHandlerInit
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization and context."""

    def test_default_context_is_empty_dict(self):
        h = BillingHandler()
        assert h.ctx == {}

    def test_context_passed_through(self):
        ctx = {"user_store": MagicMock(), "usage_tracker": MagicMock()}
        h = BillingHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_get_user_store_returns_from_context(self, handler, user_store):
        assert handler._get_user_store() is user_store

    def test_get_user_store_returns_none_when_missing(self, handler_no_store):
        assert handler_no_store._get_user_store() is None

    def test_get_usage_tracker_returns_from_context(self):
        tracker = MagicMock()
        h = BillingHandler(ctx={"usage_tracker": tracker})
        assert h._get_usage_tracker() is tracker

    def test_get_usage_tracker_returns_none_when_missing(self, handler):
        assert handler._get_usage_tracker() is None

    def test_resource_type_is_billing(self, handler):
        assert handler.RESOURCE_TYPE == "billing"

    def test_routes_list_has_14_entries(self, handler):
        assert len(handler.ROUTES) == 14


# ===========================================================================
# TestLogAudit
# ===========================================================================


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
            None, action="test.action", resource_type="subscription"
        )

    def test_log_audit_with_store_missing_method(self, handler):
        store = MagicMock(spec=[])
        handler._log_audit(store, action="test.action", resource_type="subscription")

    def test_log_audit_with_handler_extracts_ip_and_ua(self, handler):
        store = MagicMock()
        store.log_audit_event = MagicMock()
        http = MockHTTPHandler()

        with patch(
            "aragora.server.middleware.auth.extract_client_ip", return_value="10.0.0.1"
        ):
            handler._log_audit(
                store,
                action="test.action",
                resource_type="subscription",
                handler=http,
            )
        call_kwargs = store.log_audit_event.call_args[1]
        assert call_kwargs["ip_address"] == "10.0.0.1"
        assert call_kwargs["user_agent"] == "test-agent"

    def test_log_audit_exception_does_not_propagate(self, handler):
        store = MagicMock()
        store.log_audit_event.side_effect = OSError("disk full")
        # Should not raise, just log a warning
        handler._log_audit(
            store, action="test.action", resource_type="subscription"
        )


# ===========================================================================
# TestGetBillingLimiter
# ===========================================================================


class TestGetBillingLimiter:
    """Tests for the _get_billing_limiter compatibility function."""

    def test_returns_module_level_limiter_by_default(self):
        from aragora.server.handlers.billing.core import _get_billing_limiter

        limiter = _get_billing_limiter()
        assert limiter is _billing_limiter

    def test_returns_admin_billing_limiter_when_not_rate_limiter(self):
        from aragora.server.handlers.billing.core import _get_billing_limiter
        import sys

        mock_module = MagicMock()
        mock_module._billing_limiter = "not_a_real_limiter"
        with patch.dict(sys.modules, {"aragora.server.handlers.admin.billing": mock_module}):
            limiter = _get_billing_limiter()
        assert limiter == "not_a_real_limiter"
