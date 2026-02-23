"""Tests for BillingHandler in admin/billing.py.

Comprehensive coverage of all billing endpoints:
- GET  /api/v1/billing/plans           (_get_plans)
- GET  /api/v1/billing/usage           (_get_usage)
- GET  /api/v1/billing/subscription    (_get_subscription)
- POST /api/v1/billing/checkout        (_create_checkout)
- POST /api/v1/billing/portal          (_create_portal)
- POST /api/v1/billing/cancel          (_cancel_subscription)
- POST /api/v1/billing/resume          (_resume_subscription)
- GET  /api/v1/billing/audit-log       (_get_audit_log)
- GET  /api/v1/billing/usage/export    (_export_usage_csv)
- GET  /api/v1/billing/usage/forecast  (_get_usage_forecast)
- GET  /api/v1/billing/invoices        (_get_invoices)
- POST /api/v1/webhooks/stripe         (_handle_stripe_webhook)

Also covers:
- Rate limiting (429)
- Method not allowed (405)
- can_handle routing
- Webhook idempotency
- Webhook event type dispatch (checkout, subscription CRUD, invoice events)
- Stripe error variants (StripeConfigError, StripeAPIError, StripeError)
- Audit logging
- Usage forecast with tier recommendations
- CSV export
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import TIER_LIMITS, SubscriptionTier, TierLimits
from aragora.billing.stripe_client import (
    StripeAPIError,
    StripeConfigError,
    StripeError,
)
from aragora.server.handlers.admin.billing import BillingHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return json.loads(result.body)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


# ===========================================================================
# Mock objects
# ===========================================================================


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-001"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: SubscriptionTier = SubscriptionTier.PROFESSIONAL
    stripe_customer_id: str | None = "cus_test123"
    stripe_subscription_id: str | None = "sub_test123"
    debates_used_this_month: int = 15
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=10)
    )
    limits: TierLimits = field(
        default_factory=lambda: TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
    )

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)


@dataclass
class MockDbUser:
    """Mock database user."""

    id: str = "test-user-001"
    email: str = "test@example.com"
    org_id: str | None = "org-001"


@dataclass
class MockUsageSummary:
    """Mock usage tracker summary."""

    total_tokens_in: int = 50000
    total_tokens_out: int = 10000
    total_tokens: int = 60000
    total_cost_usd: float = 1.50
    total_cost: float = 1.50
    cost_by_provider: dict = field(default_factory=lambda: {"anthropic": "1.00", "openai": "0.50"})


@dataclass
class MockStripeSubscription:
    """Mock Stripe subscription."""

    id: str = "sub_test123"
    customer_id: str = "cus_test123"
    status: str = "active"
    price_id: str = "price_prof_123"
    current_period_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=10)
    )
    current_period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=20)
    )
    cancel_at_period_end: bool = False
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    is_trialing: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }


@dataclass
class MockCheckoutSession:
    """Mock Stripe checkout session."""

    id: str = "cs_test123"
    url: str = "https://checkout.stripe.com/session/cs_test123"

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


@dataclass
class MockPortalSession:
    """Mock Stripe portal session."""

    id: str = "bps_test123"
    url: str = "https://billing.stripe.com/session/bps_test123"

    def to_dict(self) -> dict:
        return {"id": self.id, "url": self.url}


class MockHTTPHandler:
    """Mock HTTP handler with body and header support."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        query_params: dict | None = None,
    ):
        self.command = method
        self.headers = {"Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)
        self.query_params = query_params or {}
        self.rfile = MagicMock()
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"

    def get(self, key, default=None):
        """Allow this to work with _resolve_params for get_string_param."""
        return self.query_params.get(key, default)


class MockUserStore:
    """Mock user store."""

    def __init__(self, user=None, org=None, owner=None):
        self._user = user or MockDbUser()
        self._org = org or MockOrg()
        self._owner = owner
        self._audit_events = []

    def get_user_by_id(self, user_id):
        if self._user and self._user.id == user_id:
            return self._user
        return None

    def get_organization_by_id(self, org_id):
        if self._org and self._org.id == org_id:
            return self._org
        return None

    def get_organization_by_subscription(self, subscription_id):
        if self._org and self._org.stripe_subscription_id == subscription_id:
            return self._org
        return None

    def get_organization_by_stripe_customer(self, customer_id):
        if self._org and self._org.stripe_customer_id == customer_id:
            return self._org
        return None

    def update_organization(self, org_id, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._org, k):
                object.__setattr__(self._org, k, v)

    def reset_org_usage(self, org_id):
        if self._org and self._org.id == org_id:
            object.__setattr__(self._org, "debates_used_this_month", 0)

    def get_organization_owner(self, org_id):
        return self._owner

    def log_audit_event(self, **kwargs):
        self._audit_events.append(kwargs)

    def get_audit_log(self, **kwargs):
        return [{"action": "subscription.created", "timestamp": "2026-01-01T00:00:00"}]

    def get_audit_log_count(self, **kwargs):
        return 1

    def has_attr(self, attr):
        return hasattr(self, attr)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _add_role_to_auth_context(mock_auth_for_handler_tests):
    """Ensure the auto-mock auth context has a .role attribute.

    The BillingHandler._get_audit_log checks user.role (singular),
    but the conftest provides AuthorizationContext which only has roles (set).
    """
    ctx = mock_auth_for_handler_tests
    if ctx is not None and not hasattr(ctx, "role"):
        object.__setattr__(ctx, "role", "admin")
    yield


@pytest.fixture
def user_store():
    return MockUserStore()


@pytest.fixture
def handler(user_store):
    return BillingHandler(ctx={"user_store": user_store})


@pytest.fixture
def http_handler():
    return MockHTTPHandler()


@pytest.fixture
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    from aragora.server.handlers.admin.billing import _billing_limiter

    _billing_limiter._requests.clear()
    yield
    _billing_limiter._requests.clear()


# ===========================================================================
# can_handle tests
# ===========================================================================


class TestCanHandle:
    def test_known_routes(self, handler):
        for route in BillingHandler.ROUTES:
            assert handler.can_handle(route), f"Should handle {route}"

    def test_unknown_route(self, handler):
        assert not handler.can_handle("/api/v1/unknown")
        assert not handler.can_handle("/api/v1/billing/nonexistent")

    def test_partial_route(self, handler):
        assert not handler.can_handle("/api/v1/billing")


# ===========================================================================
# GET /api/v1/billing/plans
# ===========================================================================


class TestGetPlans:
    def test_success(self, handler, http_handler, reset_rate_limiter):
        result = handler.handle("/api/v1/billing/plans", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert "plans" in body
        plans = body["plans"]
        assert len(plans) == len(SubscriptionTier)
        plan_ids = {p["id"] for p in plans}
        for tier in SubscriptionTier:
            assert tier.value in plan_ids

    def test_plan_structure(self, handler, http_handler, reset_rate_limiter):
        result = handler.handle("/api/v1/billing/plans", {}, http_handler, "GET")
        body = _body(result)
        plan = body["plans"][0]
        assert "id" in plan
        assert "name" in plan
        assert "price_monthly_cents" in plan
        assert "price_monthly" in plan
        assert "features" in plan
        features = plan["features"]
        assert "debates_per_month" in features
        assert "users_per_org" in features
        assert "api_access" in features

    def test_free_plan_price(self, handler, http_handler, reset_rate_limiter):
        result = handler.handle("/api/v1/billing/plans", {}, http_handler, "GET")
        body = _body(result)
        free_plan = [p for p in body["plans"] if p["id"] == "free"][0]
        assert free_plan["price_monthly_cents"] == 0
        assert free_plan["price_monthly"] == "$0.00"


# ===========================================================================
# GET /api/v1/billing/usage
# ===========================================================================


class TestGetUsage:
    def test_success_with_org(self, handler, http_handler, reset_rate_limiter):
        result = handler.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert "usage" in body
        usage = body["usage"]
        assert usage["debates_used"] == 15
        assert usage["debates_limit"] == 200

    def test_no_user_store(self, http_handler, reset_rate_limiter):
        h = BillingHandler(ctx={})
        result = h.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        assert _status(result) == 503

    def test_user_not_found(self, http_handler, reset_rate_limiter):
        store = MockUserStore()
        store._user = None
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        # The user store returns None for unknown user_id, leading to 404
        assert _status(result) == 404

    def test_user_without_org(self, http_handler, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        # Default usage when no org
        assert body["usage"]["debates_used"] == 0
        assert body["usage"]["debates_limit"] == 10

    def test_with_usage_tracker(self, http_handler, reset_rate_limiter):
        tracker = MagicMock()
        summary = MockUsageSummary()
        tracker.get_summary.return_value = summary
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store, "usage_tracker": tracker})
        result = h.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        usage = body["usage"]
        assert usage["tokens_in"] == 50000
        assert usage["tokens_out"] == 10000
        assert usage["tokens_used"] == 60000
        assert usage["estimated_cost_usd"] == 1.5
        assert "cost_breakdown" in usage
        assert usage["cost_breakdown"]["total"] == 1.5
        assert "cost_by_provider" in usage

    def test_with_usage_tracker_no_summary(self, http_handler, reset_rate_limiter):
        tracker = MagicMock()
        tracker.get_summary.return_value = None
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store, "usage_tracker": tracker})
        result = h.handle("/api/v1/billing/usage", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["usage"]["tokens_used"] == 0


# ===========================================================================
# GET /api/v1/billing/subscription
# ===========================================================================


class TestGetSubscription:
    def test_success_no_stripe(self, http_handler, reset_rate_limiter):
        org = MockOrg(stripe_subscription_id=None)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        sub = body["subscription"]
        assert sub["tier"] == "professional"
        assert "organization" in sub

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_success_with_stripe(self, mock_stripe_fn, http_handler, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = MockStripeSubscription()
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        sub = body["subscription"]
        assert sub["status"] == "active"
        assert sub["is_active"] is True
        assert sub["cancel_at_period_end"] is False
        assert sub["payment_failed"] is False

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_trialing_subscription(self, mock_stripe_fn, http_handler, reset_rate_limiter):
        stripe_sub = MockStripeSubscription(
            status="trialing",
            is_trialing=True,
            trial_start=datetime.now(timezone.utc) - timedelta(days=5),
            trial_end=datetime.now(timezone.utc) + timedelta(days=9),
        )
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = stripe_sub
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        sub = body["subscription"]
        assert sub["is_trialing"] is True
        assert "trial_start" in sub
        assert "trial_end" in sub

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_past_due_subscription(self, mock_stripe_fn, http_handler, reset_rate_limiter):
        stripe_sub = MockStripeSubscription(status="past_due")
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = stripe_sub
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        sub = body["subscription"]
        assert sub["payment_failed"] is True
        assert sub["is_active"] is False

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_error_graceful_degradation(self, mock_stripe_fn, http_handler, reset_rate_limiter):
        mock_stripe_fn.side_effect = StripeError("connection failed")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        # Should still return 200 with partial data
        assert _status(result) == 200
        body = _body(result)
        assert body["subscription"]["tier"] == "professional"

    def test_no_user_store(self, http_handler, reset_rate_limiter):
        h = BillingHandler(ctx={})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 503

    def test_user_not_found(self, http_handler, reset_rate_limiter):
        store = MockUserStore()
        store._user = None
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 404

    def test_user_without_org(self, http_handler, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        result = h.handle("/api/v1/billing/subscription", {}, http_handler, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["subscription"]["tier"] == "free"


# ===========================================================================
# POST /api/v1/billing/checkout
# ===========================================================================


class TestCreateCheckout:
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    @patch("aragora.server.handlers.admin.billing.audit_data")
    def test_success(self, mock_audit, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.create_checkout_session.return_value = MockCheckoutSession()
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={
                "tier": "starter",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 200
        body = _body(result)
        assert "checkout" in body
        assert body["checkout"]["id"] == "cs_test123"
        mock_audit.assert_called_once()

    def test_invalid_json_body(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(method="POST")
        http.rfile.read.side_effect = ValueError("bad read")
        http.headers["Content-Length"] = "invalid"
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 400

    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_schema_validation_failure(self, mock_validate, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.error = "Missing required field: tier"
        mock_validate.return_value = mock_result
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"success_url": "x"}, method="POST")
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 400

    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_invalid_tier(self, mock_validate, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"tier": "invalid_tier", "success_url": "x", "cancel_url": "y"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 400
        assert "Invalid tier" in _body(result).get("error", "")

    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_free_tier_rejected(self, mock_validate, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"tier": "free", "success_url": "x", "cancel_url": "y"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 400
        assert "free" in _body(result).get("error", "").lower()

    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_no_user_store(self, mock_validate, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        h = BillingHandler(ctx={})
        http = MockHTTPHandler(
            body={"tier": "starter", "success_url": "x", "cancel_url": "y"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_stripe_config_error(self, mock_validate, mock_stripe_fn, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        mock_stripe_fn.side_effect = StripeConfigError("missing key")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"tier": "starter", "success_url": "x", "cancel_url": "y"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.billing.validate_against_schema")
    def test_user_not_found(self, mock_validate, reset_rate_limiter):
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_validate.return_value = mock_result
        store = MockUserStore()
        store._user = None
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"tier": "starter", "success_url": "x", "cancel_url": "y"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/checkout", {}, http, "POST")
        assert _status(result) == 404


# ===========================================================================
# POST /api/v1/billing/portal
# ===========================================================================


class TestCreatePortal:
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_success(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.create_portal_session.return_value = MockPortalSession()
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(
            body={"return_url": "https://example.com/billing"},
            method="POST",
        )
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 200
        body = _body(result)
        assert "portal" in body
        assert body["portal"]["id"] == "bps_test123"

    def test_missing_return_url(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"return_url": ""}, method="POST")
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 400
        assert "Return URL" in _body(result).get("error", "")

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler(body={"return_url": "https://x.com"}, method="POST")
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 503

    def test_user_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"return_url": "https://x.com"}, method="POST")
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 404

    def test_org_no_stripe_customer(self, reset_rate_limiter):
        org = MockOrg(stripe_customer_id=None)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"return_url": "https://x.com"}, method="POST")
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 404

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_config_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_stripe_fn.side_effect = StripeConfigError("missing key")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={"return_url": "https://x.com"}, method="POST")
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 503

    def test_invalid_json_body(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(method="POST")
        http.headers["Content-Length"] = "0"
        result = h.handle("/api/v1/billing/portal", {}, http, "POST")
        assert _status(result) == 400


# ===========================================================================
# POST /api/v1/billing/cancel
# ===========================================================================


class TestCancelSubscription:
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    @patch("aragora.server.handlers.admin.billing.audit_admin")
    def test_success(self, mock_audit, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_sub = MockStripeSubscription(cancel_at_period_end=True)
        mock_client.cancel_subscription.return_value = mock_sub
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 200
        body = _body(result)
        assert "subscription" in body
        assert "canceled at period end" in body.get("message", "").lower()
        mock_audit.assert_called_once()

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 503

    def test_no_subscription(self, reset_rate_limiter):
        org = MockOrg(stripe_subscription_id=None)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 404

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_config_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_stripe_fn.side_effect = StripeConfigError("missing")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_api_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeAPIError("api fail")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 502

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_generic_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeError("unknown")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 500

    def test_user_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/cancel", {}, http, "POST")
        assert _status(result) == 404


# ===========================================================================
# POST /api/v1/billing/resume
# ===========================================================================


class TestResumeSubscription:
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    @patch("aragora.server.handlers.admin.billing.audit_admin")
    def test_success(self, mock_audit, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_sub = MockStripeSubscription(cancel_at_period_end=False)
        mock_client.resume_subscription.return_value = mock_sub
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 200
        body = _body(result)
        assert "subscription" in body
        assert "resumed" in body.get("message", "").lower()

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 503

    def test_no_subscription_to_resume(self, reset_rate_limiter):
        org = MockOrg(stripe_subscription_id=None)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 404

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_config_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_stripe_fn.side_effect = StripeConfigError("missing")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_api_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.resume_subscription.side_effect = StripeAPIError("api fail")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 502

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_generic_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.resume_subscription.side_effect = StripeError("unknown")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(body={}, method="POST")
        result = h.handle("/api/v1/billing/resume", {}, http, "POST")
        assert _status(result) == 500


# ===========================================================================
# GET /api/v1/billing/audit-log
# ===========================================================================


class TestGetAuditLog:
    def test_success(self, reset_rate_limiter):
        org = MockOrg(tier=SubscriptionTier.ENTERPRISE)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "10", "offset": "0"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert "entries" in body
        assert "total" in body
        assert body["total"] == 1

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 503

    def test_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 404

    def test_audit_logs_not_in_tier(self, reset_rate_limiter):
        org = MockOrg(tier=SubscriptionTier.FREE, limits=TIER_LIMITS[SubscriptionTier.FREE])
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 403
        assert "Enterprise" in _body(result).get("error", "")

    def test_limit_clamped_to_100(self, reset_rate_limiter):
        org = MockOrg(tier=SubscriptionTier.ENTERPRISE)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "500"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 100

    def test_with_action_filter(self, reset_rate_limiter):
        org = MockOrg(tier=SubscriptionTier.ENTERPRISE)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"action": "subscription.created"})
        result = h.handle("/api/v1/billing/audit-log", {}, http, "GET")
        assert _status(result) == 200


# ===========================================================================
# GET /api/v1/billing/usage/export
# ===========================================================================


class TestExportUsageCsv:
    def test_success(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, "GET")
        assert _status(result) == 200
        assert result.content_type == "text/csv"
        assert result.headers.get("Content-Disposition", "").startswith("attachment")
        # Parse CSV body
        csv_text = result.body.decode("utf-8")
        assert "Date" in csv_text
        assert "Summary" in csv_text
        assert "Test Org" in csv_text

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, "GET")
        assert _status(result) == 503

    def test_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, "GET")
        assert _status(result) == 404

    def test_org_not_found(self, reset_rate_limiter):
        store = MockUserStore()
        store._org = None
        user = MockDbUser(org_id="org-missing")
        store._user = user
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, "GET")
        assert _status(result) == 404

    def test_filename_format(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/export", {}, http, "GET")
        disposition = result.headers.get("Content-Disposition", "")
        assert "usage_export_test-org_" in disposition
        assert ".csv" in disposition


# ===========================================================================
# GET /api/v1/billing/usage/forecast
# ===========================================================================


class TestGetUsageForecast:
    def test_success(self, reset_rate_limiter):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        forecast = body["forecast"]
        assert "current_usage" in forecast
        assert "projection" in forecast
        assert "days_remaining" in forecast
        assert "days_elapsed" in forecast

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 503

    def test_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 404

    def test_tier_recommendation_when_hitting_limit(self, reset_rate_limiter):
        # Free tier with heavy usage should get upgrade recommendation
        org = MockOrg(
            tier=SubscriptionTier.FREE,
            limits=TIER_LIMITS[SubscriptionTier.FREE],
            debates_used_this_month=9,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=5),
        )
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        forecast = body["forecast"]
        assert forecast["will_hit_limit"] is True
        assert forecast["tier_recommendation"] is not None
        assert forecast["tier_recommendation"]["recommended_tier"] == "starter"

    def test_no_tier_recommendation_when_under_limit(self, reset_rate_limiter):
        org = MockOrg(
            tier=SubscriptionTier.ENTERPRISE,
            limits=TIER_LIMITS[SubscriptionTier.ENTERPRISE],
            debates_used_this_month=5,
            billing_cycle_start=datetime.now(timezone.utc) - timedelta(days=10),
        )
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["forecast"]["tier_recommendation"] is None

    def test_with_usage_tracker(self, reset_rate_limiter):
        tracker = MagicMock()
        summary = MagicMock()
        summary.total_tokens = 60000
        summary.total_cost = 1.50
        tracker.get_summary.return_value = summary
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store, "usage_tracker": tracker})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        forecast = body["forecast"]
        assert forecast["projection"]["cost_end_of_cycle_usd"] > 0

    def test_early_billing_cycle(self, reset_rate_limiter):
        """Day 0 of billing cycle should not crash on division by zero."""
        org = MockOrg(
            billing_cycle_start=datetime.now(timezone.utc),
            debates_used_this_month=0,
        )
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/usage/forecast", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["forecast"]["days_elapsed"] >= 1


# ===========================================================================
# GET /api/v1/billing/invoices
# ===========================================================================


class TestGetInvoices:
    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_success(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = [
            {
                "id": "inv_123",
                "number": "INV-001",
                "status": "paid",
                "amount_due": 29900,
                "amount_paid": 29900,
                "currency": "usd",
                "created": 1700000000,
                "period_start": 1700000000,
                "period_end": 1702592000,
                "hosted_invoice_url": "https://pay.stripe.com/inv_123",
                "invoice_pdf": "https://pay.stripe.com/inv_123.pdf",
            }
        ]
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "5"})
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 200
        body = _body(result)
        assert len(body["invoices"]) == 1
        inv = body["invoices"][0]
        assert inv["id"] == "inv_123"
        assert inv["amount_due"] == 299.0
        assert inv["currency"] == "USD"

    def test_no_user_store(self, reset_rate_limiter):
        h = BillingHandler(ctx={})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 503

    def test_no_org(self, reset_rate_limiter):
        user = MockDbUser(org_id=None)
        store = MockUserStore(user=user)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 404

    def test_no_stripe_customer(self, reset_rate_limiter):
        org = MockOrg(stripe_customer_id=None)
        store = MockUserStore(org=org)
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 404

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_config_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_stripe_fn.side_effect = StripeConfigError("missing key")
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_api_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeAPIError("api fail")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 502

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_stripe_generic_error(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeError("unknown")
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 500

    @patch("aragora.server.handlers.admin.billing.get_stripe_client")
    def test_limit_clamped_to_100(self, mock_stripe_fn, reset_rate_limiter):
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        mock_stripe_fn.return_value = mock_client
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(query_params={"limit": "500"})
        result = h.handle("/api/v1/billing/invoices", {}, http, "GET")
        assert _status(result) == 200
        # Verify the clamped limit was passed to Stripe
        mock_client.list_invoices.assert_called_once_with(
            customer_id="cus_test123", limit=100
        )


# ===========================================================================
# POST /api/v1/webhooks/stripe
# ===========================================================================


class TestStripeWebhook:
    def _make_webhook_handler(self, payload: bytes, signature: str = "sig_test"):
        """Create a mock handler for webhook requests."""
        h = MockHTTPHandler(method="POST")
        h.rfile.read.return_value = payload
        h.headers["Content-Length"] = str(len(payload))
        h.headers["Stripe-Signature"] = signature
        return h

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_checkout_completed(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "checkout.session.completed"
        event.event_id = "evt_123"
        event.object = {"customer": "cus_123", "subscription": "sub_123", "id": "cs_123"}
        event.metadata = {"user_id": "user-1", "org_id": "org-001", "tier": "starter"}
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        payload = b'{"type": "checkout.session.completed"}'
        http = self._make_webhook_handler(payload)
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200
        assert _body(result)["received"] is True
        mock_mark.assert_called_once_with("evt_123")

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_missing_signature(self, mock_parse):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(method="POST")
        http.headers["Content-Length"] = "2"
        http.headers["Stripe-Signature"] = ""
        http.rfile.read.return_value = b"{}"
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 400
        assert "signature" in _body(result).get("error", "").lower()

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_invalid_signature(self, mock_parse):
        mock_parse.return_value = None
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{"data": {}}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 400
        assert "signature" in _body(result).get("error", "").lower()

    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_duplicate_webhook(self, mock_parse, mock_is_dup):
        mock_is_dup.return_value = True
        event = MagicMock()
        event.event_id = "evt_dup"
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{"data": {}}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200
        assert _body(result)["duplicate"] is True

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_subscription_created(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "customer.subscription.created"
        event.event_id = "evt_sub_create"
        event.subscription_id = "sub_new"
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.billing.stripe_client.get_tier_from_price_id")
    def test_subscription_updated_with_tier_change(
        self, mock_get_tier, mock_parse, mock_is_dup, mock_mark
    ):
        mock_is_dup.return_value = False
        mock_get_tier.return_value = SubscriptionTier.ENTERPRISE
        event = MagicMock()
        event.type = "customer.subscription.updated"
        event.event_id = "evt_sub_update"
        event.object = {
            "id": "sub_test123",
            "status": "active",
            "cancel_at_period_end": False,
            "items": {"data": [{"price": {"id": "price_ent"}}]},
        }
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_subscription_deleted(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "customer.subscription.deleted"
        event.event_id = "evt_sub_del"
        event.object = {"id": "sub_test123"}
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200
        # Check org was downgraded to free
        assert store._org.tier == SubscriptionTier.FREE

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_invoice_paid(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "invoice.payment_succeeded"
        event.event_id = "evt_inv_paid"
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
            "amount_paid": 29900,
        }
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})

        with patch(
            "aragora.billing.payment_recovery.get_recovery_store"
        ) as mock_get_recovery:
            mock_recovery = MagicMock()
            mock_recovery.mark_recovered.return_value = True
            mock_get_recovery.return_value = mock_recovery

            http = self._make_webhook_handler(b'{}')
            result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
            assert _status(result) == 200
            # Usage should have been reset
            assert store._org.debates_used_this_month == 0

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_invoice_failed(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "invoice.payment_failed"
        event.event_id = "evt_inv_fail"
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
            "attempt_count": 2,
            "id": "inv_fail123",
            "hosted_invoice_url": "https://pay.stripe.com/inv_fail123",
        }
        mock_parse.return_value = event

        mock_failure = MagicMock()
        mock_failure.attempt_count = 2
        mock_failure.days_failing = 3
        mock_failure.days_until_downgrade = 11

        owner = MagicMock()
        owner.email = "owner@example.com"
        store = MockUserStore(owner=owner)
        h = BillingHandler(ctx={"user_store": store})

        with patch(
            "aragora.billing.payment_recovery.get_recovery_store"
        ) as mock_get_recovery, patch(
            "aragora.billing.notifications.get_billing_notifier"
        ) as mock_get_notifier:
            mock_recovery = MagicMock()
            mock_recovery.record_failure.return_value = mock_failure
            mock_get_recovery.return_value = mock_recovery

            mock_notifier = MagicMock()
            mock_notifier_result = MagicMock()
            mock_notifier_result.method = "email"
            mock_notifier_result.success = True
            mock_notifier.notify_payment_failed.return_value = mock_notifier_result
            mock_get_notifier.return_value = mock_notifier

            http = self._make_webhook_handler(b'{}')
            result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
            assert _status(result) == 200
            body = _body(result)
            assert body["failure_tracked"] is True

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_invoice_finalized(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "invoice.finalized"
        event.event_id = "evt_inv_final"
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
        }
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})

        with patch(
            "aragora.billing.usage_sync.get_usage_sync_service"
        ) as mock_get_sync:
            mock_sync = MagicMock()
            mock_sync.flush_period.return_value = ["record1", "record2"]
            mock_get_sync.return_value = mock_sync

            http = self._make_webhook_handler(b'{}')
            result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
            assert _status(result) == 200
            body = _body(result)
            assert body["usage_flushed"] == 2

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_unhandled_event_type(self, mock_parse, mock_is_dup, mock_mark):
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "charge.succeeded"
        event.event_id = "evt_unknown"
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200
        assert _body(result)["received"] is True

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_content_length_too_large(self, mock_parse):
        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = MockHTTPHandler(method="POST")
        # Set content length larger than 1MB
        http.headers["Content-Length"] = str(2 * 1024 * 1024)
        http.headers["Stripe-Signature"] = "sig_test"
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 400

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_webhook_no_event_id(self, mock_parse, mock_is_dup, mock_mark):
        event = MagicMock()
        event.type = "charge.succeeded"
        event.event_id = ""
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})
        http = self._make_webhook_handler(b'{}')
        result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        assert _status(result) == 200
        # Should not check for duplicates or mark processed
        mock_is_dup.assert_not_called()
        mock_mark.assert_not_called()

    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_invoice_finalized_flush_error(self, mock_parse, mock_is_dup, mock_mark):
        """Usage flush error should not prevent webhook acknowledgment."""
        mock_is_dup.return_value = False
        event = MagicMock()
        event.type = "invoice.finalized"
        event.event_id = "evt_inv_final_err"
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test123",
        }
        mock_parse.return_value = event

        store = MockUserStore()
        h = BillingHandler(ctx={"user_store": store})

        with patch(
            "aragora.billing.usage_sync.get_usage_sync_service"
        ) as mock_get_sync:
            mock_get_sync.side_effect = RuntimeError("sync unavailable")

            http = self._make_webhook_handler(b'{}')
            result = h.handle("/api/v1/webhooks/stripe", {}, http, "POST")
            assert _status(result) == 200
            body = _body(result)
            assert body["usage_flushed"] == 0


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_applies_to_billing_endpoints(self, handler, reset_rate_limiter):
        """Rate limit should apply to billing endpoints but not webhooks."""
        from aragora.server.handlers.admin.billing import _billing_limiter

        # Exhaust the rate limit
        ip = "192.168.1.1"
        http = MockHTTPHandler()
        http.client_address = (ip, 12345)
        # Fill up the limiter
        for _ in range(25):
            _billing_limiter.is_allowed(ip)

        result = handler.handle("/api/v1/billing/plans", {}, http, "GET")
        assert _status(result) == 429

    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.billing.stripe_client.parse_webhook_event")
    def test_rate_limit_skipped_for_webhooks(self, mock_parse, mock_is_dup, handler, reset_rate_limiter):
        """Webhooks should bypass rate limiting."""
        from aragora.server.handlers.admin.billing import _billing_limiter

        mock_parse.return_value = None  # Invalid signature
        ip = "192.168.1.2"
        http = MockHTTPHandler(method="POST")
        http.client_address = (ip, 12345)
        http.headers["Content-Length"] = "2"
        http.headers["Stripe-Signature"] = "sig"
        http.rfile.read.return_value = b"{}"

        # Exhaust the rate limit
        for _ in range(25):
            _billing_limiter.is_allowed(ip)

        # Webhook should still go through (to signature validation at least)
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, "POST")
        # Should get 400 (invalid signature) not 429 (rate limited)
        assert _status(result) == 400


# ===========================================================================
# Method routing
# ===========================================================================


class TestMethodRouting:
    def test_method_not_allowed(self, handler, reset_rate_limiter):
        http = MockHTTPHandler(method="DELETE")
        result = handler.handle("/api/v1/billing/plans", {}, http, "DELETE")
        assert _status(result) == 405

    def test_post_on_get_endpoint(self, handler, reset_rate_limiter):
        http = MockHTTPHandler(method="POST")
        result = handler.handle("/api/v1/billing/plans", {}, http, "POST")
        assert _status(result) == 405

    def test_get_on_post_endpoint(self, handler, reset_rate_limiter):
        http = MockHTTPHandler(method="GET")
        result = handler.handle("/api/v1/billing/checkout", {}, http, "GET")
        assert _status(result) == 405

    def test_method_from_handler_command(self, handler, reset_rate_limiter):
        """Method should be read from handler.command if present."""
        http = MockHTTPHandler(method="POST")
        http.command = "GET"
        result = handler.handle("/api/v1/billing/plans", {}, http, "POST")
        # Since handler.command is GET, it should match the plans GET route
        assert _status(result) == 200


# ===========================================================================
# Audit logging internal method
# ===========================================================================


class TestAuditLogging:
    def test_log_audit_success(self, handler, reset_rate_limiter):
        store = MockUserStore()
        handler._log_audit(
            store,
            action="test.action",
            resource_type="subscription",
            resource_id="sub_123",
            user_id="user-1",
            org_id="org-1",
        )
        assert len(store._audit_events) == 1
        assert store._audit_events[0]["action"] == "test.action"

    def test_log_audit_no_store(self, handler, reset_rate_limiter):
        """Should not crash when user_store is None."""
        handler._log_audit(
            None,
            action="test.action",
            resource_type="subscription",
        )

    def test_log_audit_store_no_method(self, handler, reset_rate_limiter):
        """Should not crash when store lacks log_audit_event."""
        store = MagicMock(spec=[])
        handler._log_audit(
            store,
            action="test.action",
            resource_type="subscription",
        )

    def test_log_audit_with_handler_ip(self, handler, reset_rate_limiter):
        """Should extract IP and user agent from handler."""
        store = MockUserStore()
        http = MockHTTPHandler()
        http.headers["User-Agent"] = "TestAgent/1.0"

        with patch(
            "aragora.server.middleware.auth.extract_client_ip",
            return_value="10.0.0.1",
        ):
            handler._log_audit(
                store,
                action="test.action",
                resource_type="subscription",
                handler=http,
            )
            assert len(store._audit_events) == 1
            assert store._audit_events[0]["ip_address"] == "10.0.0.1"
            assert store._audit_events[0]["user_agent"] == "TestAgent/1.0"

    def test_log_audit_error_swallowed(self, handler, reset_rate_limiter):
        """Audit logging errors should be swallowed, not crash the request."""
        store = MagicMock()
        store.log_audit_event.side_effect = RuntimeError("db error")
        handler._log_audit(
            store,
            action="test.action",
            resource_type="subscription",
        )
        # Should not raise


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    def test_billing_handler_exported(self):
        from aragora.server.handlers.admin.billing import __all__

        assert "BillingHandler" in __all__


# ===========================================================================
# Handler context
# ===========================================================================


class TestHandlerContext:
    def test_default_ctx_is_empty_dict(self):
        h = BillingHandler()
        assert h.ctx == {}

    def test_get_user_store_from_ctx(self, handler, user_store):
        assert handler._get_user_store() is user_store

    def test_get_usage_tracker_from_ctx(self):
        tracker = MagicMock()
        h = BillingHandler(ctx={"usage_tracker": tracker})
        assert h._get_usage_tracker() is tracker

    def test_get_user_store_none(self):
        h = BillingHandler(ctx={})
        assert h._get_user_store() is None

    def test_get_usage_tracker_none(self):
        h = BillingHandler(ctx={})
        assert h._get_usage_tracker() is None

    def test_resource_type(self):
        assert BillingHandler.RESOURCE_TYPE == "billing"
