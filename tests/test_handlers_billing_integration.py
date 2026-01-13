"""
Billing Handler Integration Tests.

Tests for billing, subscription management, and Stripe integration.
Covers:
- Route handling and validation
- Subscription tier logic
- Usage tracking
- Quota enforcement
- Webhook handling (mocked)
- Audit logging
"""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

from aragora.billing.models import (
    Organization,
    SubscriptionTier,
    TierLimits,
    TIER_LIMITS,
    User,
)
from aragora.server.handlers.billing import BillingHandler


# =============================================================================
# Fixtures
# =============================================================================


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.orgs: dict[str, Organization] = {}
        self.audit_log: list[dict] = []

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        return self.orgs.get(org_id)

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[Organization]:
        for org in self.orgs.values():
            if org.stripe_subscription_id == subscription_id:
                return org
        return None

    def get_organization_by_stripe_customer(self, customer_id: str) -> Optional[Organization]:
        for org in self.orgs.values():
            if org.stripe_customer_id == customer_id:
                return org
        return None

    def update_organization(self, org_id: str, **kwargs) -> Optional[Organization]:
        org = self.orgs.get(org_id)
        if org:
            for key, value in kwargs.items():
                setattr(org, key, value)
            org.updated_at = datetime.utcnow()
        return org

    def reset_org_usage(self, org_id: str) -> None:
        org = self.orgs.get(org_id)
        if org:
            org.reset_monthly_usage()

    def log_audit_event(self, **kwargs) -> None:
        self.audit_log.append(kwargs)

    def get_audit_log(
        self,
        org_id: str,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        entries = [e for e in self.audit_log if e.get("org_id") == org_id]
        if action:
            entries = [e for e in entries if e.get("action") == action]
        if resource_type:
            entries = [e for e in entries if e.get("resource_type") == resource_type]
        return entries[offset : offset + limit]

    def get_audit_log_count(
        self,
        org_id: str,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        return len(self.get_audit_log(org_id, action, resource_type, limit=10000))


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: Optional[dict] = None,
        headers: Optional[dict] = None,
        query_params: Optional[dict] = None,
    ):
        self.command = method
        self._body = json.dumps(body).encode() if body else b""
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.rfile = BytesIO(self._body)

        # Set content length
        if body:
            self.headers["Content-Length"] = str(len(self._body))

    def get(self, key: str, default: Any = None) -> Any:
        """Get query parameter value (for get_string_param compatibility)."""
        return self.query_params.get(key, default)


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = ""
    org_id: Optional[str] = None
    role: str = "member"


@pytest.fixture
def user_store():
    """Create mock user store."""
    return MockUserStore()


@pytest.fixture
def billing_handler(user_store):
    """Create billing handler with mock context."""
    handler = BillingHandler(server_context={"user_store": user_store})
    return handler


@pytest.fixture
def test_user(user_store):
    """Create test user."""
    user = User(
        id=str(uuid4()),
        email="test@example.com",
        name="Test User",
        role="member",
    )
    user_store.users[user.id] = user
    return user


@pytest.fixture
def test_org(user_store, test_user):
    """Create test organization."""
    org = Organization(
        id=str(uuid4()),
        name="Test Org",
        slug="test-org",
        tier=SubscriptionTier.STARTER,
        owner_id=test_user.id,
        debates_used_this_month=5,
    )
    test_user.org_id = org.id
    user_store.orgs[org.id] = org
    return org


@pytest.fixture
def enterprise_org(user_store, test_user):
    """Create enterprise organization with full features."""
    org = Organization(
        id=str(uuid4()),
        name="Enterprise Org",
        slug="enterprise-org",
        tier=SubscriptionTier.ENTERPRISE,
        owner_id=test_user.id,
        stripe_customer_id="cus_enterprise123",
        stripe_subscription_id="sub_enterprise123",
    )
    test_user.org_id = org.id
    test_user.role = "owner"
    user_store.orgs[org.id] = org
    return org


# =============================================================================
# Route Handling Tests
# =============================================================================


class TestBillingRouteHandling:
    """Tests for billing route handling."""

    def test_can_handle_plans_route(self, billing_handler):
        """Test can_handle for plans endpoint."""
        assert billing_handler.can_handle("/api/billing/plans") is True

    def test_can_handle_usage_route(self, billing_handler):
        """Test can_handle for usage endpoint."""
        assert billing_handler.can_handle("/api/billing/usage") is True

    def test_can_handle_subscription_route(self, billing_handler):
        """Test can_handle for subscription endpoint."""
        assert billing_handler.can_handle("/api/billing/subscription") is True

    def test_can_handle_checkout_route(self, billing_handler):
        """Test can_handle for checkout endpoint."""
        assert billing_handler.can_handle("/api/billing/checkout") is True

    def test_can_handle_portal_route(self, billing_handler):
        """Test can_handle for portal endpoint."""
        assert billing_handler.can_handle("/api/billing/portal") is True

    def test_can_handle_cancel_route(self, billing_handler):
        """Test can_handle for cancel endpoint."""
        assert billing_handler.can_handle("/api/billing/cancel") is True

    def test_can_handle_resume_route(self, billing_handler):
        """Test can_handle for resume endpoint."""
        assert billing_handler.can_handle("/api/billing/resume") is True

    def test_can_handle_audit_log_route(self, billing_handler):
        """Test can_handle for audit-log endpoint."""
        assert billing_handler.can_handle("/api/billing/audit-log") is True

    def test_can_handle_usage_export_route(self, billing_handler):
        """Test can_handle for usage export endpoint."""
        assert billing_handler.can_handle("/api/billing/usage/export") is True

    def test_can_handle_usage_forecast_route(self, billing_handler):
        """Test can_handle for usage forecast endpoint."""
        assert billing_handler.can_handle("/api/billing/usage/forecast") is True

    def test_can_handle_invoices_route(self, billing_handler):
        """Test can_handle for invoices endpoint."""
        assert billing_handler.can_handle("/api/billing/invoices") is True

    def test_can_handle_stripe_webhook_route(self, billing_handler):
        """Test can_handle for Stripe webhook endpoint."""
        assert billing_handler.can_handle("/api/webhooks/stripe") is True

    def test_cannot_handle_unknown_route(self, billing_handler):
        """Test can_handle returns False for unknown routes."""
        assert billing_handler.can_handle("/api/billing/unknown") is False
        assert billing_handler.can_handle("/api/other/endpoint") is False
        assert billing_handler.can_handle("/billing/plans") is False

    def test_all_routes_in_routes_constant(self, billing_handler):
        """Test all expected routes are in ROUTES constant."""
        expected_routes = [
            "/api/billing/plans",
            "/api/billing/usage",
            "/api/billing/subscription",
            "/api/billing/checkout",
            "/api/billing/portal",
            "/api/billing/cancel",
            "/api/billing/resume",
            "/api/billing/audit-log",
            "/api/billing/usage/export",
            "/api/billing/usage/forecast",
            "/api/billing/invoices",
            "/api/webhooks/stripe",
        ]
        for route in expected_routes:
            assert route in BillingHandler.ROUTES


# =============================================================================
# Subscription Tier Tests
# =============================================================================


class TestSubscriptionTiers:
    """Tests for subscription tier configuration."""

    def test_free_tier_limits(self):
        """Test free tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]
        assert limits.debates_per_month == 10
        assert limits.users_per_org == 1
        assert limits.api_access is False
        assert limits.all_agents is False
        assert limits.price_monthly_cents == 0

    def test_starter_tier_limits(self):
        """Test starter tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.STARTER]
        assert limits.debates_per_month == 50
        assert limits.users_per_org == 2
        assert limits.api_access is False
        assert limits.price_monthly_cents == 9900  # $99

    def test_professional_tier_limits(self):
        """Test professional tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
        assert limits.debates_per_month == 200
        assert limits.users_per_org == 10
        assert limits.api_access is True
        assert limits.all_agents is True
        assert limits.audit_logs is True
        assert limits.price_monthly_cents == 29900  # $299

    def test_enterprise_tier_limits(self):
        """Test enterprise tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE]
        assert limits.debates_per_month == 999999  # Unlimited
        assert limits.users_per_org == 999999
        assert limits.api_access is True
        assert limits.all_agents is True
        assert limits.custom_agents is True
        assert limits.sso_enabled is True
        assert limits.audit_logs is True
        assert limits.priority_support is True
        assert limits.price_monthly_cents == 99900  # $999

    def test_tier_limits_to_dict(self):
        """Test TierLimits.to_dict() method."""
        limits = TIER_LIMITS[SubscriptionTier.STARTER]
        data = limits.to_dict()
        assert "debates_per_month" in data
        assert "users_per_org" in data
        assert "api_access" in data
        assert "price_monthly_cents" in data
        assert data["debates_per_month"] == 50

    def test_all_tiers_have_limits(self):
        """Test all subscription tiers have defined limits."""
        for tier in SubscriptionTier:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert isinstance(limits, TierLimits)


# =============================================================================
# Organization Usage Tests
# =============================================================================


class TestOrganizationUsage:
    """Tests for organization usage tracking."""

    def test_organization_debates_remaining(self, test_org):
        """Test debates_remaining calculation."""
        test_org.tier = SubscriptionTier.STARTER
        test_org.debates_used_this_month = 10
        expected = TIER_LIMITS[SubscriptionTier.STARTER].debates_per_month - 10
        assert test_org.debates_remaining == expected

    def test_organization_debates_remaining_at_zero(self, test_org):
        """Test debates_remaining never goes negative."""
        test_org.tier = SubscriptionTier.FREE
        test_org.debates_used_this_month = 20  # Over limit
        assert test_org.debates_remaining == 0

    def test_organization_is_at_limit(self, test_org):
        """Test is_at_limit property."""
        test_org.tier = SubscriptionTier.FREE
        test_org.debates_used_this_month = 10
        assert test_org.is_at_limit is True

    def test_organization_not_at_limit(self, test_org):
        """Test is_at_limit when under limit."""
        test_org.tier = SubscriptionTier.FREE
        test_org.debates_used_this_month = 5
        assert test_org.is_at_limit is False

    def test_organization_increment_debates(self, test_org):
        """Test increment_debates method."""
        test_org.tier = SubscriptionTier.FREE
        test_org.debates_used_this_month = 5
        result = test_org.increment_debates()
        assert result is True
        assert test_org.debates_used_this_month == 6

    def test_organization_increment_debates_at_limit(self, test_org):
        """Test increment_debates at limit returns False."""
        test_org.tier = SubscriptionTier.FREE
        test_org.debates_used_this_month = 10
        result = test_org.increment_debates()
        assert result is False
        assert test_org.debates_used_this_month == 10  # Unchanged

    def test_organization_reset_monthly_usage(self, test_org):
        """Test reset_monthly_usage method."""
        test_org.debates_used_this_month = 25
        original_billing_start = test_org.billing_cycle_start
        test_org.reset_monthly_usage()
        assert test_org.debates_used_this_month == 0
        assert test_org.billing_cycle_start > original_billing_start


# =============================================================================
# Get Plans Tests
# =============================================================================


class TestGetPlans:
    """Tests for GET /api/billing/plans endpoint."""

    def test_get_plans_returns_all_tiers(self, billing_handler):
        """Test _get_plans returns all subscription tiers."""
        result = billing_handler._get_plans()
        assert result.status_code == 200

        body = json.loads(result.body)
        plans = body["plans"]

        # Should have 4 tiers
        assert len(plans) == 4

        # Check tier names
        tier_ids = [p["id"] for p in plans]
        assert "free" in tier_ids
        assert "starter" in tier_ids
        assert "professional" in tier_ids
        assert "enterprise" in tier_ids

    def test_get_plans_has_correct_structure(self, billing_handler):
        """Test plan response has correct structure."""
        result = billing_handler._get_plans()
        body = json.loads(result.body)
        plan = body["plans"][0]

        assert "id" in plan
        assert "name" in plan
        assert "price_monthly_cents" in plan
        assert "price_monthly" in plan
        assert "features" in plan

    def test_get_plans_features_structure(self, billing_handler):
        """Test plan features have correct structure."""
        result = billing_handler._get_plans()
        body = json.loads(result.body)

        for plan in body["plans"]:
            features = plan["features"]
            assert "debates_per_month" in features
            assert "users_per_org" in features
            assert "api_access" in features
            assert "all_agents" in features
            assert "custom_agents" in features
            assert "sso_enabled" in features
            assert "audit_logs" in features
            assert "priority_support" in features

    def test_get_plans_price_formatting(self, billing_handler):
        """Test price is formatted correctly."""
        result = billing_handler._get_plans()
        body = json.loads(result.body)

        # Find starter plan
        starter = next(p for p in body["plans"] if p["id"] == "starter")
        assert starter["price_monthly"] == "$99.00"
        assert starter["price_monthly_cents"] == 9900


# =============================================================================
# Checkout Validation Tests
# =============================================================================


class TestCheckoutValidation:
    """Tests for checkout validation logic."""

    def test_checkout_requires_tier(self, billing_handler, test_user, test_org):
        """Test checkout fails without tier specified."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(
                method="POST", body={"success_url": "http://x", "cancel_url": "http://y"}
            )
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Tier is required" in body["error"]

    def test_checkout_requires_urls(self, billing_handler, test_user, test_org):
        """Test checkout fails without success/cancel URLs."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={"tier": "starter"})
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Success and cancel URLs required" in body["error"]

    def test_checkout_rejects_free_tier(self, billing_handler, test_user, test_org):
        """Test checkout fails for free tier."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(
                method="POST",
                body={"tier": "free", "success_url": "http://x", "cancel_url": "http://y"},
            )
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Cannot checkout free tier" in body["error"]

    def test_checkout_rejects_invalid_tier(self, billing_handler, test_user, test_org):
        """Test checkout fails for invalid tier."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(
                method="POST",
                body={"tier": "platinum", "success_url": "http://x", "cancel_url": "http://y"},
            )
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid tier" in body["error"]


# =============================================================================
# Portal Validation Tests
# =============================================================================


class TestPortalValidation:
    """Tests for billing portal validation logic."""

    def test_portal_requires_return_url(self, billing_handler, test_user, test_org):
        """Test portal fails without return URL."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={})
            result = billing_handler._create_portal(handler)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Return URL required" in body["error"]


# =============================================================================
# Role-Based Access Tests
# =============================================================================


class TestRoleBasedAccess:
    """Tests for role-based access control."""

    def test_cancel_requires_owner_or_admin(self, billing_handler, user_store, test_user, test_org):
        """Test cancel subscription requires owner or admin role."""
        test_user.role = "member"
        test_org.stripe_subscription_id = "sub_123"
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True, role="member")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={})
            result = billing_handler._cancel_subscription(handler)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "owners" in body["error"].lower() or "Only organization" in body["error"]

    def test_resume_requires_owner_or_admin(self, billing_handler, user_store, test_user, test_org):
        """Test resume subscription requires owner or admin role."""
        test_user.role = "member"
        test_org.stripe_subscription_id = "sub_123"
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True, role="member")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={})
            result = billing_handler._resume_subscription(handler)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "owners" in body["error"].lower() or "Only organization" in body["error"]

    def test_audit_log_requires_owner_or_admin(
        self, billing_handler, user_store, test_user, enterprise_org
    ):
        """Test audit log access requires owner or admin role."""
        test_user.role = "member"
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True, role="member")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_audit_log(handler)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "Insufficient permissions" in body["error"]


# =============================================================================
# Enterprise Feature Tests
# =============================================================================


class TestEnterpriseFeatures:
    """Tests for enterprise-only features."""

    def test_audit_log_requires_enterprise_tier(
        self, billing_handler, user_store, test_user, test_org
    ):
        """Test audit log requires enterprise tier."""
        test_user.role = "owner"
        test_org.tier = SubscriptionTier.STARTER  # Not enterprise
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True, role="owner")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_audit_log(handler)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "Enterprise tier" in body["error"]

    def test_audit_log_allowed_for_enterprise(
        self, billing_handler, user_store, test_user, enterprise_org
    ):
        """Test audit log access allowed for enterprise tier with owner role."""
        test_user.role = "owner"
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True, role="owner")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            handler.query_params = {}
            with patch.object(billing_handler, "ctx", {"user_store": user_store}):
                result = billing_handler._get_audit_log(handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "entries" in body
        assert "total" in body

    def test_professional_tier_has_audit_logs(self):
        """Test professional tier has audit logs enabled."""
        limits = TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
        assert limits.audit_logs is True

    def test_free_tier_no_audit_logs(self):
        """Test free tier does not have audit logs."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]
        assert limits.audit_logs is False


# =============================================================================
# Authentication Tests
# =============================================================================


class TestBillingAuthentication:
    """Tests for authentication requirements."""

    def test_get_usage_requires_auth(self, billing_handler):
        """Test get usage requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_usage(handler)

        assert result.status_code == 401
        body = json.loads(result.body)
        assert "Not authenticated" in body["error"]

    def test_get_subscription_requires_auth(self, billing_handler):
        """Test get subscription requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_subscription(handler)

        assert result.status_code == 401

    def test_create_checkout_requires_auth(self, billing_handler):
        """Test create checkout requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={})
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 401

    def test_cancel_subscription_requires_auth(self, billing_handler):
        """Test cancel subscription requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body={})
            result = billing_handler._cancel_subscription(handler)

        assert result.status_code == 401


# =============================================================================
# User Not Found Tests
# =============================================================================


class TestUserNotFound:
    """Tests for user not found scenarios."""

    def test_get_usage_user_not_found(self, billing_handler, user_store):
        """Test get usage returns 404 for non-existent user."""
        mock_auth = MockAuthContext(is_authenticated=True, user_id="nonexistent")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_usage(handler)

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "User not found" in body["error"]

    def test_get_subscription_user_not_found(self, billing_handler, user_store):
        """Test get subscription returns 404 for non-existent user."""
        mock_auth = MockAuthContext(is_authenticated=True, user_id="nonexistent")

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_subscription(handler)

        assert result.status_code == 404


# =============================================================================
# Usage Forecast Tests
# =============================================================================


class TestUsageForecast:
    """Tests for usage forecast endpoint."""

    def test_forecast_requires_auth(self, billing_handler):
        """Test forecast requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_usage_forecast(handler)

        assert result.status_code == 401

    def test_forecast_requires_organization(self, billing_handler, user_store, test_user):
        """Test forecast requires organization membership."""
        test_user.org_id = None
        mock_auth = MockAuthContext(is_authenticated=True, user_id=test_user.id)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_usage_forecast(handler)

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "No organization found" in body["error"]


# =============================================================================
# Invoices Tests
# =============================================================================


class TestInvoices:
    """Tests for invoices endpoint."""

    def test_invoices_requires_auth(self, billing_handler):
        """Test invoices requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_invoices(handler)

        assert result.status_code == 401

    def test_invoices_requires_billing_account(
        self, billing_handler, user_store, test_user, test_org
    ):
        """Test invoices requires Stripe customer ID."""
        test_org.stripe_customer_id = None
        mock_auth = MockAuthContext(is_authenticated=True, user_id=test_user.id)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_invoices(handler)

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "No billing account found" in body["error"]


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestAuditLogging:
    """Tests for audit logging functionality."""

    def test_log_audit_records_event(self, billing_handler, user_store):
        """Test _log_audit records events to store."""
        billing_handler._log_audit(
            user_store,
            action="subscription.created",
            resource_type="subscription",
            resource_id="sub_123",
            user_id="user_123",
            org_id="org_123",
            old_value={"tier": "free"},
            new_value={"tier": "starter"},
        )

        assert len(user_store.audit_log) == 1
        event = user_store.audit_log[0]
        assert event["action"] == "subscription.created"
        assert event["resource_type"] == "subscription"
        assert event["org_id"] == "org_123"

    def test_log_audit_handles_missing_store(self, billing_handler):
        """Test _log_audit handles missing user store gracefully."""
        # Should not raise exception
        billing_handler._log_audit(
            None,
            action="test",
            resource_type="test",
        )

    def test_log_audit_extracts_client_info(self, billing_handler, user_store):
        """Test _log_audit extracts IP and user agent from handler."""
        handler = MockHandler(
            headers={
                "User-Agent": "TestBrowser/1.0",
                "X-Forwarded-For": "192.168.1.1",
            }
        )

        with patch("aragora.server.middleware.auth.extract_client_ip", return_value="192.168.1.1"):
            billing_handler._log_audit(
                user_store,
                action="test",
                resource_type="test",
                handler=handler,
            )

        assert len(user_store.audit_log) == 1
        event = user_store.audit_log[0]
        assert event["ip_address"] == "192.168.1.1"


# =============================================================================
# Webhook Handling Tests (Mocked)
# =============================================================================


class TestWebhookHandling:
    """Tests for Stripe webhook handling logic."""

    def test_webhook_requires_signature(self, billing_handler):
        """Test webhook requires Stripe signature."""
        handler = MockHandler(
            method="POST",
            headers={"Content-Length": "10"},
            body={"test": "data"},
        )
        handler.headers.pop("Stripe-Signature", None)

        result = billing_handler._handle_stripe_webhook(handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Missing signature" in body["error"]

    def test_webhook_subscription_deleted_downgrades_org(
        self, billing_handler, user_store, test_org
    ):
        """Test subscription deleted webhook downgrades organization."""
        test_org.stripe_subscription_id = "sub_to_delete"
        test_org.tier = SubscriptionTier.PROFESSIONAL

        @dataclass
        class MockEvent:
            type: str = "customer.subscription.deleted"
            object: dict = field(default_factory=lambda: {"id": "sub_to_delete"})
            data: dict = field(default_factory=dict)
            metadata: dict = field(default_factory=dict)
            subscription_id: str = "sub_to_delete"

        result = billing_handler._handle_subscription_deleted(MockEvent(), user_store)

        assert result.status_code == 200
        assert test_org.tier == SubscriptionTier.FREE
        assert test_org.stripe_subscription_id is None

    def test_webhook_invoice_paid_resets_usage(self, billing_handler, user_store, test_org):
        """Test invoice paid webhook resets organization usage."""
        test_org.stripe_customer_id = "cus_123"
        test_org.debates_used_this_month = 45

        @dataclass
        class MockEvent:
            type: str = "invoice.payment_succeeded"
            object: dict = field(
                default_factory=lambda: {
                    "customer": "cus_123",
                    "subscription": "sub_123",
                    "amount_paid": 9900,
                }
            )
            data: dict = field(default_factory=dict)
            metadata: dict = field(default_factory=dict)

        result = billing_handler._handle_invoice_paid(MockEvent(), user_store)

        assert result.status_code == 200
        assert test_org.debates_used_this_month == 0


# =============================================================================
# Handler Context Tests
# =============================================================================


class TestHandlerContext:
    """Tests for handler context access."""

    def test_get_user_store_from_context(self, user_store):
        """Test _get_user_store retrieves from context."""
        handler = BillingHandler(server_context={"user_store": user_store})
        assert handler._get_user_store() is user_store

    def test_get_user_store_returns_none_if_missing(self):
        """Test _get_user_store returns None if not in context."""
        handler = BillingHandler(server_context={})
        assert handler._get_user_store() is None

    def test_get_usage_tracker_from_context(self):
        """Test _get_usage_tracker retrieves from context."""
        mock_tracker = MagicMock()
        handler = BillingHandler(server_context={"usage_tracker": mock_tracker})
        assert handler._get_usage_tracker() is mock_tracker


# =============================================================================
# Method Not Allowed Tests
# =============================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_plans_post_not_allowed(self, billing_handler):
        """Test POST to plans returns 405."""
        handler = MockHandler(method="POST")
        result = billing_handler.handle("/api/billing/plans", {}, handler, "POST")
        assert result.status_code == 405

    def test_usage_post_not_allowed(self, billing_handler):
        """Test POST to usage returns 405."""
        handler = MockHandler(method="POST")
        result = billing_handler.handle("/api/billing/usage", {}, handler, "POST")
        assert result.status_code == 405

    def test_checkout_get_not_allowed(self, billing_handler):
        """Test GET to checkout returns 405."""
        handler = MockHandler(method="GET")
        result = billing_handler.handle("/api/billing/checkout", {}, handler, "GET")
        assert result.status_code == 405


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestBillingEdgeCases:
    """Tests for edge cases in billing handler."""

    def test_empty_body_checkout(self, billing_handler, test_user, test_org):
        """Test checkout with empty/invalid body."""
        mock_auth = MockAuthContext(user_id=test_user.id, is_authenticated=True)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="POST", body=None)
            handler._body = b"not json"
            handler.rfile = BytesIO(handler._body)
            handler.headers["Content-Length"] = str(len(handler._body))
            result = billing_handler._create_checkout(handler)

        assert result.status_code == 400

    def test_organization_limits_property(self, test_org):
        """Test organization.limits returns correct tier limits."""
        test_org.tier = SubscriptionTier.PROFESSIONAL
        limits = test_org.limits
        assert limits.debates_per_month == 200
        assert limits.api_access is True

    def test_user_without_org_for_usage(self, billing_handler, user_store, test_user):
        """Test usage for user without organization."""
        test_user.org_id = None
        mock_auth = MockAuthContext(is_authenticated=True, user_id=test_user.id)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_usage(handler)

        # Should return 200 with default usage data
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["usage"]["debates_used"] == 0

    def test_subscription_without_stripe(self, billing_handler, user_store, test_user, test_org):
        """Test subscription response without Stripe integration."""
        test_org.stripe_subscription_id = None
        mock_auth = MockAuthContext(is_authenticated=True, user_id=test_user.id)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._get_subscription(handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["subscription"]["tier"] == "starter"
        assert body["subscription"]["is_active"] is True


# =============================================================================
# CSV Export Tests
# =============================================================================


class TestUsageExport:
    """Tests for usage CSV export."""

    def test_export_requires_auth(self, billing_handler):
        """Test export requires authentication."""
        mock_auth = MockAuthContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._export_usage_csv(handler)

        assert result.status_code == 401

    def test_export_requires_organization(self, billing_handler, user_store, test_user):
        """Test export requires organization membership."""
        test_user.org_id = None
        mock_auth = MockAuthContext(is_authenticated=True, user_id=test_user.id)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_auth):
            handler = MockHandler(method="GET")
            result = billing_handler._export_usage_csv(handler)

        assert result.status_code == 404


__all__ = [
    "TestBillingRouteHandling",
    "TestSubscriptionTiers",
    "TestOrganizationUsage",
    "TestGetPlans",
    "TestCheckoutValidation",
    "TestPortalValidation",
    "TestRoleBasedAccess",
    "TestEnterpriseFeatures",
    "TestBillingAuthentication",
    "TestUserNotFound",
    "TestUsageForecast",
    "TestInvoices",
    "TestAuditLogging",
    "TestWebhookHandling",
    "TestHandlerContext",
    "TestMethodNotAllowed",
    "TestBillingEdgeCases",
    "TestUsageExport",
]
