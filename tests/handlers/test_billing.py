"""Tests for billing handler endpoints.

Tests the billing API endpoints including:
- Subscription plans listing
- Usage tracking and reporting
- Checkout session creation
- Billing portal access
- Subscription management (cancel/resume)
- Audit logging (Enterprise feature)
- Usage export and forecasting
- Invoice history
- Stripe webhook handling
"""

import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import SubscriptionTier
from aragora.server.handlers.admin.billing import BillingHandler


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


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


class MockSubscriptionTier:
    """Mock subscription tier enum."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

    def __init__(self, value: str):
        self.value = value
        self.name = value.upper()


class MockUser:
    """Mock user object for billing tests."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str = "Test User",
        role: str = "member",
        org_id: Optional[str] = None,
        is_active: bool = True,
    ):
        self.id = id
        self.user_id = id  # Alias for compatibility
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id
        self.is_active = is_active


class MockOrganization:
    """Mock organization object for billing tests."""

    def __init__(
        self,
        id: str,
        name: str,
        slug: str = "test-org",
        tier: MockSubscriptionTier = None,
        debates_used_this_month: int = 0,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        billing_cycle_start: Optional[datetime] = None,
        limits: Optional[MockTierLimits] = None,
    ):
        self.id = id
        self.name = name
        self.slug = slug
        self.tier = tier or MockSubscriptionTier("free")
        self.debates_used_this_month = debates_used_this_month
        self.stripe_customer_id = stripe_customer_id
        self.stripe_subscription_id = stripe_subscription_id
        self.billing_cycle_start = billing_cycle_start or datetime.now(timezone.utc).replace(day=1)
        self.limits = limits or MockTierLimits()

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)


class MockUsageSummary:
    """Mock usage summary for tracking."""

    def __init__(
        self,
        total_tokens: int = 0,
        total_tokens_in: int = 0,
        total_tokens_out: int = 0,
        total_cost_usd: Decimal = Decimal("0.00"),
        total_cost: Decimal = Decimal("0.00"),
        cost_by_provider: Optional[Dict[str, Decimal]] = None,
    ):
        self.total_tokens = total_tokens
        self.total_tokens_in = total_tokens_in
        self.total_tokens_out = total_tokens_out
        self.total_cost_usd = total_cost_usd
        self.total_cost = total_cost
        self.cost_by_provider = cost_by_provider or {}


class MockUsageTracker:
    """Mock usage tracker for testing."""

    def __init__(self, summary: Optional[MockUsageSummary] = None):
        self._summary = summary

    def get_summary(
        self,
        org_id: str = None,
        period_start: datetime = None,
        start_time: datetime = None,
    ) -> Optional[MockUsageSummary]:
        return self._summary


class MockStripeSubscription:
    """Mock Stripe subscription object."""

    def __init__(
        self,
        id: str,
        status: str = "active",
        current_period_end: Optional[datetime] = None,
        cancel_at_period_end: bool = False,
        trial_start: Optional[datetime] = None,
        trial_end: Optional[datetime] = None,
    ):
        self.id = id
        self.status = status
        self.current_period_end = current_period_end or (datetime.now(timezone.utc) + timedelta(days=30))
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
        subscription: Optional[MockStripeSubscription] = None,
        checkout_session: Optional[MockCheckoutSession] = None,
        portal_session: Optional[MockPortalSession] = None,
        invoices: Optional[List[dict]] = None,
    ):
        self._subscription = subscription
        self._checkout_session = checkout_session or MockCheckoutSession("cs_test_123")
        self._portal_session = portal_session or MockPortalSession("bps_test_123")
        self._invoices = invoices or []

    def get_subscription(self, subscription_id: str) -> Optional[MockStripeSubscription]:
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

    def list_invoices(self, customer_id: str, limit: int = 10) -> List[dict]:
        return self._invoices[:limit]


class MockUserStore:
    """Mock user store for billing tests."""

    def __init__(self):
        self._users: Dict[str, MockUser] = {}
        self._orgs: Dict[str, MockOrganization] = {}
        self._orgs_by_subscription: Dict[str, MockOrganization] = {}
        self._orgs_by_customer: Dict[str, MockOrganization] = {}
        self._audit_log: List[dict] = []

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org
        if org.stripe_subscription_id:
            self._orgs_by_subscription[org.stripe_subscription_id] = org
        if org.stripe_customer_id:
            self._orgs_by_customer[org.stripe_customer_id] = org

    def get_user_by_id(self, user_id: str) -> Optional[MockUser]:
        return self._users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> Optional[MockOrganization]:
        return self._orgs.get(org_id)

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[MockOrganization]:
        return self._orgs_by_subscription.get(subscription_id)

    def get_organization_by_stripe_customer(self, customer_id: str) -> Optional[MockOrganization]:
        return self._orgs_by_customer.get(customer_id)

    def get_organization_owner(self, org_id: str) -> Optional[MockUser]:
        for user in self._users.values():
            if user.org_id == org_id and user.role == "owner":
                return user
        return None

    def update_organization(self, org_id: str, **kwargs) -> Optional[MockOrganization]:
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

    def get_audit_log(
        self,
        org_id: str = None,
        action: str = None,
        resource_type: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        entries = self._audit_log
        if org_id:
            entries = [e for e in entries if e.get("org_id") == org_id]
        if action:
            entries = [e for e in entries if e.get("action") == action]
        if resource_type:
            entries = [e for e in entries if e.get("resource_type") == resource_type]
        return entries[offset : offset + limit]

    def get_audit_log_count(
        self,
        org_id: str = None,
        action: str = None,
        resource_type: str = None,
    ) -> int:
        return len(self.get_audit_log(org_id, action, resource_type))

    def log_audit_event(self, **kwargs):
        self._audit_log.append(kwargs)


class MockAuthContext:
    """Mock authentication context."""

    def __init__(
        self,
        user_id: str,
        is_authenticated: bool = True,
        org_id: Optional[str] = None,
        role: str = "member",
        permissions: Optional[List[str]] = None,
        error_reason: Optional[str] = None,
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.org_id = org_id
        self.role = role
        self.permissions = permissions or []
        self.error_reason = error_reason


class MockHandler:
    """Mock HTTP handler."""

    def __init__(
        self,
        body: Optional[dict] = None,
        command: str = "GET",
        query_string: str = "",
        user_store=None,
        query_params: Optional[dict] = None,
    ):
        self.command = command
        self.headers = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""
        self.user_store = user_store
        self._query_params = query_params or {}

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"

    def get(self, key: str, default=None):
        """Get query parameter - for compatibility with get_string_param."""
        return self._query_params.get(key, default)


class MockWebhookEvent:
    """Mock Stripe webhook event."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        object_data: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        self.event_id = event_id
        self.type = event_type
        self.object = object_data or {}
        self.metadata = metadata or {}
        self.subscription_id = object_data.get("id") if object_data else None


@pytest.fixture
def user_store():
    """Create a mock user store with test data."""
    store = MockUserStore()

    # Add owner user
    owner = MockUser(
        id="owner_1",
        email="owner@example.com",
        name="Org Owner",
        role="owner",
        org_id="org_1",
    )
    store.add_user(owner)

    # Add admin user
    admin = MockUser(
        id="admin_1",
        email="admin@example.com",
        name="Admin User",
        role="admin",
        org_id="org_1",
    )
    store.add_user(admin)

    # Add regular member
    member = MockUser(
        id="member_1",
        email="member@example.com",
        name="Member User",
        role="member",
        org_id="org_1",
    )
    store.add_user(member)

    # Add user without org
    no_org_user = MockUser(
        id="no_org_1",
        email="noorg@example.com",
        name="No Org User",
        role="member",
        org_id=None,
    )
    store.add_user(no_org_user)

    # Add organization with free tier - use actual SubscriptionTier
    free_org = MockOrganization(
        id="org_1",
        name="Free Org",
        slug="free-org",
        tier=SubscriptionTier.FREE,
        debates_used_this_month=5,
        limits=MockTierLimits(debates_per_month=10),
    )
    store.add_organization(free_org)

    # Add organization with enterprise tier - use actual SubscriptionTier
    enterprise_org = MockOrganization(
        id="org_enterprise",
        name="Enterprise Org",
        slug="enterprise-org",
        tier=SubscriptionTier.ENTERPRISE,
        stripe_customer_id="cus_enterprise_123",
        stripe_subscription_id="sub_enterprise_123",
        limits=MockTierLimits(
            debates_per_month=10000,
            users_per_org=100,
            price_monthly_cents=99900,
            api_access=True,
            all_agents=True,
            custom_agents=True,
            sso_enabled=True,
            audit_logs=True,
            priority_support=True,
        ),
    )
    store.add_organization(enterprise_org)

    # Add enterprise owner
    enterprise_owner = MockUser(
        id="ent_owner_1",
        email="entowner@example.com",
        name="Enterprise Owner",
        role="owner",
        org_id="org_enterprise",
    )
    store.add_user(enterprise_owner)

    return store


@pytest.fixture
def usage_tracker():
    """Create mock usage tracker."""
    summary = MockUsageSummary(
        total_tokens=50000,
        total_tokens_in=30000,
        total_tokens_out=20000,
        total_cost_usd=Decimal("1.25"),
        total_cost=Decimal("1.25"),
        cost_by_provider={"anthropic": Decimal("1.00"), "openai": Decimal("0.25")},
    )
    return MockUsageTracker(summary)


@pytest.fixture
def stripe_client():
    """Create mock Stripe client."""
    subscription = MockStripeSubscription(
        id="sub_test_123",
        status="active",
    )
    invoices = [
        {
            "id": "in_test_1",
            "number": "INV-001",
            "status": "paid",
            "amount_due": 2900,
            "amount_paid": 2900,
            "currency": "usd",
            "created": 1700000000,
            "period_start": 1697000000,
            "period_end": 1700000000,
            "hosted_invoice_url": "https://invoice.stripe.com/1",
            "invoice_pdf": "https://invoice.stripe.com/1.pdf",
        },
    ]
    return MockStripeClient(subscription=subscription, invoices=invoices)


@pytest.fixture
def billing_handler(user_store, usage_tracker):
    """Create billing handler with mock context."""
    ctx = {
        "user_store": user_store,
        "usage_tracker": usage_tracker,
    }
    return BillingHandler(ctx)


class TestBillingHandlerRouting:
    """Tests for billing handler routing."""

    def test_can_handle_billing_paths(self, billing_handler):
        """Test can_handle identifies billing paths."""
        assert billing_handler.can_handle("/api/billing/plans")
        assert billing_handler.can_handle("/api/billing/usage")
        assert billing_handler.can_handle("/api/billing/subscription")
        assert billing_handler.can_handle("/api/billing/checkout")
        assert billing_handler.can_handle("/api/billing/portal")
        assert billing_handler.can_handle("/api/billing/cancel")
        assert billing_handler.can_handle("/api/billing/resume")
        assert billing_handler.can_handle("/api/billing/audit-log")
        assert billing_handler.can_handle("/api/billing/usage/export")
        assert billing_handler.can_handle("/api/billing/usage/forecast")
        assert billing_handler.can_handle("/api/billing/invoices")
        assert billing_handler.can_handle("/api/webhooks/stripe")

    def test_cannot_handle_non_billing_paths(self, billing_handler):
        """Test can_handle rejects non-billing paths."""
        assert not billing_handler.can_handle("/api/debates")
        assert not billing_handler.can_handle("/api/users")
        assert not billing_handler.can_handle("/api/admin/stats")


class TestRateLimiting:
    """Tests for billing rate limiting."""

    def test_rate_limit_allows_requests(self, billing_handler):
        """Test that rate limiter allows normal requests."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = billing_handler.handle("/api/billing/plans", {}, mock_handler)

            assert result.status_code == 200

    def test_rate_limit_blocks_excessive_requests(self, billing_handler):
        """Test that rate limiter blocks excessive requests."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = billing_handler.handle("/api/billing/plans", {}, mock_handler)

            assert result.status_code == 429
            assert "Rate limit" in parse_body(result)["error"]

    def test_webhooks_bypass_rate_limit(self, billing_handler):
        """Test that webhooks bypass rate limiting."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "test_sig"
        mock_handler.headers["Content-Length"] = "100"

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
                mock_limiter.is_allowed.return_value = False  # Rate limit would block
                mock_parse.return_value = MockWebhookEvent("evt_1", "unhandled.event")

                result = billing_handler.handle(
                    "/api/webhooks/stripe", {}, mock_handler, method="POST"
                )

                # Should still succeed (200) even with rate limit active
                assert result.status_code == 200
                mock_limiter.is_allowed.assert_not_called()


class TestGetPlans:
    """Tests for get subscription plans endpoint."""

    def test_get_plans_success(self, billing_handler):
        """Test successful plans listing."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = billing_handler.handle("/api/billing/plans", {}, mock_handler)
            body = parse_body(result)

            assert result.status_code == 200
            assert "plans" in body
            assert len(body["plans"]) > 0

            # Verify plan structure
            for plan in body["plans"]:
                assert "id" in plan
                assert "name" in plan
                assert "price_monthly" in plan
                assert "features" in plan

    def test_get_plans_no_auth_required(self, billing_handler):
        """Test that plans endpoint doesn't require authentication."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            # No auth mocking needed - endpoint should work without auth
            result = billing_handler.handle("/api/billing/plans", {}, mock_handler)

            assert result.status_code == 200


class TestGetUsage:
    """Tests for get usage endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_get_usage_success(self, billing_handler, user_store):
        """Test successful usage retrieval."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle("/api/billing/usage", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "usage" in body
                assert "debates_used" in body["usage"]
                assert "debates_limit" in body["usage"]
                assert "debates_remaining" in body["usage"]

    def test_get_usage_with_token_tracking(self, billing_handler, user_store):
        """Test usage includes token and cost data when available."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle("/api/billing/usage", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                usage = body["usage"]
                assert "tokens_used" in usage
                assert "tokens_in" in usage
                assert "tokens_out" in usage
                assert "estimated_cost_usd" in usage

    def test_get_usage_without_org_returns_defaults(self, billing_handler, user_store):
        """Test usage for user without organization returns defaults."""
        mock_handler = MockHandler(user_store=user_store)
        no_org_user = user_store.get_user_by_id("no_org_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                # Make the user an owner so they have billing permission
                mock_extract.return_value = self.make_auth_context(no_org_user, role="owner")

                result = billing_handler.handle("/api/billing/usage", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                # Should return default values
                assert body["usage"]["debates_used"] == 0
                assert body["usage"]["debates_limit"] == 10

    def test_get_usage_requires_billing_permission(self, billing_handler, user_store):
        """Test that usage endpoint requires org:billing permission."""
        mock_handler = MockHandler(user_store=user_store)
        member = user_store.get_user_by_id("member_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                # Member role doesn't have org:billing permission
                mock_extract.return_value = self.make_auth_context(member)

                result = billing_handler.handle("/api/billing/usage", {}, mock_handler)

                # Should be rejected due to missing permission
                assert result.status_code == 403

    def test_get_usage_unauthenticated(self, billing_handler, user_store):
        """Test usage returns 401 when not authenticated."""
        mock_handler = MockHandler(user_store=user_store)

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = MockAuthContext("", is_authenticated=False)

                result = billing_handler.handle("/api/billing/usage", {}, mock_handler)

                assert result.status_code == 401


class TestGetSubscription:
    """Tests for get subscription endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_get_subscription_success(self, billing_handler, user_store):
        """Test successful subscription retrieval."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle("/api/billing/subscription", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "subscription" in body
                assert "tier" in body["subscription"]
                assert "status" in body["subscription"]
                assert "is_active" in body["subscription"]

    def test_get_subscription_with_stripe_data(self, billing_handler, user_store, stripe_client):
        """Test subscription includes Stripe data when available."""
        mock_handler = MockHandler(user_store=user_store)
        enterprise_owner = user_store.get_user_by_id("ent_owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(enterprise_owner)
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle("/api/billing/subscription", {}, mock_handler)
                    body = parse_body(result)

                    assert result.status_code == 200
                    sub = body["subscription"]
                    assert sub["tier"] == "enterprise"
                    assert "current_period_end" in sub
                    assert "cancel_at_period_end" in sub

    def test_get_subscription_stripe_error_graceful_degradation(self, billing_handler, user_store):
        """Test that Stripe errors degrade gracefully."""
        mock_handler = MockHandler(user_store=user_store)
        enterprise_owner = user_store.get_user_by_id("ent_owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    from aragora.billing.stripe_client import StripeError

                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(enterprise_owner)
                    mock_stripe = MagicMock()
                    mock_stripe.get_subscription.side_effect = StripeError("API Error")
                    mock_get_stripe.return_value = mock_stripe

                    result = billing_handler.handle("/api/billing/subscription", {}, mock_handler)
                    body = parse_body(result)

                    # Should still succeed with partial data
                    assert result.status_code == 200
                    assert "subscription" in body


class TestCreateCheckout:
    """Tests for create checkout session endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_create_checkout_success(self, billing_handler, user_store, stripe_client):
        """Test successful checkout session creation."""
        mock_handler = MockHandler(
            body={
                "tier": "starter",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    with patch(
                        "aragora.server.handlers.admin.billing.validate_against_schema"
                    ) as mock_validate:
                        mock_limiter.is_allowed.return_value = True
                        mock_extract.return_value = self.make_auth_context(owner)
                        mock_get_stripe.return_value = stripe_client
                        mock_validate.return_value = MagicMock(is_valid=True)

                        result = billing_handler.handle(
                            "/api/billing/checkout", {}, mock_handler, method="POST"
                        )
                        body = parse_body(result)

                        assert result.status_code == 200
                        assert "checkout" in body
                        assert "id" in body["checkout"]
                        assert "url" in body["checkout"]

    def test_create_checkout_free_tier_rejected(self, billing_handler, user_store):
        """Test that checkout for free tier is rejected."""
        mock_handler = MockHandler(
            body={
                "tier": "free",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.validate_against_schema"
                ) as mock_validate:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_validate.return_value = MagicMock(is_valid=True)

                    result = billing_handler.handle(
                        "/api/billing/checkout", {}, mock_handler, method="POST"
                    )

                    assert result.status_code == 400
                    assert "free" in parse_body(result)["error"].lower()

    def test_create_checkout_invalid_tier(self, billing_handler, user_store):
        """Test that invalid tier is rejected."""
        mock_handler = MockHandler(
            body={
                "tier": "invalid_tier",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.validate_against_schema"
                ) as mock_validate:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_validate.return_value = MagicMock(is_valid=True)

                    result = billing_handler.handle(
                        "/api/billing/checkout", {}, mock_handler, method="POST"
                    )

                    assert result.status_code == 400
                    assert "Invalid tier" in parse_body(result)["error"]

    def test_create_checkout_invalid_json(self, billing_handler, user_store):
        """Test that invalid JSON body is rejected."""
        mock_handler = MockHandler(command="POST", user_store=user_store)
        mock_handler.rfile.read.return_value = b"not valid json"
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle(
                    "/api/billing/checkout", {}, mock_handler, method="POST"
                )

                assert result.status_code == 400

    def test_create_checkout_stripe_unavailable(self, billing_handler, user_store):
        """Test handling of Stripe configuration error."""
        mock_handler = MockHandler(
            body={
                "tier": "starter",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    with patch(
                        "aragora.server.handlers.admin.billing.validate_against_schema"
                    ) as mock_validate:
                        from aragora.billing.stripe_client import StripeConfigError

                        mock_limiter.is_allowed.return_value = True
                        mock_extract.return_value = self.make_auth_context(owner)
                        mock_get_stripe.side_effect = StripeConfigError("No API key")
                        mock_validate.return_value = MagicMock(is_valid=True)

                        result = billing_handler.handle(
                            "/api/billing/checkout", {}, mock_handler, method="POST"
                        )

                        assert result.status_code == 503
                        assert "unavailable" in parse_body(result)["error"].lower()


class TestCreatePortal:
    """Tests for create billing portal session endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_create_portal_success(self, billing_handler, user_store, stripe_client):
        """Test successful portal session creation."""
        # Add Stripe customer ID to org
        org = user_store.get_organization_by_id("org_1")
        org.stripe_customer_id = "cus_test_123"

        mock_handler = MockHandler(
            body={"return_url": "https://example.com/settings"},
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle(
                        "/api/billing/portal", {}, mock_handler, method="POST"
                    )
                    body = parse_body(result)

                    assert result.status_code == 200
                    assert "portal" in body
                    assert "url" in body["portal"]

    def test_create_portal_missing_return_url(self, billing_handler, user_store):
        """Test that missing return URL is rejected."""
        mock_handler = MockHandler(body={}, command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle(
                    "/api/billing/portal", {}, mock_handler, method="POST"
                )

                assert result.status_code == 400
                assert "Return URL" in parse_body(result)["error"]

    def test_create_portal_no_stripe_customer(self, billing_handler, user_store):
        """Test portal creation fails without Stripe customer."""
        mock_handler = MockHandler(
            body={"return_url": "https://example.com/settings"},
            command="POST",
            user_store=user_store,
        )
        owner = user_store.get_user_by_id("owner_1")
        # Ensure org has no Stripe customer
        org = user_store.get_organization_by_id("org_1")
        org.stripe_customer_id = None

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle(
                    "/api/billing/portal", {}, mock_handler, method="POST"
                )

                assert result.status_code == 404
                assert "billing account" in parse_body(result)["error"].lower()


class TestCancelSubscription:
    """Tests for cancel subscription endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_cancel_subscription_success(self, billing_handler, user_store, stripe_client):
        """Test successful subscription cancellation."""
        # Setup org with subscription
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = "sub_test_123"

        mock_handler = MockHandler(command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle(
                        "/api/billing/cancel", {}, mock_handler, method="POST"
                    )
                    body = parse_body(result)

                    assert result.status_code == 200
                    assert "message" in body
                    assert "subscription" in body

    def test_cancel_subscription_no_active_subscription(self, billing_handler, user_store):
        """Test cancellation fails without active subscription."""
        mock_handler = MockHandler(command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")
        # Ensure no subscription
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = None

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle(
                    "/api/billing/cancel", {}, mock_handler, method="POST"
                )

                assert result.status_code == 404
                assert "subscription" in parse_body(result)["error"].lower()

    def test_cancel_subscription_logs_audit_event(self, billing_handler, user_store, stripe_client):
        """Test that cancellation logs an audit event."""
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = "sub_test_123"

        mock_handler = MockHandler(command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle(
                        "/api/billing/cancel", {}, mock_handler, method="POST"
                    )

                    assert result.status_code == 200
                    # Check audit log
                    audit_entries = user_store.get_audit_log(org_id="org_1")
                    assert len(audit_entries) == 1
                    assert audit_entries[0]["action"] == "subscription.canceled"


class TestResumeSubscription:
    """Tests for resume subscription endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_resume_subscription_success(self, billing_handler, user_store, stripe_client):
        """Test successful subscription resumption."""
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = "sub_test_123"

        mock_handler = MockHandler(command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_extract.return_value = self.make_auth_context(owner)
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle(
                        "/api/billing/resume", {}, mock_handler, method="POST"
                    )
                    body = parse_body(result)

                    assert result.status_code == 200
                    assert "message" in body
                    assert "resumed" in body["message"].lower()

    def test_resume_subscription_no_subscription(self, billing_handler, user_store):
        """Test resumption fails without subscription."""
        mock_handler = MockHandler(command="POST", user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = None

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle(
                    "/api/billing/resume", {}, mock_handler, method="POST"
                )

                assert result.status_code == 404


class TestAuditLog:
    """Tests for billing audit log endpoint."""

    def make_auth_context(self, user, role=None):
        """Create an auth context for a user."""
        return MockAuthContext(
            user_id=user.id,
            is_authenticated=True,
            org_id=user.org_id,
            role=role or user.role,
        )

    def test_get_audit_log_success(self, billing_handler, user_store):
        """Test successful audit log retrieval."""
        mock_handler = MockHandler(user_store=user_store)
        enterprise_owner = user_store.get_user_by_id("ent_owner_1")

        # Add some audit entries
        user_store.log_audit_event(
            action="subscription.created",
            resource_type="subscription",
            org_id="org_enterprise",
        )

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(enterprise_owner)

                result = billing_handler.handle("/api/billing/audit-log", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "entries" in body
                assert "total" in body

    def test_get_audit_log_requires_enterprise_tier(self, billing_handler, user_store):
        """Test that audit log requires Enterprise tier."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")  # Free tier org owner

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(owner)

                result = billing_handler.handle("/api/billing/audit-log", {}, mock_handler)

                assert result.status_code == 403
                assert "Enterprise" in parse_body(result)["error"]

    def test_get_audit_log_requires_admin_role(self, billing_handler, user_store):
        """Test that audit log requires admin or owner role."""
        # Add a member to enterprise org
        enterprise_member = MockUser(
            id="ent_member_1",
            email="entmember@example.com",
            name="Enterprise Member",
            role="member",
            org_id="org_enterprise",
        )
        user_store.add_user(enterprise_member)

        mock_handler = MockHandler(user_store=user_store)

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_extract.return_value = self.make_auth_context(enterprise_member)

                result = billing_handler.handle("/api/billing/audit-log", {}, mock_handler)

                assert result.status_code == 403
                assert "permission" in parse_body(result)["error"].lower()


class TestUsageExport:
    """Tests for usage export endpoint."""

    def test_export_usage_csv_success(self, billing_handler, user_store):
        """Test successful CSV export."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            # Patch at the billing module level since it imports directly
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_ctx = MockAuthContext(
                    user_id=owner.id,
                    is_authenticated=True,
                )
                mock_extract.return_value = mock_ctx

                result = billing_handler.handle("/api/billing/usage/export", {}, mock_handler)

                assert result.status_code == 200
                assert result.content_type == "text/csv"
                assert "Content-Disposition" in result.headers

    def test_export_usage_requires_auth(self, billing_handler, user_store):
        """Test that export requires authentication."""
        mock_handler = MockHandler(user_store=user_store)

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_ctx = MockAuthContext("", is_authenticated=False)
                mock_extract.return_value = mock_ctx

                result = billing_handler.handle("/api/billing/usage/export", {}, mock_handler)

                assert result.status_code == 401


class TestUsageForecast:
    """Tests for usage forecast endpoint."""

    def test_get_usage_forecast_success(self, billing_handler, user_store):
        """Test successful forecast retrieval."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_ctx = MockAuthContext(
                    user_id=owner.id,
                    is_authenticated=True,
                )
                mock_extract.return_value = mock_ctx

                result = billing_handler.handle("/api/billing/usage/forecast", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "forecast" in body
                assert "current_usage" in body["forecast"]
                assert "projection" in body["forecast"]
                assert "will_hit_limit" in body["forecast"]

    def test_forecast_includes_tier_recommendation(self, billing_handler, user_store):
        """Test that forecast includes tier recommendation when hitting limits."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")

        # Set high usage to trigger recommendation
        org = user_store.get_organization_by_id("org_1")
        org.debates_used_this_month = 9  # Near limit
        org.billing_cycle_start = datetime.now(timezone.utc) - timedelta(days=5)  # High rate

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_ctx = MockAuthContext(
                    user_id=owner.id,
                    is_authenticated=True,
                )
                mock_extract.return_value = mock_ctx

                result = billing_handler.handle("/api/billing/usage/forecast", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert body["forecast"]["will_hit_limit"] is True


class TestInvoices:
    """Tests for invoices endpoint."""

    def test_get_invoices_success(self, billing_handler, user_store, stripe_client):
        """Test successful invoice retrieval."""
        mock_handler = MockHandler(user_store=user_store)
        enterprise_owner = user_store.get_user_by_id("ent_owner_1")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                with patch(
                    "aragora.server.handlers.admin.billing.get_stripe_client"
                ) as mock_get_stripe:
                    mock_limiter.is_allowed.return_value = True
                    mock_ctx = MockAuthContext(
                        user_id=enterprise_owner.id,
                        is_authenticated=True,
                    )
                    mock_extract.return_value = mock_ctx
                    mock_get_stripe.return_value = stripe_client

                    result = billing_handler.handle("/api/billing/invoices", {}, mock_handler)
                    body = parse_body(result)

                    assert result.status_code == 200
                    assert "invoices" in body
                    assert len(body["invoices"]) == 1
                    assert "id" in body["invoices"][0]
                    assert "amount_due" in body["invoices"][0]

    def test_get_invoices_no_billing_account(self, billing_handler, user_store):
        """Test invoices fails without billing account."""
        mock_handler = MockHandler(user_store=user_store)
        owner = user_store.get_user_by_id("owner_1")
        # Ensure no Stripe customer
        org = user_store.get_organization_by_id("org_1")
        org.stripe_customer_id = None

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            with patch(
                "aragora.server.handlers.admin.billing.extract_user_from_request"
            ) as mock_extract:
                mock_limiter.is_allowed.return_value = True
                mock_ctx = MockAuthContext(
                    user_id=owner.id,
                    is_authenticated=True,
                )
                mock_extract.return_value = mock_ctx

                result = billing_handler.handle("/api/billing/invoices", {}, mock_handler)

                assert result.status_code == 404
                assert "billing account" in parse_body(result)["error"].lower()


class TestStripeWebhook:
    """Tests for Stripe webhook handling."""

    def test_webhook_missing_signature_rejected(self, billing_handler):
        """Test that webhooks without signature are rejected."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Content-Length"] = "100"
        # No Stripe-Signature header

        result = billing_handler.handle("/api/webhooks/stripe", {}, mock_handler, method="POST")

        assert result.status_code == 400
        assert "signature" in parse_body(result)["error"].lower()

    def test_webhook_invalid_signature_rejected(self, billing_handler):
        """Test that webhooks with invalid signature are rejected."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "invalid_sig"
        mock_handler.headers["Content-Length"] = "100"

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            mock_parse.return_value = None  # Invalid signature

            result = billing_handler.handle("/api/webhooks/stripe", {}, mock_handler, method="POST")

            assert result.status_code == 400
            assert "signature" in parse_body(result)["error"].lower()

    def test_webhook_duplicate_event_skipped(self, billing_handler):
        """Test that duplicate webhook events are skipped."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                mock_parse.return_value = MockWebhookEvent("evt_123", "test.event")
                mock_dup.return_value = True  # Duplicate

                result = billing_handler.handle(
                    "/api/webhooks/stripe", {}, mock_handler, method="POST"
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["duplicate"] is True

    def test_webhook_checkout_completed(self, billing_handler, user_store):
        """Test checkout.session.completed webhook handling."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        event = MockWebhookEvent(
            event_id="evt_checkout_1",
            event_type="checkout.session.completed",
            object_data={
                "id": "cs_test_123",
                "customer": "cus_new_123",
                "subscription": "sub_new_123",
            },
            metadata={
                "user_id": "owner_1",
                "org_id": "org_1",
                "tier": "starter",
            },
        )

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                with patch("aragora.server.handlers.admin.billing._mark_webhook_processed"):
                    mock_parse.return_value = event
                    mock_dup.return_value = False

                    result = billing_handler.handle(
                        "/api/webhooks/stripe", {}, mock_handler, method="POST"
                    )

                    assert result.status_code == 200

                    # Verify org was updated
                    org = user_store.get_organization_by_id("org_1")
                    assert org.stripe_customer_id == "cus_new_123"
                    assert org.stripe_subscription_id == "sub_new_123"

    def test_webhook_subscription_deleted(self, billing_handler, user_store):
        """Test customer.subscription.deleted webhook handling."""
        # Setup org with subscription - use actual SubscriptionTier
        org = user_store.get_organization_by_id("org_1")
        org.stripe_subscription_id = "sub_to_delete"
        org.tier = SubscriptionTier.STARTER
        user_store._orgs_by_subscription["sub_to_delete"] = org

        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        event = MockWebhookEvent(
            event_id="evt_sub_deleted_1",
            event_type="customer.subscription.deleted",
            object_data={"id": "sub_to_delete"},
        )

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                with patch("aragora.server.handlers.admin.billing._mark_webhook_processed"):
                    mock_parse.return_value = event
                    mock_dup.return_value = False

                    result = billing_handler.handle(
                        "/api/webhooks/stripe", {}, mock_handler, method="POST"
                    )

                    assert result.status_code == 200

                    # Verify org was downgraded
                    org = user_store.get_organization_by_id("org_1")
                    # Tier might be SubscriptionTier enum or have .value
                    tier_value = org.tier.value if hasattr(org.tier, "value") else str(org.tier)
                    assert tier_value == "free"
                    assert org.stripe_subscription_id is None

    def test_webhook_invoice_paid_resets_usage(self, billing_handler, user_store):
        """Test invoice.payment_succeeded webhook resets usage."""
        # Setup org with usage
        org = user_store.get_organization_by_id("org_1")
        org.stripe_customer_id = "cus_invoice_test"
        org.debates_used_this_month = 50
        user_store._orgs_by_customer["cus_invoice_test"] = org

        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        event = MockWebhookEvent(
            event_id="evt_invoice_paid_1",
            event_type="invoice.payment_succeeded",
            object_data={
                "id": "in_test_paid",
                "customer": "cus_invoice_test",
                "subscription": "sub_test",
                "amount_paid": 2900,
            },
        )

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                with patch("aragora.server.handlers.admin.billing._mark_webhook_processed"):
                    with patch(
                        "aragora.billing.payment_recovery.get_recovery_store"
                    ) as mock_recovery:
                        mock_parse.return_value = event
                        mock_dup.return_value = False
                        mock_recovery_store = MagicMock()
                        mock_recovery_store.mark_recovered.return_value = False
                        mock_recovery.return_value = mock_recovery_store

                        result = billing_handler.handle(
                            "/api/webhooks/stripe", {}, mock_handler, method="POST"
                        )

                        assert result.status_code == 200

                        # Verify usage was reset
                        org = user_store.get_organization_by_id("org_1")
                        assert org.debates_used_this_month == 0

    def test_webhook_invoice_failed_records_failure(self, billing_handler, user_store):
        """Test invoice.payment_failed webhook records failure."""
        # Setup org
        org = user_store.get_organization_by_id("org_1")
        org.stripe_customer_id = "cus_fail_test"
        user_store._orgs_by_customer["cus_fail_test"] = org

        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        event = MockWebhookEvent(
            event_id="evt_invoice_failed_1",
            event_type="invoice.payment_failed",
            object_data={
                "id": "in_test_failed",
                "customer": "cus_fail_test",
                "subscription": "sub_test",
                "attempt_count": 1,
                "hosted_invoice_url": "https://invoice.stripe.com/failed",
            },
        )

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                with patch("aragora.server.handlers.admin.billing._mark_webhook_processed"):
                    with patch(
                        "aragora.billing.payment_recovery.get_recovery_store"
                    ) as mock_recovery:
                        with patch(
                            "aragora.billing.notifications.get_billing_notifier"
                        ) as mock_notifier:
                            mock_parse.return_value = event
                            mock_dup.return_value = False

                            # Setup recovery store mock
                            mock_failure = MagicMock()
                            mock_failure.attempt_count = 1
                            mock_failure.days_failing = 1
                            mock_failure.days_until_downgrade = 13
                            mock_recovery_store = MagicMock()
                            mock_recovery_store.record_failure.return_value = mock_failure
                            mock_recovery.return_value = mock_recovery_store

                            # Setup notifier mock
                            mock_notify_result = MagicMock()
                            mock_notify_result.method = "email"
                            mock_notify_result.success = True
                            mock_notifier_instance = MagicMock()
                            mock_notifier_instance.notify_payment_failed.return_value = (
                                mock_notify_result
                            )
                            mock_notifier.return_value = mock_notifier_instance

                            result = billing_handler.handle(
                                "/api/webhooks/stripe", {}, mock_handler, method="POST"
                            )
                            body = parse_body(result)

                            assert result.status_code == 200
                            assert body["failure_tracked"] is True

    def test_webhook_unhandled_event_acknowledged(self, billing_handler):
        """Test that unhandled webhook events are acknowledged."""
        mock_handler = MockHandler(command="POST")
        mock_handler.headers["Stripe-Signature"] = "valid_sig"
        mock_handler.headers["Content-Length"] = "100"

        event = MockWebhookEvent(
            event_id="evt_unknown_1",
            event_type="unknown.event.type",
        )

        with patch("aragora.billing.stripe_client.parse_webhook_event") as mock_parse:
            with patch("aragora.server.handlers.admin.billing._is_duplicate_webhook") as mock_dup:
                mock_parse.return_value = event
                mock_dup.return_value = False

                result = billing_handler.handle(
                    "/api/webhooks/stripe", {}, mock_handler, method="POST"
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["received"] is True


class TestMethodNotAllowed:
    """Tests for method not allowed handling."""

    def test_unsupported_method_returns_405(self, billing_handler):
        """Test that unsupported methods return 405."""
        mock_handler = MockHandler(command="DELETE")

        with patch("aragora.server.handlers.admin.billing._billing_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = billing_handler.handle("/api/billing/plans", {}, mock_handler, method="DELETE")

            assert result.status_code == 405
