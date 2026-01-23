"""
Tests for BillingHandler - subscription and billing endpoints.

Tests cover:
- GET /api/billing/plans - List subscription plans
- GET /api/billing/usage - Get current usage
- GET /api/billing/subscription - Get subscription status
- POST /api/billing/checkout - Create checkout session
- POST /api/billing/portal - Create billing portal
- POST /api/billing/cancel - Cancel subscription
- POST /api/billing/resume - Resume subscription
- POST /api/webhooks/stripe - Stripe webhooks

Security tests:
- Authentication required for usage/subscription endpoints
- Webhook signature validation
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from aragora.server.handlers.admin import BillingHandler


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = Mock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.org_id = "org-456"
    user.role = "member"
    user.is_active = True
    user.to_dict = Mock(
        return_value={
            "id": "user-123",
            "email": "test@example.com",
        }
    )
    return user


@pytest.fixture
def mock_org():
    """Create a mock organization object."""
    limits = Mock()
    limits.debates_per_month = 100
    limits.users_per_org = 10
    limits.api_access = True
    limits.all_agents = True
    limits.to_dict = Mock(
        return_value={
            "debates_per_month": 100,
            "users_per_org": 10,
        }
    )

    org = Mock()
    org.id = "org-456"
    org.name = "Test Org"
    org.tier = Mock(value="pro")
    org.limits = limits
    org.debates_used_this_month = 25
    org.debates_remaining = 75
    org.billing_cycle_start = datetime(2026, 1, 1)
    org.stripe_customer_id = "cus_123"
    org.stripe_subscription_id = None
    org.to_dict = Mock(
        return_value={
            "id": "org-456",
            "name": "Test Org",
        }
    )
    return org


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    """Create a mock user store."""
    store = Mock()
    store.get_user_by_id = Mock(return_value=mock_user)
    store.get_organization_by_id = Mock(return_value=mock_org)
    return store


@pytest.fixture
def mock_usage_tracker():
    """Create a mock usage tracker."""
    tracker = Mock()
    summary = Mock()
    summary.total_tokens = 50000
    summary.total_tokens_in = 40000
    summary.total_tokens_out = 10000
    summary.total_cost = 2.50
    summary.total_cost_usd = 2.50
    summary.by_provider = {"anthropic": 40000, "openai": 10000}
    summary.cost_by_provider = {"anthropic": "2.00", "openai": "0.50"}
    tracker.get_summary = Mock(return_value=summary)
    return tracker


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.command = "GET"
    handler.headers = {"Content-Type": "application/json"}
    handler.rfile = Mock()
    return handler


@pytest.fixture
def billing_handler(mock_user_store, mock_usage_tracker):
    """Create BillingHandler with mock dependencies."""
    ctx = {
        "user_store": mock_user_store,
        "usage_tracker": mock_usage_tracker,
    }
    return BillingHandler(ctx)


# ============================================================================
# Route Tests
# ============================================================================


class TestBillingHandlerRoutes:
    """Tests for BillingHandler routing."""

    def test_can_handle_plans(self, billing_handler):
        """Test handler recognizes plans route."""
        assert billing_handler.can_handle("/api/v1/billing/plans")

    def test_can_handle_usage(self, billing_handler):
        """Test handler recognizes usage route."""
        assert billing_handler.can_handle("/api/v1/billing/usage")

    def test_can_handle_subscription(self, billing_handler):
        """Test handler recognizes subscription route."""
        assert billing_handler.can_handle("/api/v1/billing/subscription")

    def test_can_handle_checkout(self, billing_handler):
        """Test handler recognizes checkout route."""
        assert billing_handler.can_handle("/api/v1/billing/checkout")

    def test_can_handle_portal(self, billing_handler):
        """Test handler recognizes portal route."""
        assert billing_handler.can_handle("/api/v1/billing/portal")

    def test_can_handle_cancel(self, billing_handler):
        """Test handler recognizes cancel route."""
        assert billing_handler.can_handle("/api/v1/billing/cancel")

    def test_can_handle_resume(self, billing_handler):
        """Test handler recognizes resume route."""
        assert billing_handler.can_handle("/api/v1/billing/resume")

    def test_can_handle_stripe_webhook(self, billing_handler):
        """Test handler recognizes Stripe webhook route."""
        assert billing_handler.can_handle("/api/v1/webhooks/stripe")

    def test_cannot_handle_unknown_route(self, billing_handler):
        """Test handler rejects unknown routes."""
        assert not billing_handler.can_handle("/api/v1/billing/unknown")
        assert not billing_handler.can_handle("/api/v1/payments")


# ============================================================================
# Plans Endpoint Tests
# ============================================================================


class TestGetPlans:
    """Tests for get plans endpoint."""

    @patch("aragora.server.handlers.billing.TIER_LIMITS")
    @patch("aragora.server.handlers.billing.SubscriptionTier")
    def test_get_plans_returns_all_tiers(self, mock_tier_enum, mock_limits, billing_handler):
        """Test that all subscription tiers are returned."""
        # Mock the tier enum
        mock_free = Mock(value="free", name="FREE")
        mock_pro = Mock(value="pro", name="PRO")
        mock_tier_enum.__iter__ = Mock(return_value=iter([mock_free, mock_pro]))

        # Mock limits
        mock_free_limits = Mock()
        mock_free_limits.price_monthly_cents = 0
        mock_free_limits.debates_per_month = 10
        mock_free_limits.users_per_org = 1
        mock_free_limits.api_access = False
        mock_free_limits.all_agents = False
        mock_free_limits.custom_agents = False
        mock_free_limits.sso_enabled = False
        mock_free_limits.audit_logs = False
        mock_free_limits.priority_support = False

        mock_pro_limits = Mock()
        mock_pro_limits.price_monthly_cents = 2900
        mock_pro_limits.debates_per_month = 100
        mock_pro_limits.users_per_org = 10
        mock_pro_limits.api_access = True
        mock_pro_limits.all_agents = True
        mock_pro_limits.custom_agents = True
        mock_pro_limits.sso_enabled = False
        mock_pro_limits.audit_logs = True
        mock_pro_limits.priority_support = False

        mock_limits.__getitem__ = Mock(
            side_effect=lambda x: {
                mock_free: mock_free_limits,
                mock_pro: mock_pro_limits,
            }[x]
        )

        result = billing_handler._get_plans()

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "plans" in data
        assert len(data["plans"]) == 2


# ============================================================================
# Usage Endpoint Tests
# ============================================================================


class TestGetUsage:
    """Tests for get usage endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_usage_success(self, mock_extract, billing_handler, mock_handler):
        """Test successful usage retrieval."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",  # Required for org:billing permission
        )

        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "usage" in data
        assert "debates_used" in data["usage"]
        assert "debates_limit" in data["usage"]
        assert "debates_remaining" in data["usage"]

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_usage_with_token_tracking(self, mock_extract, billing_handler, mock_handler):
        """Test usage includes token tracking data."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage"]["tokens_used"] == 50000
        assert data["usage"]["estimated_cost_usd"] == 2.50

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_usage_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test usage requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 401

    def test_get_usage_user_store_unavailable(self, mock_handler):
        """Test usage when user store unavailable."""
        handler = BillingHandler({})  # No user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = Mock(
                is_authenticated=True, user_id="user-123", role="owner"
            )
            result = handler._get_usage(mock_handler)

        assert result.status_code == 503


# ============================================================================
# Subscription Endpoint Tests
# ============================================================================


class TestGetSubscription:
    """Tests for get subscription endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_subscription_success(self, mock_extract, billing_handler, mock_handler):
        """Test successful subscription retrieval."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        result = billing_handler._get_subscription(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "subscription" in data
        assert data["subscription"]["tier"] == "pro"
        assert data["subscription"]["is_active"] is True

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_subscription_free_tier(
        self, mock_extract, billing_handler, mock_handler, mock_user_store
    ):
        """Test subscription for free tier user."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_user_store.get_organization_by_id.return_value = None

        result = billing_handler._get_subscription(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["subscription"]["tier"] == "free"

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_subscription_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test subscription requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = billing_handler._get_subscription(mock_handler)

        assert result.status_code == 401


# ============================================================================
# Checkout Endpoint Tests
# ============================================================================


class TestCreateCheckout:
    """Tests for create checkout session endpoint."""

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.server.handlers.billing.SubscriptionTier")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_checkout_success(
        self, mock_extract, mock_tier_enum, mock_stripe, billing_handler, mock_handler
    ):
        """Test successful checkout session creation."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_tier_enum.return_value = Mock(value="pro")
        mock_tier_enum.FREE = Mock(value="free")

        mock_session = Mock()
        mock_session.id = "cs_123"
        mock_session.to_dict = Mock(
            return_value={
                "id": "cs_123",
                "url": "https://checkout.stripe.com/...",
            }
        )
        mock_stripe.return_value.create_checkout_session.return_value = mock_session

        billing_handler.read_json_body = Mock(
            return_value={
                "tier": "professional",
                "success_url": "https://app.example.com/success",
                "cancel_url": "https://app.example.com/cancel",
            }
        )

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "checkout" in data

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_checkout_missing_tier(self, mock_extract, billing_handler, mock_handler):
        """Test checkout without tier."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        billing_handler.read_json_body = Mock(
            return_value={
                "success_url": "https://app.example.com/success",
                "cancel_url": "https://app.example.com/cancel",
            }
        )

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "tier" in data["error"].lower()

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_checkout_missing_urls(self, mock_extract, billing_handler, mock_handler):
        """Test checkout without success/cancel URLs."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        billing_handler.read_json_body = Mock(
            return_value={
                "tier": "professional",
                # Missing success_url and cancel_url
            }
        )

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        # Schema validation requires success_url and cancel_url
        assert "required" in data["error"].lower() or "success_url" in data["error"]

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_checkout_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test checkout requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 401

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_checkout_free_tier_rejected(self, mock_extract, billing_handler, mock_handler):
        """Test cannot checkout free tier - rejected by schema validation."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        # "free" is not a valid tier in CHECKOUT_SESSION_SCHEMA
        # Valid tiers are: starter, professional, enterprise
        billing_handler.read_json_body = Mock(
            return_value={
                "tier": "free",
                "success_url": "https://app.example.com/success",
                "cancel_url": "https://app.example.com/cancel",
            }
        )

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        # Schema validation rejects "free" as invalid tier
        assert "tier" in data["error"].lower()


# ============================================================================
# Portal Endpoint Tests
# ============================================================================


class TestCreatePortal:
    """Tests for create billing portal endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_portal_missing_return_url(self, mock_extract, billing_handler, mock_handler):
        """Test portal requires return URL."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        billing_handler.read_json_body = Mock(return_value={})

        result = billing_handler._create_portal(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "url" in data["error"].lower()

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_create_portal_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test portal requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        result = billing_handler._create_portal(mock_handler)

        assert result.status_code == 401


# ============================================================================
# Cancel/Resume Tests
# ============================================================================


class TestCancelSubscription:
    """Tests for cancel subscription endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cancel_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test cancel requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        # Need to set method to POST
        mock_handler.command = "POST"
        billing_handler.read_json_body = Mock(return_value={})

        result = billing_handler.handle("/api/billing/cancel", {}, mock_handler, "POST")

        assert result.status_code == 401


class TestResumeSubscription:
    """Tests for resume subscription endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_resume_not_authenticated(self, mock_extract, billing_handler, mock_handler):
        """Test resume requires authentication."""
        mock_extract.return_value = Mock(is_authenticated=False)

        mock_handler.command = "POST"
        billing_handler.read_json_body = Mock(return_value={})

        result = billing_handler.handle("/api/billing/resume", {}, mock_handler, "POST")

        assert result.status_code == 401


# ============================================================================
# Webhook Tests
# ============================================================================


class TestStripeWebhook:
    """Tests for Stripe webhook handling."""

    def test_webhook_route_recognized(self, billing_handler):
        """Test webhook route is recognized."""
        assert billing_handler.can_handle("/api/v1/webhooks/stripe")


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurityMeasures:
    """Tests for security measures in billing handler."""

    def test_all_endpoints_check_authentication(self, billing_handler, mock_handler):
        """Test that all user-specific endpoints require authentication."""
        auth_required_endpoints = [
            ("/api/billing/usage", "GET"),
            ("/api/billing/subscription", "GET"),
            ("/api/billing/checkout", "POST"),
            ("/api/billing/portal", "POST"),
            ("/api/billing/cancel", "POST"),
            ("/api/billing/resume", "POST"),
        ]

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = Mock(is_authenticated=False)

            for path, method in auth_required_endpoints:
                mock_handler.command = method
                if method == "POST":
                    billing_handler.read_json_body = Mock(return_value={})

                result = billing_handler.handle(path, {}, mock_handler, method)

                assert result.status_code == 401, f"Endpoint {path} should require authentication"

    def test_plans_endpoint_public(self, billing_handler):
        """Test that plans endpoint is publicly accessible."""
        # Plans should not require authentication
        with (
            patch("aragora.server.handlers.billing.TIER_LIMITS") as mock_limits,
            patch("aragora.server.handlers.billing.SubscriptionTier") as mock_tier,
        ):
            mock_tier.__iter__ = Mock(return_value=iter([]))
            result = billing_handler._get_plans()

            assert result.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in billing handler."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_invalid_json_body(self, mock_extract, billing_handler, mock_handler):
        """Test handling of invalid JSON body."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )

        billing_handler.read_json_body = Mock(return_value=None)

        result = billing_handler._create_checkout(mock_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "JSON" in data["error"]

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_user_not_found(self, mock_extract, billing_handler, mock_handler, mock_user_store):
        """Test handling when user not found."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="nonexistent",
            role="owner",  # Required for org:billing permission
        )
        mock_user_store.get_user_by_id.return_value = None

        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 404


# ============================================================================
# Stripe Exception Handling Tests
# ============================================================================


class TestStripeExceptionHandling:
    """Tests for Stripe exception handling in billing handler.

    These tests verify that Stripe exceptions are properly caught and
    mapped to appropriate HTTP status codes.

    Note: The tests patch at the source module (aragora.billing.jwt_auth)
    since the decorator imports extract_user_from_request at runtime.
    """

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_subscription_handles_stripe_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _get_subscription gracefully handles StripeError."""
        from aragora.billing.stripe_client import StripeError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        # Make Stripe client raise error
        mock_stripe.return_value.get_subscription.side_effect = StripeError("Connection failed")

        result = billing_handler._get_subscription(mock_handler, user=user_ctx)

        # Should succeed with partial data (graceful degradation)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "subscription" in data
        # The Stripe-specific fields won't be populated

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.server.handlers.billing.extract_user_from_request")
    def test_get_invoices_handles_stripe_config_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _get_invoices returns 503 on StripeConfigError."""
        from aragora.billing.stripe_client import StripeConfigError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_customer_id = "cus_123"

        # Mock query param extraction
        mock_handler.path = "/api/billing/invoices"
        mock_handler.headers = {"Authorization": "Bearer test"}

        mock_stripe.return_value.list_invoices.side_effect = StripeConfigError("Not configured")

        with patch("aragora.server.handlers.billing.get_string_param", return_value="10"):
            result = billing_handler._get_invoices(mock_handler)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "unavailable" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.server.handlers.billing.extract_user_from_request")
    def test_get_invoices_handles_stripe_api_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _get_invoices returns 502 on StripeAPIError."""
        from aragora.billing.stripe_client import StripeAPIError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_customer_id = "cus_123"

        mock_stripe.return_value.list_invoices.side_effect = StripeAPIError(
            "API error", code="api_error"
        )

        with patch("aragora.server.handlers.billing.get_string_param", return_value="10"):
            result = billing_handler._get_invoices(mock_handler)

        assert result.status_code == 502
        data = json.loads(result.body)
        assert "payment provider" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.server.handlers.billing.extract_user_from_request")
    def test_get_invoices_handles_generic_stripe_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _get_invoices returns 500 on generic StripeError."""
        from aragora.billing.stripe_client import StripeError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_customer_id = "cus_123"

        mock_stripe.return_value.list_invoices.side_effect = StripeError("Unknown error")

        with patch("aragora.server.handlers.billing.get_string_param", return_value="10"):
            result = billing_handler._get_invoices(mock_handler)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cancel_subscription_handles_stripe_config_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _cancel_subscription returns 503 on StripeConfigError."""
        from aragora.billing.stripe_client import StripeConfigError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        mock_stripe.return_value.cancel_subscription.side_effect = StripeConfigError(
            "Not configured"
        )

        result = billing_handler._cancel_subscription(mock_handler, user=user_ctx)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "unavailable" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cancel_subscription_handles_stripe_api_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _cancel_subscription returns 502 on StripeAPIError."""
        from aragora.billing.stripe_client import StripeAPIError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        mock_stripe.return_value.cancel_subscription.side_effect = StripeAPIError(
            "Subscription not found", code="resource_missing"
        )

        result = billing_handler._cancel_subscription(mock_handler, user=user_ctx)

        assert result.status_code == 502
        data = json.loads(result.body)
        assert "payment provider" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_resume_subscription_handles_stripe_config_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _resume_subscription returns 503 on StripeConfigError."""
        from aragora.billing.stripe_client import StripeConfigError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        mock_stripe.return_value.resume_subscription.side_effect = StripeConfigError(
            "Not configured"
        )

        result = billing_handler._resume_subscription(mock_handler, user=user_ctx)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "unavailable" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_resume_subscription_handles_stripe_api_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _resume_subscription returns 502 on StripeAPIError."""
        from aragora.billing.stripe_client import StripeAPIError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        mock_stripe.return_value.resume_subscription.side_effect = StripeAPIError(
            "Cannot resume", code="subscription_status_invalid"
        )

        result = billing_handler._resume_subscription(mock_handler, user=user_ctx)

        assert result.status_code == 502
        data = json.loads(result.body)
        assert "payment provider" in data["error"].lower()

    @patch("aragora.server.handlers.billing.get_stripe_client")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_resume_subscription_handles_generic_stripe_error(
        self, mock_extract, mock_stripe, billing_handler, mock_handler, mock_org
    ):
        """Test _resume_subscription returns 500 on generic StripeError."""
        from aragora.billing.stripe_client import StripeError

        user_ctx = Mock(
            is_authenticated=True,
            user_id="user-123",
            role="owner",
        )
        mock_extract.return_value = user_ctx
        mock_org.stripe_subscription_id = "sub_123"

        mock_stripe.return_value.resume_subscription.side_effect = StripeError("Unknown error")

        result = billing_handler._resume_subscription(mock_handler, user=user_ctx)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "resume" in data["error"].lower()
