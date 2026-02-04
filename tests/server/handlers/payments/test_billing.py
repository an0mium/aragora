"""
Tests for payments billing handlers.

Tests cover:
- Customer profile CRUD (create, read, update, delete)
- Subscription management (create, read, update, cancel)
- RBAC permission enforcement
- Rate limiting
- Error handling
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext, PermissionDecision


# Mock for StripeConnector
class MockStripeCustomer:
    def __init__(
        self,
        id="cus_123",
        email="test@example.com",
        name="Test User",
        created=1640000000,
        metadata=None,
        deleted=False,
    ):
        self.id = id
        self.email = email
        self.name = name
        self.created = created
        self.metadata = metadata or {}
        self.deleted = deleted


class MockStripeSubscription:
    def __init__(
        self,
        id="sub_123",
        status="active",
        current_period_start=1640000000,
        current_period_end=1642678800,
        customer="cus_123",
    ):
        self.id = id
        self.status = status
        self.current_period_start = current_period_start
        self.current_period_end = current_period_end
        self.customer = customer
        self.items = MagicMock()
        self.items.data = [MagicMock(id="si_123", price=MagicMock(id="price_123"), quantity=1)]


class MockAuthnetProfile:
    def __init__(
        self,
        profile_id="profile_123",
        merchant_customer_id="mcid_123",
        email="test@example.com",
        description="Test User",
        payment_profiles=None,
    ):
        self.profile_id = profile_id
        self.merchant_customer_id = merchant_customer_id
        self.email = email
        self.description = description
        self.payment_profiles = payment_profiles or []


class MockAuthnetSubscription:
    def __init__(self, subscription_id="arb_123", name="Premium Plan", status=None, amount=None):
        self.subscription_id = subscription_id
        self.name = name
        self.status = status or MagicMock(value="active")
        self.amount = amount or Decimal("99.99")


def create_mock_request(match_info=None, query=None, body=None, user_id="test_user"):
    """Create a mock aiohttp request."""
    request = MagicMock()
    request.get.return_value = user_id
    request.match_info = match_info or {}
    request.query = query or {}
    request.headers = {}
    request.transport = None

    if body is not None:

        async def read_json():
            return body

        request.json = read_json
        request.read = AsyncMock(return_value=json.dumps(body).encode())
        request.content_type = "application/json"

    return request


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter to always allow."""
    with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
        mock_pkg.return_value._check_rate_limit.return_value = None
        mock_pkg.return_value._get_provider_from_request.side_effect = lambda r, b: (
            "authorize_net" if b.get("provider") in ("authorize_net", "authnet") else "stripe"
        )
        yield mock_pkg


@pytest.fixture
def mock_permission_allowed():
    """Mock permission checker to always allow."""
    with patch("aragora.rbac.decorators.get_permission_checker") as mock_checker:
        checker = MagicMock()
        checker.check_permission.return_value = PermissionDecision(allowed=True)
        mock_checker.return_value = checker
        yield checker


class TestCustomerHandlers:
    """Tests for customer profile handlers."""

    @pytest.mark.asyncio
    async def test_create_customer_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe customer creation."""
        from aragora.server.handlers.payments.billing import handle_create_customer

        request = create_mock_request(
            body={
                "email": "new@example.com",
                "name": "New Customer",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.create_customer.return_value = MockStripeCustomer(
            id="cus_new", email="new@example.com"
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = MagicMock(
                value="stripe"
            )
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            # Set PaymentProvider.STRIPE for comparison
            from aragora.server.handlers.payments import PaymentProvider

            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_create_customer(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["customer_id"] == "cus_new"

    @pytest.mark.asyncio
    async def test_create_customer_connector_unavailable(self, mock_permission_allowed):
        """Test customer creation when connector unavailable."""
        from aragora.server.handlers.payments.billing import handle_create_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "email": "new@example.com",
                "name": "New Customer",
            }
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=None)

            response = await handle_create_customer(request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_get_customer_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe customer retrieval."""
        from aragora.server.handlers.payments.billing import handle_get_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"customer_id": "cus_123"},
            query={"provider": "stripe"},
        )

        mock_connector = AsyncMock()
        mock_connector.retrieve_customer.return_value = MockStripeCustomer()

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_customer(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["customer"]["id"] == "cus_123"

    @pytest.mark.asyncio
    async def test_get_customer_missing_id(self, mock_permission_allowed):
        """Test customer retrieval with missing ID."""
        from aragora.server.handlers.payments.billing import handle_get_customer

        request = create_mock_request(match_info={})

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None

            response = await handle_get_customer(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_delete_customer_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe customer deletion."""
        from aragora.server.handlers.payments.billing import handle_delete_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"customer_id": "cus_123"},
            query={"provider": "stripe"},
        )

        mock_connector = AsyncMock()
        mock_connector.delete_customer.return_value = MagicMock(deleted=True)

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_delete_customer(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_update_customer_no_params(self, mock_permission_allowed):
        """Test customer update with no parameters."""
        from aragora.server.handlers.payments.billing import handle_update_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"customer_id": "cus_123"},
            body={"provider": "stripe"},
        )

        mock_connector = AsyncMock()

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_update_customer(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_update_customer_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe customer update."""
        from aragora.server.handlers.payments.billing import handle_update_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"customer_id": "cus_123"},
            body={
                "provider": "stripe",
                "email": "updated@example.com",
                "name": "Updated Name",
            },
        )

        mock_connector = AsyncMock()
        mock_connector.update_customer.return_value = MockStripeCustomer(
            id="cus_123", email="updated@example.com", name="Updated Name"
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_update_customer(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True


class TestSubscriptionHandlers:
    """Tests for subscription handlers."""

    @pytest.mark.asyncio
    async def test_get_subscription_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe subscription retrieval."""
        from aragora.server.handlers.payments.billing import handle_get_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"subscription_id": "sub_123"},
            query={"provider": "stripe"},
        )

        mock_connector = AsyncMock()
        mock_connector.retrieve_subscription.return_value = MockStripeSubscription()

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_subscription(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["subscription"]["id"] == "sub_123"
        assert data["subscription"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_subscription_missing_id(self, mock_permission_allowed):
        """Test subscription retrieval with missing ID."""
        from aragora.server.handlers.payments.billing import handle_get_subscription

        request = create_mock_request(match_info={})

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None

            response = await handle_get_subscription(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_create_subscription_missing_customer(self, mock_permission_allowed):
        """Test subscription creation with missing customer."""
        from aragora.server.handlers.payments.billing import handle_create_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "provider": "stripe",
                "amount": 99.99,
            }
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_create_subscription(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_create_subscription_invalid_amount(self, mock_permission_allowed):
        """Test subscription creation with invalid amount."""
        from aragora.server.handlers.payments.billing import handle_create_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "provider": "stripe",
                "customer_id": "cus_123",
                "amount": 0,
            }
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_create_subscription(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_create_subscription_stripe_requires_price_id(self, mock_permission_allowed):
        """Test Stripe subscription requires price_id."""
        from aragora.server.handlers.payments.billing import handle_create_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "provider": "stripe",
                "customer_id": "cus_123",
                "amount": 99.99,
            }
        )

        mock_connector = AsyncMock()

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_create_subscription(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "price_id" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_cancel_subscription_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe subscription cancellation."""
        from aragora.server.handlers.payments.billing import handle_cancel_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"subscription_id": "sub_123"},
            query={"provider": "stripe"},
        )

        mock_connector = AsyncMock()
        mock_connector.cancel_subscription.return_value = MockStripeSubscription(
            id="sub_123", status="canceled"
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_cancel_subscription(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_update_subscription_no_params(self, mock_permission_allowed):
        """Test subscription update with no parameters."""
        from aragora.server.handlers.payments.billing import handle_update_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"subscription_id": "sub_123"},
            body={"provider": "stripe"},
        )

        mock_connector = AsyncMock()

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_update_subscription(request)

        assert response.status == 400


class TestAuthNetIntegration:
    """Tests for Authorize.net specific functionality."""

    @pytest.mark.asyncio
    async def test_create_customer_authnet_success(self, mock_permission_allowed):
        """Test successful Authorize.net customer creation."""
        from aragora.server.handlers.payments.billing import handle_create_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "email": "authnet@example.com",
                "name": "AuthNet Customer",
                "provider": "authorize_net",
                "merchant_customer_id": "mcid_456",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.create_customer_profile.return_value = MockAuthnetProfile(
            profile_id="profile_456", merchant_customer_id="mcid_456"
        )
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = (
                PaymentProvider.AUTHORIZE_NET
            )
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_create_customer(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["customer_id"] == "profile_456"

    @pytest.mark.asyncio
    async def test_get_customer_authnet_not_found(self, mock_permission_allowed):
        """Test Authorize.net customer not found."""
        from aragora.server.handlers.payments.billing import handle_get_customer
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"customer_id": "profile_invalid"},
            query={"provider": "authnet"},
        )

        mock_connector = AsyncMock()
        mock_connector.get_customer_profile.return_value = None
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_customer(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_get_subscription_authnet_not_found(self, mock_permission_allowed):
        """Test Authorize.net subscription not found."""
        from aragora.server.handlers.payments.billing import handle_get_subscription
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"subscription_id": "arb_invalid"},
            query={"provider": "authnet"},
        )

        mock_connector = AsyncMock()
        mock_connector.get_subscription.return_value = None
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_subscription(request)

        assert response.status == 404


class TestRateLimiting:
    """Tests for rate limiting in billing handlers."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_on_create(self, mock_permission_allowed):
        """Test rate limit exceeded on customer creation."""
        from aragora.server.handlers.payments.billing import handle_create_customer
        from aiohttp import web

        request = create_mock_request(body={"email": "test@example.com"})

        rate_limit_response = web.json_response(
            {"error": "Rate limit exceeded"},
            status=429,
        )

        with patch("aragora.server.handlers.payments.billing._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = rate_limit_response

            response = await handle_create_customer(request)

        assert response.status == 429


class TestPermissionEnforcement:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_permission_denied_on_create(self):
        """Test permission denied on customer creation."""
        from aragora.server.handlers.payments.billing import (
            handle_create_customer,
            _enforce_permission_if_context,
        )

        auth_ctx = AuthorizationContext(
            user_id="user_123",
            roles=["viewer"],
            permissions=set(),
        )

        with patch("aragora.rbac.decorators.get_permission_checker") as mock_checker:
            checker = MagicMock()
            checker.check_permission.return_value = PermissionDecision(
                allowed=False, reason="Permission denied: payments:customer:create"
            )
            mock_checker.return_value = checker

            with pytest.raises(PermissionDeniedError):
                _enforce_permission_if_context(auth_ctx, "payments:customer:create")
