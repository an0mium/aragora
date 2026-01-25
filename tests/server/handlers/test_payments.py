"""
Tests for aragora.server.handlers.payments - Payment processing handler.

Tests cover:
- Payment charge operations (Stripe and Authorize.net)
- Payment authorization and capture
- Refund and void operations
- Customer profile management
- Subscription management
- Webhook handling
- Error handling and validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from aragora.server.handlers.payments import (
    PaymentProvider,
    PaymentStatus,
    PaymentRequest,
    PaymentResult,
    handle_charge,
    handle_authorize,
    handle_capture,
    handle_refund,
    handle_void,
    handle_get_transaction,
    handle_create_customer,
    handle_get_customer,
    handle_delete_customer,
    handle_create_subscription,
    handle_cancel_subscription,
    handle_stripe_webhook,
    handle_authnet_webhook,
    _get_provider_from_request,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockStripePaymentIntent:
    """Mock Stripe PaymentIntent."""

    id: str = "pi_test123"
    status: str = "succeeded"
    amount: int = 10000
    currency: str = "usd"
    client_secret: str = "pi_test123_secret"
    created: int = 1704067200
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockStripeCustomer:
    """Mock Stripe Customer."""

    id: str = "cus_test123"
    email: str = "test@example.com"
    name: str = "Test Customer"
    created: int = 1704067200
    metadata: dict[str, Any] = field(default_factory=dict)
    deleted: bool = False


@dataclass
class MockStripeSubscription:
    """Mock Stripe Subscription."""

    id: str = "sub_test123"
    status: str = "active"
    current_period_end: int = 1706745600


@dataclass
class MockStripeRefund:
    """Mock Stripe Refund."""

    id: str = "re_test123"
    status: str = "succeeded"


@dataclass
class MockAuthnetResult:
    """Mock Authorize.net transaction result."""

    transaction_id: str = "123456789"
    approved: bool = True
    message: str = "Transaction approved"
    auth_code: str = "AUTH123"
    avs_result: str = "Y"
    cvv_result: str = "M"


@dataclass
class MockAuthnetCustomerProfile:
    """Mock Authorize.net customer profile."""

    profile_id: str = "12345678"
    merchant_customer_id: str = "cust_123"
    email: str = "test@example.com"
    description: str = "Test Customer"
    payment_profiles: list[Any] = field(default_factory=list)


@dataclass
class MockAuthnetSubscription:
    """Mock Authorize.net subscription."""

    subscription_id: str = "987654321"
    name: str = "Test Subscription"
    status: MagicMock = field(default_factory=lambda: MagicMock(value="active"))


@pytest.fixture
def mock_stripe_connector():
    """Create a mock Stripe connector."""
    connector = AsyncMock()
    connector.create_payment_intent = AsyncMock(return_value=MockStripePaymentIntent())
    connector.capture_payment_intent = AsyncMock(return_value=MockStripePaymentIntent())
    connector.cancel_payment_intent = AsyncMock(
        return_value=MockStripePaymentIntent(status="canceled")
    )
    connector.retrieve_payment_intent = AsyncMock(return_value=MockStripePaymentIntent())
    connector.create_refund = AsyncMock(return_value=MockStripeRefund())
    connector.create_customer = AsyncMock(return_value=MockStripeCustomer())
    connector.retrieve_customer = AsyncMock(return_value=MockStripeCustomer())
    connector.delete_customer = AsyncMock(return_value=MockStripeCustomer(deleted=True))
    connector.create_subscription = AsyncMock(return_value=MockStripeSubscription())
    connector.cancel_subscription = AsyncMock(
        return_value=MockStripeSubscription(status="canceled")
    )
    return connector


@pytest.fixture
def mock_authnet_connector():
    """Create a mock Authorize.net connector."""
    connector = AsyncMock()
    connector.__aenter__ = AsyncMock(return_value=connector)
    connector.__aexit__ = AsyncMock(return_value=None)
    connector.charge = AsyncMock(return_value=MockAuthnetResult())
    connector.authorize = AsyncMock(return_value=MockAuthnetResult())
    connector.capture = AsyncMock(return_value=MockAuthnetResult())
    connector.refund = AsyncMock(return_value=MockAuthnetResult())
    connector.void = AsyncMock(return_value=MockAuthnetResult())
    connector.get_transaction_details = AsyncMock(
        return_value={"id": "123456789", "status": "settled"}
    )
    connector.create_customer_profile = AsyncMock(return_value=MockAuthnetCustomerProfile())
    connector.get_customer_profile = AsyncMock(return_value=MockAuthnetCustomerProfile())
    connector.delete_customer_profile = AsyncMock(return_value=True)
    connector.create_subscription = AsyncMock(return_value=MockAuthnetSubscription())
    connector.cancel_subscription = AsyncMock(return_value=True)
    connector.verify_webhook_signature = AsyncMock(return_value=True)
    return connector


def create_mock_request(body: dict[str, Any] | None = None, method: str = "POST") -> MagicMock:
    """Create a mock aiohttp request with JSON body."""
    request = MagicMock(spec=web.Request)
    request.method = method
    request.match_info = {}
    request.query = {}

    if body is not None:

        async def json_func():
            return body

        request.json = json_func
    else:

        async def json_error():
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        request.json = json_error

    return request


# ===========================================================================
# Test PaymentResult Dataclass
# ===========================================================================


class TestPaymentResult:
    """Tests for PaymentResult dataclass."""

    def test_to_dict(self):
        """PaymentResult serializes to dict."""
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("100.00"),
            currency="USD",
            message="Payment successful",
        )

        data = result.to_dict()

        assert data["transaction_id"] == "txn_123"
        assert data["provider"] == "stripe"
        assert data["status"] == "approved"
        assert data["amount"] == "100.00"
        assert data["currency"] == "USD"

    def test_to_dict_with_auth_code(self):
        """PaymentResult includes auth code when present."""
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.APPROVED,
            amount=Decimal("50.00"),
            currency="USD",
            auth_code="AUTH123",
            avs_result="Y",
            cvv_result="M",
        )

        data = result.to_dict()

        assert data["auth_code"] == "AUTH123"
        assert data["avs_result"] == "Y"
        assert data["cvv_result"] == "M"


# ===========================================================================
# Test Provider Detection
# ===========================================================================


class TestProviderDetection:
    """Tests for payment provider detection."""

    def test_default_to_stripe(self):
        """Defaults to Stripe when no provider specified."""
        request = create_mock_request({})
        provider = _get_provider_from_request(request, {})
        assert provider == PaymentProvider.STRIPE

    def test_explicit_stripe(self):
        """Recognizes explicit Stripe provider."""
        request = create_mock_request({})
        provider = _get_provider_from_request(request, {"provider": "stripe"})
        assert provider == PaymentProvider.STRIPE

    def test_authorize_net(self):
        """Recognizes Authorize.net provider."""
        request = create_mock_request({})
        provider = _get_provider_from_request(request, {"provider": "authorize_net"})
        assert provider == PaymentProvider.AUTHORIZE_NET

    def test_authnet_shorthand(self):
        """Recognizes authnet shorthand."""
        request = create_mock_request({})
        provider = _get_provider_from_request(request, {"provider": "authnet"})
        assert provider == PaymentProvider.AUTHORIZE_NET


# ===========================================================================
# Test Charge Handler
# ===========================================================================


class TestChargeHandler:
    """Tests for charge payment handler."""

    @pytest.mark.asyncio
    async def test_charge_stripe_success(self, mock_stripe_connector):
        """Successful Stripe charge returns approved result."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "amount": 100.00,
                "currency": "USD",
                "description": "Test charge",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_charge(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True
        assert data["transaction"]["provider"] == "stripe"

    @pytest.mark.asyncio
    async def test_charge_invalid_amount(self):
        """Charge with zero/negative amount returns error."""
        request = create_mock_request(
            {
                "amount": 0,
            }
        )

        response = await handle_charge(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Amount must be greater than 0" in data["error"]

    @pytest.mark.asyncio
    async def test_charge_negative_amount(self):
        """Charge with negative amount returns error."""
        request = create_mock_request(
            {
                "amount": -50.00,
            }
        )

        response = await handle_charge(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Amount must be greater than 0" in data["error"]

    @pytest.mark.asyncio
    async def test_charge_invalid_json(self):
        """Charge with invalid JSON returns error."""
        request = create_mock_request(None)  # Will raise JSONDecodeError

        response = await handle_charge(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Invalid JSON" in data["error"]

    @pytest.mark.asyncio
    async def test_charge_stripe_not_available(self):
        """Charge returns error when Stripe connector unavailable."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "amount": 100.00,
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=None,
        ):
            response = await handle_charge(request)

        assert response.status == 200  # Returns success=false in body
        data = json.loads(response.text)
        assert data["success"] is False
        assert "not available" in data["transaction"]["message"]


# ===========================================================================
# Test Authorize Handler
# ===========================================================================


class TestAuthorizeHandler:
    """Tests for authorize payment handler."""

    @pytest.mark.asyncio
    async def test_authorize_stripe_success(self, mock_stripe_connector):
        """Successful Stripe authorization."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "amount": 100.00,
                "payment_method": "pm_test123",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_authorize(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_authorize_invalid_amount(self):
        """Authorize with zero amount returns error."""
        request = create_mock_request(
            {
                "amount": 0,
            }
        )

        response = await handle_authorize(request)

        assert response.status == 400


# ===========================================================================
# Test Capture Handler
# ===========================================================================


class TestCaptureHandler:
    """Tests for capture payment handler."""

    @pytest.mark.asyncio
    async def test_capture_stripe_success(self, mock_stripe_connector):
        """Successful Stripe capture."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "transaction_id": "pi_test123",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_capture(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_capture_missing_transaction_id(self):
        """Capture without transaction_id returns error."""
        request = create_mock_request(
            {
                "provider": "stripe",
            }
        )

        response = await handle_capture(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Missing transaction_id" in data["error"]


# ===========================================================================
# Test Refund Handler
# ===========================================================================


class TestRefundHandler:
    """Tests for refund payment handler."""

    @pytest.mark.asyncio
    async def test_refund_stripe_success(self, mock_stripe_connector):
        """Successful Stripe refund."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "transaction_id": "pi_test123",
                "amount": 50.00,
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_refund(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_refund_missing_transaction_id(self):
        """Refund without transaction_id returns error."""
        request = create_mock_request(
            {
                "amount": 50.00,
            }
        )

        response = await handle_refund(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_refund_invalid_amount(self):
        """Refund with zero amount returns error."""
        request = create_mock_request(
            {
                "transaction_id": "pi_test123",
                "amount": 0,
            }
        )

        response = await handle_refund(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_refund_authnet_requires_card_last_four(self, mock_authnet_connector):
        """Authorize.net refund requires card_last_four."""
        request = create_mock_request(
            {
                "provider": "authorize_net",
                "transaction_id": "123456789",
                "amount": 50.00,
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_authnet_connector",
            return_value=mock_authnet_connector,
        ):
            response = await handle_refund(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "card_last_four required" in data["error"]


# ===========================================================================
# Test Void Handler
# ===========================================================================


class TestVoidHandler:
    """Tests for void transaction handler."""

    @pytest.mark.asyncio
    async def test_void_stripe_success(self, mock_stripe_connector):
        """Successful Stripe void."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "transaction_id": "pi_test123",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_void(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_void_missing_transaction_id(self):
        """Void without transaction_id returns error."""
        request = create_mock_request({})

        response = await handle_void(request)

        assert response.status == 400


# ===========================================================================
# Test Customer Handlers
# ===========================================================================


class TestCustomerHandlers:
    """Tests for customer profile handlers."""

    @pytest.mark.asyncio
    async def test_create_customer_stripe(self, mock_stripe_connector):
        """Can create Stripe customer."""
        request = create_mock_request(
            {
                "provider": "stripe",
                "email": "test@example.com",
                "name": "Test Customer",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_create_customer(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True
        assert "customer_id" in data

    @pytest.mark.asyncio
    async def test_get_customer_stripe(self, mock_stripe_connector):
        """Can get Stripe customer."""
        request = create_mock_request({})
        request.match_info = {"customer_id": "cus_test123"}
        request.query = {"provider": "stripe"}

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_get_customer(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "customer" in data

    @pytest.mark.asyncio
    async def test_delete_customer_stripe(self, mock_stripe_connector):
        """Can delete Stripe customer."""
        request = create_mock_request({})
        request.match_info = {"customer_id": "cus_test123"}
        request.query = {"provider": "stripe"}

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_delete_customer(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True


# ===========================================================================
# Test Subscription Handlers
# ===========================================================================


class TestSubscriptionHandlers:
    """Tests for subscription handlers."""

    @pytest.mark.asyncio
    async def test_create_subscription_authnet(self, mock_authnet_connector):
        """Can create Authorize.net subscription."""
        request = create_mock_request(
            {
                "provider": "authorize_net",
                "customer_id": "12345678",
                "name": "Monthly Plan",
                "amount": 29.99,
                "interval": "month",
            }
        )

        with patch(
            "aragora.server.handlers.payments.get_authnet_connector",
            return_value=mock_authnet_connector,
        ):
            response = await handle_create_subscription(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_create_subscription_missing_customer(self):
        """Subscription without customer_id returns error."""
        request = create_mock_request(
            {
                "amount": 29.99,
            }
        )

        response = await handle_create_subscription(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_cancel_subscription_stripe(self, mock_stripe_connector):
        """Can cancel Stripe subscription."""
        request = create_mock_request({})
        request.match_info = {"subscription_id": "sub_test123"}
        request.query = {"provider": "stripe"}

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_cancel_subscription(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True


# ===========================================================================
# Test Webhook Handlers
# ===========================================================================


class TestWebhookHandlers:
    """Tests for webhook handlers."""

    @pytest.mark.asyncio
    async def test_stripe_webhook_success(self, mock_stripe_connector):
        """Stripe webhook processes valid event."""
        request = MagicMock(spec=web.Request)
        request.headers = {"Stripe-Signature": "test_sig"}

        async def read_func():
            return b'{"type": "payment_intent.succeeded"}'

        request.read = read_func

        mock_event = MagicMock()
        mock_event.type = "payment_intent.succeeded"
        mock_event.data = MagicMock()
        mock_event.data.object = MagicMock(id="pi_test123")

        mock_stripe_connector.construct_webhook_event = AsyncMock(return_value=mock_event)

        with patch(
            "aragora.server.handlers.payments.get_stripe_connector",
            return_value=mock_stripe_connector,
        ):
            response = await handle_stripe_webhook(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["received"] is True

    @pytest.mark.asyncio
    async def test_authnet_webhook_success(self, mock_authnet_connector):
        """Authorize.net webhook processes valid event."""
        request = create_mock_request(
            {
                "eventType": "net.authorize.payment.authcapture.created",
                "payload": {"id": "123456789"},
            }
        )
        request.headers = {"X-ANET-Signature": "test_sig"}

        with patch(
            "aragora.server.handlers.payments.get_authnet_connector",
            return_value=mock_authnet_connector,
        ):
            response = await handle_authnet_webhook(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["received"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
