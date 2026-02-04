"""
Tests for Stripe payment processing handlers.

Tests cover:
- Charge processing (Stripe and Authorize.net)
- Authorization, capture, refund, void operations
- Transaction retrieval
- Webhook handling (Stripe and Authorize.net)
- Error handling and resilience
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationDecision


class MockPaymentIntent:
    """Mock Stripe PaymentIntent."""

    def __init__(
        self,
        id="pi_123",
        status="succeeded",
        amount=10000,
        currency="usd",
        created=1640000000,
        metadata=None,
        client_secret="pi_secret_123",
    ):
        self.id = id
        self.status = status
        self.amount = amount
        self.currency = currency
        self.created = created
        self.metadata = metadata or {}
        self.client_secret = client_secret


class MockRefund:
    """Mock Stripe Refund."""

    def __init__(self, id="re_123", status="succeeded"):
        self.id = id
        self.status = status


class MockStripeEvent:
    """Mock Stripe webhook event."""

    def __init__(self, id="evt_123", type="payment_intent.succeeded"):
        self.id = id
        self.type = type
        self.data = MagicMock()
        self.data.object = MagicMock(id="pi_123")


class MockAuthnetResult:
    """Mock Authorize.net transaction result."""

    def __init__(
        self,
        transaction_id="tx_123",
        approved=True,
        message="Approved",
        auth_code="AUTH123",
        avs_result="Y",
        cvv_result="M",
    ):
        self.transaction_id = transaction_id
        self.approved = approved
        self.message = message
        self.auth_code = auth_code
        self.avs_result = avs_result
        self.cvv_result = cvv_result


def create_mock_request(match_info=None, query=None, body=None, user_id="test_user", headers=None):
    """Create a mock aiohttp request."""
    request = MagicMock()
    request.get.return_value = user_id
    request.match_info = match_info or {}
    request.query = query or {}
    request.headers = headers or {}
    request.transport = None

    if body is not None:

        async def read_json():
            return body

        request.json = read_json
        request.read = AsyncMock(return_value=json.dumps(body).encode())
        request.content_type = "application/json"

    return request


@pytest.fixture
def mock_permission_allowed():
    """Mock permission checker to always allow."""
    with patch("aragora.rbac.decorators.get_permission_checker") as mock_checker:
        checker = MagicMock()
        checker.check_permission.return_value = AuthorizationDecision(
            allowed=True, reason="Allowed", permission_key="test:permission"
        )
        mock_checker.return_value = checker
        yield checker


class TestChargeHandler:
    """Tests for payment charge handler."""

    @pytest.mark.asyncio
    async def test_charge_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe charge."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "description": "Test charge",
                "customer_id": "cus_123",
                "payment_method": "pm_card_visa",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._resilient_stripe_call = AsyncMock(
                return_value=MockPaymentIntent()
            )

            response = await handle_charge(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["transaction"]["transaction_id"] == "pi_123"

    @pytest.mark.asyncio
    async def test_charge_invalid_amount(self, mock_permission_allowed):
        """Test charge with invalid amount."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 0,
                "currency": "USD",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_charge(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_charge_negative_amount(self, mock_permission_allowed):
        """Test charge with negative amount."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": -50.00,
                "currency": "USD",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_charge(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_charge_connector_unavailable(self, mock_permission_allowed):
        """Test charge when connector unavailable."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider, PaymentStatus

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "provider": "stripe",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=None)

            response = await handle_charge(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is False
        assert data["transaction"]["status"] == PaymentStatus.ERROR.value

    @pytest.mark.asyncio
    async def test_charge_connection_error(self, mock_permission_allowed):
        """Test charge with connection error returns error result."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider, PaymentStatus

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._resilient_stripe_call = AsyncMock(
                side_effect=ConnectionError("API timeout")
            )

            response = await handle_charge(request)

        # Connection error is caught inside _charge_stripe and returns error result
        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is False
        assert data["transaction"]["status"] == PaymentStatus.ERROR.value


class TestAuthorizeHandler:
    """Tests for payment authorization handler."""

    @pytest.mark.asyncio
    async def test_authorize_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe authorization."""
        from aragora.server.handlers.payments.stripe import handle_authorize
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "payment_method": "pm_card_visa",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.create_payment_intent.return_value = MockPaymentIntent(
            status="requires_capture"
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_authorize(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["transaction_id"] == "pi_123"

    @pytest.mark.asyncio
    async def test_authorize_invalid_amount(self, mock_permission_allowed):
        """Test authorization with invalid amount."""
        from aragora.server.handlers.payments.stripe import handle_authorize
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 0,
                "currency": "USD",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_authorize(request)

        assert response.status == 400


class TestCaptureHandler:
    """Tests for payment capture handler."""

    @pytest.mark.asyncio
    async def test_capture_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe capture."""
        from aragora.server.handlers.payments.stripe import handle_capture
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "pi_123",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.capture_payment_intent.return_value = MockPaymentIntent()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_capture(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_capture_missing_transaction_id(self, mock_permission_allowed):
        """Test capture with missing transaction ID."""
        from aragora.server.handlers.payments.stripe import handle_capture
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "provider": "stripe",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_capture(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_capture_partial_amount(self, mock_permission_allowed):
        """Test partial capture with amount."""
        from aragora.server.handlers.payments.stripe import handle_capture
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "pi_123",
                "amount": 50.00,
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.capture_payment_intent.return_value = MockPaymentIntent(amount=5000)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_capture(request)

        assert response.status == 200
        mock_connector.capture_payment_intent.assert_called_once()


class TestRefundHandler:
    """Tests for payment refund handler."""

    @pytest.mark.asyncio
    async def test_refund_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe refund."""
        from aragora.server.handlers.payments.stripe import handle_refund
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "pi_123",
                "amount": 50.00,
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.create_refund.return_value = MockRefund()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value.audit_data = MagicMock()

            response = await handle_refund(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["refund_id"] == "re_123"

    @pytest.mark.asyncio
    async def test_refund_missing_transaction_id(self, mock_permission_allowed):
        """Test refund with missing transaction ID."""
        from aragora.server.handlers.payments.stripe import handle_refund
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 50.00,
                "provider": "stripe",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_refund(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_refund_invalid_amount(self, mock_permission_allowed):
        """Test refund with invalid amount."""
        from aragora.server.handlers.payments.stripe import handle_refund
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "pi_123",
                "amount": 0,
                "provider": "stripe",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_refund(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_refund_authnet_requires_card_last_four(self, mock_permission_allowed):
        """Test Authorize.net refund requires card_last_four."""
        from aragora.server.handlers.payments.stripe import handle_refund
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "tx_123",
                "amount": 50.00,
                "provider": "authorize_net",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = (
                PaymentProvider.AUTHORIZE_NET
            )
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_refund(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "card_last_four" in data.get("error", "").lower()


class TestVoidHandler:
    """Tests for payment void handler."""

    @pytest.mark.asyncio
    async def test_void_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe void."""
        from aragora.server.handlers.payments.stripe import handle_void
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "transaction_id": "pi_123",
                "provider": "stripe",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.cancel_payment_intent.return_value = MockPaymentIntent(status="canceled")

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_void(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_void_missing_transaction_id(self, mock_permission_allowed):
        """Test void with missing transaction ID."""
        from aragora.server.handlers.payments.stripe import handle_void
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "provider": "stripe",
            }
        )

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = PaymentProvider.STRIPE

            response = await handle_void(request)

        assert response.status == 400


class TestGetTransactionHandler:
    """Tests for transaction retrieval handler."""

    @pytest.mark.asyncio
    async def test_get_transaction_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe transaction retrieval."""
        from aragora.server.handlers.payments.stripe import handle_get_transaction
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"transaction_id": "pi_123"},
            query={"provider": "stripe"},
        )

        mock_connector = AsyncMock()
        mock_connector.retrieve_payment_intent.return_value = MockPaymentIntent()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_transaction(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["transaction"]["id"] == "pi_123"

    @pytest.mark.asyncio
    async def test_get_transaction_missing_id(self, mock_permission_allowed):
        """Test transaction retrieval with missing ID."""
        from aragora.server.handlers.payments.stripe import handle_get_transaction

        request = create_mock_request(match_info={})

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None

            response = await handle_get_transaction(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_get_transaction_authnet_not_found(self, mock_permission_allowed):
        """Test Authorize.net transaction not found."""
        from aragora.server.handlers.payments.stripe import handle_get_transaction
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            match_info={"transaction_id": "tx_invalid"},
            query={"provider": "authnet"},
        )

        mock_connector = AsyncMock()
        mock_connector.get_transaction_details.return_value = None
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_get_transaction(request)

        assert response.status == 404


class TestStripeWebhook:
    """Tests for Stripe webhook handler."""

    @pytest.mark.asyncio
    async def test_webhook_stripe_success(self, mock_permission_allowed):
        """Test successful Stripe webhook processing."""
        from aragora.server.handlers.payments.stripe import handle_stripe_webhook

        payload = json.dumps({"type": "payment_intent.succeeded", "id": "evt_123"})
        request = MagicMock()
        request.read = AsyncMock(return_value=payload.encode())
        request.headers = {"Stripe-Signature": "sig_123"}
        request.get.return_value = "test_user"
        request.transport = None

        mock_connector = AsyncMock()
        mock_connector.construct_webhook_event.return_value = MockStripeEvent()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._is_duplicate_webhook.return_value = False
            mock_pkg.return_value._mark_webhook_processed = MagicMock()

            response = await handle_stripe_webhook(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["received"] is True

    @pytest.mark.asyncio
    async def test_webhook_stripe_duplicate(self, mock_permission_allowed):
        """Test Stripe webhook duplicate detection."""
        from aragora.server.handlers.payments.stripe import handle_stripe_webhook

        payload = json.dumps({"type": "payment_intent.succeeded", "id": "evt_123"})
        request = MagicMock()
        request.read = AsyncMock(return_value=payload.encode())
        request.headers = {"Stripe-Signature": "sig_123"}
        request.get.return_value = "test_user"
        request.transport = None

        mock_connector = AsyncMock()
        mock_connector.construct_webhook_event.return_value = MockStripeEvent()

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._is_duplicate_webhook.return_value = True

            response = await handle_stripe_webhook(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["received"] is True
        assert data["duplicate"] is True

    @pytest.mark.asyncio
    async def test_webhook_stripe_invalid_signature(self, mock_permission_allowed):
        """Test Stripe webhook with invalid signature."""
        from aragora.server.handlers.payments.stripe import handle_stripe_webhook

        payload = json.dumps({"type": "payment_intent.succeeded"})
        request = MagicMock()
        request.read = AsyncMock(return_value=payload.encode())
        request.headers = {"Stripe-Signature": "invalid_sig"}
        request.get.return_value = "test_user"
        request.transport = None

        mock_connector = AsyncMock()
        mock_connector.construct_webhook_event.side_effect = Exception("Invalid signature")

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_stripe_connector = AsyncMock(return_value=mock_connector)

            response = await handle_stripe_webhook(request)

        # Invalid signature may result in 400 (bad request) or 500 (if caught as unexpected error)
        assert response.status in (400, 500)


class TestAuthnetWebhook:
    """Tests for Authorize.net webhook handler."""

    @pytest.mark.asyncio
    async def test_webhook_authnet_success(self, mock_permission_allowed):
        """Test successful Authorize.net webhook processing."""
        from aragora.server.handlers.payments.stripe import handle_authnet_webhook

        request = create_mock_request(
            body={
                "eventType": "net.authorize.payment.authcapture.created",
                "notificationId": "notif_123",
                "payload": {"id": "tx_123"},
            },
            headers={"X-ANET-Signature": "sig_123"},
        )

        mock_connector = AsyncMock()
        mock_connector.verify_webhook_signature.return_value = True
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._is_duplicate_webhook.return_value = False
            mock_pkg.return_value._mark_webhook_processed = MagicMock()

            response = await handle_authnet_webhook(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["received"] is True

    @pytest.mark.asyncio
    async def test_webhook_authnet_invalid_signature(self, mock_permission_allowed):
        """Test Authorize.net webhook with invalid signature."""
        from aragora.server.handlers.payments.stripe import handle_authnet_webhook

        request = create_mock_request(
            body={"eventType": "net.authorize.payment.authcapture.created"},
            headers={"X-ANET-Signature": "invalid_sig"},
        )

        mock_connector = AsyncMock()
        mock_connector.verify_webhook_signature.return_value = False
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_authnet_webhook(request)

        assert response.status == 400


class TestAuthNetCharging:
    """Tests for Authorize.net specific charging."""

    @pytest.mark.asyncio
    async def test_charge_authnet_success(self, mock_permission_allowed):
        """Test successful Authorize.net charge."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "payment_method": {
                    "type": "card",
                    "card_number": "4111111111111111",
                    "exp_month": "12",
                    "exp_year": "2025",
                    "cvv": "123",
                },
                "provider": "authorize_net",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.charge.return_value = MockAuthnetResult()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = (
                PaymentProvider.AUTHORIZE_NET
            )
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)
            mock_pkg.return_value._resilient_authnet_call = AsyncMock(
                return_value=MockAuthnetResult()
            )

            # Need to also mock the imports for CreditCard
            with patch.dict(
                "sys.modules", {"aragora.connectors.payments.authorize_net": MagicMock()}
            ):
                response = await handle_charge(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_charge_authnet_invalid_payment_method(self, mock_permission_allowed):
        """Test Authorize.net charge with invalid payment method."""
        from aragora.server.handlers.payments.stripe import handle_charge
        from aragora.server.handlers.payments import PaymentProvider, PaymentStatus

        request = create_mock_request(
            body={
                "amount": 100.00,
                "currency": "USD",
                "payment_method": "pm_card_visa",  # Stripe-style, not valid for authnet
                "provider": "authorize_net",
            }
        )

        mock_connector = AsyncMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.payments.stripe._pkg") as mock_pkg:
            mock_pkg.return_value._check_rate_limit.return_value = None
            mock_pkg.return_value._get_provider_from_request.return_value = (
                PaymentProvider.AUTHORIZE_NET
            )
            mock_pkg.return_value.get_authnet_connector = AsyncMock(return_value=mock_connector)

            response = await handle_charge(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["success"] is False
        assert data["transaction"]["status"] == PaymentStatus.ERROR.value


class TestRouteRegistration:
    """Tests for payment route registration."""

    def test_register_payment_routes(self):
        """Test that all payment routes are registered."""
        from aragora.server.handlers.payments.plans import register_payment_routes
        from aiohttp import web

        app = web.Application()
        register_payment_routes(app)

        routes = [
            r.resource.canonical for r in app.router.routes() if hasattr(r.resource, "canonical")
        ]

        # Verify v1 routes
        assert "/api/v1/payments/charge" in routes
        assert "/api/v1/payments/authorize" in routes
        assert "/api/v1/payments/capture" in routes
        assert "/api/v1/payments/refund" in routes
        assert "/api/v1/payments/void" in routes
        assert "/api/v1/payments/customer" in routes
        assert "/api/v1/payments/subscription" in routes

        # Verify legacy routes
        assert "/api/payments/charge" in routes
        assert "/api/payments/customer" in routes
