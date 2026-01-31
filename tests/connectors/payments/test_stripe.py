"""
Tests for Stripe Payment Connector.

Tests cover:
- Client initialization
- API authentication
- Customers API
- Products and Prices API
- Subscriptions API
- Invoices API
- Payment Intents API
- Balance API
- Refund processing
- Webhook signature verification
- Error handling (declined cards, network errors, invalid amounts)
- Idempotency key handling
- Currency conversion
- Mock data generators
"""

from datetime import datetime
import hashlib
import hmac
import json
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def stripe_credentials():
    """Create test credentials."""
    from aragora.connectors.payments.stripe import StripeCredentials

    return StripeCredentials(
        secret_key="sk_test_123456789",
        webhook_secret="whsec_test_webhook_secret",
    )


@pytest.fixture
def stripe_connector(stripe_credentials):
    """Create connector instance."""
    from aragora.connectors.payments.stripe import StripeConnector

    return StripeConnector(stripe_credentials)


@pytest.fixture
def mock_customer_api_response():
    """Mock customer API response."""
    return {
        "id": "cus_test_12345",
        "email": "customer@example.com",
        "name": "Test Customer",
        "phone": "+15551234567",
        "description": "Test customer description",
        "balance": 0,
        "currency": "usd",
        "delinquent": False,
        "default_source": None,
        "metadata": {"user_id": "internal_123"},
        "created": int(time.time()),
    }


@pytest.fixture
def mock_payment_intent_api_response():
    """Mock payment intent API response."""
    return {
        "id": "pi_test_12345",
        "amount": 5000,
        "currency": "usd",
        "status": "requires_capture",
        "customer": "cus_test_12345",
        "description": "Test payment",
        "receipt_email": "receipt@example.com",
        "payment_method": "pm_card_visa",
        "client_secret": "pi_test_12345_secret_xyz",
        "metadata": {"order_id": "ORD-001"},
        "created": int(time.time()),
    }


@pytest.fixture
def mock_refund_api_response():
    """Mock refund API response."""
    return {
        "id": "re_test_12345",
        "amount": 2500,
        "currency": "usd",
        "payment_intent": "pi_test_12345",
        "status": "succeeded",
        "reason": "requested_by_customer",
        "created": int(time.time()),
    }


@pytest.fixture
def mock_subscription_api_response():
    """Mock subscription API response."""
    return {
        "id": "sub_test_12345",
        "customer": "cus_test_12345",
        "status": "active",
        "current_period_start": int(time.time()),
        "current_period_end": int(time.time()) + 2592000,
        "cancel_at_period_end": False,
        "canceled_at": None,
        "ended_at": None,
        "trial_start": None,
        "trial_end": None,
        "items": {"data": [{"id": "si_123", "price": {"id": "price_123"}}]},
        "metadata": {"plan_type": "premium"},
        "created": int(time.time()),
    }


@pytest.fixture
def mock_invoice_api_response():
    """Mock invoice API response."""
    return {
        "id": "in_test_12345",
        "customer": "cus_test_12345",
        "subscription": "sub_test_12345",
        "status": "paid",
        "currency": "usd",
        "amount_due": 4999,
        "amount_paid": 4999,
        "amount_remaining": 0,
        "subtotal": 4999,
        "tax": 0,
        "total": 4999,
        "number": "INV-0001",
        "invoice_pdf": "https://stripe.com/invoice.pdf",
        "hosted_invoice_url": "https://stripe.com/invoice",
        "paid": True,
        "due_date": None,
        "created": int(time.time()),
    }


# =============================================================================
# Enum Tests
# =============================================================================


class TestStripeEnums:
    """Tests for Stripe enums."""

    def test_payment_status_values(self):
        """PaymentStatus enum has expected values."""
        from aragora.connectors.payments.stripe import PaymentStatus

        assert PaymentStatus.REQUIRES_PAYMENT_METHOD.value == "requires_payment_method"
        assert PaymentStatus.REQUIRES_CONFIRMATION.value == "requires_confirmation"
        assert PaymentStatus.REQUIRES_ACTION.value == "requires_action"
        assert PaymentStatus.PROCESSING.value == "processing"
        assert PaymentStatus.REQUIRES_CAPTURE.value == "requires_capture"
        assert PaymentStatus.CANCELED.value == "canceled"
        assert PaymentStatus.SUCCEEDED.value == "succeeded"

    def test_subscription_status_values(self):
        """SubscriptionStatus enum has expected values."""
        from aragora.connectors.payments.stripe import SubscriptionStatus

        assert SubscriptionStatus.INCOMPLETE.value == "incomplete"
        assert SubscriptionStatus.INCOMPLETE_EXPIRED.value == "incomplete_expired"
        assert SubscriptionStatus.TRIALING.value == "trialing"
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.PAST_DUE.value == "past_due"
        assert SubscriptionStatus.CANCELED.value == "canceled"
        assert SubscriptionStatus.UNPAID.value == "unpaid"
        assert SubscriptionStatus.PAUSED.value == "paused"

    def test_invoice_status_values(self):
        """InvoiceStatus enum has expected values."""
        from aragora.connectors.payments.stripe import InvoiceStatus

        assert InvoiceStatus.DRAFT.value == "draft"
        assert InvoiceStatus.OPEN.value == "open"
        assert InvoiceStatus.PAID.value == "paid"
        assert InvoiceStatus.UNCOLLECTIBLE.value == "uncollectible"
        assert InvoiceStatus.VOID.value == "void"

    def test_price_type_values(self):
        """PriceType enum has expected values."""
        from aragora.connectors.payments.stripe import PriceType

        assert PriceType.ONE_TIME.value == "one_time"
        assert PriceType.RECURRING.value == "recurring"


# =============================================================================
# Credentials Tests
# =============================================================================


class TestStripeCredentials:
    """Tests for StripeCredentials."""

    def test_credentials_init(self):
        """Create credentials with secret key."""
        from aragora.connectors.payments.stripe import StripeCredentials

        creds = StripeCredentials(secret_key="sk_test_123456")

        assert creds.secret_key == "sk_test_123456"
        assert creds.webhook_secret is None

    def test_credentials_with_webhook_secret(self):
        """Create credentials with webhook secret."""
        from aragora.connectors.payments.stripe import StripeCredentials

        creds = StripeCredentials(
            secret_key="sk_test_123456",
            webhook_secret="whsec_test_secret",
        )

        assert creds.secret_key == "sk_test_123456"
        assert creds.webhook_secret == "whsec_test_secret"


# =============================================================================
# Data Model Tests
# =============================================================================


class TestStripeCustomer:
    """Tests for StripeCustomer dataclass."""

    def test_customer_from_api(self):
        """Parse StripeCustomer from API response."""
        from aragora.connectors.payments.stripe import StripeCustomer

        data = {
            "id": "cus_ABC123",
            "email": "customer@example.com",
            "name": "John Doe",
            "phone": "+15551234567",
            "description": "Test customer",
            "balance": 1000,
            "currency": "usd",
            "delinquent": False,
            "default_source": "card_123",
            "metadata": {"user_id": "123"},
            "created": 1704067200,  # 2024-01-01 00:00:00 UTC
        }

        customer = StripeCustomer.from_api(data)

        assert customer.id == "cus_ABC123"
        assert customer.email == "customer@example.com"
        assert customer.name == "John Doe"
        assert customer.phone == "+15551234567"
        assert customer.balance == 1000
        assert customer.delinquent is False
        assert customer.metadata == {"user_id": "123"}
        assert customer.created is not None

    def test_customer_from_api_minimal(self):
        """Parse StripeCustomer with minimal data."""
        from aragora.connectors.payments.stripe import StripeCustomer

        data = {"id": "cus_minimal"}

        customer = StripeCustomer.from_api(data)

        assert customer.id == "cus_minimal"
        assert customer.email is None
        assert customer.name is None
        assert customer.balance == 0
        assert customer.currency == "usd"


class TestStripeProduct:
    """Tests for StripeProduct dataclass."""

    def test_product_from_api(self):
        """Parse StripeProduct from API response."""
        from aragora.connectors.payments.stripe import StripeProduct

        data = {
            "id": "prod_ABC123",
            "name": "Premium Plan",
            "active": True,
            "description": "Our best plan",
            "metadata": {"tier": "premium"},
            "created": 1704067200,
            "updated": 1704153600,
        }

        product = StripeProduct.from_api(data)

        assert product.id == "prod_ABC123"
        assert product.name == "Premium Plan"
        assert product.active is True
        assert product.description == "Our best plan"
        assert product.metadata == {"tier": "premium"}

    def test_product_from_api_minimal(self):
        """Parse StripeProduct with minimal data."""
        from aragora.connectors.payments.stripe import StripeProduct

        data = {"id": "prod_123", "name": "Basic"}

        product = StripeProduct.from_api(data)

        assert product.id == "prod_123"
        assert product.name == "Basic"
        assert product.active is True
        assert product.description is None


class TestStripePrice:
    """Tests for StripePrice dataclass."""

    def test_price_from_api_one_time(self):
        """Parse one-time StripePrice from API response."""
        from aragora.connectors.payments.stripe import StripePrice, PriceType

        data = {
            "id": "price_ABC123",
            "product": "prod_123",
            "active": True,
            "currency": "usd",
            "unit_amount": 2999,
            "type": "one_time",
            "metadata": {},
            "created": 1704067200,
        }

        price = StripePrice.from_api(data)

        assert price.id == "price_ABC123"
        assert price.product_id == "prod_123"
        assert price.unit_amount == 2999
        assert price.type == PriceType.ONE_TIME
        assert price.recurring_interval is None

    def test_price_from_api_recurring(self):
        """Parse recurring StripePrice from API response."""
        from aragora.connectors.payments.stripe import StripePrice, PriceType

        data = {
            "id": "price_recurring",
            "product": {"id": "prod_456"},  # Expanded product object
            "active": True,
            "currency": "eur",
            "unit_amount": 999,
            "type": "recurring",
            "recurring": {
                "interval": "month",
                "interval_count": 1,
            },
            "metadata": {},
            "created": 1704067200,
        }

        price = StripePrice.from_api(data)

        assert price.id == "price_recurring"
        assert price.product_id == "prod_456"
        assert price.type == PriceType.RECURRING
        assert price.recurring_interval == "month"
        assert price.recurring_interval_count == 1


class TestStripeSubscription:
    """Tests for StripeSubscription dataclass."""

    def test_subscription_from_api(self):
        """Parse StripeSubscription from API response."""
        from aragora.connectors.payments.stripe import StripeSubscription, SubscriptionStatus

        data = {
            "id": "sub_ABC123",
            "customer": "cus_123",
            "status": "active",
            "current_period_start": 1704067200,
            "current_period_end": 1706745600,
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_123", "price": {"id": "price_123"}}]},
            "metadata": {"plan": "premium"},
            "created": 1704067200,
        }

        subscription = StripeSubscription.from_api(data)

        assert subscription.id == "sub_ABC123"
        assert subscription.customer_id == "cus_123"
        assert subscription.status == SubscriptionStatus.ACTIVE
        assert subscription.cancel_at_period_end is False
        assert len(subscription.items) == 1

    def test_subscription_from_api_with_expanded_customer(self):
        """Parse subscription with expanded customer object."""
        from aragora.connectors.payments.stripe import StripeSubscription, SubscriptionStatus

        data = {
            "id": "sub_expanded",
            "customer": {"id": "cus_expanded", "email": "test@example.com"},
            "status": "trialing",
            "current_period_start": 1704067200,
            "current_period_end": 1706745600,
            "trial_start": 1704067200,
            "trial_end": 1705276800,
            "items": {"data": []},
        }

        subscription = StripeSubscription.from_api(data)

        assert subscription.id == "sub_expanded"
        assert subscription.customer_id == "cus_expanded"
        assert subscription.status == SubscriptionStatus.TRIALING
        assert subscription.trial_start is not None
        assert subscription.trial_end is not None

    def test_subscription_canceled_status(self):
        """Parse canceled subscription."""
        from aragora.connectors.payments.stripe import StripeSubscription, SubscriptionStatus

        data = {
            "id": "sub_canceled",
            "customer": "cus_123",
            "status": "canceled",
            "current_period_start": 1704067200,
            "current_period_end": 1706745600,
            "canceled_at": 1705000000,
            "ended_at": 1706745600,
            "items": {"data": []},
        }

        subscription = StripeSubscription.from_api(data)

        assert subscription.status == SubscriptionStatus.CANCELED
        assert subscription.canceled_at is not None
        assert subscription.ended_at is not None


class TestStripeInvoice:
    """Tests for StripeInvoice dataclass."""

    def test_invoice_from_api(self):
        """Parse StripeInvoice from API response."""
        from aragora.connectors.payments.stripe import StripeInvoice, InvoiceStatus

        data = {
            "id": "in_ABC123",
            "customer": "cus_123",
            "subscription": "sub_123",
            "status": "paid",
            "currency": "usd",
            "amount_due": 2999,
            "amount_paid": 2999,
            "amount_remaining": 0,
            "subtotal": 2999,
            "tax": 0,
            "total": 2999,
            "number": "INV-0001",
            "invoice_pdf": "https://stripe.com/invoice.pdf",
            "hosted_invoice_url": "https://stripe.com/invoice",
            "paid": True,
            "created": 1704067200,
        }

        invoice = StripeInvoice.from_api(data)

        assert invoice.id == "in_ABC123"
        assert invoice.customer_id == "cus_123"
        assert invoice.subscription_id == "sub_123"
        assert invoice.status == InvoiceStatus.PAID
        assert invoice.amount_due == 2999
        assert invoice.paid is True
        assert invoice.number == "INV-0001"

    def test_invoice_from_api_draft(self):
        """Parse draft invoice."""
        from aragora.connectors.payments.stripe import StripeInvoice, InvoiceStatus

        data = {
            "id": "in_draft",
            "customer": {"id": "cus_expanded"},
            "status": "draft",
            "currency": "usd",
            "amount_due": 0,
            "total": 0,
        }

        invoice = StripeInvoice.from_api(data)

        assert invoice.id == "in_draft"
        assert invoice.customer_id == "cus_expanded"
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.paid is False


class TestPaymentIntent:
    """Tests for PaymentIntent dataclass."""

    def test_payment_intent_from_api(self):
        """Parse PaymentIntent from API response."""
        from aragora.connectors.payments.stripe import PaymentIntent, PaymentStatus

        data = {
            "id": "pi_ABC123",
            "amount": 9999,
            "currency": "usd",
            "status": "succeeded",
            "customer": "cus_123",
            "description": "Test payment",
            "receipt_email": "receipt@example.com",
            "payment_method": "pm_123",
            "client_secret": "pi_ABC123_secret_xyz",
            "metadata": {"order_id": "ORD-001"},
            "created": 1704067200,
        }

        pi = PaymentIntent.from_api(data)

        assert pi.id == "pi_ABC123"
        assert pi.amount == 9999
        assert pi.currency == "usd"
        assert pi.status == PaymentStatus.SUCCEEDED
        assert pi.customer_id == "cus_123"
        assert pi.client_secret == "pi_ABC123_secret_xyz"

    def test_payment_intent_requires_action(self):
        """Parse payment intent requiring action (3DS)."""
        from aragora.connectors.payments.stripe import PaymentIntent, PaymentStatus

        data = {
            "id": "pi_3ds",
            "amount": 5000,
            "currency": "eur",
            "status": "requires_action",
            "client_secret": "pi_3ds_secret",
            "created": 1704067200,
        }

        pi = PaymentIntent.from_api(data)

        assert pi.id == "pi_3ds"
        assert pi.status == PaymentStatus.REQUIRES_ACTION


class TestBalanceTransaction:
    """Tests for BalanceTransaction dataclass."""

    def test_balance_transaction_from_api(self):
        """Parse BalanceTransaction from API response."""
        from aragora.connectors.payments.stripe import BalanceTransaction

        data = {
            "id": "txn_ABC123",
            "amount": 9999,
            "currency": "usd",
            "type": "charge",
            "description": "Payment for order ORD-001",
            "fee": 319,
            "net": 9680,
            "status": "available",
            "created": 1704067200,
        }

        txn = BalanceTransaction.from_api(data)

        assert txn.id == "txn_ABC123"
        assert txn.amount == 9999
        assert txn.type == "charge"
        assert txn.fee == 319
        assert txn.net == 9680
        assert txn.status == "available"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestStripeError:
    """Tests for StripeError."""

    def test_error_creation(self):
        """Create error with details."""
        from aragora.connectors.payments.stripe import StripeError

        error = StripeError(
            message="Your card was declined",
            code="card_declined",
            status_code=402,
        )

        assert str(error) == "Your card was declined"
        assert error.code == "card_declined"
        assert error.status_code == 402

    def test_error_creation_minimal(self):
        """Create error with minimal details."""
        from aragora.connectors.payments.stripe import StripeError

        error = StripeError(message="Unknown error")

        assert str(error) == "Unknown error"
        assert error.code is None
        assert error.status_code is None


# =============================================================================
# Client Tests
# =============================================================================


class TestStripeConnectorInit:
    """Tests for StripeConnector initialization."""

    def test_connector_creation(self):
        """Create connector with credentials."""
        from aragora.connectors.payments.stripe import StripeConnector, StripeCredentials

        creds = StripeCredentials(secret_key="sk_test_123")
        connector = StripeConnector(creds)

        assert connector.credentials == creds
        assert connector.BASE_URL == "https://api.stripe.com/v1"

    def test_connector_client_not_initialized(self):
        """Connector raises error when client not initialized."""
        from aragora.connectors.payments.stripe import (
            StripeConnector,
            StripeCredentials,
            StripeError,
        )

        creds = StripeCredentials(secret_key="sk_test_123")
        connector = StripeConnector(creds)

        with pytest.raises(StripeError, match="not initialized"):
            _ = connector.client


# =============================================================================
# Mock Data Generator Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data generators."""

    def test_get_mock_customer(self):
        """Get mock customer for testing."""
        from aragora.connectors.payments.stripe import get_mock_customer

        customer = get_mock_customer()

        assert customer.id == "cus_123456789"
        assert customer.email == "customer@example.com"
        assert customer.name == "John Doe"

    def test_get_mock_subscription(self):
        """Get mock subscription for testing."""
        from aragora.connectors.payments.stripe import get_mock_subscription, SubscriptionStatus

        subscription = get_mock_subscription()

        assert subscription.id == "sub_123456789"
        assert subscription.customer_id == "cus_123456789"
        assert subscription.status == SubscriptionStatus.ACTIVE

    def test_get_mock_invoice(self):
        """Get mock invoice for testing."""
        from aragora.connectors.payments.stripe import get_mock_invoice, InvoiceStatus

        invoice = get_mock_invoice()

        assert invoice.id == "in_123456789"
        assert invoice.customer_id == "cus_123456789"
        assert invoice.status == InvoiceStatus.PAID
        assert invoice.paid is True


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_payments_module(self):
        """Import Stripe from payments module."""
        from aragora.connectors.payments import (
            StripeConnector,
            StripeCredentials,
            StripeError,
            StripePaymentStatus,
            StripeSubscriptionStatus,
            StripeInvoiceStatus,
            StripePriceType,
            StripeCustomer,
            StripeProduct,
            StripePrice,
            StripeSubscription,
            StripeInvoice,
            StripePaymentIntent,
            StripeBalanceTransaction,
        )

        # Verify imports work
        assert StripeConnector is not None
        assert StripeCredentials is not None
        assert StripeError is not None
        assert StripePaymentStatus is not None
        assert StripeCustomer is not None

    def test_import_directly(self):
        """Import directly from stripe module."""
        from aragora.connectors.payments.stripe import (
            StripeConnector,
            StripeCredentials,
            StripeError,
            PaymentStatus,
            SubscriptionStatus,
            InvoiceStatus,
            PriceType,
            StripeCustomer,
            StripeProduct,
            StripePrice,
            StripeSubscription,
            StripeInvoice,
            PaymentIntent,
            BalanceTransaction,
            get_mock_customer,
            get_mock_subscription,
            get_mock_invoice,
        )

        # Verify imports work
        assert StripeConnector is not None
        assert PaymentStatus is not None
        assert get_mock_customer is not None


# =============================================================================
# Payment Intent Creation and Capture Tests
# =============================================================================


class TestPaymentIntentCreation:
    """Tests for payment intent creation and capture."""

    @pytest.mark.asyncio
    async def test_create_payment_intent_basic(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test basic payment intent creation."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=5000,
                    currency="usd",
                )

            assert result.id == "pi_test_12345"
            assert result.amount == 5000
            assert result.currency == "usd"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_payment_intent_with_customer(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test payment intent creation with customer."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=5000,
                    currency="usd",
                    customer_id="cus_test_12345",
                )

            assert result.customer_id == "cus_test_12345"
            call_args = mock_request.call_args
            assert call_args[1]["data"]["customer"] == "cus_test_12345"

    @pytest.mark.asyncio
    async def test_create_payment_intent_with_metadata(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test payment intent creation with metadata."""
        mock_payment_intent_api_response["metadata"] = {"order_id": "ORD-123"}

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=5000,
                    currency="usd",
                    metadata={"order_id": "ORD-123"},
                )

            assert result.metadata == {"order_id": "ORD-123"}

    @pytest.mark.asyncio
    async def test_confirm_payment_intent(self, stripe_connector, mock_payment_intent_api_response):
        """Test confirming a payment intent."""
        from aragora.connectors.payments.stripe import PaymentStatus

        mock_payment_intent_api_response["status"] = "succeeded"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.confirm_payment_intent("pi_test_12345")

            assert result.status == PaymentStatus.SUCCEEDED
            mock_request.assert_called_with("POST", "/payment_intents/pi_test_12345/confirm")

    @pytest.mark.asyncio
    async def test_cancel_payment_intent(self, stripe_connector, mock_payment_intent_api_response):
        """Test canceling a payment intent."""
        from aragora.connectors.payments.stripe import PaymentStatus

        mock_payment_intent_api_response["status"] = "canceled"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.cancel_payment_intent("pi_test_12345")

            assert result.status == PaymentStatus.CANCELED


# =============================================================================
# Refund Processing Tests
# =============================================================================


class TestRefundProcessing:
    """Tests for refund processing - simulated via payment intent operations."""

    @pytest.mark.asyncio
    async def test_get_payment_intent_for_refund(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test getting payment intent details before refund."""
        mock_payment_intent_api_response["status"] = "succeeded"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.get_payment_intent("pi_test_12345")

            assert result.id == "pi_test_12345"
            mock_request.assert_called_with("GET", "/payment_intents/pi_test_12345")


# =============================================================================
# Webhook Signature Verification Tests
# =============================================================================


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    @pytest.mark.asyncio
    async def test_valid_webhook_signature(self, stripe_connector):
        """Test verification of valid webhook signature."""
        payload = b'{"id": "evt_test_123", "type": "payment_intent.succeeded"}'
        timestamp = str(int(time.time()))

        # Compute valid signature
        signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
        signature = hmac.new(
            stripe_connector.credentials.webhook_secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        sig_header = f"t={timestamp},v1={signature}"

        async with stripe_connector:
            event = await stripe_connector.construct_webhook_event(payload, sig_header)

        assert event.id == "evt_test_123"
        assert event.type == "payment_intent.succeeded"

    @pytest.mark.asyncio
    async def test_invalid_webhook_signature(self, stripe_connector):
        """Test rejection of invalid webhook signature."""
        payload = b'{"id": "evt_test_123"}'
        timestamp = str(int(time.time()))
        sig_header = f"t={timestamp},v1=invalid_signature_here"

        async with stripe_connector:
            with pytest.raises(Exception) as exc_info:
                await stripe_connector.construct_webhook_event(payload, sig_header)

        assert "Signature verification failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_signature_header(self, stripe_connector):
        """Test missing signature header raises exception."""
        payload = b'{"id": "evt_test_123"}'

        async with stripe_connector:
            with pytest.raises(Exception) as exc_info:
                await stripe_connector.construct_webhook_event(payload, None)

        assert "Missing Stripe-Signature header" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_expired_webhook_timestamp(self, stripe_connector):
        """Test expired timestamp raises exception."""
        payload = b'{"id": "evt_test_123"}'
        # Timestamp older than 5 minutes
        old_timestamp = str(int(time.time()) - 400)

        signed_payload = f"{old_timestamp}.{payload.decode('utf-8')}"
        signature = hmac.new(
            stripe_connector.credentials.webhook_secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        sig_header = f"t={old_timestamp},v1={signature}"

        async with stripe_connector:
            with pytest.raises(Exception) as exc_info:
                await stripe_connector.construct_webhook_event(payload, sig_header)

        assert "timestamp too old" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_webhook_no_secret_configured(self, stripe_credentials):
        """Test webhook without secret configured."""
        from aragora.connectors.payments.stripe import StripeConnector

        stripe_credentials.webhook_secret = None
        connector = StripeConnector(stripe_credentials)

        payload = b'{"id": "evt_test_123"}'
        sig_header = "t=123,v1=sig"

        async with connector:
            with pytest.raises(Exception) as exc_info:
                await connector.construct_webhook_event(payload, sig_header)

        assert "Webhook secret not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_json_payload(self, stripe_connector):
        """Test invalid JSON payload raises ValueError."""
        payload = b"not valid json at all"
        timestamp = str(int(time.time()))

        signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
        signature = hmac.new(
            stripe_connector.credentials.webhook_secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        sig_header = f"t={timestamp},v1={signature}"

        async with stripe_connector:
            with pytest.raises(ValueError) as exc_info:
                await stripe_connector.construct_webhook_event(payload, sig_header)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_signature_format(self, stripe_connector):
        """Test invalid signature header format."""
        payload = b'{"id": "evt_test"}'
        sig_header = "invalid_format_no_equals"

        async with stripe_connector:
            with pytest.raises(Exception) as exc_info:
                await stripe_connector.construct_webhook_event(payload, sig_header)

        assert "Invalid" in str(exc_info.value) or "format" in str(exc_info.value).lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_declined_card_error(self, stripe_connector):
        """Test handling of declined card error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="Your card was declined.",
                code="card_declined",
                status_code=402,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_payment_intent(amount=5000, currency="usd")

            assert exc_info.value.code == "card_declined"
            assert exc_info.value.status_code == 402

    @pytest.mark.asyncio
    async def test_network_error(self, stripe_connector):
        """Test handling of network error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError("HTTP error: Connection refused")

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_customer(email="test@example.com")

            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_amount_error(self, stripe_connector):
        """Test handling of invalid amount error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="Amount must be at least 50 cents",
                code="amount_too_small",
                status_code=400,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_payment_intent(amount=10, currency="usd")

            assert exc_info.value.code == "amount_too_small"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, stripe_connector):
        """Test handling of rate limit error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="Rate limit exceeded",
                code="rate_limit",
                status_code=429,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_customer(email="test@example.com")

            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_authentication_error(self, stripe_connector):
        """Test handling of authentication error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="Invalid API Key provided",
                code="invalid_api_key",
                status_code=401,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_customer(email="test@example.com")

            assert exc_info.value.status_code == 401
            assert exc_info.value.code == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_resource_not_found_error(self, stripe_connector):
        """Test handling of resource not found error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="No such customer: cus_invalid",
                code="resource_missing",
                status_code=404,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.get_customer("cus_invalid")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_insufficient_funds_error(self, stripe_connector):
        """Test handling of insufficient funds error."""
        from aragora.connectors.payments.stripe import StripeError

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = StripeError(
                message="Your card has insufficient funds.",
                code="insufficient_funds",
                status_code=402,
            )

            async with stripe_connector:
                with pytest.raises(StripeError) as exc_info:
                    await stripe_connector.create_payment_intent(amount=100000, currency="usd")

            assert exc_info.value.code == "insufficient_funds"


# =============================================================================
# Idempotency Key Handling Tests
# =============================================================================


class TestIdempotencyKeyHandling:
    """Tests for idempotency key handling in payment operations."""

    @pytest.mark.asyncio
    async def test_customer_creation_with_same_data(
        self, stripe_connector, mock_customer_api_response
    ):
        """Test that creating customer with same data returns consistent result."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with stripe_connector:
                result1 = await stripe_connector.create_customer(
                    email="test@example.com",
                    name="Test User",
                )
                result2 = await stripe_connector.create_customer(
                    email="test@example.com",
                    name="Test User",
                )

            # Both calls should succeed (idempotency handled by Stripe)
            assert result1.email == result2.email

    @pytest.mark.asyncio
    async def test_payment_intent_idempotency(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test payment intent creation handles idempotency."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=5000,
                    currency="usd",
                    metadata={"idempotency_key": "unique_key_123"},
                )

            assert result.id == "pi_test_12345"


# =============================================================================
# Currency Handling Tests
# =============================================================================


class TestCurrencyHandling:
    """Tests for multi-currency handling."""

    @pytest.mark.asyncio
    async def test_payment_in_eur(self, stripe_connector, mock_payment_intent_api_response):
        """Test payment in EUR currency."""
        mock_payment_intent_api_response["currency"] = "eur"
        mock_payment_intent_api_response["amount"] = 4500

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=4500,
                    currency="eur",
                )

            assert result.currency == "eur"
            assert result.amount == 4500

    @pytest.mark.asyncio
    async def test_payment_in_gbp(self, stripe_connector, mock_payment_intent_api_response):
        """Test payment in GBP currency."""
        mock_payment_intent_api_response["currency"] = "gbp"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=5000,
                    currency="gbp",
                )

            assert result.currency == "gbp"

    @pytest.mark.asyncio
    async def test_zero_decimal_currency_jpy(
        self, stripe_connector, mock_payment_intent_api_response
    ):
        """Test zero-decimal currency (JPY) handling."""
        mock_payment_intent_api_response["currency"] = "jpy"
        mock_payment_intent_api_response["amount"] = 500  # 500 yen, not cents

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_intent_api_response

            async with stripe_connector:
                result = await stripe_connector.create_payment_intent(
                    amount=500,
                    currency="jpy",
                )

            assert result.currency == "jpy"
            assert result.amount == 500


# =============================================================================
# Customer Operations Tests (Async)
# =============================================================================


class TestCustomerOperationsAsync:
    """Async tests for customer operations."""

    @pytest.mark.asyncio
    async def test_create_customer_async(self, stripe_connector, mock_customer_api_response):
        """Test async customer creation."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with stripe_connector:
                result = await stripe_connector.create_customer(
                    email="customer@example.com",
                    name="Test Customer",
                )

            assert result.id == "cus_test_12345"
            assert result.email == "customer@example.com"

    @pytest.mark.asyncio
    async def test_get_customer_async(self, stripe_connector, mock_customer_api_response):
        """Test async customer retrieval."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with stripe_connector:
                result = await stripe_connector.get_customer("cus_test_12345")

            assert result.id == "cus_test_12345"

    @pytest.mark.asyncio
    async def test_update_customer_async(self, stripe_connector, mock_customer_api_response):
        """Test async customer update."""
        mock_customer_api_response["name"] = "Updated Name"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with stripe_connector:
                result = await stripe_connector.update_customer(
                    "cus_test_12345", name="Updated Name"
                )

            assert result.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_list_customers_async(self, stripe_connector, mock_customer_api_response):
        """Test async customer listing."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": [mock_customer_api_response]}

            async with stripe_connector:
                result = await stripe_connector.list_customers(limit=10)

            assert len(result) == 1
            assert result[0].id == "cus_test_12345"

    @pytest.mark.asyncio
    async def test_delete_customer_async(self, stripe_connector):
        """Test async customer deletion."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "cus_test_12345", "deleted": True}

            async with stripe_connector:
                await stripe_connector.delete_customer("cus_test_12345")

            mock_request.assert_called_with("DELETE", "/customers/cus_test_12345")


# =============================================================================
# Subscription Operations Tests (Async)
# =============================================================================


class TestSubscriptionOperationsAsync:
    """Async tests for subscription operations."""

    @pytest.mark.asyncio
    async def test_create_subscription_async(
        self, stripe_connector, mock_subscription_api_response
    ):
        """Test async subscription creation."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with stripe_connector:
                result = await stripe_connector.create_subscription(
                    customer_id="cus_test_12345",
                    price_id="price_test_123",
                )

            assert result.id == "sub_test_12345"

    @pytest.mark.asyncio
    async def test_get_subscription_async(self, stripe_connector, mock_subscription_api_response):
        """Test async subscription retrieval."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with stripe_connector:
                result = await stripe_connector.get_subscription("sub_test_12345")

            assert result.id == "sub_test_12345"

    @pytest.mark.asyncio
    async def test_cancel_subscription_immediately_async(
        self, stripe_connector, mock_subscription_api_response
    ):
        """Test async immediate subscription cancellation."""
        from aragora.connectors.payments.stripe import SubscriptionStatus

        mock_subscription_api_response["status"] = "canceled"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with stripe_connector:
                result = await stripe_connector.cancel_subscription(
                    "sub_test_12345", at_period_end=False
                )

            assert result.status == SubscriptionStatus.CANCELED

    @pytest.mark.asyncio
    async def test_cancel_subscription_at_period_end_async(
        self, stripe_connector, mock_subscription_api_response
    ):
        """Test async subscription cancellation at period end."""
        mock_subscription_api_response["cancel_at_period_end"] = True

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with stripe_connector:
                result = await stripe_connector.cancel_subscription(
                    "sub_test_12345", at_period_end=True
                )

            assert result.cancel_at_period_end is True


# =============================================================================
# Invoice Operations Tests (Async)
# =============================================================================


class TestInvoiceOperationsAsync:
    """Async tests for invoice operations."""

    @pytest.mark.asyncio
    async def test_create_invoice_async(self, stripe_connector, mock_invoice_api_response):
        """Test async invoice creation."""
        mock_invoice_api_response["status"] = "draft"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with stripe_connector:
                result = await stripe_connector.create_invoice(customer_id="cus_test_12345")

            assert result.customer_id == "cus_test_12345"

    @pytest.mark.asyncio
    async def test_finalize_invoice_async(self, stripe_connector, mock_invoice_api_response):
        """Test async invoice finalization."""
        from aragora.connectors.payments.stripe import InvoiceStatus

        mock_invoice_api_response["status"] = "open"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with stripe_connector:
                result = await stripe_connector.finalize_invoice("in_test_12345")

            assert result.status == InvoiceStatus.OPEN

    @pytest.mark.asyncio
    async def test_pay_invoice_async(self, stripe_connector, mock_invoice_api_response):
        """Test async invoice payment."""
        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with stripe_connector:
                result = await stripe_connector.pay_invoice("in_test_12345")

            assert result.paid is True

    @pytest.mark.asyncio
    async def test_void_invoice_async(self, stripe_connector, mock_invoice_api_response):
        """Test async invoice voiding."""
        from aragora.connectors.payments.stripe import InvoiceStatus

        mock_invoice_api_response["status"] = "void"

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with stripe_connector:
                result = await stripe_connector.void_invoice("in_test_12345")

            assert result.status == InvoiceStatus.VOID


# =============================================================================
# Balance Operations Tests (Async)
# =============================================================================


class TestBalanceOperationsAsync:
    """Async tests for balance operations."""

    @pytest.mark.asyncio
    async def test_get_balance_async(self, stripe_connector):
        """Test async balance retrieval."""
        mock_response = {
            "available": [{"amount": 100000, "currency": "usd"}],
            "pending": [{"amount": 25000, "currency": "usd"}],
        }

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with stripe_connector:
                result = await stripe_connector.get_balance()

            assert result["available"][0]["amount"] == 100000

    @pytest.mark.asyncio
    async def test_list_balance_transactions_async(self, stripe_connector):
        """Test async balance transaction listing."""
        mock_response = {
            "data": [
                {
                    "id": "txn_test_123",
                    "amount": 5000,
                    "currency": "usd",
                    "type": "charge",
                    "fee": 175,
                    "net": 4825,
                    "status": "available",
                    "created": int(time.time()),
                }
            ]
        }

        with patch.object(stripe_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with stripe_connector:
                result = await stripe_connector.list_balance_transactions(limit=10)

            assert len(result) == 1
            assert result[0].id == "txn_test_123"
            assert result[0].fee == 175


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_client(self, stripe_connector):
        """Test context manager properly initializes HTTP client."""
        async with stripe_connector:
            assert stripe_connector._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, stripe_connector):
        """Test context manager properly closes HTTP client."""
        async with stripe_connector:
            pass

        assert stripe_connector._client is None

    def test_client_access_without_context_raises_error(self, stripe_connector):
        """Test accessing client without context manager raises error."""
        from aragora.connectors.payments.stripe import StripeError

        with pytest.raises(StripeError) as exc_info:
            _ = stripe_connector.client

        assert "not initialized" in str(exc_info.value)
