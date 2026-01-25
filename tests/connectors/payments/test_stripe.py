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
- Error handling
- Mock data generators
"""

from datetime import datetime

import pytest


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
