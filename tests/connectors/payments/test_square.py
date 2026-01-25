"""
Tests for Square Payment Connector.

Tests cover:
- Client initialization
- API authentication
- Payments API
- Customers API
- Subscriptions API
- Invoices API
- Catalog API
- Error handling
- Mock data generators
"""

from datetime import datetime, timezone

import pytest


# =============================================================================
# Enum Tests
# =============================================================================


class TestSquareEnums:
    """Tests for Square enums."""

    def test_environment_values(self):
        """SquareEnvironment enum has expected values."""
        from aragora.connectors.payments.square import SquareEnvironment

        assert SquareEnvironment.SANDBOX.value == "sandbox"
        assert SquareEnvironment.PRODUCTION.value == "production"

    def test_payment_status_values(self):
        """PaymentStatus enum has expected values."""
        from aragora.connectors.payments.square import PaymentStatus

        assert PaymentStatus.APPROVED.value == "APPROVED"
        assert PaymentStatus.COMPLETED.value == "COMPLETED"
        assert PaymentStatus.CANCELED.value == "CANCELED"
        assert PaymentStatus.FAILED.value == "FAILED"

    def test_card_brand_values(self):
        """CardBrand enum has expected values."""
        from aragora.connectors.payments.square import CardBrand

        assert CardBrand.VISA.value == "VISA"
        assert CardBrand.MASTERCARD.value == "MASTERCARD"
        assert CardBrand.AMERICAN_EXPRESS.value == "AMERICAN_EXPRESS"

    def test_subscription_status_values(self):
        """SubscriptionStatus enum has expected values."""
        from aragora.connectors.payments.square import SubscriptionStatus

        assert SubscriptionStatus.ACTIVE.value == "ACTIVE"
        assert SubscriptionStatus.CANCELED.value == "CANCELED"
        assert SubscriptionStatus.PAUSED.value == "PAUSED"

    def test_invoice_status_values(self):
        """InvoiceStatus enum has expected values."""
        from aragora.connectors.payments.square import InvoiceStatus

        assert InvoiceStatus.DRAFT.value == "DRAFT"
        assert InvoiceStatus.PUBLISHED.value == "PUBLISHED"
        assert InvoiceStatus.PAID.value == "PAID"
        assert InvoiceStatus.CANCELED.value == "CANCELED"

    def test_catalog_object_type_values(self):
        """CatalogObjectType enum has expected values."""
        from aragora.connectors.payments.square import CatalogObjectType

        assert CatalogObjectType.ITEM.value == "ITEM"
        assert CatalogObjectType.ITEM_VARIATION.value == "ITEM_VARIATION"
        assert CatalogObjectType.CATEGORY.value == "CATEGORY"


# =============================================================================
# Credentials Tests
# =============================================================================


class TestSquareCredentials:
    """Tests for SquareCredentials."""

    def test_credentials_init(self):
        """Create credentials with access token."""
        from aragora.connectors.payments.square import SquareCredentials, SquareEnvironment

        creds = SquareCredentials(access_token="sq0atp-test_token")

        assert creds.access_token == "sq0atp-test_token"
        assert creds.environment == SquareEnvironment.SANDBOX

    def test_sandbox_base_url(self):
        """Sandbox environment uses sandbox URL."""
        from aragora.connectors.payments.square import SquareCredentials, SquareEnvironment

        creds = SquareCredentials(
            access_token="token",
            environment=SquareEnvironment.SANDBOX,
        )

        assert creds.base_url == "https://connect.squareupsandbox.com"

    def test_production_base_url(self):
        """Production environment uses production URL."""
        from aragora.connectors.payments.square import SquareCredentials, SquareEnvironment

        creds = SquareCredentials(
            access_token="token",
            environment=SquareEnvironment.PRODUCTION,
        )

        assert creds.base_url == "https://connect.squareup.com"

    def test_credentials_with_location(self):
        """Create credentials with location ID."""
        from aragora.connectors.payments.square import SquareCredentials

        creds = SquareCredentials(
            access_token="token",
            location_id="LOC123",
        )

        assert creds.location_id == "LOC123"


# =============================================================================
# Data Model Tests
# =============================================================================


class TestMoney:
    """Tests for Money dataclass."""

    def test_money_from_api(self):
        """Parse Money from API response."""
        from aragora.connectors.payments.square import Money

        data = {"amount": 9999, "currency": "USD"}
        money = Money.from_api(data)

        assert money.amount == 9999
        assert money.currency == "USD"

    def test_money_to_api(self):
        """Convert Money to API format."""
        from aragora.connectors.payments.square import Money

        money = Money(amount=5000, currency="EUR")
        result = money.to_api()

        assert result == {"amount": 5000, "currency": "EUR"}

    def test_money_usd_helper(self):
        """Create USD money from dollars."""
        from aragora.connectors.payments.square import Money

        # Use exact dollar amounts to avoid floating point issues
        money = Money.usd(20.00)

        assert money.amount == 2000
        assert money.currency == "USD"

    def test_money_as_dollars(self):
        """Convert cents to dollars."""
        from aragora.connectors.payments.square import Money

        money = Money(amount=9999, currency="USD")
        assert money.as_dollars == 99.99

    def test_money_from_api_none(self):
        """Handle None input."""
        from aragora.connectors.payments.square import Money

        assert Money.from_api(None) is None


class TestAddress:
    """Tests for Address dataclass."""

    def test_address_from_api(self):
        """Parse Address from API response."""
        from aragora.connectors.payments.square import Address

        data = {
            "address_line_1": "123 Main St",
            "locality": "San Francisco",
            "administrative_district_level_1": "CA",
            "postal_code": "94102",
            "country": "US",
        }

        address = Address.from_api(data)

        assert address.address_line_1 == "123 Main St"
        assert address.locality == "San Francisco"
        assert address.postal_code == "94102"

    def test_address_to_api(self):
        """Convert Address to API format."""
        from aragora.connectors.payments.square import Address

        address = Address(
            address_line_1="456 Oak Ave",
            locality="New York",
            administrative_district_level_1="NY",
            postal_code="10001",
        )

        result = address.to_api()

        assert result["address_line_1"] == "456 Oak Ave"
        assert result["locality"] == "New York"


class TestCard:
    """Tests for Card dataclass."""

    def test_card_from_api(self):
        """Parse Card from API response."""
        from aragora.connectors.payments.square import Card, CardBrand

        data = {
            "id": "CARD123",
            "card_brand": "VISA",
            "last_4": "1234",
            "exp_month": 12,
            "exp_year": 2027,
            "cardholder_name": "John Doe",
        }

        card = Card.from_api(data)

        assert card.id == "CARD123"
        assert card.card_brand == CardBrand.VISA
        assert card.last_4 == "1234"
        assert card.exp_month == 12


class TestCustomer:
    """Tests for Customer dataclass."""

    def test_customer_from_api(self):
        """Parse Customer from API response."""
        from aragora.connectors.payments.square import Customer

        data = {
            "id": "CUST123",
            "given_name": "John",
            "family_name": "Doe",
            "email_address": "john@example.com",
            "phone_number": "+15551234567",
            "created_at": "2025-01-01T12:00:00Z",
        }

        customer = Customer.from_api(data)

        assert customer.id == "CUST123"
        assert customer.given_name == "John"
        assert customer.family_name == "Doe"
        assert customer.email_address == "john@example.com"

    def test_customer_full_name(self):
        """Get full name property."""
        from aragora.connectors.payments.square import Customer

        customer = Customer(
            id="CUST",
            given_name="Jane",
            family_name="Smith",
        )

        assert customer.full_name == "Jane Smith"


class TestPayment:
    """Tests for Payment dataclass."""

    def test_payment_from_api(self):
        """Parse Payment from API response."""
        from aragora.connectors.payments.square import Payment, PaymentStatus

        data = {
            "id": "PAY123",
            "status": "COMPLETED",
            "amount_money": {"amount": 9999, "currency": "USD"},
            "total_money": {"amount": 9999, "currency": "USD"},
            "source_type": "CARD",
            "receipt_url": "https://squareup.com/receipt/123",
        }

        payment = Payment.from_api(data)

        assert payment.id == "PAY123"
        assert payment.status == PaymentStatus.COMPLETED
        assert payment.amount_money.amount == 9999
        assert payment.receipt_url == "https://squareup.com/receipt/123"


class TestRefund:
    """Tests for Refund dataclass."""

    def test_refund_from_api(self):
        """Parse Refund from API response."""
        from aragora.connectors.payments.square import Refund

        data = {
            "id": "REF123",
            "payment_id": "PAY123",
            "status": "COMPLETED",
            "amount_money": {"amount": 5000, "currency": "USD"},
            "reason": "Customer request",
        }

        refund = Refund.from_api(data)

        assert refund.id == "REF123"
        assert refund.payment_id == "PAY123"
        assert refund.status == "COMPLETED"
        assert refund.reason == "Customer request"


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_subscription_from_api(self):
        """Parse Subscription from API response."""
        from aragora.connectors.payments.square import Subscription, SubscriptionStatus

        data = {
            "id": "SUB123",
            "status": "ACTIVE",
            "plan_id": "PLAN123",
            "customer_id": "CUST123",
            "start_date": "2025-01-01",
        }

        subscription = Subscription.from_api(data)

        assert subscription.id == "SUB123"
        assert subscription.status == SubscriptionStatus.ACTIVE
        assert subscription.plan_id == "PLAN123"
        assert subscription.start_date == "2025-01-01"


class TestInvoice:
    """Tests for Invoice dataclass."""

    def test_invoice_from_api(self):
        """Parse Invoice from API response."""
        from aragora.connectors.payments.square import Invoice, InvoiceStatus

        data = {
            "id": "INV123",
            "version": 1,
            "status": "DRAFT",
            "location_id": "LOC123",
            "order_id": "ORD123",
            "invoice_number": "INV-001",
        }

        invoice = Invoice.from_api(data)

        assert invoice.id == "INV123"
        assert invoice.version == 1
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.invoice_number == "INV-001"


class TestCatalogItem:
    """Tests for CatalogItem dataclass."""

    def test_catalog_item_from_api(self):
        """Parse CatalogItem from API response."""
        from aragora.connectors.payments.square import CatalogItem, CatalogObjectType

        data = {
            "id": "CAT123",
            "type": "ITEM",
            "item_data": {
                "name": "Coffee",
                "description": "Fresh brewed coffee",
                "variations": [],
            },
            "is_deleted": False,
        }

        item = CatalogItem.from_api(data)

        assert item.id == "CAT123"
        assert item.type == CatalogObjectType.ITEM
        assert item.name == "Coffee"
        assert item.description == "Fresh brewed coffee"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSquareError:
    """Tests for SquareError."""

    def test_error_creation(self):
        """Create error with details."""
        from aragora.connectors.payments.square import SquareError

        error = SquareError(
            message="Card declined",
            status_code=400,
            code="CARD_DECLINED",
            category="PAYMENT_METHOD_ERROR",
        )

        assert str(error) == "Card declined"
        assert error.status_code == 400
        assert error.code == "CARD_DECLINED"
        assert error.category == "PAYMENT_METHOD_ERROR"


# =============================================================================
# Client Tests
# =============================================================================


class TestSquareClientInit:
    """Tests for SquareClient initialization."""

    def test_client_creation(self):
        """Create client with credentials."""
        from aragora.connectors.payments.square import SquareClient, SquareCredentials

        creds = SquareCredentials(access_token="test_token")
        client = SquareClient(creds)

        assert client.credentials == creds
        assert client.API_VERSION == "2024-01-18"


# =============================================================================
# Mock Data Generator Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data generators."""

    def test_get_mock_customer(self):
        """Get mock customer for testing."""
        from aragora.connectors.payments.square import get_mock_customer

        customer = get_mock_customer()

        assert customer.id == "CUSTOMER_ID_12345"
        assert customer.given_name == "John"
        assert customer.family_name == "Doe"
        assert customer.email_address == "john.doe@example.com"

    def test_get_mock_payment(self):
        """Get mock payment for testing."""
        from aragora.connectors.payments.square import get_mock_payment, PaymentStatus

        payment = get_mock_payment()

        assert payment.id == "PAYMENT_ID_12345"
        assert payment.status == PaymentStatus.COMPLETED
        assert payment.amount_money.amount == 9999

    def test_get_mock_subscription(self):
        """Get mock subscription for testing."""
        from aragora.connectors.payments.square import get_mock_subscription, SubscriptionStatus

        subscription = get_mock_subscription()

        assert subscription.id == "SUBSCRIPTION_ID_12345"
        assert subscription.status == SubscriptionStatus.ACTIVE


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_payments_module(self):
        """Import Square from payments module."""
        from aragora.connectors.payments import (
            SquareClient,
            SquareCredentials,
            SquareEnvironment,
            SquareError,
            SquarePaymentStatus,
            CardBrand,
            SquareSubscriptionStatus,
            SquareInvoiceStatus,
            CatalogObjectType,
            SquareMoney,
            SquareAddress,
            SquareCard,
            SquareCustomer,
            SquarePayment,
            SquareRefund,
            SubscriptionPlan,
            SquareSubscription,
            SquareInvoice,
            CatalogItem,
        )

        # Verify imports work
        assert SquareClient is not None
        assert SquareCredentials is not None
        assert SquareError is not None
        assert SquarePaymentStatus is not None
        assert SquareMoney is not None
