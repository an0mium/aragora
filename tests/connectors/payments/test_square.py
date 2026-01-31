"""
Tests for Square Payment Connector.

Tests cover:
- Client initialization
- API authentication
- Payments API (creation and capture)
- Customers API
- Subscriptions API
- Invoices API
- Catalog API
- Refund processing
- Webhook signature verification
- Error handling (declined cards, network errors, invalid amounts)
- Idempotency key handling
- Currency handling
- Mock data generators
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def square_credentials():
    """Create test credentials."""
    from aragora.connectors.payments.square import SquareCredentials, SquareEnvironment

    return SquareCredentials(
        access_token="sq0atp-test_access_token_12345",
        environment=SquareEnvironment.SANDBOX,
        location_id="LOC_TEST_12345",
        webhook_signature_key="webhook_sig_key_test",
    )


@pytest.fixture
def square_client(square_credentials):
    """Create client instance."""
    from aragora.connectors.payments.square import SquareClient

    return SquareClient(square_credentials)


@pytest.fixture
def mock_payment_api_response():
    """Mock payment API response."""
    return {
        "payment": {
            "id": "PAYMENT_TEST_12345",
            "status": "COMPLETED",
            "amount_money": {"amount": 9999, "currency": "USD"},
            "total_money": {"amount": 9999, "currency": "USD"},
            "source_type": "CARD",
            "card_details": {
                "card": {"card_brand": "VISA", "last_4": "1234"},
                "status": "CAPTURED",
            },
            "location_id": "LOC_TEST_12345",
            "order_id": "ORDER_12345",
            "receipt_url": "https://squareup.com/receipt/PAYMENT_TEST_12345",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }


@pytest.fixture
def mock_refund_api_response():
    """Mock refund API response."""
    return {
        "refund": {
            "id": "REFUND_TEST_12345",
            "payment_id": "PAYMENT_TEST_12345",
            "status": "COMPLETED",
            "amount_money": {"amount": 5000, "currency": "USD"},
            "reason": "Customer request",
            "location_id": "LOC_TEST_12345",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }


@pytest.fixture
def mock_customer_api_response():
    """Mock customer API response."""
    return {
        "customer": {
            "id": "CUSTOMER_TEST_12345",
            "given_name": "John",
            "family_name": "Doe",
            "email_address": "john.doe@example.com",
            "phone_number": "+15551234567",
            "address": {
                "address_line_1": "123 Main St",
                "locality": "San Francisco",
                "administrative_district_level_1": "CA",
                "postal_code": "94102",
                "country": "US",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }


@pytest.fixture
def mock_subscription_api_response():
    """Mock subscription API response."""
    return {
        "subscription": {
            "id": "SUBSCRIPTION_TEST_12345",
            "status": "ACTIVE",
            "plan_id": "PLAN_TEST_12345",
            "customer_id": "CUSTOMER_TEST_12345",
            "location_id": "LOC_TEST_12345",
            "start_date": "2024-01-15",
            "charged_through_date": "2024-02-15",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }


@pytest.fixture
def mock_invoice_api_response():
    """Mock invoice API response."""
    return {
        "invoice": {
            "id": "INVOICE_TEST_12345",
            "version": 1,
            "status": "DRAFT",
            "location_id": "LOC_TEST_12345",
            "order_id": "ORDER_12345",
            "invoice_number": "INV-001",
            "payment_requests": [
                {
                    "request_type": "BALANCE",
                    "due_date": "2024-02-01",
                }
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }


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


# =============================================================================
# Payment Creation and Capture Tests
# =============================================================================


class TestPaymentCreation:
    """Tests for payment creation and capture."""

    @pytest.mark.asyncio
    async def test_create_payment_basic(self, square_client, mock_payment_api_response):
        """Test basic payment creation."""
        from aragora.connectors.payments.square import PaymentStatus

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.create_payment(
                    source_id="cnon:card-nonce-ok",
                    amount_money={"amount": 9999, "currency": "USD"},
                    location_id="LOC_TEST_12345",
                )

            assert result.id == "PAYMENT_TEST_12345"
            assert result.status == PaymentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_create_payment_with_idempotency_key(
        self, square_client, mock_payment_api_response
    ):
        """Test payment creation with idempotency key."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.create_payment(
                    source_id="cnon:card-nonce-ok",
                    amount_money={"amount": 9999, "currency": "USD"},
                    location_id="LOC_TEST_12345",
                    idempotency_key="unique_key_12345",
                )

            assert result.id == "PAYMENT_TEST_12345"

    @pytest.mark.asyncio
    async def test_get_payment(self, square_client, mock_payment_api_response):
        """Test getting payment details."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.get_payment("PAYMENT_TEST_12345")

            assert result.id == "PAYMENT_TEST_12345"

    @pytest.mark.asyncio
    async def test_complete_payment(self, square_client, mock_payment_api_response):
        """Test completing (capturing) a payment."""
        from aragora.connectors.payments.square import PaymentStatus

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.complete_payment("PAYMENT_TEST_12345")

            assert result.status == PaymentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_payment(self, square_client, mock_payment_api_response):
        """Test canceling a payment."""
        from aragora.connectors.payments.square import PaymentStatus

        mock_payment_api_response["payment"]["status"] = "CANCELED"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.cancel_payment("PAYMENT_TEST_12345")

            assert result.status == PaymentStatus.CANCELED


# =============================================================================
# Refund Processing Tests
# =============================================================================


class TestRefundProcessing:
    """Tests for refund processing."""

    @pytest.mark.asyncio
    async def test_refund_payment_full(self, square_client, mock_refund_api_response):
        """Test full refund of a payment."""
        mock_refund_api_response["refund"]["amount_money"]["amount"] = 9999

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with square_client:
                result = await square_client.refund_payment(
                    payment_id="PAYMENT_TEST_12345",
                    amount_money={"amount": 9999, "currency": "USD"},
                    idempotency_key="refund_key_12345",
                )

            assert result.id == "REFUND_TEST_12345"
            assert result.status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_refund_payment_partial(self, square_client, mock_refund_api_response):
        """Test partial refund of a payment."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with square_client:
                result = await square_client.refund_payment(
                    payment_id="PAYMENT_TEST_12345",
                    amount_money={"amount": 5000, "currency": "USD"},
                    idempotency_key="partial_refund_key",
                )

            assert result.amount_money.amount == 5000

    @pytest.mark.asyncio
    async def test_refund_with_reason(self, square_client, mock_refund_api_response):
        """Test refund with reason."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with square_client:
                result = await square_client.refund_payment(
                    payment_id="PAYMENT_TEST_12345",
                    amount_money={"amount": 5000, "currency": "USD"},
                    idempotency_key="refund_with_reason",
                    reason="Customer request",
                )

            assert result.reason == "Customer request"


# =============================================================================
# Webhook Signature Verification Tests
# =============================================================================


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    def test_verify_webhook_signature_valid(self, square_client):
        """Test verification of valid webhook signature."""
        import base64

        notification_url = "https://webhook.example.com/square"
        payload = b'{"type":"payment.created","event_id":"evt_12345"}'

        string_to_sign = notification_url + payload.decode()
        signature = hmac.new(
            square_client.credentials.webhook_signature_key.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_sig = base64.b64encode(signature).decode()

        result = square_client.verify_webhook_signature(
            payload=payload,
            signature=expected_sig,
            notification_url=notification_url,
        )

        assert result is True

    def test_verify_webhook_signature_invalid(self, square_client):
        """Test rejection of invalid webhook signature."""
        payload = b'{"type":"payment.created"}'
        invalid_signature = "invalid_signature_base64"
        notification_url = "https://webhook.example.com/square"

        result = square_client.verify_webhook_signature(
            payload=payload,
            signature=invalid_signature,
            notification_url=notification_url,
        )

        assert result is False

    def test_parse_webhook_event(self, square_client):
        """Test parsing webhook event payload."""
        webhook_data = {
            "merchant_id": "MERCHANT_12345",
            "type": "payment.created",
            "event_id": "EVT_12345",
            "created_at": "2024-01-15T12:00:00Z",
            "data": {
                "type": "payment",
                "id": "PAYMENT_12345",
            },
        }

        event = square_client.parse_webhook_event(webhook_data)

        assert event["type"] == "payment.created"
        assert event["data"]["id"] == "PAYMENT_12345"


# =============================================================================
# Error Handling Tests (Async)
# =============================================================================


class TestErrorHandlingAsync:
    """Async tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_declined_card_error(self, square_client):
        """Test handling of declined card error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError(
                message="Card declined",
                status_code=400,
                code="CARD_DECLINED",
                category="PAYMENT_METHOD_ERROR",
            )

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.create_payment(
                        source_id="cnon:card-nonce-declined",
                        amount_money={"amount": 9999, "currency": "USD"},
                        location_id="LOC_TEST_12345",
                    )

            assert exc_info.value.code == "CARD_DECLINED"

    @pytest.mark.asyncio
    async def test_network_error(self, square_client):
        """Test handling of network error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError("HTTP error: Connection refused")

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.get_payment("PAYMENT_12345")

            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_amount_error(self, square_client):
        """Test handling of invalid amount error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError(
                message="Amount is invalid",
                status_code=400,
                code="INVALID_VALUE",
                category="INVALID_REQUEST_ERROR",
            )

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.create_payment(
                        source_id="cnon:card-nonce-ok",
                        amount_money={"amount": -100, "currency": "USD"},
                        location_id="LOC_TEST_12345",
                    )

            assert exc_info.value.code == "INVALID_VALUE"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, square_client):
        """Test handling of rate limit error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError(
                message="Rate limit exceeded",
                status_code=429,
                code="RATE_LIMITED",
                category="RATE_LIMIT_ERROR",
            )

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.create_payment(
                        source_id="cnon:card-nonce-ok",
                        amount_money={"amount": 9999, "currency": "USD"},
                        location_id="LOC_TEST_12345",
                    )

            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_authentication_error(self, square_client):
        """Test handling of authentication error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError(
                message="Invalid access token",
                status_code=401,
                code="UNAUTHORIZED",
                category="AUTHENTICATION_ERROR",
            )

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.get_payment("PAYMENT_12345")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_resource_not_found_error(self, square_client):
        """Test handling of resource not found error."""
        from aragora.connectors.payments.square import SquareError

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = SquareError(
                message="Payment not found",
                status_code=404,
                code="NOT_FOUND",
                category="INVALID_REQUEST_ERROR",
            )

            async with square_client:
                with pytest.raises(SquareError) as exc_info:
                    await square_client.get_payment("INVALID_PAYMENT_ID")

            assert exc_info.value.status_code == 404


# =============================================================================
# Currency Handling Tests
# =============================================================================


class TestCurrencyHandling:
    """Tests for currency handling."""

    @pytest.mark.asyncio
    async def test_payment_in_gbp(self, square_client, mock_payment_api_response):
        """Test payment in GBP."""
        mock_payment_api_response["payment"]["amount_money"]["currency"] = "GBP"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.create_payment(
                    source_id="cnon:card-nonce-ok",
                    amount_money={"amount": 7500, "currency": "GBP"},
                    location_id="LOC_TEST_12345",
                )

            assert result.amount_money.currency == "GBP"

    @pytest.mark.asyncio
    async def test_payment_in_cad(self, square_client, mock_payment_api_response):
        """Test payment in CAD."""
        mock_payment_api_response["payment"]["amount_money"]["currency"] = "CAD"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.create_payment(
                    source_id="cnon:card-nonce-ok",
                    amount_money={"amount": 12500, "currency": "CAD"},
                    location_id="LOC_TEST_12345",
                )

            assert result.amount_money.currency == "CAD"

    @pytest.mark.asyncio
    async def test_zero_decimal_currency_jpy(self, square_client, mock_payment_api_response):
        """Test zero-decimal currency (JPY)."""
        mock_payment_api_response["payment"]["amount_money"] = {
            "amount": 5000,
            "currency": "JPY",
        }

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payment_api_response

            async with square_client:
                result = await square_client.create_payment(
                    source_id="cnon:card-nonce-ok",
                    amount_money={"amount": 5000, "currency": "JPY"},
                    location_id="LOC_TEST_12345",
                )

            assert result.amount_money.currency == "JPY"
            assert result.amount_money.amount == 5000


# =============================================================================
# Customer Operations Tests (Async)
# =============================================================================


class TestCustomerOperationsAsync:
    """Async tests for customer operations."""

    @pytest.mark.asyncio
    async def test_create_customer(self, square_client, mock_customer_api_response):
        """Test customer creation."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with square_client:
                result = await square_client.create_customer(
                    given_name="John",
                    family_name="Doe",
                    email_address="john.doe@example.com",
                )

            assert result.id == "CUSTOMER_TEST_12345"
            assert result.given_name == "John"

    @pytest.mark.asyncio
    async def test_get_customer(self, square_client, mock_customer_api_response):
        """Test getting customer details."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with square_client:
                result = await square_client.get_customer("CUSTOMER_TEST_12345")

            assert result.id == "CUSTOMER_TEST_12345"

    @pytest.mark.asyncio
    async def test_update_customer(self, square_client, mock_customer_api_response):
        """Test updating customer."""
        mock_customer_api_response["customer"]["phone_number"] = "+15559876543"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_customer_api_response

            async with square_client:
                result = await square_client.update_customer(
                    customer_id="CUSTOMER_TEST_12345",
                    phone_number="+15559876543",
                )

            assert result.phone_number == "+15559876543"

    @pytest.mark.asyncio
    async def test_delete_customer(self, square_client):
        """Test deleting customer."""
        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}

            async with square_client:
                await square_client.delete_customer("CUSTOMER_TEST_12345")

            mock_request.assert_called_once()


# =============================================================================
# Subscription Operations Tests (Async)
# =============================================================================


class TestSubscriptionOperationsAsync:
    """Async tests for subscription operations."""

    @pytest.mark.asyncio
    async def test_create_subscription(self, square_client, mock_subscription_api_response):
        """Test subscription creation."""
        from aragora.connectors.payments.square import SubscriptionStatus

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with square_client:
                result = await square_client.create_subscription(
                    location_id="LOC_TEST_12345",
                    plan_id="PLAN_TEST_12345",
                    customer_id="CUSTOMER_TEST_12345",
                )

            assert result.id == "SUBSCRIPTION_TEST_12345"
            assert result.status == SubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, square_client, mock_subscription_api_response):
        """Test canceling subscription."""
        from aragora.connectors.payments.square import SubscriptionStatus

        mock_subscription_api_response["subscription"]["status"] = "CANCELED"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with square_client:
                result = await square_client.cancel_subscription("SUBSCRIPTION_TEST_12345")

            assert result.status == SubscriptionStatus.CANCELED


# =============================================================================
# Invoice Operations Tests (Async)
# =============================================================================


class TestInvoiceOperationsAsync:
    """Async tests for invoice operations."""

    @pytest.mark.asyncio
    async def test_create_invoice(self, square_client, mock_invoice_api_response):
        """Test invoice creation."""
        from aragora.connectors.payments.square import InvoiceStatus

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with square_client:
                result = await square_client.create_invoice(
                    location_id="LOC_TEST_12345",
                    order_id="ORDER_12345",
                    idempotency_key="invoice_key_12345",
                )

            assert result.id == "INVOICE_TEST_12345"
            assert result.status == InvoiceStatus.DRAFT

    @pytest.mark.asyncio
    async def test_publish_invoice(self, square_client, mock_invoice_api_response):
        """Test publishing invoice."""
        from aragora.connectors.payments.square import InvoiceStatus

        mock_invoice_api_response["invoice"]["status"] = "PUBLISHED"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with square_client:
                result = await square_client.publish_invoice(
                    invoice_id="INVOICE_TEST_12345",
                    version=1,
                    idempotency_key="publish_key_12345",
                )

            assert result.status == InvoiceStatus.PUBLISHED

    @pytest.mark.asyncio
    async def test_cancel_invoice(self, square_client, mock_invoice_api_response):
        """Test canceling invoice."""
        from aragora.connectors.payments.square import InvoiceStatus

        mock_invoice_api_response["invoice"]["status"] = "CANCELED"

        with patch.object(square_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_invoice_api_response

            async with square_client:
                result = await square_client.cancel_invoice(
                    invoice_id="INVOICE_TEST_12345",
                    version=1,
                )

            assert result.status == InvoiceStatus.CANCELED


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_client(self, square_client):
        """Test context manager properly initializes HTTP client."""
        async with square_client:
            assert square_client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, square_client):
        """Test context manager properly closes HTTP client."""
        async with square_client:
            pass

        assert square_client._client is None
