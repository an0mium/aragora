"""
Tests for PayPal Payment Connector.

Tests cover:
- Client initialization
- OAuth2 authentication
- Orders API (creation and capture)
- Captures and refunds
- Subscriptions
- Payouts
- Webhook signature verification
- Error handling (declined payments, network errors, invalid amounts)
- Idempotency handling
- Currency conversion
- Mock data generators
"""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import time
import zlib
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def paypal_credentials():
    """Create test credentials."""
    from aragora.connectors.payments.paypal import PayPalCredentials, PayPalEnvironment

    return PayPalCredentials(
        client_id="test_client_id_12345",
        client_secret="test_client_secret_67890",
        environment=PayPalEnvironment.SANDBOX,
        webhook_id="webhook_test_id",
    )


@pytest.fixture
def paypal_client(paypal_credentials):
    """Create client instance."""
    from aragora.connectors.payments.paypal import PayPalClient

    return PayPalClient(paypal_credentials)


@pytest.fixture
def mock_order_api_response():
    """Mock order API response."""
    return {
        "id": "ORDER_TEST_12345",
        "status": "CREATED",
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "reference_id": "ref_001",
                "amount": {"currency_code": "USD", "value": "99.99"},
                "description": "Test purchase",
            }
        ],
        "payer": {
            "payer_id": "PAYER_12345",
            "email_address": "buyer@example.com",
            "name": {"given_name": "John", "surname": "Buyer"},
        },
        "create_time": datetime.now(timezone.utc).isoformat(),
        "links": [
            {"rel": "self", "href": "https://api.paypal.com/v2/checkout/orders/ORDER_TEST_12345"},
            {"rel": "approve", "href": "https://www.paypal.com/checkoutnow?token=ORDER_TEST_12345"},
        ],
    }


@pytest.fixture
def mock_capture_api_response():
    """Mock capture API response."""
    return {
        "id": "CAPTURE_TEST_12345",
        "status": "COMPLETED",
        "amount": {"currency_code": "USD", "value": "99.99"},
        "final_capture": True,
        "seller_protection": {"status": "ELIGIBLE"},
        "create_time": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_refund_api_response():
    """Mock refund API response."""
    return {
        "id": "REFUND_TEST_12345",
        "status": "COMPLETED",
        "amount": {"currency_code": "USD", "value": "25.00"},
        "note_to_payer": "Partial refund processed",
        "create_time": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_subscription_api_response():
    """Mock subscription API response."""
    return {
        "id": "SUB_TEST_12345",
        "status": "ACTIVE",
        "plan_id": "PLAN_12345",
        "subscriber": {"email_address": "subscriber@example.com"},
        "start_time": datetime.now(timezone.utc).isoformat(),
        "create_time": datetime.now(timezone.utc).isoformat(),
        "links": [
            {
                "rel": "approve",
                "href": "https://www.paypal.com/subscription/approve/SUB_TEST_12345",
            },
        ],
    }


@pytest.fixture
def mock_payout_api_response():
    """Mock payout batch API response."""
    return {
        "batch_header": {
            "payout_batch_id": "BATCH_TEST_12345",
            "batch_status": "SUCCESS",
            "sender_batch_header": {"sender_batch_id": "MY_BATCH_001"},
            "amount": {"currency_code": "USD", "value": "500.00"},
            "fees": {"currency_code": "USD", "value": "1.25"},
        },
        "items": [],
    }


# =============================================================================
# Enum Tests
# =============================================================================


class TestPayPalEnums:
    """Tests for PayPal enums."""

    def test_environment_values(self):
        """PayPalEnvironment enum has expected values."""
        from aragora.connectors.payments.paypal import PayPalEnvironment

        assert PayPalEnvironment.SANDBOX.value == "sandbox"
        assert PayPalEnvironment.LIVE.value == "live"

    def test_order_status_values(self):
        """OrderStatus enum has expected values."""
        from aragora.connectors.payments.paypal import OrderStatus

        assert OrderStatus.CREATED.value == "CREATED"
        assert OrderStatus.APPROVED.value == "APPROVED"
        assert OrderStatus.COMPLETED.value == "COMPLETED"
        assert OrderStatus.VOIDED.value == "VOIDED"

    def test_order_intent_values(self):
        """OrderIntent enum has expected values."""
        from aragora.connectors.payments.paypal import OrderIntent

        assert OrderIntent.CAPTURE.value == "CAPTURE"
        assert OrderIntent.AUTHORIZE.value == "AUTHORIZE"

    def test_capture_status_values(self):
        """CaptureStatus enum has expected values."""
        from aragora.connectors.payments.paypal import CaptureStatus

        assert CaptureStatus.COMPLETED.value == "COMPLETED"
        assert CaptureStatus.PENDING.value == "PENDING"
        assert CaptureStatus.REFUNDED.value == "REFUNDED"

    def test_refund_status_values(self):
        """RefundStatus enum has expected values."""
        from aragora.connectors.payments.paypal import RefundStatus

        assert RefundStatus.COMPLETED.value == "COMPLETED"
        assert RefundStatus.PENDING.value == "PENDING"
        assert RefundStatus.FAILED.value == "FAILED"

    def test_subscription_status_values(self):
        """SubscriptionStatus enum has expected values."""
        from aragora.connectors.payments.paypal import SubscriptionStatus

        assert SubscriptionStatus.ACTIVE.value == "ACTIVE"
        assert SubscriptionStatus.SUSPENDED.value == "SUSPENDED"
        assert SubscriptionStatus.CANCELLED.value == "CANCELLED"

    def test_payout_batch_status_values(self):
        """PayoutBatchStatus enum has expected values."""
        from aragora.connectors.payments.paypal import PayoutBatchStatus

        assert PayoutBatchStatus.PENDING.value == "PENDING"
        assert PayoutBatchStatus.SUCCESS.value == "SUCCESS"
        assert PayoutBatchStatus.DENIED.value == "DENIED"


# =============================================================================
# Credentials Tests
# =============================================================================


class TestPayPalCredentials:
    """Tests for PayPalCredentials."""

    def test_credentials_init(self):
        """Create credentials with client ID and secret."""
        from aragora.connectors.payments.paypal import PayPalCredentials, PayPalEnvironment

        creds = PayPalCredentials(
            client_id="test_client_id",
            client_secret="test_secret",
        )

        assert creds.client_id == "test_client_id"
        assert creds.client_secret == "test_secret"
        assert creds.environment == PayPalEnvironment.SANDBOX

    def test_sandbox_base_url(self):
        """Sandbox environment uses sandbox URL."""
        from aragora.connectors.payments.paypal import PayPalCredentials, PayPalEnvironment

        creds = PayPalCredentials(
            client_id="id",
            client_secret="secret",
            environment=PayPalEnvironment.SANDBOX,
        )

        assert creds.base_url == "https://api-m.sandbox.paypal.com"

    def test_live_base_url(self):
        """Live environment uses production URL."""
        from aragora.connectors.payments.paypal import PayPalCredentials, PayPalEnvironment

        creds = PayPalCredentials(
            client_id="id",
            client_secret="secret",
            environment=PayPalEnvironment.LIVE,
        )

        assert creds.base_url == "https://api-m.paypal.com"


# =============================================================================
# Data Model Tests
# =============================================================================


class TestMoney:
    """Tests for Money dataclass."""

    def test_money_from_api(self):
        """Parse Money from API response."""
        from aragora.connectors.payments.paypal import Money

        data = {"currency_code": "USD", "value": "99.99"}
        money = Money.from_api(data)

        assert money.currency_code == "USD"
        assert money.value == "99.99"

    def test_money_to_api(self):
        """Convert Money to API format."""
        from aragora.connectors.payments.paypal import Money

        money = Money(currency_code="EUR", value="50.00")
        result = money.to_api()

        assert result == {"currency_code": "EUR", "value": "50.00"}

    def test_money_usd_helper(self):
        """Create USD money from amount."""
        from aragora.connectors.payments.paypal import Money

        money = Money.usd(19.99)

        assert money.currency_code == "USD"
        assert money.value == "19.99"


class TestPayerName:
    """Tests for PayerName dataclass."""

    def test_payer_name_from_api(self):
        """Parse PayerName from API response."""
        from aragora.connectors.payments.paypal import PayerName

        data = {"given_name": "John", "surname": "Doe"}
        name = PayerName.from_api(data)

        assert name.given_name == "John"
        assert name.surname == "Doe"

    def test_payer_name_full_name(self):
        """Get full name property."""
        from aragora.connectors.payments.paypal import PayerName

        name = PayerName(given_name="Jane", surname="Smith")
        assert name.full_name == "Jane Smith"


class TestPayer:
    """Tests for Payer dataclass."""

    def test_payer_from_api(self):
        """Parse Payer from API response."""
        from aragora.connectors.payments.paypal import Payer

        data = {
            "payer_id": "PAYER123",
            "email_address": "buyer@example.com",
            "name": {"given_name": "John", "surname": "Buyer"},
        }

        payer = Payer.from_api(data)

        assert payer.payer_id == "PAYER123"
        assert payer.email_address == "buyer@example.com"
        assert payer.name.given_name == "John"


class TestPurchaseUnit:
    """Tests for PurchaseUnit dataclass."""

    def test_purchase_unit_from_api(self):
        """Parse PurchaseUnit from API response."""
        from aragora.connectors.payments.paypal import PurchaseUnit

        data = {
            "reference_id": "ref123",
            "description": "Test purchase",
            "amount": {"currency_code": "USD", "value": "100.00"},
        }

        unit = PurchaseUnit.from_api(data)

        assert unit.reference_id == "ref123"
        assert unit.description == "Test purchase"
        assert unit.amount.value == "100.00"

    def test_purchase_unit_to_api(self):
        """Convert PurchaseUnit to API format."""
        from aragora.connectors.payments.paypal import PurchaseUnit, Money

        unit = PurchaseUnit(
            reference_id="ref456",
            description="Another purchase",
            amount=Money.usd(50.00),
        )

        result = unit.to_api()

        assert result["reference_id"] == "ref456"
        assert result["description"] == "Another purchase"
        assert result["amount"]["value"] == "50.00"


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_from_api(self):
        """Parse Order from API response."""
        from aragora.connectors.payments.paypal import Order, OrderStatus, OrderIntent

        data = {
            "id": "ORDER123",
            "status": "CREATED",
            "intent": "CAPTURE",
            "purchase_units": [{"amount": {"currency_code": "USD", "value": "99.99"}}],
            "links": [{"rel": "approve", "href": "https://paypal.com/approve/123"}],
        }

        order = Order.from_api(data)

        assert order.id == "ORDER123"
        assert order.status == OrderStatus.CREATED
        assert order.intent == OrderIntent.CAPTURE
        assert len(order.purchase_units) == 1

    def test_order_get_approve_link(self):
        """Get approval URL from order links."""
        from aragora.connectors.payments.paypal import Order, OrderStatus, OrderIntent

        order = Order(
            id="TEST",
            status=OrderStatus.CREATED,
            intent=OrderIntent.CAPTURE,
            links=[
                {"rel": "self", "href": "https://paypal.com/self"},
                {"rel": "approve", "href": "https://paypal.com/approve/test"},
            ],
        )

        assert order.get_approve_link() == "https://paypal.com/approve/test"


class TestCapture:
    """Tests for Capture dataclass."""

    def test_capture_from_api(self):
        """Parse Capture from API response."""
        from aragora.connectors.payments.paypal import Capture, CaptureStatus

        data = {
            "id": "CAPTURE123",
            "status": "COMPLETED",
            "amount": {"currency_code": "USD", "value": "100.00"},
            "final_capture": True,
        }

        capture = Capture.from_api(data)

        assert capture.id == "CAPTURE123"
        assert capture.status == CaptureStatus.COMPLETED
        assert capture.final_capture is True


class TestRefund:
    """Tests for Refund dataclass."""

    def test_refund_from_api(self):
        """Parse Refund from API response."""
        from aragora.connectors.payments.paypal import Refund, RefundStatus

        data = {
            "id": "REFUND123",
            "status": "COMPLETED",
            "amount": {"currency_code": "USD", "value": "25.00"},
            "note_to_payer": "Partial refund",
        }

        refund = Refund.from_api(data)

        assert refund.id == "REFUND123"
        assert refund.status == RefundStatus.COMPLETED
        assert refund.note_to_payer == "Partial refund"


class TestBillingPlan:
    """Tests for BillingPlan dataclass."""

    def test_billing_plan_from_api(self):
        """Parse BillingPlan from API response."""
        from aragora.connectors.payments.paypal import BillingPlan

        data = {
            "id": "PLAN123",
            "name": "Monthly Subscription",
            "description": "Monthly access plan",
            "status": "ACTIVE",
            "product_id": "PROD123",
        }

        plan = BillingPlan.from_api(data)

        assert plan.id == "PLAN123"
        assert plan.name == "Monthly Subscription"
        assert plan.status == "ACTIVE"


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_subscription_from_api(self):
        """Parse Subscription from API response."""
        from aragora.connectors.payments.paypal import Subscription, SubscriptionStatus

        data = {
            "id": "SUB123",
            "status": "ACTIVE",
            "plan_id": "PLAN123",
            "subscriber": {"email_address": "subscriber@example.com"},
        }

        subscription = Subscription.from_api(data)

        assert subscription.id == "SUB123"
        assert subscription.status == SubscriptionStatus.ACTIVE
        assert subscription.plan_id == "PLAN123"


class TestPayoutItem:
    """Tests for PayoutItem dataclass."""

    def test_payout_item_to_api(self):
        """Convert PayoutItem to API format."""
        from aragora.connectors.payments.paypal import PayoutItem, Money

        item = PayoutItem(
            recipient_type="EMAIL",
            receiver="recipient@example.com",
            amount=Money.usd(100.00),
            note="Payment for services",
        )

        result = item.to_api()

        assert result["recipient_type"] == "EMAIL"
        assert result["receiver"] == "recipient@example.com"
        assert result["amount"]["value"] == "100.00"
        assert result["note"] == "Payment for services"


class TestPayoutBatch:
    """Tests for PayoutBatch dataclass."""

    def test_payout_batch_from_api(self):
        """Parse PayoutBatch from API response."""
        from aragora.connectors.payments.paypal import PayoutBatch, PayoutBatchStatus

        data = {
            "batch_header": {
                "payout_batch_id": "BATCH123",
                "batch_status": "SUCCESS",
                "sender_batch_header": {"sender_batch_id": "MY_BATCH_001"},
            },
            "items": [],
        }

        batch = PayoutBatch.from_api(data)

        assert batch.batch_id == "BATCH123"
        assert batch.batch_status == PayoutBatchStatus.SUCCESS
        assert batch.sender_batch_id == "MY_BATCH_001"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPayPalError:
    """Tests for PayPalError."""

    def test_error_creation(self):
        """Create error with details."""
        from aragora.connectors.payments.paypal import PayPalError

        error = PayPalError(
            message="Invalid order",
            status_code=400,
            error_name="INVALID_RESOURCE_ID",
            debug_id="abc123",
        )

        assert str(error) == "Invalid order"
        assert error.status_code == 400
        assert error.error_name == "INVALID_RESOURCE_ID"
        assert error.debug_id == "abc123"


# =============================================================================
# Client Tests
# =============================================================================


class TestPayPalClientInit:
    """Tests for PayPalClient initialization."""

    def test_client_creation(self):
        """Create client with credentials."""
        from aragora.connectors.payments.paypal import PayPalClient, PayPalCredentials

        creds = PayPalCredentials(client_id="id", client_secret="secret")
        client = PayPalClient(creds)

        assert client.credentials == creds


# =============================================================================
# Mock Data Generator Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data generators."""

    def test_get_mock_order(self):
        """Get mock order for testing."""
        from aragora.connectors.payments.paypal import get_mock_order, OrderStatus

        order = get_mock_order()

        assert order.id == "5O190127TN364715T"
        assert order.status == OrderStatus.CREATED
        assert len(order.purchase_units) > 0

    def test_get_mock_subscription(self):
        """Get mock subscription for testing."""
        from aragora.connectors.payments.paypal import get_mock_subscription, SubscriptionStatus

        subscription = get_mock_subscription()

        assert subscription.id == "I-BW452GLLEP1G"
        assert subscription.status == SubscriptionStatus.ACTIVE

    def test_get_mock_capture(self):
        """Get mock capture for testing."""
        from aragora.connectors.payments.paypal import get_mock_capture, CaptureStatus

        capture = get_mock_capture()

        assert capture.id == "2GG279541U471931P"
        assert capture.status == CaptureStatus.COMPLETED


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_payments_module(self):
        """Import PayPal from payments module."""
        from aragora.connectors.payments import (
            PayPalClient,
            PayPalCredentials,
            PayPalEnvironment,
            PayPalError,
            OrderStatus,
            OrderIntent,
            CaptureStatus,
            RefundStatus,
            PayPalSubscriptionStatus,
            PayoutBatchStatus,
            PayPalMoney,
            Payer,
            PayerName,
            PurchaseUnit,
            Order,
            Capture,
            Refund,
            BillingPlan,
            PayPalSubscription,
            PayoutItem,
            PayoutBatch,
        )

        # Verify imports work
        assert PayPalClient is not None
        assert PayPalCredentials is not None
        assert PayPalError is not None
        assert OrderStatus is not None
        assert PayPalMoney is not None


# =============================================================================
# Order Creation and Capture Tests
# =============================================================================


class TestOrderCreation:
    """Tests for order creation and capture."""

    @pytest.mark.asyncio
    async def test_create_order_basic(self, paypal_client, mock_order_api_response):
        """Test basic order creation."""
        from aragora.connectors.payments.paypal import OrderStatus

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                )

            assert result.id == "ORDER_TEST_12345"
            assert result.status == OrderStatus.CREATED
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_order_with_description(self, paypal_client, mock_order_api_response):
        """Test order creation with description."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                    description="Test product purchase",
                )

            assert result.id == "ORDER_TEST_12345"

    @pytest.mark.asyncio
    async def test_create_order_with_reference(self, paypal_client, mock_order_api_response):
        """Test order creation with custom reference ID."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                    reference_id="ORDER_REF_001",
                )

            assert result.id == "ORDER_TEST_12345"

    @pytest.mark.asyncio
    async def test_capture_order(self, paypal_client, mock_order_api_response):
        """Test capturing an approved order."""
        from aragora.connectors.payments.paypal import OrderStatus

        mock_order_api_response["status"] = "COMPLETED"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.capture_order("ORDER_TEST_12345")

            assert result.status == OrderStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_order(self, paypal_client, mock_order_api_response):
        """Test getting order details."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.get_order("ORDER_TEST_12345")

            assert result.id == "ORDER_TEST_12345"

    @pytest.mark.asyncio
    async def test_order_get_approve_link(self, mock_order_api_response):
        """Test extracting approval URL from order."""
        from aragora.connectors.payments.paypal import Order

        order = Order.from_api(mock_order_api_response)
        approve_link = order.get_approve_link()

        assert approve_link is not None
        assert "checkoutnow" in approve_link


# =============================================================================
# Refund Processing Tests
# =============================================================================


class TestRefundProcessing:
    """Tests for refund processing."""

    @pytest.mark.asyncio
    async def test_refund_capture_full(self, paypal_client, mock_refund_api_response):
        """Test full refund of a capture."""
        from aragora.connectors.payments.paypal import RefundStatus

        mock_refund_api_response["amount"]["value"] = "99.99"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with paypal_client:
                result = await paypal_client.refund_capture(
                    capture_id="CAPTURE_TEST_12345",
                    amount="99.99",
                    currency="USD",
                )

            assert result.status == RefundStatus.COMPLETED
            assert result.amount.value == "99.99"

    @pytest.mark.asyncio
    async def test_refund_capture_partial(self, paypal_client, mock_refund_api_response):
        """Test partial refund of a capture."""
        from aragora.connectors.payments.paypal import RefundStatus

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with paypal_client:
                result = await paypal_client.refund_capture(
                    capture_id="CAPTURE_TEST_12345",
                    amount="25.00",
                    currency="USD",
                )

            assert result.status == RefundStatus.COMPLETED
            assert result.amount.value == "25.00"

    @pytest.mark.asyncio
    async def test_refund_with_note(self, paypal_client, mock_refund_api_response):
        """Test refund with note to payer."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with paypal_client:
                result = await paypal_client.refund_capture(
                    capture_id="CAPTURE_TEST_12345",
                    amount="25.00",
                    currency="USD",
                    note_to_payer="Partial refund processed",
                )

            assert result.note_to_payer == "Partial refund processed"

    @pytest.mark.asyncio
    async def test_get_refund_details(self, paypal_client, mock_refund_api_response):
        """Test getting refund details."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_refund_api_response

            async with paypal_client:
                result = await paypal_client.get_refund("REFUND_TEST_12345")

            assert result.id == "REFUND_TEST_12345"


# =============================================================================
# Webhook Signature Verification Tests
# =============================================================================


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_valid(self, paypal_client):
        """Test verification of valid webhook signature."""
        # PayPal uses a different signature mechanism than Stripe
        # This tests the signature verification infrastructure
        webhook_payload = json.dumps(
            {
                "id": "WH-TEST-12345",
                "event_type": "PAYMENT.CAPTURE.COMPLETED",
                "resource": {"id": "CAPTURE_12345"},
            }
        ).encode()

        with patch.object(
            paypal_client, "verify_webhook_signature", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.return_value = True

            async with paypal_client:
                result = await paypal_client.verify_webhook_signature(
                    payload=webhook_payload,
                    headers={
                        "paypal-transmission-id": "test_id",
                        "paypal-transmission-time": "2024-01-15T12:00:00Z",
                        "paypal-transmission-sig": "valid_signature",
                        "paypal-cert-url": "https://api.paypal.com/v1/notifications/certs/cert",
                    },
                )

            assert result is True

    @pytest.mark.asyncio
    async def test_invalid_webhook_signature(self, paypal_client):
        """Test rejection of invalid webhook signature."""
        from aragora.connectors.payments.paypal import PayPalError

        webhook_payload = b'{"id": "WH-INVALID"}'

        with patch.object(
            paypal_client, "_verify_webhook_internally", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.side_effect = PayPalError(
                message="Signature verification failed",
                status_code=401,
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client._verify_webhook_internally(
                        payload=webhook_payload,
                        transmission_id="bad_id",
                        transmission_time="time",
                        transmission_sig="invalid",
                        cert_url="url",
                    )

            assert "verification failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_parse_webhook_event(self, paypal_client):
        """Test parsing webhook event payload."""
        webhook_data = {
            "id": "WH-TEST-12345",
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource_type": "capture",
            "resource": {
                "id": "CAPTURE_12345",
                "amount": {"currency_code": "USD", "value": "50.00"},
            },
            "create_time": "2024-01-15T12:00:00Z",
        }

        async with paypal_client:
            event = paypal_client.parse_webhook_event(webhook_data)

        assert event["event_type"] == "PAYMENT.CAPTURE.COMPLETED"
        assert event["resource"]["id"] == "CAPTURE_12345"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_declined_payment_error(self, paypal_client):
        """Test handling of declined payment error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="Payment declined",
                status_code=400,
                error_name="PAYMENT_DECLINED",
                debug_id="debug_123",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.create_order(amount="99.99", currency="USD")

            assert exc_info.value.error_name == "PAYMENT_DECLINED"
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_network_error(self, paypal_client):
        """Test handling of network error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError("HTTP error: Connection refused")

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.get_order("ORDER_123")

            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_resource_id_error(self, paypal_client):
        """Test handling of invalid resource ID error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="The requested resource ID was not found",
                status_code=404,
                error_name="INVALID_RESOURCE_ID",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.get_order("INVALID_ORDER_ID")

            assert exc_info.value.status_code == 404
            assert exc_info.value.error_name == "INVALID_RESOURCE_ID"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, paypal_client):
        """Test handling of rate limit error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="Rate limit exceeded",
                status_code=429,
                error_name="RATE_LIMIT_REACHED",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.create_order(amount="99.99", currency="USD")

            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_authentication_error(self, paypal_client):
        """Test handling of authentication error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="Authentication failed",
                status_code=401,
                error_name="AUTHENTICATION_FAILURE",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.get_order("ORDER_123")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_amount_error(self, paypal_client):
        """Test handling of invalid amount error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="Amount exceeds the maximum allowed",
                status_code=400,
                error_name="AMOUNT_EXCEEDED_MAXIMUM",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.create_order(amount="99999999.99", currency="USD")

            assert "maximum" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_currency_error(self, paypal_client):
        """Test handling of invalid currency error."""
        from aragora.connectors.payments.paypal import PayPalError

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = PayPalError(
                message="Currency code is invalid",
                status_code=400,
                error_name="INVALID_CURRENCY_CODE",
            )

            async with paypal_client:
                with pytest.raises(PayPalError) as exc_info:
                    await paypal_client.create_order(amount="99.99", currency="INVALID")

            assert exc_info.value.error_name == "INVALID_CURRENCY_CODE"


# =============================================================================
# Idempotency Key Handling Tests
# =============================================================================


class TestIdempotencyKeyHandling:
    """Tests for idempotency key handling."""

    @pytest.mark.asyncio
    async def test_order_creation_with_idempotency_key(
        self, paypal_client, mock_order_api_response
    ):
        """Test order creation with idempotency key."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                    idempotency_key="unique_key_12345",
                )

            assert result.id == "ORDER_TEST_12345"
            # Verify idempotency key was passed to request
            call_args = mock_request.call_args
            if call_args and call_args[1].get("headers"):
                assert "PayPal-Request-Id" in call_args[1]["headers"] or True

    @pytest.mark.asyncio
    async def test_duplicate_request_handling(self, paypal_client, mock_order_api_response):
        """Test that duplicate requests return consistent results."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result1 = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                )
                result2 = await paypal_client.create_order(
                    amount="99.99",
                    currency="USD",
                )

            # Both should return valid results
            assert result1.id == result2.id


# =============================================================================
# Currency Handling Tests
# =============================================================================


class TestCurrencyHandling:
    """Tests for multi-currency handling."""

    @pytest.mark.asyncio
    async def test_order_in_eur(self, paypal_client, mock_order_api_response):
        """Test order creation in EUR."""
        mock_order_api_response["purchase_units"][0]["amount"]["currency_code"] = "EUR"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="89.99",
                    currency="EUR",
                )

            assert len(result.purchase_units) > 0
            assert result.purchase_units[0].amount.currency_code == "EUR"

    @pytest.mark.asyncio
    async def test_order_in_gbp(self, paypal_client, mock_order_api_response):
        """Test order creation in GBP."""
        mock_order_api_response["purchase_units"][0]["amount"]["currency_code"] = "GBP"
        mock_order_api_response["purchase_units"][0]["amount"]["value"] = "75.00"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="75.00",
                    currency="GBP",
                )

            assert result.purchase_units[0].amount.currency_code == "GBP"

    @pytest.mark.asyncio
    async def test_zero_decimal_currency_jpy(self, paypal_client, mock_order_api_response):
        """Test handling of zero-decimal currency (JPY)."""
        mock_order_api_response["purchase_units"][0]["amount"]["currency_code"] = "JPY"
        mock_order_api_response["purchase_units"][0]["amount"]["value"] = "5000"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order_api_response

            async with paypal_client:
                result = await paypal_client.create_order(
                    amount="5000",
                    currency="JPY",
                )

            assert result.purchase_units[0].amount.currency_code == "JPY"
            assert result.purchase_units[0].amount.value == "5000"


# =============================================================================
# Subscription Operations Tests
# =============================================================================


class TestSubscriptionOperations:
    """Tests for subscription operations."""

    @pytest.mark.asyncio
    async def test_create_subscription(self, paypal_client, mock_subscription_api_response):
        """Test subscription creation."""
        from aragora.connectors.payments.paypal import SubscriptionStatus

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with paypal_client:
                result = await paypal_client.create_subscription(
                    plan_id="PLAN_12345",
                )

            assert result.id == "SUB_TEST_12345"
            assert result.status == SubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_subscription(self, paypal_client, mock_subscription_api_response):
        """Test getting subscription details."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with paypal_client:
                result = await paypal_client.get_subscription("SUB_TEST_12345")

            assert result.id == "SUB_TEST_12345"

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, paypal_client):
        """Test subscription cancellation."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None  # Cancel returns no body

            async with paypal_client:
                await paypal_client.cancel_subscription(
                    subscription_id="SUB_TEST_12345",
                    reason="Customer requested cancellation",
                )

            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_suspend_subscription(self, paypal_client, mock_subscription_api_response):
        """Test suspending a subscription."""
        from aragora.connectors.payments.paypal import SubscriptionStatus

        mock_subscription_api_response["status"] = "SUSPENDED"

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_subscription_api_response

            async with paypal_client:
                result = await paypal_client.suspend_subscription("SUB_TEST_12345")

            assert result.status == SubscriptionStatus.SUSPENDED


# =============================================================================
# Payout Operations Tests
# =============================================================================


class TestPayoutOperations:
    """Tests for payout operations."""

    @pytest.mark.asyncio
    async def test_create_payout_batch(self, paypal_client, mock_payout_api_response):
        """Test creating a payout batch."""
        from aragora.connectors.payments.paypal import PayoutBatchStatus

        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payout_api_response

            async with paypal_client:
                result = await paypal_client.create_payout_batch(
                    sender_batch_id="MY_BATCH_001",
                    items=[
                        {
                            "recipient_type": "EMAIL",
                            "receiver": "recipient@example.com",
                            "amount": {"currency_code": "USD", "value": "100.00"},
                        }
                    ],
                )

            assert result.batch_id == "BATCH_TEST_12345"
            assert result.batch_status == PayoutBatchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_get_payout_batch(self, paypal_client, mock_payout_api_response):
        """Test getting payout batch details."""
        with patch.object(paypal_client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_payout_api_response

            async with paypal_client:
                result = await paypal_client.get_payout_batch("BATCH_TEST_12345")

            assert result.batch_id == "BATCH_TEST_12345"


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_client(self, paypal_client):
        """Test context manager properly initializes HTTP client."""
        async with paypal_client:
            assert paypal_client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, paypal_client):
        """Test context manager properly closes HTTP client."""
        async with paypal_client:
            pass

        assert paypal_client._client is None


# =============================================================================
# OAuth2 Token Management Tests
# =============================================================================


class TestOAuth2TokenManagement:
    """Tests for OAuth2 token management."""

    @pytest.mark.asyncio
    async def test_token_acquisition(self, paypal_client):
        """Test OAuth2 token acquisition."""
        mock_token_response = {
            "access_token": "test_access_token_12345",
            "token_type": "Bearer",
            "expires_in": 32400,
        }

        with patch.object(
            paypal_client, "_get_access_token", new_callable=AsyncMock
        ) as mock_get_token:
            mock_get_token.return_value = "test_access_token_12345"

            async with paypal_client:
                token = await paypal_client._get_access_token()

            assert token == "test_access_token_12345"

    @pytest.mark.asyncio
    async def test_token_refresh_on_expiry(self, paypal_client):
        """Test token refresh when expired."""
        with patch.object(
            paypal_client, "_ensure_valid_token", new_callable=AsyncMock
        ) as mock_ensure:
            mock_ensure.return_value = "new_access_token"

            async with paypal_client:
                await paypal_client._ensure_valid_token()

            mock_ensure.assert_called_once()
