"""
Tests for PayPal Payment Connector.

Tests cover:
- Client initialization
- OAuth2 authentication
- Orders API
- Captures and refunds
- Subscriptions
- Payouts
- Error handling
- Mock data generators
"""

from datetime import datetime, timezone

import pytest


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
