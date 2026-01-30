"""
Tests for Stripe billing integration.

Tests cover:
- Data class serialization
- Form data encoding
- Webhook signature verification
- Error handling
- Client configuration
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.stripe_client import (
    BillingPortalSession,
    CheckoutSession,
    StripeAPIError,
    StripeClient,
    StripeConfigError,
    StripeCustomer,
    StripeSubscription,
    UsageRecord,
    WebhookEvent,
    parse_webhook_event,
    verify_webhook_signature,
)


class TestStripeDataClasses:
    """Tests for Stripe data class serialization."""

    def test_stripe_customer_to_dict(self):
        """Test StripeCustomer serialization."""
        customer = StripeCustomer(
            id="cus_test123",
            email="test@example.com",
            name="Test User",
            metadata={"user_id": "123"},
        )
        result = customer.to_dict()

        assert result["id"] == "cus_test123"
        assert result["email"] == "test@example.com"
        assert result["name"] == "Test User"
        assert result["metadata"]["user_id"] == "123"

    def test_stripe_customer_empty_metadata(self):
        """Test StripeCustomer with no metadata."""
        customer = StripeCustomer(
            id="cus_test123",
            email="test@example.com",
        )
        result = customer.to_dict()

        assert result["metadata"] == {}

    def test_stripe_subscription_to_dict(self):
        """Test StripeSubscription serialization."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=30)

        sub = StripeSubscription(
            id="sub_test123",
            customer_id="cus_test123",
            status="active",
            price_id="price_test123",
            current_period_start=now,
            current_period_end=end,
        )
        result = sub.to_dict()

        assert result["id"] == "sub_test123"
        assert result["customer_id"] == "cus_test123"
        assert result["status"] == "active"
        assert result["price_id"] == "price_test123"
        assert result["cancel_at_period_end"] is False
        assert result["is_trialing"] is False

    def test_stripe_subscription_trialing(self):
        """Test StripeSubscription trialing detection."""
        now = datetime.now(timezone.utc)
        trial_end = now + timedelta(days=7)

        sub = StripeSubscription(
            id="sub_test123",
            customer_id="cus_test123",
            status="trialing",
            price_id="price_test123",
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            trial_start=now,
            trial_end=trial_end,
        )

        assert sub.is_trialing is True

    def test_stripe_subscription_trial_ended(self):
        """Test StripeSubscription trialing detection when trial ended."""
        now = datetime.now(timezone.utc)
        trial_end = now - timedelta(days=1)  # Trial ended yesterday

        sub = StripeSubscription(
            id="sub_test123",
            customer_id="cus_test123",
            status="trialing",  # Status not yet updated
            price_id="price_test123",
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            trial_start=now - timedelta(days=8),
            trial_end=trial_end,
        )

        assert sub.is_trialing is False

    def test_checkout_session_to_dict(self):
        """Test CheckoutSession serialization."""
        session = CheckoutSession(
            id="cs_test123",
            url="https://checkout.stripe.com/pay/cs_test123",
            customer_id="cus_test123",
            subscription_id="sub_test123",
        )
        result = session.to_dict()

        assert result["id"] == "cs_test123"
        assert result["url"] == "https://checkout.stripe.com/pay/cs_test123"
        assert result["customer_id"] == "cus_test123"
        assert result["subscription_id"] == "sub_test123"

    def test_billing_portal_session_to_dict(self):
        """Test BillingPortalSession serialization."""
        session = BillingPortalSession(
            id="bps_test123",
            url="https://billing.stripe.com/session/bps_test123",
        )
        result = session.to_dict()

        assert result["id"] == "bps_test123"
        assert result["url"] == "https://billing.stripe.com/session/bps_test123"

    def test_usage_record_to_dict(self):
        """Test UsageRecord serialization."""
        now = datetime.now(timezone.utc)
        record = UsageRecord(
            id="mbur_test123",
            subscription_item_id="si_test123",
            quantity=100,
            timestamp=now,
            action="increment",
        )
        result = record.to_dict()

        assert result["id"] == "mbur_test123"
        assert result["subscription_item_id"] == "si_test123"
        assert result["quantity"] == 100
        assert result["action"] == "increment"


class TestStripeClientConfiguration:
    """Tests for StripeClient configuration."""

    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        client = StripeClient(api_key="sk_test_abc123")
        assert client._is_configured() is True

    def test_client_without_api_key(self):
        """Test client initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("aragora.billing.stripe_client.STRIPE_SECRET_KEY", ""):
                client = StripeClient()
                assert client._is_configured() is False

    def test_unconfigured_client_raises_error(self):
        """Test that unconfigured client raises StripeConfigError."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("aragora.billing.stripe_client.STRIPE_SECRET_KEY", ""):
                client = StripeClient()
                with pytest.raises(StripeConfigError, match="not configured"):
                    client._request("GET", "/customers")


class TestStripeFormEncoding:
    """Tests for Stripe form data encoding."""

    def test_encode_simple_data(self):
        """Test encoding simple key-value pairs (URL-encoded)."""
        client = StripeClient(api_key="sk_test_abc123")
        result = client._encode_form_data({"email": "test@example.com", "name": "Test"})

        # @ is URL-encoded to %40 for form data safety
        assert "email=test%40example.com" in result
        assert "name=Test" in result

    def test_encode_nested_data(self):
        """Test encoding nested objects."""
        client = StripeClient(api_key="sk_test_abc123")
        result = client._encode_form_data(
            {
                "metadata": {"user_id": "123", "org_id": "456"},
            }
        )

        assert "metadata[user_id]=123" in result
        assert "metadata[org_id]=456" in result

    def test_encode_list_data(self):
        """Test encoding list values."""
        client = StripeClient(api_key="sk_test_abc123")
        result = client._encode_form_data(
            {
                "items": [{"price": "price_123"}, {"price": "price_456"}],
            }
        )

        assert "items[0][price]=price_123" in result
        assert "items[1][price]=price_456" in result

    def test_encode_none_values_skipped(self):
        """Test that None values are skipped."""
        client = StripeClient(api_key="sk_test_abc123")
        result = client._encode_form_data({"email": "test@example.com", "name": None})

        # @ is URL-encoded to %40 for form data safety
        assert "email=test%40example.com" in result
        assert "name" not in result


class TestWebhookSignatureVerification:
    """Tests for Stripe webhook signature verification."""

    def _create_signature(self, payload: bytes, secret: str, timestamp: int) -> str:
        """Create a valid Stripe webhook signature."""
        signed_payload = f"{timestamp}.".encode() + payload
        sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
        return f"t={timestamp},v1={sig}"

    def test_valid_signature(self):
        """Test verification of valid signature."""
        secret = "whsec_test_secret"
        payload = b'{"type": "checkout.session.completed"}'
        timestamp = int(time.time())
        signature = self._create_signature(payload, secret, timestamp)

        result = verify_webhook_signature(payload, signature, secret)
        assert result is True

    def test_invalid_signature(self):
        """Test verification of invalid signature."""
        secret = "whsec_test_secret"
        payload = b'{"type": "checkout.session.completed"}'
        signature = "t=123,v1=invalid_signature"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_expired_timestamp(self):
        """Test rejection of expired timestamp."""
        secret = "whsec_test_secret"
        payload = b'{"type": "checkout.session.completed"}'
        old_timestamp = int(time.time()) - 600  # 10 minutes ago
        signature = self._create_signature(payload, secret, old_timestamp)

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_missing_secret(self):
        """Test verification without secret configured."""
        payload = b'{"type": "checkout.session.completed"}'
        signature = "t=123,v1=sig"

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", ""):
            result = verify_webhook_signature(payload, signature, None)
            assert result is False

    def test_malformed_signature_header(self):
        """Test verification with malformed signature header."""
        secret = "whsec_test_secret"
        payload = b'{"type": "checkout.session.completed"}'

        # Missing timestamp
        result = verify_webhook_signature(payload, "v1=sig", secret)
        assert result is False

        # Missing signature
        result = verify_webhook_signature(payload, "t=123", secret)
        assert result is False

        # Invalid format
        result = verify_webhook_signature(payload, "invalid", secret)
        assert result is False


class TestWebhookEventParsing:
    """Tests for webhook event parsing."""

    def _create_signature(self, payload: bytes, secret: str) -> str:
        """Create a valid signature for the payload."""
        timestamp = int(time.time())
        signed_payload = f"{timestamp}.".encode() + payload
        sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
        return f"t={timestamp},v1={sig}"

    def test_parse_valid_event(self):
        """Test parsing a valid webhook event."""
        secret = "whsec_test_secret"
        event_data = {
            "id": "evt_test123",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test123",
                    "customer": "cus_test123",
                }
            },
        }
        payload = json.dumps(event_data).encode()
        signature = self._create_signature(payload, secret)

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "checkout.session.completed"
        assert event.event_id == "evt_test123"

    def test_parse_invalid_signature(self):
        """Test parsing with invalid signature returns None."""
        secret = "whsec_test_secret"
        payload = b'{"type": "checkout.session.completed"}'
        signature = "t=123,v1=invalid"

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is None


class TestWebhookEvent:
    """Tests for WebhookEvent class."""

    def test_webhook_event_properties(self):
        """Test WebhookEvent property accessors."""
        event = WebhookEvent(
            event_type="checkout.session.completed",
            data={
                "object": {
                    "id": "cs_test123",
                    "customer": "cus_test123",
                    "subscription": "sub_test123",
                    "metadata": {"user_id": "123"},
                }
            },
            event_id="evt_test123",
        )

        assert event.object["id"] == "cs_test123"
        assert event.customer_id == "cus_test123"
        assert event.subscription_id == "sub_test123"
        assert event.metadata["user_id"] == "123"

    def test_webhook_event_missing_properties(self):
        """Test WebhookEvent with missing properties."""
        event = WebhookEvent(
            event_type="customer.created",
            data={"object": {"id": "cus_test123"}},
            event_id="evt_test123",
        )

        # customer_id falls back to object.id if customer is not set
        assert event.customer_id == "cus_test123"
        assert event.subscription_id is None
        assert event.metadata == {}


class TestStripeErrors:
    """Tests for Stripe error classes."""

    def test_stripe_error_attributes(self):
        """Test StripeError attributes."""
        error = StripeAPIError("Payment failed", "card_declined", 402)

        assert str(error) == "Payment failed"
        assert error.code == "card_declined"
        assert error.status == 402

    def test_stripe_config_error(self):
        """Test StripeConfigError."""
        error = StripeConfigError("API key not set")
        assert str(error) == "API key not set"


class TestStripeClientMockedRequests:
    """Tests for StripeClient with mocked HTTP requests."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_create_customer_success(self, mock_urlopen):
        """Test successful customer creation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cus_test123",
                "email": "test@example.com",
                "name": "Test User",
                "metadata": {},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        customer = client.create_customer(
            email="test@example.com",
            name="Test User",
        )

        assert customer.id == "cus_test123"
        assert customer.email == "test@example.com"
        assert customer.name == "Test User"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_api_error_handling(self, mock_urlopen):
        """Test API error handling for non-404 errors."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {
                "error": {
                    "message": "Invalid API Key",
                    "code": "api_key_invalid",
                }
            }
        ).encode()

        # HTTPError.read() returns bytes directly from fp
        http_error = HTTPError(
            url="https://api.stripe.com/v1/customers",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")

        # create_customer should raise for 401 errors
        with pytest.raises(StripeAPIError) as exc_info:
            client.create_customer(email="test@example.com")

        assert "Invalid API Key" in str(exc_info.value)
        assert exc_info.value.code == "api_key_invalid"
        assert exc_info.value.status == 401

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_customer_returns_none_for_404(self, mock_urlopen):
        """Test that get_customer returns None for 404 errors."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {
                "error": {
                    "message": "No such customer",
                    "code": "resource_missing",
                }
            }
        ).encode()

        http_error = HTTPError(
            url="https://api.stripe.com/v1/customers/invalid",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")
        result = client.get_customer("cus_invalid")

        assert result is None

    @patch("aragora.billing.stripe_client.urlopen")
    def test_update_customer_success(self, mock_urlopen):
        """Test successful customer update."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cus_test123",
                "email": "updated@example.com",
                "name": "Updated Name",
                "metadata": {"user_id": "123"},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        customer = client.update_customer(
            customer_id="cus_test123",
            email="updated@example.com",
            name="Updated Name",
            metadata={"user_id": "123"},
        )

        assert customer.id == "cus_test123"
        assert customer.email == "updated@example.com"
        assert customer.name == "Updated Name"
        assert customer.metadata == {"user_id": "123"}


class TestSubscriptionManagement:
    """Tests for subscription creation, updates, and cancellation."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_subscription_success(self, mock_urlopen):
        """Test successful subscription retrieval."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = now_ts + (30 * 24 * 60 * 60)  # 30 days later

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_test123",
                "customer": "cus_test123",
                "status": "active",
                "current_period_start": now_ts,
                "current_period_end": end_ts,
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_test123"}}]},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        subscription = client.get_subscription("sub_test123")

        assert subscription is not None
        assert subscription.id == "sub_test123"
        assert subscription.customer_id == "cus_test123"
        assert subscription.status == "active"
        assert subscription.price_id == "price_test123"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_subscription_not_found(self, mock_urlopen):
        """Test subscription retrieval returns None for 404."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {"error": {"message": "No such subscription", "code": "resource_missing"}}
        ).encode()

        http_error = HTTPError(
            url="https://api.stripe.com/v1/subscriptions/invalid",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")
        result = client.get_subscription("sub_invalid")

        assert result is None

    @patch("aragora.billing.stripe_client.urlopen")
    def test_cancel_subscription_at_period_end(self, mock_urlopen):
        """Test subscription cancellation at period end."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = now_ts + (30 * 24 * 60 * 60)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_test123",
                "customer": "cus_test123",
                "status": "active",
                "current_period_start": now_ts,
                "current_period_end": end_ts,
                "cancel_at_period_end": True,
                "items": {"data": [{"price": {"id": "price_test123"}}]},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        subscription = client.cancel_subscription("sub_test123", at_period_end=True)

        assert subscription.cancel_at_period_end is True

    @patch("aragora.billing.stripe_client.urlopen")
    def test_cancel_subscription_immediately(self, mock_urlopen):
        """Test immediate subscription cancellation."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = now_ts + (30 * 24 * 60 * 60)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_test123",
                "customer": "cus_test123",
                "status": "canceled",
                "current_period_start": now_ts,
                "current_period_end": end_ts,
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_test123"}}]},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        subscription = client.cancel_subscription("sub_test123", at_period_end=False)

        assert subscription.status == "canceled"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_resume_subscription(self, mock_urlopen):
        """Test resuming a subscription scheduled for cancellation."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = now_ts + (30 * 24 * 60 * 60)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_test123",
                "customer": "cus_test123",
                "status": "active",
                "current_period_start": now_ts,
                "current_period_end": end_ts,
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_test123"}}]},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        subscription = client.resume_subscription("sub_test123")

        assert subscription.cancel_at_period_end is False
        assert subscription.status == "active"


class TestInvoiceGeneration:
    """Tests for invoice listing and retrieval."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_list_invoices_success(self, mock_urlopen):
        """Test successful invoice listing."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {
                        "id": "in_test123",
                        "customer": "cus_test123",
                        "amount_due": 9900,
                        "status": "paid",
                    },
                    {
                        "id": "in_test456",
                        "customer": "cus_test123",
                        "amount_due": 9900,
                        "status": "open",
                    },
                ],
                "has_more": False,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        invoices = client.list_invoices("cus_test123", limit=10)

        assert len(invoices) == 2
        assert invoices[0]["id"] == "in_test123"
        assert invoices[1]["status"] == "open"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_list_invoices_pagination(self, mock_urlopen):
        """Test invoice listing with pagination cursor."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [{"id": "in_test789", "status": "paid"}],
                "has_more": False,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        invoices = client.list_invoices("cus_test123", limit=5, starting_after="in_test456")

        assert len(invoices) == 1
        # Verify the starting_after parameter was passed
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "starting_after=in_test456" in request.full_url


class TestWebhookEventHandling:
    """Tests for webhook event handling (payment succeeded, failed, subscription updated)."""

    def _create_signature(self, payload: bytes, secret: str) -> str:
        """Create a valid signature for the payload."""
        timestamp = int(time.time())
        signed_payload = f"{timestamp}.".encode() + payload
        sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
        return f"t={timestamp},v1={sig}"

    def test_payment_succeeded_event(self):
        """Test parsing payment_intent.succeeded event."""
        secret = "whsec_test_secret"
        event_data = {
            "id": "evt_payment_succeeded",
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test123",
                    "customer": "cus_test123",
                    "amount": 9900,
                    "currency": "usd",
                    "status": "succeeded",
                    "metadata": {"org_id": "org_test123"},
                }
            },
        }
        payload = json.dumps(event_data).encode()
        signature = self._create_signature(payload, secret)

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "payment_intent.succeeded"
        assert event.customer_id == "cus_test123"
        assert event.metadata.get("org_id") == "org_test123"

    def test_payment_failed_event(self):
        """Test parsing payment_intent.payment_failed event."""
        secret = "whsec_test_secret"
        event_data = {
            "id": "evt_payment_failed",
            "type": "payment_intent.payment_failed",
            "data": {
                "object": {
                    "id": "pi_test456",
                    "customer": "cus_test123",
                    "amount": 9900,
                    "status": "requires_payment_method",
                    "last_payment_error": {
                        "code": "card_declined",
                        "message": "Your card was declined.",
                    },
                }
            },
        }
        payload = json.dumps(event_data).encode()
        signature = self._create_signature(payload, secret)

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "payment_intent.payment_failed"
        assert event.object.get("last_payment_error", {}).get("code") == "card_declined"

    def test_subscription_updated_event(self):
        """Test parsing customer.subscription.updated event."""
        secret = "whsec_test_secret"
        now_ts = int(datetime.now(timezone.utc).timestamp())
        event_data = {
            "id": "evt_sub_updated",
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_test123",
                    "customer": "cus_test123",
                    "status": "active",
                    "current_period_end": now_ts + 86400 * 30,
                    "cancel_at_period_end": True,
                    "metadata": {"tier": "professional"},
                }
            },
        }
        payload = json.dumps(event_data).encode()
        signature = self._create_signature(payload, secret)

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "customer.subscription.updated"
        # For subscription events, subscription_id is the object id
        assert event.subscription_id == "sub_test123"
        assert event.metadata.get("tier") == "professional"

    def test_invoice_payment_succeeded_event(self):
        """Test parsing invoice.payment_succeeded event."""
        secret = "whsec_test_secret"
        event_data = {
            "id": "evt_invoice_paid",
            "type": "invoice.payment_succeeded",
            "data": {
                "object": {
                    "id": "in_test123",
                    "customer": "cus_test123",
                    "subscription": "sub_test123",
                    "amount_paid": 9900,
                    "status": "paid",
                }
            },
        }
        payload = json.dumps(event_data).encode()
        signature = self._create_signature(payload, secret)

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "invoice.payment_succeeded"
        assert event.subscription_id == "sub_test123"


class TestRefundProcessing:
    """Tests for refund operations (via API requests)."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_refund_via_stripe_api(self, mock_urlopen):
        """Test that client can make refund requests to Stripe API."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "re_test123",
                "amount": 5000,
                "charge": "ch_test123",
                "status": "succeeded",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        # Use _request directly since refund isn't a dedicated method
        result = client._request("POST", "/refunds", {"charge": "ch_test123", "amount": 5000})

        assert result["id"] == "re_test123"
        assert result["status"] == "succeeded"


class TestProrationHandling:
    """Tests for proration during subscription changes."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_subscription_with_proration_behavior(self, mock_urlopen):
        """Test subscription update respects proration settings."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = now_ts + (30 * 24 * 60 * 60)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_test123",
                "customer": "cus_test123",
                "status": "active",
                "current_period_start": now_ts,
                "current_period_end": end_ts,
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_professional"}}]},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        # Proration would be handled via subscription update with proration_behavior param
        result = client._request(
            "POST",
            "/subscriptions/sub_test123",
            {"proration_behavior": "create_prorations"},
        )

        assert result["id"] == "sub_test123"


class TestErrorHandling:
    """Tests for various error scenarios."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_card_declined_error(self, mock_urlopen):
        """Test handling of card_declined error."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {
                "error": {
                    "type": "card_error",
                    "code": "card_declined",
                    "message": "Your card was declined.",
                    "decline_code": "generic_decline",
                }
            }
        ).encode()

        http_error = HTTPError(
            url="https://api.stripe.com/v1/payment_intents",
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")

        with pytest.raises(StripeAPIError) as exc_info:
            client._request("POST", "/payment_intents", {"amount": 1000})

        assert exc_info.value.code == "card_declined"
        assert exc_info.value.status == 402
        assert "declined" in str(exc_info.value)

    @patch("aragora.billing.stripe_client.urlopen")
    def test_invalid_api_key_error(self, mock_urlopen):
        """Test handling of invalid API key error."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {
                "error": {
                    "type": "authentication_error",
                    "code": "api_key_invalid",
                    "message": "Invalid API Key provided: sk_test_***",
                }
            }
        ).encode()

        http_error = HTTPError(
            url="https://api.stripe.com/v1/customers",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_invalid")

        with pytest.raises(StripeAPIError) as exc_info:
            client.create_customer(email="test@example.com")

        assert exc_info.value.code == "api_key_invalid"
        assert exc_info.value.status == 401

    @patch("aragora.billing.stripe_client.urlopen")
    def test_network_error_handling(self, mock_urlopen):
        """Test handling of network connection errors."""
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")

        client = StripeClient(api_key="sk_test_abc123")

        with pytest.raises(StripeAPIError) as exc_info:
            client.create_customer(email="test@example.com")

        assert "Connection error" in str(exc_info.value)

    @patch("aragora.billing.stripe_client.urlopen")
    def test_rate_limit_error(self, mock_urlopen):
        """Test handling of rate limit (429) error."""
        from io import BytesIO
        from urllib.error import HTTPError

        error_response = json.dumps(
            {
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit",
                    "message": "Too many requests. Please slow down.",
                }
            }
        ).encode()

        http_error = HTTPError(
            url="https://api.stripe.com/v1/customers",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")

        with pytest.raises(StripeAPIError) as exc_info:
            client.create_customer(email="test@example.com")

        assert exc_info.value.status == 429

    @patch("aragora.billing.stripe_client.urlopen")
    def test_malformed_json_error_response(self, mock_urlopen):
        """Test handling of malformed JSON in error response."""
        from io import BytesIO
        from urllib.error import HTTPError

        # Non-JSON error response
        error_response = b"Internal Server Error"

        http_error = HTTPError(
            url="https://api.stripe.com/v1/customers",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(error_response),
        )
        mock_urlopen.side_effect = http_error

        client = StripeClient(api_key="sk_test_abc123")

        with pytest.raises(StripeAPIError) as exc_info:
            client.create_customer(email="test@example.com")

        assert exc_info.value.status == 500


class TestIdempotencyKeys:
    """Tests for idempotency key handling."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_idempotency_key_in_request_header(self, mock_urlopen):
        """Test that idempotency key is included in request headers."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"id": "cus_test123", "email": "test@example.com"}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        client._request(
            "POST",
            "/customers",
            {"email": "test@example.com"},
            idempotency_key="unique_key_123",
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Idempotency-key") == "unique_key_123"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_usage_report_with_idempotency_key(self, mock_urlopen):
        """Test that usage reporting supports idempotency keys."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "mbur_test123",
                "subscription_item": "si_test123",
                "quantity": 100,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        now = datetime.now(timezone.utc)
        record = client.report_usage(
            subscription_item_id="si_test123",
            quantity=100,
            timestamp=now,
            idempotency_key="usage_report_unique_123",
        )

        assert record.quantity == 100
        # Verify idempotency key was passed
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Idempotency-key") == "usage_report_unique_123"


class TestUsageMetering:
    """Tests for metered billing usage reporting."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_subscription_items(self, mock_urlopen):
        """Test retrieving subscription items for metered billing."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {
                        "id": "si_base_plan",
                        "price": {"id": "price_base"},
                        "quantity": 1,
                    },
                    {
                        "id": "si_tokens",
                        "price": {"id": "price_tokens_metered"},
                        "quantity": None,  # Metered items don't have quantity
                    },
                ],
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        items = client.get_subscription_items("sub_test123")

        assert len(items) == 2
        assert items[1]["price"]["id"] == "price_tokens_metered"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_find_metered_subscription_item(self, mock_urlopen):
        """Test finding specific subscription item by price ID."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {"id": "si_base", "price": {"id": "price_base"}},
                    {"id": "si_tokens", "price": {"id": "price_tokens_metered"}},
                ],
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        item_id = client.find_metered_subscription_item("sub_test123", "price_tokens_metered")

        assert item_id == "si_tokens"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_find_metered_subscription_item_not_found(self, mock_urlopen):
        """Test finding subscription item returns None when not found."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"data": [{"id": "si_base", "price": {"id": "price_base"}}]}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        item_id = client.find_metered_subscription_item("sub_test123", "price_nonexistent")

        assert item_id is None

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_usage_summary(self, mock_urlopen):
        """Test retrieving usage summary for a subscription item."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {
                        "id": "sis_test123",
                        "total_usage": 5000,
                        "period": {
                            "start": 1704067200,
                            "end": 1706745600,
                        },
                    }
                ],
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        summary = client.get_usage_summary("si_test123")

        assert summary["total_usage"] == 5000
        assert summary["period_start"] == 1704067200
        assert summary["period_end"] == 1706745600

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_usage_summary_empty(self, mock_urlopen):
        """Test usage summary returns defaults when no data."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        summary = client.get_usage_summary("si_test123")

        assert summary["total_usage"] == 0
        assert summary["period_start"] is None
        assert summary["period_end"] is None


class TestCheckoutSession:
    """Tests for checkout session creation."""

    @patch("aragora.billing.stripe_client.urlopen")
    @patch("aragora.billing.stripe_client.STRIPE_PRICES")
    def test_create_checkout_session_success(self, mock_prices, mock_urlopen):
        """Test successful checkout session creation."""
        from aragora.billing.models import SubscriptionTier

        mock_prices.get = MagicMock(return_value="price_starter_test")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cs_test123",
                "url": "https://checkout.stripe.com/pay/cs_test123",
                "customer": "cus_test123",
                "subscription": None,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        session = client.create_checkout_session(
            tier=SubscriptionTier.STARTER,
            customer_email="test@example.com",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )

        assert session.id == "cs_test123"
        assert "checkout.stripe.com" in session.url

    @patch("aragora.billing.stripe_client.STRIPE_PRICES", {})
    def test_create_checkout_session_missing_price(self):
        """Test checkout session fails when price not configured."""
        from aragora.billing.models import SubscriptionTier

        client = StripeClient(api_key="sk_test_abc123")

        with pytest.raises(StripeConfigError) as exc_info:
            client.create_checkout_session(
                tier=SubscriptionTier.STARTER,
                customer_email="test@example.com",
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
            )

        assert "No price configured" in str(exc_info.value)


class TestBillingPortal:
    """Tests for billing portal session creation."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_create_portal_session_success(self, mock_urlopen):
        """Test successful billing portal session creation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "bps_test123",
                "url": "https://billing.stripe.com/session/bps_test123",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_abc123")
        session = client.create_portal_session(
            customer_id="cus_test123",
            return_url="https://example.com/account",
        )

        assert session.id == "bps_test123"
        assert "billing.stripe.com" in session.url


class TestTierMapping:
    """Tests for tier-to-price mapping utilities."""

    def test_get_tier_from_price_id(self):
        """Test mapping price ID back to tier."""
        from aragora.billing.models import SubscriptionTier
        from aragora.billing.stripe_client import get_tier_from_price_id

        # Test with actual module function
        with patch.dict(
            "aragora.billing.stripe_client.STRIPE_PRICES",
            {
                SubscriptionTier.STARTER: "price_starter",
                SubscriptionTier.PROFESSIONAL: "price_professional",
            },
        ):
            tier = get_tier_from_price_id("price_starter")
            assert tier == SubscriptionTier.STARTER

            tier = get_tier_from_price_id("price_unknown")
            assert tier is None

    def test_get_price_id_for_tier(self):
        """Test mapping tier to price ID."""
        from aragora.billing.models import SubscriptionTier
        from aragora.billing.stripe_client import get_price_id_for_tier

        with patch.dict(
            "aragora.billing.stripe_client.STRIPE_PRICES",
            {
                SubscriptionTier.PROFESSIONAL: "price_professional_test",
            },
        ):
            price_id = get_price_id_for_tier(SubscriptionTier.PROFESSIONAL)
            assert price_id == "price_professional_test"
