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
from datetime import datetime, timedelta
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
        now = datetime.utcnow()
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
        now = datetime.utcnow()
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
        now = datetime.utcnow()
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
        now = datetime.utcnow()
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
        result = client._encode_form_data({
            "metadata": {"user_id": "123", "org_id": "456"},
        })

        assert "metadata[user_id]=123" in result
        assert "metadata[org_id]=456" in result

    def test_encode_list_data(self):
        """Test encoding list values."""
        client = StripeClient(api_key="sk_test_abc123")
        result = client._encode_form_data({
            "items": [{"price": "price_123"}, {"price": "price_456"}],
        })

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
        mock_response.read.return_value = json.dumps({
            "id": "cus_test123",
            "email": "test@example.com",
            "name": "Test User",
            "metadata": {},
        }).encode()
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

        error_response = json.dumps({
            "error": {
                "message": "Invalid API Key",
                "code": "api_key_invalid",
            }
        }).encode()

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

        error_response = json.dumps({
            "error": {
                "message": "No such customer",
                "code": "resource_missing",
            }
        }).encode()

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
