"""
Tests for Stripe client integration.

Security-critical tests for webhook signature verification and payment processing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from aragora.billing.stripe_client import (
    BillingPortalSession,
    CheckoutSession,
    StripeAPIError,
    StripeClient,
    StripeConfigError,
    StripeCustomer,
    StripeSubscription,
    WebhookEvent,
    get_price_id_for_tier,
    get_stripe_client,
    get_tier_from_price_id,
    parse_webhook_event,
    verify_webhook_signature,
    STRIPE_PRICES,
)
from aragora.billing.models import SubscriptionTier


# =============================================================================
# StripeClient Tests
# =============================================================================


class TestStripeClientInit:
    """Tests for StripeClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = StripeClient(api_key="sk_test_123")
        assert client.api_key == "sk_test_123"
        assert client._is_configured() is True

    def test_init_without_api_key(self):
        """Test initialization without API key (uses env var)."""
        with patch.dict("os.environ", {"STRIPE_SECRET_KEY": ""}, clear=False):
            with patch("aragora.billing.stripe_client.STRIPE_SECRET_KEY", ""):
                client = StripeClient()
                # Falls back to empty string if no env var
                assert client._is_configured() is False

    def test_init_with_env_var(self):
        """Test initialization uses STRIPE_SECRET_KEY env var."""
        with patch("aragora.billing.stripe_client.STRIPE_SECRET_KEY", "sk_test_from_env"):
            client = StripeClient()
            assert client.api_key == "sk_test_from_env"

    def test_is_configured_true(self):
        """Test _is_configured returns True when API key is set."""
        client = StripeClient(api_key="sk_test_abc")
        assert client._is_configured() is True

    def test_is_configured_false(self):
        """Test _is_configured returns False when API key is empty."""
        client = StripeClient(api_key="")
        assert client._is_configured() is False


class TestStripeClientFormEncoding:
    """Tests for form data encoding."""

    def test_encode_simple_data(self):
        """Test encoding simple flat data."""
        client = StripeClient(api_key="sk_test")
        result = client._encode_form_data({"email": "test@example.com", "name": "Test"})
        assert "email=test@example.com" in result
        assert "name=Test" in result

    def test_encode_nested_data(self):
        """Test encoding nested metadata."""
        client = StripeClient(api_key="sk_test")
        result = client._encode_form_data({"metadata": {"user_id": "123", "org_id": "456"}})
        assert "metadata[user_id]=123" in result
        assert "metadata[org_id]=456" in result

    def test_encode_list_data(self):
        """Test encoding list items."""
        client = StripeClient(api_key="sk_test")
        result = client._encode_form_data({"items": ["a", "b", "c"]})
        assert "items[0]=a" in result
        assert "items[1]=b" in result
        assert "items[2]=c" in result

    def test_encode_list_of_dicts(self):
        """Test encoding list of dictionaries (line_items format)."""
        client = StripeClient(api_key="sk_test")
        result = client._encode_form_data({"line_items": [{"price": "price_123", "quantity": 1}]})
        assert "line_items[0][price]=price_123" in result
        assert "line_items[0][quantity]=1" in result

    def test_encode_none_values_skipped(self):
        """Test that None values are skipped."""
        client = StripeClient(api_key="sk_test")
        result = client._encode_form_data({"email": "test@example.com", "name": None})
        assert "email=test@example.com" in result
        assert "name" not in result


class TestStripeClientRequest:
    """Tests for Stripe API requests."""

    def test_request_unconfigured_raises(self):
        """Test that request fails when not configured."""
        client = StripeClient(api_key="")
        with pytest.raises(StripeConfigError, match="not configured"):
            client._request("GET", "/customers")

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_success(self, mock_urlopen):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"id": "cus_123"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_123")
        result = client._request("GET", "/customers/cus_123")

        assert result == {"id": "cus_123"}

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_includes_auth_header(self, mock_urlopen):
        """Test that Authorization header is included."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test_secret")
        client._request("GET", "/customers")

        # Check the request was made with proper auth
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Authorization") == "Bearer sk_test_secret"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_includes_idempotency_key(self, mock_urlopen):
        """Test that Idempotency-Key header is included when provided."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        client._request("POST", "/customers", idempotency_key="idem_123")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Idempotency-key") == "idem_123"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_http_error(self, mock_urlopen):
        """Test HTTP error handling."""
        error = HTTPError(
            "https://api.stripe.com/v1/test",
            400,
            "Bad Request",
            {},
            None,
        )
        error.read = MagicMock(return_value=b'{"error": {"message": "Invalid card"}}')
        mock_urlopen.side_effect = error

        client = StripeClient(api_key="sk_test")
        with pytest.raises(StripeAPIError, match="Invalid card"):
            client._request("POST", "/charges")

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_network_error(self, mock_urlopen):
        """Test network error handling."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = StripeClient(api_key="sk_test")
        with pytest.raises(StripeAPIError, match="Connection error"):
            client._request("GET", "/customers")

    @patch("aragora.billing.stripe_client.urlopen")
    def test_request_malformed_error_response(self, mock_urlopen):
        """Test handling of malformed error response."""
        error = HTTPError(
            "https://api.stripe.com/v1/test",
            500,
            "Internal Error",
            {},
            None,
        )
        error.read = MagicMock(return_value=b"Not JSON")
        mock_urlopen.side_effect = error

        client = StripeClient(api_key="sk_test")
        with pytest.raises(StripeAPIError):
            client._request("GET", "/customers")


class TestStripeClientCustomers:
    """Tests for customer management."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_create_customer_success(self, mock_urlopen):
        """Test creating a customer."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cus_123",
                "email": "test@example.com",
                "name": "Test User",
                "metadata": {"user_id": "u_123"},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        customer = client.create_customer(
            email="test@example.com",
            name="Test User",
            metadata={"user_id": "u_123"},
        )

        assert isinstance(customer, StripeCustomer)
        assert customer.id == "cus_123"
        assert customer.email == "test@example.com"
        assert customer.name == "Test User"
        assert customer.metadata == {"user_id": "u_123"}

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_customer_found(self, mock_urlopen):
        """Test getting an existing customer."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cus_123",
                "email": "test@example.com",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        customer = client.get_customer("cus_123")

        assert customer is not None
        assert customer.id == "cus_123"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_customer_not_found(self, mock_urlopen):
        """Test getting a non-existent customer."""
        error = HTTPError(
            "https://api.stripe.com/v1/customers/cus_invalid",
            404,
            "Not Found",
            {},
            None,
        )
        error.read = MagicMock(return_value=b'{"error": {"message": "Not found"}}')
        mock_urlopen.side_effect = error

        client = StripeClient(api_key="sk_test")
        customer = client.get_customer("cus_invalid")

        assert customer is None

    @patch("aragora.billing.stripe_client.urlopen")
    def test_update_customer(self, mock_urlopen):
        """Test updating a customer."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cus_123",
                "email": "new@example.com",
                "name": "New Name",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        customer = client.update_customer("cus_123", email="new@example.com", name="New Name")

        assert customer.email == "new@example.com"
        assert customer.name == "New Name"


class TestStripeClientCheckout:
    """Tests for checkout sessions."""

    @patch("aragora.billing.stripe_client.urlopen")
    @patch(
        "aragora.billing.stripe_client.STRIPE_PRICES", {SubscriptionTier.PROFESSIONAL: "price_pro"}
    )
    def test_create_checkout_session_success(self, mock_urlopen):
        """Test creating a checkout session."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cs_123",
                "url": "https://checkout.stripe.com/pay/cs_123",
                "customer": "cus_123",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        session = client.create_checkout_session(
            tier=SubscriptionTier.PROFESSIONAL,
            customer_email="test@example.com",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )

        assert isinstance(session, CheckoutSession)
        assert session.id == "cs_123"
        assert "checkout.stripe.com" in session.url

    def test_create_checkout_session_no_price_configured(self):
        """Test checkout fails when price not configured."""
        with patch("aragora.billing.stripe_client.STRIPE_PRICES", {SubscriptionTier.STARTER: ""}):
            client = StripeClient(api_key="sk_test")
            with pytest.raises(StripeConfigError, match="No price configured"):
                client.create_checkout_session(
                    tier=SubscriptionTier.STARTER,
                    customer_email="test@example.com",
                    success_url="https://example.com/success",
                    cancel_url="https://example.com/cancel",
                )

    @patch("aragora.billing.stripe_client.urlopen")
    @patch(
        "aragora.billing.stripe_client.STRIPE_PRICES", {SubscriptionTier.STARTER: "price_starter"}
    )
    def test_create_checkout_session_with_trial(self, mock_urlopen):
        """Test creating checkout session with trial period."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "cs_trial",
                "url": "https://checkout.stripe.com/pay/cs_trial",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        session = client.create_checkout_session(
            tier=SubscriptionTier.STARTER,
            customer_email="test@example.com",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
            trial_days=14,
        )

        assert session.id == "cs_trial"


class TestStripeClientBillingPortal:
    """Tests for billing portal sessions."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_create_portal_session(self, mock_urlopen):
        """Test creating a billing portal session."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "bps_123",
                "url": "https://billing.stripe.com/session/bps_123",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        session = client.create_portal_session(
            customer_id="cus_123",
            return_url="https://example.com/account",
        )

        assert isinstance(session, BillingPortalSession)
        assert session.id == "bps_123"
        assert "billing.stripe.com" in session.url


class TestStripeClientSubscriptions:
    """Tests for subscription management."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_subscription_found(self, mock_urlopen):
        """Test getting an existing subscription."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_123",
                "customer": "cus_123",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_pro"}}]},
                "current_period_start": 1704067200,
                "current_period_end": 1706745600,
                "cancel_at_period_end": False,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        sub = client.get_subscription("sub_123")

        assert sub is not None
        assert isinstance(sub, StripeSubscription)
        assert sub.id == "sub_123"
        assert sub.status == "active"
        assert sub.price_id == "price_pro"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_get_subscription_not_found(self, mock_urlopen):
        """Test getting a non-existent subscription."""
        error = HTTPError(
            "https://api.stripe.com/v1/subscriptions/sub_invalid",
            404,
            "Not Found",
            {},
            None,
        )
        error.read = MagicMock(return_value=b'{"error": {"message": "Not found"}}')
        mock_urlopen.side_effect = error

        client = StripeClient(api_key="sk_test")
        sub = client.get_subscription("sub_invalid")

        assert sub is None

    @patch("aragora.billing.stripe_client.urlopen")
    def test_cancel_subscription_at_period_end(self, mock_urlopen):
        """Test canceling subscription at period end."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_123",
                "customer": "cus_123",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_pro"}}]},
                "current_period_start": 1704067200,
                "current_period_end": 1706745600,
                "cancel_at_period_end": True,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        sub = client.cancel_subscription("sub_123", at_period_end=True)

        assert sub.cancel_at_period_end is True

    @patch("aragora.billing.stripe_client.urlopen")
    def test_cancel_subscription_immediately(self, mock_urlopen):
        """Test canceling subscription immediately."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_123",
                "customer": "cus_123",
                "status": "canceled",
                "items": {"data": []},
                "current_period_start": 1704067200,
                "current_period_end": 1706745600,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        sub = client.cancel_subscription("sub_123", at_period_end=False)

        assert sub.status == "canceled"

    @patch("aragora.billing.stripe_client.urlopen")
    def test_resume_subscription(self, mock_urlopen):
        """Test resuming a subscription."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "id": "sub_123",
                "customer": "cus_123",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_pro"}}]},
                "current_period_start": 1704067200,
                "current_period_end": 1706745600,
                "cancel_at_period_end": False,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        sub = client.resume_subscription("sub_123")

        assert sub.cancel_at_period_end is False


class TestStripeClientInvoices:
    """Tests for invoice listing."""

    @patch("aragora.billing.stripe_client.urlopen")
    def test_list_invoices(self, mock_urlopen):
        """Test listing invoices."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {"id": "in_1", "amount_paid": 2000, "status": "paid"},
                    {"id": "in_2", "amount_paid": 2000, "status": "paid"},
                ]
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = StripeClient(api_key="sk_test")
        invoices = client.list_invoices("cus_123", limit=10)

        assert len(invoices) == 2
        assert invoices[0]["id"] == "in_1"


# =============================================================================
# Webhook Signature Verification Tests (SECURITY-CRITICAL)
# =============================================================================


class TestWebhookSignatureVerification:
    """Security-critical tests for webhook signature verification."""

    def test_verify_valid_signature(self):
        """Test verification of valid webhook signature."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test_secret"
        timestamp = str(int(time.time()))

        # Generate valid signature
        signed_payload = f"{timestamp}.".encode() + payload
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={timestamp},v1={expected_sig}"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is True

    def test_verify_invalid_signature(self):
        """SECURITY: Test that invalid signatures are rejected."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test_secret"
        timestamp = str(int(time.time()))

        # Use wrong signature
        signature = f"t={timestamp},v1=invalid_signature_hash"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_verify_tampered_payload(self):
        """SECURITY: Test that tampered payloads are rejected."""
        original_payload = b'{"type": "test.event", "amount": 100}'
        tampered_payload = b'{"type": "test.event", "amount": 10000}'
        secret = "whsec_test_secret"
        timestamp = str(int(time.time()))

        # Sign original payload
        signed_payload = f"{timestamp}.".encode() + original_payload
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={timestamp},v1={expected_sig}"

        # Verify with tampered payload should fail
        result = verify_webhook_signature(tampered_payload, signature, secret)
        assert result is False

    def test_verify_expired_timestamp(self):
        """SECURITY: Test that expired timestamps are rejected."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test_secret"
        # Timestamp from 10 minutes ago (beyond 5 min tolerance)
        old_timestamp = str(int(time.time()) - 600)

        signed_payload = f"{old_timestamp}.".encode() + payload
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={old_timestamp},v1={expected_sig}"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_verify_future_timestamp_tolerance(self):
        """Test that timestamps slightly in the future are accepted (clock skew)."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test_secret"
        # Timestamp 2 minutes in the future (within 5 min tolerance)
        future_timestamp = str(int(time.time()) + 120)

        signed_payload = f"{future_timestamp}.".encode() + payload
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={future_timestamp},v1={expected_sig}"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is True

    def test_verify_missing_secret(self):
        """Test that missing secret returns False."""
        payload = b'{"type": "test.event"}'
        signature = "t=123,v1=abc"

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", ""):
            result = verify_webhook_signature(payload, signature, None)
        assert result is False

    def test_verify_malformed_signature_header(self):
        """SECURITY: Test that malformed signature headers are rejected."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test"

        # Various malformed signatures
        assert verify_webhook_signature(payload, "", secret) is False
        assert verify_webhook_signature(payload, "invalid", secret) is False
        assert verify_webhook_signature(payload, "t=,v1=", secret) is False
        assert verify_webhook_signature(payload, "only_timestamp=123", secret) is False

    def test_verify_missing_timestamp(self):
        """SECURITY: Test that missing timestamp is rejected."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test"
        signature = "v1=somehash"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_verify_missing_v1_signature(self):
        """SECURITY: Test that missing v1 signature is rejected."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test"
        signature = f"t={int(time.time())}"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is False

    def test_verify_multiple_signatures(self):
        """Test that verification works with multiple v1 signatures (Stripe format)."""
        payload = b'{"type": "test.event"}'
        secret = "whsec_test_secret"
        timestamp = str(int(time.time()))

        signed_payload = f"{timestamp}.".encode() + payload
        valid_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        # Stripe can send multiple v1 signatures
        signature = f"t={timestamp},v1=invalid_old_sig,v1={valid_sig}"

        result = verify_webhook_signature(payload, signature, secret)
        assert result is True

    def test_verify_wrong_secret(self):
        """SECURITY: Test that wrong secret is rejected."""
        payload = b'{"type": "test.event"}'
        correct_secret = "whsec_correct"
        wrong_secret = "whsec_wrong"
        timestamp = str(int(time.time()))

        # Sign with correct secret
        signed_payload = f"{timestamp}.".encode() + payload
        sig = hmac.new(
            correct_secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={timestamp},v1={sig}"

        # Verify with wrong secret
        result = verify_webhook_signature(payload, signature, wrong_secret)
        assert result is False


class TestWebhookEventParsing:
    """Tests for webhook event parsing."""

    def test_parse_valid_event(self):
        """Test parsing a valid webhook event."""
        payload = json.dumps(
            {
                "type": "customer.subscription.created",
                "data": {
                    "object": {
                        "id": "sub_123",
                        "customer": "cus_123",
                        "metadata": {"org_id": "org_456"},
                    }
                },
            }
        ).encode()
        secret = "whsec_test"
        timestamp = str(int(time.time()))

        signed_payload = f"{timestamp}.".encode() + payload
        sig = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
        signature = f"t={timestamp},v1={sig}"

        # Need to patch the global STRIPE_WEBHOOK_SECRET for parse_webhook_event
        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            event = parse_webhook_event(payload, signature)

        assert event is not None
        assert event.type == "customer.subscription.created"
        assert event.subscription_id == "sub_123"
        assert event.customer_id == "cus_123"
        assert event.metadata == {"org_id": "org_456"}

    def test_parse_invalid_signature_returns_none(self):
        """Test that invalid signature returns None."""
        payload = b'{"type": "test.event", "data": {}}'
        signature = "t=123,v1=invalid"

        with patch("aragora.billing.stripe_client.verify_webhook_signature", return_value=False):
            event = parse_webhook_event(payload, signature)

        assert event is None

    def test_parse_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        payload = b"not json"
        signature = "t=123,v1=sig"

        with patch("aragora.billing.stripe_client.verify_webhook_signature", return_value=True):
            event = parse_webhook_event(payload, signature)

        assert event is None


class TestWebhookEvent:
    """Tests for WebhookEvent class."""

    def test_customer_id_from_customer_field(self):
        """Test extracting customer_id from customer field."""
        event = WebhookEvent("invoice.paid", {"object": {"customer": "cus_123"}})
        assert event.customer_id == "cus_123"

    def test_customer_id_from_id_field(self):
        """Test extracting customer_id from id field (for customer.* events)."""
        event = WebhookEvent("customer.created", {"object": {"id": "cus_456"}})
        assert event.customer_id == "cus_456"

    def test_subscription_id_from_subscription_event(self):
        """Test extracting subscription_id from subscription events."""
        event = WebhookEvent("customer.subscription.updated", {"object": {"id": "sub_123"}})
        assert event.subscription_id == "sub_123"

    def test_subscription_id_from_subscription_field(self):
        """Test extracting subscription_id from subscription field."""
        event = WebhookEvent("invoice.paid", {"object": {"subscription": "sub_456"}})
        assert event.subscription_id == "sub_456"

    def test_metadata_extraction(self):
        """Test metadata extraction."""
        event = WebhookEvent("test.event", {"object": {"metadata": {"key": "value"}}})
        assert event.metadata == {"key": "value"}

    def test_metadata_empty_when_missing(self):
        """Test that missing metadata returns empty dict."""
        event = WebhookEvent("test.event", {"object": {}})
        assert event.metadata == {}


# =============================================================================
# Tier Mapping Tests
# =============================================================================


class TestTierMapping:
    """Tests for tier/price ID mapping."""

    def test_get_tier_from_price_id_found(self):
        """Test getting tier from price ID."""
        with patch.dict(STRIPE_PRICES, {SubscriptionTier.PROFESSIONAL: "price_pro_123"}):
            tier = get_tier_from_price_id("price_pro_123")
            assert tier == SubscriptionTier.PROFESSIONAL

    def test_get_tier_from_price_id_not_found(self):
        """Test getting tier from unknown price ID."""
        tier = get_tier_from_price_id("price_unknown")
        assert tier is None

    def test_get_price_id_for_tier_found(self):
        """Test getting price ID for tier."""
        with patch.dict(STRIPE_PRICES, {SubscriptionTier.STARTER: "price_starter_123"}):
            price_id = get_price_id_for_tier(SubscriptionTier.STARTER)
            assert price_id == "price_starter_123"

    def test_get_price_id_for_tier_not_configured(self):
        """Test getting price ID when not configured."""
        with patch.dict(STRIPE_PRICES, {SubscriptionTier.ENTERPRISE: ""}, clear=True):
            price_id = get_price_id_for_tier(SubscriptionTier.ENTERPRISE)
            assert price_id == ""


# =============================================================================
# Default Client Tests
# =============================================================================


class TestDefaultClient:
    """Tests for default client singleton."""

    def test_get_stripe_client_returns_client(self):
        """Test that get_stripe_client returns a StripeClient."""
        with patch("aragora.billing.stripe_client._default_client", None):
            client = get_stripe_client()
            assert isinstance(client, StripeClient)

    def test_get_stripe_client_returns_same_instance(self):
        """Test that get_stripe_client returns singleton."""
        with patch("aragora.billing.stripe_client._default_client", None):
            client1 = get_stripe_client()
            client2 = get_stripe_client()
            assert client1 is client2


# =============================================================================
# Data Class Tests
# =============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_stripe_customer_to_dict(self):
        """Test StripeCustomer serialization."""
        customer = StripeCustomer(
            id="cus_123",
            email="test@example.com",
            name="Test",
            metadata={"key": "value"},
        )
        d = customer.to_dict()
        assert d["id"] == "cus_123"
        assert d["email"] == "test@example.com"
        assert d["metadata"] == {"key": "value"}

    def test_stripe_subscription_to_dict(self):
        """Test StripeSubscription serialization."""
        sub = StripeSubscription(
            id="sub_123",
            customer_id="cus_123",
            status="active",
            price_id="price_pro",
            current_period_start=datetime(2024, 1, 1),
            current_period_end=datetime(2024, 2, 1),
            cancel_at_period_end=False,
        )
        d = sub.to_dict()
        assert d["id"] == "sub_123"
        assert d["status"] == "active"
        assert "2024-01-01" in d["current_period_start"]

    def test_checkout_session_to_dict(self):
        """Test CheckoutSession serialization."""
        session = CheckoutSession(
            id="cs_123",
            url="https://checkout.stripe.com/pay/cs_123",
            customer_id="cus_123",
            subscription_id="sub_123",
        )
        d = session.to_dict()
        assert d["id"] == "cs_123"
        assert d["url"] == "https://checkout.stripe.com/pay/cs_123"

    def test_billing_portal_session_to_dict(self):
        """Test BillingPortalSession serialization."""
        session = BillingPortalSession(
            id="bps_123",
            url="https://billing.stripe.com/session/bps_123",
        )
        d = session.to_dict()
        assert d["id"] == "bps_123"
        assert d["url"] == "https://billing.stripe.com/session/bps_123"
