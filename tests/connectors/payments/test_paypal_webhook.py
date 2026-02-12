"""
Tests for PayPal webhook signature verification.

These tests verify the security-critical webhook signature verification
that prevents attackers from sending forged webhook events.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import os
import zlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from aragora.connectors.payments.paypal import PayPalClient, PayPalCredentials


class TestPayPalWebhookSignature:
    """Tests for PayPal webhook signature verification."""

    def _compute_valid_signature(
        self,
        secret: str,
        transmission_id: str,
        timestamp: str,
        webhook_id: str,
        event_body: str,
    ) -> str:
        """Compute a valid HMAC-SHA256 signature for testing."""
        crc = zlib.crc32(event_body.encode("utf-8")) & 0xFFFFFFFF
        sig_input = f"{transmission_id}|{timestamp}|{webhook_id}|{crc}"
        return base64.b64encode(
            hmac.new(secret.encode("utf-8"), sig_input.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")

    def test_valid_signature_accepted(self):
        """Valid signature should return True."""
        secret = "test_webhook_secret_12345"
        webhook_id = "WH-12345678"

        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=webhook_id,
            webhook_secret=secret,
        )
        client = PayPalClient(creds)

        transmission_id = "trans-abc-123"
        timestamp = datetime.now(timezone.utc).isoformat()
        event_body = '{"event_type": "PAYMENT.CAPTURE.COMPLETED", "id": "evt123"}'

        valid_signature = self._compute_valid_signature(
            secret, transmission_id, timestamp, webhook_id, event_body
        )

        result = client.verify_webhook_signature(
            transmission_id=transmission_id,
            timestamp=timestamp,
            webhook_id=webhook_id,
            event_body=event_body,
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature=valid_signature,
        )

        assert result is True

    def test_invalid_signature_rejected(self):
        """Invalid signature should return False."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id="WH-12345678",
            webhook_secret="test_secret_12345",
        )
        client = PayPalClient(creds)

        result = client.verify_webhook_signature(
            transmission_id="trans-123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            webhook_id="WH-12345678",
            event_body='{"event": "test"}',
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature="invalid_signature_here",
        )

        assert result is False

    def test_missing_signature_rejected(self):
        """Missing signature should return False."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id="WH-12345678",
            webhook_secret="test_secret_12345",
        )
        client = PayPalClient(creds)

        result = client.verify_webhook_signature(
            transmission_id="trans-123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            webhook_id="WH-12345678",
            event_body='{"event": "test"}',
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature="",  # Empty signature
        )

        assert result is False

    def test_webhook_id_mismatch_rejected(self):
        """Mismatched webhook ID should return False."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id="WH-EXPECTED",
            webhook_secret="test_secret_12345",
        )
        client = PayPalClient(creds)

        result = client.verify_webhook_signature(
            transmission_id="trans-123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            webhook_id="WH-DIFFERENT",  # Different from configured
            event_body='{"event": "test"}',
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature="some_signature",
        )

        assert result is False

    def test_expired_timestamp_rejected(self):
        """Old timestamp should return False (replay attack protection)."""
        secret = "test_secret_12345"
        webhook_id = "WH-12345678"

        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=webhook_id,
            webhook_secret=secret,
        )
        client = PayPalClient(creds)

        # Timestamp 10 minutes in the past (exceeds 5-minute window)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        event_body = '{"event": "test"}'

        # Even with valid signature, old timestamp should be rejected
        valid_signature = self._compute_valid_signature(
            secret, "trans-123", old_timestamp, webhook_id, event_body
        )

        result = client.verify_webhook_signature(
            transmission_id="trans-123",
            timestamp=old_timestamp,
            webhook_id=webhook_id,
            event_body=event_body,
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature=valid_signature,
        )

        assert result is False

    def test_production_rejects_missing_webhook_id(self):
        """Production should reject if webhook_id not configured."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=None,  # Not configured
            webhook_secret="test_secret",
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = client.verify_webhook_signature(
                transmission_id="trans-123",
                timestamp=datetime.now(timezone.utc).isoformat(),
                webhook_id="WH-12345678",
                event_body='{"event": "test"}',
                cert_url="https://api.paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="some_signature",
            )

        assert result is False

    def test_production_rejects_missing_webhook_secret(self):
        """Production should reject if webhook_secret not configured."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id="WH-12345678",
            webhook_secret=None,  # Not configured
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = client.verify_webhook_signature(
                transmission_id="trans-123",
                timestamp=datetime.now(timezone.utc).isoformat(),
                webhook_id="WH-12345678",
                event_body='{"event": "test"}',
                cert_url="https://api.paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="some_signature",
            )

        assert result is False

    def test_development_allows_missing_config(self):
        """Development mode allows missing webhook config (with warnings)."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=None,  # Not configured
            webhook_secret=None,  # Not configured
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            result = client.verify_webhook_signature(
                transmission_id="trans-123",
                timestamp=datetime.now(timezone.utc).isoformat(),
                webhook_id="WH-12345678",
                event_body='{"event": "test"}',
                cert_url="https://api.paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="some_signature",
            )

        # In dev mode, missing config allows bypass (for testing)
        assert result is True

    def test_timing_safe_comparison_used(self):
        """Verify hmac.compare_digest is used for timing-safe comparison."""
        source = inspect.getsource(PayPalClient.verify_webhook_signature)

        # Must use hmac.compare_digest for timing-safe comparison
        assert "hmac.compare_digest" in source

        # Must NOT use simple equality for signature comparison
        assert "actual_signature ==" not in source
        assert "== actual_signature" not in source

    def test_crc32_unsigned(self):
        """Verify CRC32 is computed as unsigned integer."""
        source = inspect.getsource(PayPalClient.verify_webhook_signature)

        # Should mask CRC32 to ensure unsigned (0xFFFFFFFF)
        assert "0xFFFFFFFF" in source or "0xffffffff" in source

    def test_signature_with_special_characters_in_body(self):
        """Test signature verification with special characters in event body."""
        secret = "test_webhook_secret"
        webhook_id = "WH-12345678"

        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=webhook_id,
            webhook_secret=secret,
        )
        client = PayPalClient(creds)

        transmission_id = "trans-123"
        timestamp = datetime.now(timezone.utc).isoformat()
        # Body with unicode, newlines, and special characters
        event_body = '{"message": "Payment for café — €50.00\\n\\tThank you! 🎉"}'

        valid_signature = self._compute_valid_signature(
            secret, transmission_id, timestamp, webhook_id, event_body
        )

        result = client.verify_webhook_signature(
            transmission_id=transmission_id,
            timestamp=timestamp,
            webhook_id=webhook_id,
            event_body=event_body,
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature=valid_signature,
        )

        assert result is True

    def test_credentials_from_env_loads_webhook_secret(self):
        """Test that from_env loads webhook_secret from environment."""
        with patch.dict(
            os.environ,
            {
                "PAYPAL_CLIENT_ID": "env_client_id",
                "PAYPAL_CLIENT_SECRET": "env_client_secret",
                "PAYPAL_WEBHOOK_ID": "WH-ENV-12345",
                "PAYPAL_WEBHOOK_SECRET": "env_webhook_secret",
                "PAYPAL_ENVIRONMENT": "sandbox",
            },
        ):
            creds = PayPalCredentials.from_env()

            assert creds.client_id == "env_client_id"
            assert creds.webhook_id == "WH-ENV-12345"
            assert creds.webhook_secret == "env_webhook_secret"


class TestPayPalWebhookEdgeCases:
    """Edge case tests for PayPal webhook verification."""

    def test_malformed_timestamp_in_production(self):
        """Malformed timestamp should be rejected in production."""
        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id="WH-12345678",
            webhook_secret="test_secret_12345",
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = client.verify_webhook_signature(
                transmission_id="trans-123",
                timestamp="not-a-valid-timestamp",
                webhook_id="WH-12345678",
                event_body='{"event": "test"}',
                cert_url="https://api.paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="some_signature",
            )

        assert result is False

    def test_future_timestamp_accepted(self):
        """Future timestamp within tolerance should be accepted."""
        secret = "test_secret_12345"
        webhook_id = "WH-12345678"

        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=webhook_id,
            webhook_secret=secret,
        )
        client = PayPalClient(creds)

        # 2 minutes in the future (within 5-minute window)
        future_timestamp = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
        event_body = '{"event": "test"}'

        valid_signature = base64.b64encode(
            hmac.new(
                secret.encode("utf-8"),
                f"trans-123|{future_timestamp}|{webhook_id}|{zlib.crc32(event_body.encode()) & 0xFFFFFFFF}".encode(),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        result = client.verify_webhook_signature(
            transmission_id="trans-123",
            timestamp=future_timestamp,
            webhook_id=webhook_id,
            event_body=event_body,
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature=valid_signature,
        )

        assert result is True

    def test_empty_event_body(self):
        """Empty event body should be handled correctly."""
        secret = "test_secret_12345"
        webhook_id = "WH-12345678"

        creds = PayPalCredentials(
            client_id="test_id",
            client_secret="test_secret",
            webhook_id=webhook_id,
            webhook_secret=secret,
        )
        client = PayPalClient(creds)

        transmission_id = "trans-123"
        timestamp = datetime.now(timezone.utc).isoformat()
        event_body = ""  # Empty body

        crc = zlib.crc32(event_body.encode("utf-8")) & 0xFFFFFFFF
        sig_input = f"{transmission_id}|{timestamp}|{webhook_id}|{crc}"
        valid_signature = base64.b64encode(
            hmac.new(secret.encode("utf-8"), sig_input.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")

        result = client.verify_webhook_signature(
            transmission_id=transmission_id,
            timestamp=timestamp,
            webhook_id=webhook_id,
            event_body=event_body,
            cert_url="https://api.paypal.com/cert",
            auth_algo="SHA256withRSA",
            actual_signature=valid_signature,
        )

        assert result is True
