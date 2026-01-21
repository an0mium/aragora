"""
Tests for webhook signature verification utilities.
"""

import json
import time

import pytest


class TestGenerateSignature:
    """Tests for generate_signature function."""

    def test_format(self):
        """Signature should be sha256= prefixed hex."""
        from aragora.events.webhook_verify import generate_signature

        sig = generate_signature('{"test": "data"}', "secret")

        assert sig.startswith("sha256=")
        assert len(sig) == 7 + 64  # "sha256=" + 64 hex chars

    def test_deterministic(self):
        """Same input should produce same signature."""
        from aragora.events.webhook_verify import generate_signature

        sig1 = generate_signature("payload", "secret")
        sig2 = generate_signature("payload", "secret")

        assert sig1 == sig2

    def test_different_secrets(self):
        """Different secrets should produce different signatures."""
        from aragora.events.webhook_verify import generate_signature

        sig1 = generate_signature("payload", "secret1")
        sig2 = generate_signature("payload", "secret2")

        assert sig1 != sig2

    def test_different_payloads(self):
        """Different payloads should produce different signatures."""
        from aragora.events.webhook_verify import generate_signature

        sig1 = generate_signature("payload1", "secret")
        sig2 = generate_signature("payload2", "secret")

        assert sig1 != sig2


class TestVerifySignature:
    """Tests for verify_signature function."""

    def test_valid_signature(self):
        """Should return True for valid signature."""
        from aragora.events.webhook_verify import generate_signature, verify_signature

        payload = '{"event": "test"}'
        secret = "webhook-secret"
        signature = generate_signature(payload, secret)

        assert verify_signature(payload, signature, secret) is True

    def test_invalid_signature(self):
        """Should return False for invalid signature."""
        from aragora.events.webhook_verify import verify_signature

        assert verify_signature("payload", "sha256=invalid", "secret") is False

    def test_tampered_payload(self):
        """Should return False for tampered payload."""
        from aragora.events.webhook_verify import generate_signature, verify_signature

        original = '{"event": "test"}'
        tampered = '{"event": "tampered"}'
        secret = "secret"
        sig = generate_signature(original, secret)

        assert verify_signature(tampered, sig, secret) is False

    def test_missing_prefix(self):
        """Should return False if signature missing sha256= prefix."""
        from aragora.events.webhook_verify import verify_signature

        assert verify_signature("payload", "abcd1234", "secret") is False

    def test_empty_signature(self):
        """Should return False for empty signature."""
        from aragora.events.webhook_verify import verify_signature

        assert verify_signature("payload", "", "secret") is False
        assert verify_signature("payload", None, "secret") is False


class TestVerifyTimestamp:
    """Tests for verify_timestamp function."""

    def test_valid_recent_timestamp(self):
        """Should accept recent timestamp."""
        from aragora.events.webhook_verify import verify_timestamp

        valid, error = verify_timestamp(time.time())

        assert valid is True
        assert error is None

    def test_old_timestamp(self):
        """Should reject old timestamp."""
        from aragora.events.webhook_verify import verify_timestamp

        old_ts = time.time() - 600  # 10 minutes ago
        valid, error = verify_timestamp(old_ts, tolerance_seconds=300)

        assert valid is False
        assert "too old" in error.lower()

    def test_future_timestamp(self):
        """Should reject future timestamp."""
        from aragora.events.webhook_verify import verify_timestamp

        future_ts = time.time() + 120  # 2 minutes in future
        valid, error = verify_timestamp(future_ts)

        assert valid is False
        assert "future" in error.lower()

    def test_slight_future_allowed(self):
        """Should allow slight future timestamp (clock skew)."""
        from aragora.events.webhook_verify import verify_timestamp

        # 30 seconds in future should be allowed (within 60s tolerance)
        slight_future = time.time() + 30
        valid, error = verify_timestamp(slight_future)

        assert valid is True

    def test_invalid_format(self):
        """Should reject invalid timestamp format."""
        from aragora.events.webhook_verify import verify_timestamp

        valid, error = verify_timestamp("not-a-number")

        assert valid is False
        assert "format" in error.lower()

    def test_string_timestamp(self):
        """Should accept string timestamp."""
        from aragora.events.webhook_verify import verify_timestamp

        valid, error = verify_timestamp(str(int(time.time())))

        assert valid is True

    def test_custom_tolerance(self):
        """Should respect custom tolerance."""
        from aragora.events.webhook_verify import verify_timestamp

        # 4 minutes ago should fail with 3 minute tolerance
        ts = time.time() - 240
        valid, error = verify_timestamp(ts, tolerance_seconds=180)
        assert valid is False

        # But should pass with 5 minute tolerance
        valid2, error2 = verify_timestamp(ts, tolerance_seconds=300)
        assert valid2 is True


class TestVerifyWebhookRequest:
    """Tests for verify_webhook_request function."""

    def test_valid_request(self):
        """Should accept valid request with signature and timestamp."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload = '{"event": "test"}'
        secret = "whsec_test"
        timestamp = str(int(time.time()))
        signature = generate_signature(payload, secret)

        result = verify_webhook_request(
            payload=payload,
            signature=signature,
            timestamp=timestamp,
            secret=secret,
        )

        assert result.valid is True
        assert result.error is None
        assert bool(result) is True

    def test_missing_signature(self):
        """Should reject missing signature."""
        from aragora.events.webhook_verify import verify_webhook_request

        result = verify_webhook_request(
            payload="{}",
            signature=None,
            timestamp=str(int(time.time())),
            secret="secret",
        )

        assert result.valid is False
        assert "signature" in result.error.lower()

    def test_missing_timestamp(self):
        """Should reject missing timestamp when check_timestamp=True."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload = "{}"
        signature = generate_signature(payload, "secret")

        result = verify_webhook_request(
            payload=payload,
            signature=signature,
            timestamp=None,
            secret="secret",
            check_timestamp=True,
        )

        assert result.valid is False
        assert "timestamp" in result.error.lower()

    def test_skip_timestamp_check(self):
        """Should skip timestamp check when disabled."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload = "{}"
        signature = generate_signature(payload, "secret")

        result = verify_webhook_request(
            payload=payload,
            signature=signature,
            timestamp=None,
            secret="secret",
            check_timestamp=False,
        )

        assert result.valid is True

    def test_invalid_signature(self):
        """Should reject invalid signature."""
        from aragora.events.webhook_verify import verify_webhook_request

        result = verify_webhook_request(
            payload="{}",
            signature="sha256=invalid",
            timestamp=str(int(time.time())),
            secret="secret",
        )

        assert result.valid is False
        assert "signature" in result.error.lower()

    def test_expired_timestamp(self):
        """Should reject expired timestamp."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload = "{}"
        signature = generate_signature(payload, "secret")
        old_timestamp = str(int(time.time() - 600))  # 10 minutes ago

        result = verify_webhook_request(
            payload=payload,
            signature=signature,
            timestamp=old_timestamp,
            secret="secret",
            timestamp_tolerance=300,  # 5 minute tolerance
        )

        assert result.valid is False
        assert "timestamp" in result.error.lower()

    def test_bytes_payload(self):
        """Should accept bytes payload."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload_str = '{"event": "test"}'
        payload_bytes = payload_str.encode("utf-8")
        secret = "secret"
        signature = generate_signature(payload_str, secret)

        result = verify_webhook_request(
            payload=payload_bytes,
            signature=signature,
            timestamp=str(int(time.time())),
            secret=secret,
        )

        assert result.valid is True

    def test_missing_secret(self):
        """Should reject missing secret."""
        from aragora.events.webhook_verify import verify_webhook_request

        result = verify_webhook_request(
            payload="{}",
            signature="sha256=abc",
            timestamp=str(int(time.time())),
            secret="",
        )

        assert result.valid is False
        assert "secret" in result.error.lower()

    def test_verification_result_bool(self):
        """VerificationResult should work as boolean."""
        from aragora.events.webhook_verify import VerificationResult

        valid = VerificationResult(True)
        invalid = VerificationResult(False, "error")

        assert bool(valid) is True
        assert bool(invalid) is False

        # Can be used in if statements
        if valid:
            passed = True
        else:
            passed = False
        assert passed is True


class TestCreateTestWebhookPayload:
    """Tests for create_test_webhook_payload function."""

    def test_creates_valid_payload(self):
        """Should create payload that passes verification."""
        from aragora.events.webhook_verify import (
            create_test_webhook_payload,
            verify_webhook_request,
        )

        secret = "whsec_test123"
        payload, headers = create_test_webhook_payload(
            event_type="debate_end",
            data={"debate_id": "deb_123"},
            secret=secret,
        )

        # Verify the payload
        payload_json = json.dumps(payload)
        result = verify_webhook_request(
            payload=payload_json,
            signature=headers["X-Aragora-Signature"],
            timestamp=headers["X-Aragora-Timestamp"],
            secret=secret,
        )

        assert result.valid is True

    def test_includes_required_fields(self):
        """Should include all required payload fields."""
        from aragora.events.webhook_verify import create_test_webhook_payload

        payload, headers = create_test_webhook_payload(
            "test_event",
            {"key": "value"},
            "secret",
        )

        assert payload["event"] == "test_event"
        assert "timestamp" in payload
        assert "delivery_id" in payload
        assert payload["data"] == {"key": "value"}

    def test_includes_required_headers(self):
        """Should include all required headers."""
        from aragora.events.webhook_verify import create_test_webhook_payload

        payload, headers = create_test_webhook_payload(
            "test_event",
            {},
            "secret",
        )

        assert "X-Aragora-Signature" in headers
        assert "X-Aragora-Event" in headers
        assert "X-Aragora-Timestamp" in headers
        assert "X-Aragora-Delivery" in headers
        assert headers["Content-Type"] == "application/json"


class TestReplayAttackPrevention:
    """Tests for replay attack prevention."""

    def test_reused_old_request_rejected(self):
        """Should reject reused request with old timestamp."""
        from aragora.events.webhook_verify import (
            generate_signature,
            verify_webhook_request,
        )

        payload = '{"event": "sensitive_action"}'
        secret = "secret"

        # Create a request that was valid 10 minutes ago
        old_timestamp = str(int(time.time() - 600))
        signature = generate_signature(payload, secret)

        # Attempt to replay the request
        result = verify_webhook_request(
            payload=payload,
            signature=signature,
            timestamp=old_timestamp,  # Old timestamp
            secret=secret,
            timestamp_tolerance=300,  # 5 minute tolerance
        )

        assert result.valid is False
        assert "timestamp" in result.error.lower()
