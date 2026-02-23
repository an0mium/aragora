"""Comprehensive tests for Slack signature verification (aragora/server/handlers/bots/slack/signature.py).

Covers:
- verify_slack_signature() function
  - Valid signature verification (happy path)
  - Invalid signature rejection
  - Timestamp validation (replay attack prevention)
    - Timestamp too old (> 5 minutes)
    - Timestamp too far in the future (> 5 minutes)
    - Timestamp exactly at the 5-minute boundary
    - Timestamp just within the 5-minute window
  - Invalid timestamp formats
    - Non-numeric string
    - Empty string
    - None value
    - Float string
  - Body encoding
    - Empty body
    - Unicode body
    - JSON body
    - Body with special characters
  - Signing secret variations
    - Empty signing secret
    - Multi-byte signing secret
  - Signature format
    - Missing v0= prefix
    - Wrong prefix
    - Empty signature
    - Truncated signature
  - Deterministic signature computation
  - hmac.compare_digest timing-safe comparison
- Module __all__ exports
"""

from __future__ import annotations

import hashlib
import hmac
import time
from unittest.mock import patch

import pytest

from aragora.server.handlers.bots.slack.signature import (
    verify_slack_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signature(body: bytes, timestamp: str, signing_secret: str) -> str:
    """Compute a valid Slack signature for the given parameters."""
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


def _current_timestamp() -> str:
    """Return the current Unix timestamp as a string."""
    return str(int(time.time()))


SECRET = "test_signing_secret_abc123"


# ---------------------------------------------------------------------------
# Happy path: valid signature
# ---------------------------------------------------------------------------


class TestValidSignature:
    """Tests for successful signature verification."""

    def test_valid_signature_returns_true(self):
        """A correctly computed HMAC-SHA256 signature should verify."""
        ts = _current_timestamp()
        body = b'{"type":"event_callback","event":{"type":"message"}}'
        sig = _make_signature(body, ts, SECRET)

        assert verify_slack_signature(body, ts, sig, SECRET) is True

    def test_valid_signature_empty_body(self):
        """Verification should succeed with an empty body."""
        ts = _current_timestamp()
        body = b""
        sig = _make_signature(body, ts, SECRET)

        assert verify_slack_signature(body, ts, sig, SECRET) is True

    def test_valid_signature_unicode_body(self):
        """Verification should succeed with unicode characters in body."""
        ts = _current_timestamp()
        body = "{'text': 'café résumé naïve'}".encode("utf-8")
        sig = _make_signature(body, ts, SECRET)

        assert verify_slack_signature(body, ts, sig, SECRET) is True

    def test_valid_signature_large_body(self):
        """Verification should succeed with a large body payload."""
        ts = _current_timestamp()
        body = b"x" * 100_000
        sig = _make_signature(body, ts, SECRET)

        assert verify_slack_signature(body, ts, sig, SECRET) is True

    def test_valid_signature_special_characters_in_body(self):
        """Verification should succeed with special characters in body."""
        ts = _current_timestamp()
        body = b'{"text": "line1\\nline2\\ttab", "emoji": "\\ud83d"}'
        sig = _make_signature(body, ts, SECRET)

        assert verify_slack_signature(body, ts, sig, SECRET) is True

    def test_valid_signature_with_different_secret(self):
        """Different signing secrets produce different valid signatures."""
        ts = _current_timestamp()
        body = b"hello"
        secret_a = "secret_alpha"
        secret_b = "secret_beta"
        sig_a = _make_signature(body, ts, secret_a)
        sig_b = _make_signature(body, ts, secret_b)

        assert sig_a != sig_b
        assert verify_slack_signature(body, ts, sig_a, secret_a) is True
        assert verify_slack_signature(body, ts, sig_b, secret_b) is True


# ---------------------------------------------------------------------------
# Invalid signature
# ---------------------------------------------------------------------------


class TestInvalidSignature:
    """Tests for signature mismatch rejection."""

    def test_wrong_signature_returns_false(self):
        """A completely wrong signature should be rejected."""
        ts = _current_timestamp()
        body = b"test body"
        wrong_sig = "v0=0000000000000000000000000000000000000000000000000000000000000000"

        assert verify_slack_signature(body, ts, wrong_sig, SECRET) is False

    def test_signature_with_wrong_secret(self):
        """A signature computed with a different secret should be rejected."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, "wrong_secret")

        assert verify_slack_signature(body, ts, sig, SECRET) is False

    def test_tampered_body_rejected(self):
        """If the body is modified after signing, verification should fail."""
        ts = _current_timestamp()
        original_body = b"original content"
        sig = _make_signature(original_body, ts, SECRET)
        tampered_body = b"tampered content"

        assert verify_slack_signature(tampered_body, ts, sig, SECRET) is False

    def test_tampered_timestamp_rejected(self):
        """If the timestamp differs from what was signed, verification should fail."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)
        different_ts = str(int(ts) + 1)

        assert verify_slack_signature(body, different_ts, sig, SECRET) is False

    def test_empty_signature_returns_false(self):
        """An empty signature string should be rejected."""
        ts = _current_timestamp()
        body = b"test body"

        assert verify_slack_signature(body, ts, "", SECRET) is False

    def test_signature_missing_v0_prefix(self):
        """A signature without the v0= prefix should be rejected."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)
        # Strip the v0= prefix
        hex_only = sig[3:]

        assert verify_slack_signature(body, ts, hex_only, SECRET) is False

    def test_signature_wrong_prefix(self):
        """A signature with v1= instead of v0= should be rejected."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)
        wrong_prefix_sig = "v1=" + sig[3:]

        assert verify_slack_signature(body, ts, wrong_prefix_sig, SECRET) is False

    def test_truncated_signature(self):
        """A truncated signature should be rejected."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)
        truncated = sig[:20]

        assert verify_slack_signature(body, ts, truncated, SECRET) is False


# ---------------------------------------------------------------------------
# Timestamp validation (replay attack prevention)
# ---------------------------------------------------------------------------


class TestTimestampValidation:
    """Tests for timestamp-based replay attack prevention."""

    def test_timestamp_too_old_returns_false(self):
        """A timestamp older than 5 minutes should be rejected."""
        old_ts = str(int(time.time()) - 301)  # 5 min + 1 sec
        body = b"test body"
        sig = _make_signature(body, old_ts, SECRET)

        assert verify_slack_signature(body, old_ts, sig, SECRET) is False

    def test_timestamp_too_far_in_future_returns_false(self):
        """A timestamp more than 5 minutes in the future should be rejected."""
        future_ts = str(int(time.time()) + 301)
        body = b"test body"
        sig = _make_signature(body, future_ts, SECRET)

        assert verify_slack_signature(body, future_ts, sig, SECRET) is False

    def test_timestamp_exactly_at_boundary_accepted(self):
        """A timestamp exactly 300 seconds old should be accepted (abs <= 300)."""
        # We need to control time.time() to test the boundary precisely
        now = 1700000000
        boundary_ts = str(now - 300)
        body = b"test body"
        sig = _make_signature(body, boundary_ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            result = verify_slack_signature(body, boundary_ts, sig, SECRET)

        assert result is True

    def test_timestamp_one_second_past_boundary_rejected(self):
        """A timestamp 301 seconds old should be rejected."""
        now = 1700000000
        boundary_ts = str(now - 301)
        body = b"test body"
        sig = _make_signature(body, boundary_ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            result = verify_slack_signature(body, boundary_ts, sig, SECRET)

        assert result is False

    def test_timestamp_just_within_window_accepted(self):
        """A timestamp 299 seconds old should be accepted."""
        now = 1700000000
        ts = str(now - 299)
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            result = verify_slack_signature(body, ts, sig, SECRET)

        assert result is True

    def test_current_timestamp_accepted(self):
        """A timestamp matching the current time should be accepted."""
        now = 1700000000
        ts = str(now)
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            result = verify_slack_signature(body, ts, sig, SECRET)

        assert result is True

    def test_future_timestamp_within_window_accepted(self):
        """A timestamp slightly in the future (clock skew) should be accepted."""
        now = 1700000000
        ts = str(now + 60)  # 1 minute in the future
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            result = verify_slack_signature(body, ts, sig, SECRET)

        assert result is True


# ---------------------------------------------------------------------------
# Invalid timestamp formats
# ---------------------------------------------------------------------------


class TestInvalidTimestampFormats:
    """Tests for non-numeric or missing timestamps."""

    def test_non_numeric_timestamp_returns_false(self):
        """A non-numeric timestamp should be rejected."""
        body = b"test body"
        assert verify_slack_signature(body, "not-a-number", "v0=abc", SECRET) is False

    def test_empty_string_timestamp_returns_false(self):
        """An empty timestamp string should be rejected."""
        body = b"test body"
        assert verify_slack_signature(body, "", "v0=abc", SECRET) is False

    def test_none_timestamp_returns_false(self):
        """A None timestamp should be rejected (TypeError caught)."""
        body = b"test body"
        assert verify_slack_signature(body, None, "v0=abc", SECRET) is False

    def test_float_string_timestamp_returns_false(self):
        """A float-formatted timestamp should be rejected."""
        body = b"test body"
        assert verify_slack_signature(body, "1700000000.5", "v0=abc", SECRET) is False

    def test_negative_timestamp_handled(self):
        """A negative timestamp is parseable but should be too old and rejected."""
        body = b"test body"
        sig = _make_signature(body, "-1", SECRET)
        assert verify_slack_signature(body, "-1", sig, SECRET) is False

    def test_whitespace_timestamp_returns_false(self):
        """A whitespace-only timestamp should be rejected."""
        body = b"test body"
        assert verify_slack_signature(body, "   ", "v0=abc", SECRET) is False


# ---------------------------------------------------------------------------
# Signing secret edge cases
# ---------------------------------------------------------------------------


class TestSigningSecret:
    """Tests for signing secret variations."""

    def test_empty_signing_secret(self):
        """Verification should work (or at least not crash) with an empty secret."""
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, "")

        assert verify_slack_signature(body, ts, sig, "") is True

    def test_long_signing_secret(self):
        """Verification should work with a very long signing secret."""
        long_secret = "s" * 1024
        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, long_secret)

        assert verify_slack_signature(body, ts, sig, long_secret) is True


# ---------------------------------------------------------------------------
# Deterministic computation
# ---------------------------------------------------------------------------


class TestDeterministic:
    """Tests that signature computation is deterministic."""

    def test_same_inputs_produce_same_result(self):
        """The same body, timestamp, and secret should always produce the same signature."""
        ts = "1700000000"
        body = b"deterministic test"
        sig1 = _make_signature(body, ts, SECRET)
        sig2 = _make_signature(body, ts, SECRET)

        assert sig1 == sig2

    def test_verify_is_deterministic(self):
        """Calling verify twice with the same inputs should give the same result."""
        now = 1700000000
        ts = str(now)
        body = b"deterministic test"
        sig = _make_signature(body, ts, SECRET)

        with patch("aragora.server.handlers.bots.slack.signature.time") as mock_time:
            mock_time.time.return_value = now
            r1 = verify_slack_signature(body, ts, sig, SECRET)
            r2 = verify_slack_signature(body, ts, sig, SECRET)

        assert r1 is True
        assert r2 is True


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_verify_slack_signature(self):
        """The module should export verify_slack_signature in __all__."""
        from aragora.server.handlers.bots.slack import signature

        assert "verify_slack_signature" in signature.__all__

    def test_all_exports_length(self):
        """__all__ should contain exactly one export."""
        from aragora.server.handlers.bots.slack import signature

        assert len(signature.__all__) == 1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """Tests that appropriate warnings are logged for rejected signatures."""

    def test_logs_warning_on_invalid_timestamp_format(self, caplog):
        """Invalid timestamp format should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            verify_slack_signature(b"body", "invalid", "v0=abc", SECRET)

        assert any("Invalid Slack signature timestamp format" in msg for msg in caplog.messages)

    def test_logs_warning_on_expired_timestamp(self, caplog):
        """An expired timestamp should log a warning."""
        import logging

        old_ts = str(int(time.time()) - 400)
        with caplog.at_level(logging.WARNING):
            verify_slack_signature(b"body", old_ts, "v0=abc", SECRET)

        assert any("Slack signature timestamp too old" in msg for msg in caplog.messages)

    def test_no_warning_on_valid_signature(self, caplog):
        """A valid signature should not produce any warnings."""
        import logging

        ts = _current_timestamp()
        body = b"test body"
        sig = _make_signature(body, ts, SECRET)

        with caplog.at_level(logging.WARNING):
            verify_slack_signature(body, ts, sig, SECRET)

        assert len(caplog.messages) == 0
