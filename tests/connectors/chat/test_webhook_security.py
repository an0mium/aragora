"""
Tests for webhook security utilities.

Tests cover:
- WebhookVerificationError exception
- Environment detection (production vs development)
- Webhook verification requirement checks
- Unverified webhook bypass in development
- WebhookVerificationResult dataclass
- Verification logging
- Slack HMAC-SHA256 signature verification
- Replay attack prevention (timestamp validation)
- WebhookVerifier abstract base class
- HMACVerifier for configurable HMAC verification
- Ed25519Verifier for Discord-style verification
"""

import pytest
import time
import hashlib
import hmac
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestWebhookVerificationError:
    """Tests for WebhookVerificationError exception."""

    def test_init(self):
        """Should initialize with source and reason."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationError

        error = WebhookVerificationError(source="slack", reason="signing_secret not configured")

        assert error.source == "slack"
        assert error.reason == "signing_secret not configured"

    def test_message_format(self):
        """Should format message with source, reason, and instructions."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationError

        error = WebhookVerificationError(source="teams", reason="invalid signature")

        message = str(error)
        assert "teams" in message
        assert "invalid signature" in message
        assert "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS" in message

    def test_is_exception(self):
        """Should be an Exception subclass."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationError

        error = WebhookVerificationError(source="test", reason="test")

        assert isinstance(error, Exception)


class TestGetEnvironment:
    """Tests for get_environment function."""

    def test_default_production(self):
        """Should default to production environment (secure default)."""
        from aragora.connectors.chat.webhook_security import get_environment

        with patch.dict("os.environ", {}, clear=True):
            env = get_environment()

        assert env == "production"

    def test_reads_env_var(self):
        """Should read ARAGORA_ENV variable."""
        from aragora.connectors.chat.webhook_security import get_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            env = get_environment()

        assert env == "production"

    def test_lowercases_value(self):
        """Should lowercase environment value."""
        from aragora.connectors.chat.webhook_security import get_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "PRODUCTION"}):
            env = get_environment()

        assert env == "production"


class TestIsProductionEnvironment:
    """Tests for is_production_environment function."""

    def test_production(self):
        """Should detect production environment."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            assert is_production_environment() is True

    def test_prod(self):
        """Should detect prod as production."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "prod"}):
            assert is_production_environment() is True

    def test_staging(self):
        """Should detect staging as production."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "staging"}):
            assert is_production_environment() is True

    def test_stage(self):
        """Should detect stage as production."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "stage"}):
            assert is_production_environment() is True

    def test_development(self):
        """Should not detect development as production."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            assert is_production_environment() is False

    def test_default(self):
        """Should default to production (secure default)."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        with patch.dict("os.environ", {}, clear=True):
            assert is_production_environment() is True


class TestIsWebhookVerificationRequired:
    """Tests for is_webhook_verification_required function."""

    def test_required_in_production(self):
        """Should always require verification in production."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert is_webhook_verification_required() is True

    def test_required_by_default_in_dev(self):
        """Should require verification by default in development."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}, clear=True):
            assert is_webhook_verification_required() is True

    def test_not_required_with_bypass_in_dev(self):
        """Should not require verification when bypass enabled in dev."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert is_webhook_verification_required() is False

    def test_bypass_values(self):
        """Should accept multiple truthy values for bypass."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        for value in ["1", "true", "yes"]:
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": value,
                },
            ):
                assert is_webhook_verification_required() is False


class TestShouldAllowUnverified:
    """Tests for should_allow_unverified function."""

    def test_never_in_production(self):
        """Should never allow unverified in production."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert should_allow_unverified("slack") is False

    def test_never_in_staging(self):
        """Should never allow unverified in staging."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "staging",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert should_allow_unverified("teams") is False

    def test_not_by_default_in_dev(self):
        """Should not allow unverified by default in dev."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}, clear=True):
            assert should_allow_unverified("slack") is False

    def test_allowed_with_env_var(self):
        """Should allow unverified in dev with env var."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert should_allow_unverified("slack") is True

    def test_logs_warning_in_production(self):
        """Should log warning when attempted in production."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            with patch("aragora.connectors.chat.webhook_security.logger") as mock_logger:
                should_allow_unverified("slack")
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args[0][0]
                assert "slack" in call_args
                assert "ignored" in call_args.lower()

    def test_logs_warning_when_bypassed(self):
        """Should log warning when verification bypassed in dev."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            with patch("aragora.connectors.chat.webhook_security.logger") as mock_logger:
                should_allow_unverified("teams")
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args[0][0]
                assert "teams" in call_args
                assert "security risk" in call_args.lower()


class TestWebhookVerificationResult:
    """Tests for WebhookVerificationResult dataclass."""

    def test_init(self):
        """Should initialize with all fields."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationResult

        result = WebhookVerificationResult(
            verified=True,
            source="slack",
            method="hmac-sha256",
            error=None,
        )

        assert result.verified is True
        assert result.source == "slack"
        assert result.method == "hmac-sha256"
        assert result.error is None

    def test_bool_true(self):
        """Should be truthy when verified."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationResult

        result = WebhookVerificationResult(
            verified=True,
            source="slack",
            method="hmac-sha256",
        )

        assert bool(result) is True

    def test_bool_false(self):
        """Should be falsy when not verified."""
        from aragora.connectors.chat.webhook_security import WebhookVerificationResult

        result = WebhookVerificationResult(
            verified=False,
            source="slack",
            method="hmac-sha256",
            error="Signature mismatch",
        )

        assert bool(result) is False


class TestLogVerificationAttempt:
    """Tests for log_verification_attempt function."""

    def test_returns_result(self):
        """Should return WebhookVerificationResult."""
        from aragora.connectors.chat.webhook_security import (
            log_verification_attempt,
            WebhookVerificationResult,
        )

        result = log_verification_attempt(
            source="slack",
            success=True,
            method="hmac-sha256",
        )

        assert isinstance(result, WebhookVerificationResult)
        assert result.verified is True
        assert result.source == "slack"
        assert result.method == "hmac-sha256"

    def test_logs_success(self):
        """Should log debug message on success."""
        from aragora.connectors.chat.webhook_security import log_verification_attempt

        with patch("aragora.connectors.chat.webhook_security.logger") as mock_logger:
            log_verification_attempt(
                source="slack",
                success=True,
                method="hmac-sha256",
            )
            mock_logger.debug.assert_called_once()

    def test_logs_failure(self):
        """Should log warning on failure."""
        from aragora.connectors.chat.webhook_security import log_verification_attempt

        with patch("aragora.connectors.chat.webhook_security.logger") as mock_logger:
            log_verification_attempt(
                source="slack",
                success=False,
                method="hmac-sha256",
                error="Signature mismatch",
            )
            mock_logger.warning.assert_called_once()


class TestVerifySlackSignature:
    """Tests for verify_slack_signature function."""

    def _generate_valid_signature(self, timestamp: str, body: str, secret: str) -> str:
        """Helper to generate valid Slack signature."""
        sig_basestring = f"v0:{timestamp}:{body}"
        expected_sig = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )
        return expected_sig

    def test_valid_signature(self):
        """Should verify valid signature."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        timestamp = str(int(time.time()))
        body = '{"type":"message"}'
        secret = "test_secret_123"
        signature = self._generate_valid_signature(timestamp, body, secret)

        result = verify_slack_signature(
            timestamp=timestamp,
            body=body,
            signature=signature,
            signing_secret=secret,
        )

        assert result.verified is True
        assert result.method == "hmac-sha256"

    def test_invalid_signature(self):
        """Should reject invalid signature."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        timestamp = str(int(time.time()))
        body = '{"type":"message"}'
        secret = "test_secret_123"

        result = verify_slack_signature(
            timestamp=timestamp,
            body=body,
            signature="v0=invalid_signature",
            signing_secret=secret,
        )

        assert result.verified is False
        assert "mismatch" in result.error.lower()

    def test_missing_timestamp(self):
        """Should reject missing timestamp."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        result = verify_slack_signature(
            timestamp="",
            body="body",
            signature="v0=abc",
            signing_secret="secret",
        )

        assert result.verified is False
        assert "Missing" in result.error

    def test_missing_signature(self):
        """Should reject missing signature."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        result = verify_slack_signature(
            timestamp=str(int(time.time())),
            body="body",
            signature="",
            signing_secret="secret",
        )

        assert result.verified is False
        assert "Missing" in result.error

    def test_missing_secret_in_production(self):
        """Should reject missing secret in production."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            result = verify_slack_signature(
                timestamp=str(int(time.time())),
                body="body",
                signature="v0=abc",
                signing_secret="",
            )

        assert result.verified is False
        assert "not configured" in result.error

    def test_missing_secret_bypassed_in_dev(self):
        """Should bypass missing secret in dev with flag."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            result = verify_slack_signature(
                timestamp=str(int(time.time())),
                body="body",
                signature="v0=abc",
                signing_secret="",
            )

        assert result.verified is True
        assert result.method == "bypassed"

    def test_old_timestamp(self):
        """Should reject old timestamp (replay attack protection)."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        # Timestamp from 10 minutes ago
        old_timestamp = str(int(time.time()) - 600)
        body = '{"type":"message"}'
        secret = "test_secret"
        signature = self._generate_valid_signature(old_timestamp, body, secret)

        result = verify_slack_signature(
            timestamp=old_timestamp,
            body=body,
            signature=signature,
            signing_secret=secret,
        )

        assert result.verified is False
        assert "too old" in result.error.lower()

    def test_invalid_timestamp_format(self):
        """Should reject invalid timestamp format."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        result = verify_slack_signature(
            timestamp="not_a_number",
            body="body",
            signature="v0=abc",
            signing_secret="secret",
        )

        assert result.verified is False
        assert "Invalid timestamp" in result.error

    def test_bytes_body(self):
        """Should handle bytes body."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        timestamp = str(int(time.time()))
        body = b'{"type":"message"}'
        secret = "test_secret_123"
        signature = self._generate_valid_signature(timestamp, body.decode(), secret)

        result = verify_slack_signature(
            timestamp=timestamp,
            body=body,
            signature=signature,
            signing_secret=secret,
        )

        assert result.verified is True


class TestHMACVerifier:
    """Tests for HMACVerifier class."""

    def test_init(self):
        """Should initialize with all parameters."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        verifier = HMACVerifier(
            secret="test_secret",
            source="slack",
            algorithm="sha256",
            signature_header="X-Slack-Signature",
            timestamp_header="X-Slack-Request-Timestamp",
            signature_prefix="v0=",
            body_template="v0:{timestamp}:{body}",
            max_timestamp_age=300,
        )

        assert verifier.secret == "test_secret"
        assert verifier.source == "slack"
        assert verifier.algorithm == "sha256"

    def test_verify_slack_style(self):
        """Should verify Slack-style signature."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        secret = "test_secret"
        timestamp = str(int(time.time()))
        body = b'{"type":"message"}'

        # Generate signature
        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        expected_sig = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        verifier = HMACVerifier(
            secret=secret,
            source="slack",
            algorithm="sha256",
            signature_header="X-Slack-Signature",
            timestamp_header="X-Slack-Request-Timestamp",
            signature_prefix="v0=",
            body_template="v0:{timestamp}:{body}",
        )

        result = verifier.verify(
            headers={
                "X-Slack-Signature": expected_sig,
                "X-Slack-Request-Timestamp": timestamp,
            },
            body=body,
        )

        assert result.verified is True

    def test_verify_whatsapp_style(self):
        """Should verify WhatsApp-style signature."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        secret = "app_secret"
        body = b'{"entry":[{"id":"123"}]}'

        # Generate signature
        expected_sig = (
            "sha256="
            + hmac.new(
                secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
        )

        verifier = HMACVerifier(
            secret=secret,
            source="whatsapp",
            algorithm="sha256",
            signature_header="X-Hub-Signature-256",
            signature_prefix="sha256=",
        )

        result = verifier.verify(
            headers={"X-Hub-Signature-256": expected_sig},
            body=body,
        )

        assert result.verified is True

    def test_missing_secret_production(self):
        """Should fail when secret missing in production."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            verifier = HMACVerifier(
                secret="",
                source="slack",
            )

            result = verifier.verify(
                headers={"X-Signature": "abc"},
                body=b"body",
            )

        assert result.verified is False
        assert "not configured" in result.error

    def test_missing_signature_header(self):
        """Should fail when signature header missing."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        verifier = HMACVerifier(
            secret="secret",
            source="test",
            signature_header="X-Signature",
        )

        result = verifier.verify(
            headers={},
            body=b"body",
        )

        assert result.verified is False
        assert "Missing" in result.error

    def test_case_insensitive_header_lookup(self):
        """Should perform case-insensitive header lookup."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        secret = "test"
        body = b"test"
        expected_sig = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        verifier = HMACVerifier(
            secret=secret,
            source="test",
            signature_header="X-Signature",
        )

        result = verifier.verify(
            headers={"x-signature": expected_sig},  # lowercase
            body=body,
        )

        assert result.verified is True

    def test_timestamp_validation(self):
        """Should validate timestamp age."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        old_timestamp = str(int(time.time()) - 400)

        verifier = HMACVerifier(
            secret="secret",
            source="test",
            signature_header="X-Sig",
            timestamp_header="X-Time",
            max_timestamp_age=300,
        )

        result = verifier.verify(
            headers={
                "X-Sig": "abc",
                "X-Time": old_timestamp,
            },
            body=b"body",
        )

        assert result.verified is False
        assert "too old" in result.error.lower()


class TestEd25519Verifier:
    """Tests for Ed25519Verifier class."""

    def test_init(self):
        """Should initialize with public key."""
        from aragora.connectors.chat.webhook_security import Ed25519Verifier

        verifier = Ed25519Verifier(
            public_key="abc123",
            source="discord",
        )

        assert verifier.public_key == "abc123"
        assert verifier.source == "discord"

    def test_missing_nacl_production(self):
        """Should fail when PyNaCl not available in production."""
        from aragora.connectors.chat.webhook_security import Ed25519Verifier

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            verifier = Ed25519Verifier(
                public_key="abc",
                source="discord",
            )

            # Mock import failure
            with patch.dict("sys.modules", {"nacl.signing": None}):
                import sys

                original_import = (
                    __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
                )

                def mock_import(name, *args, **kwargs):
                    if "nacl" in name:
                        raise ImportError("No module named 'nacl'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", mock_import):
                    result = verifier.verify(
                        headers={
                            "X-Signature-Ed25519": "abc",
                            "X-Signature-Timestamp": "123",
                        },
                        body=b"body",
                    )

        # Should fail since PyNaCl is not available
        assert result.verified is False

    def test_missing_public_key_production(self):
        """Should fail when public key missing in production.

        Note: If PyNaCl is not installed, the error will be about PyNaCl
        since that check comes first in the implementation.
        """
        from aragora.connectors.chat.webhook_security import Ed25519Verifier

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            verifier = Ed25519Verifier(
                public_key="",
                source="discord",
            )

            result = verifier.verify(
                headers={
                    "X-Signature-Ed25519": "abc",
                    "X-Signature-Timestamp": "123",
                },
                body=b"body",
            )

        assert result.verified is False
        # Either PyNaCl not installed OR public key not configured
        assert "not configured" in result.error or "PyNaCl" in result.error

    def test_missing_headers(self):
        """Should fail when required headers missing.

        Note: If PyNaCl is not installed, the error will be about PyNaCl
        since that check comes first in the implementation.
        """
        from aragora.connectors.chat.webhook_security import Ed25519Verifier

        verifier = Ed25519Verifier(
            public_key="abc",
            source="discord",
        )

        result = verifier.verify(
            headers={},
            body=b"body",
        )

        assert result.verified is False
        # Either missing headers OR PyNaCl not installed
        assert "Missing" in result.error or "PyNaCl" in result.error


class TestWebhookVerifierAbstract:
    """Tests for WebhookVerifier abstract base class."""

    def test_get_header_exact_match(self):
        """Should get header with exact match."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        verifier = HMACVerifier(secret="test", source="test")

        result = verifier._get_header(
            {"X-Signature": "value"},
            "X-Signature",
        )

        assert result == "value"

    def test_get_header_case_insensitive(self):
        """Should get header case-insensitively."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        verifier = HMACVerifier(secret="test", source="test")

        result = verifier._get_header(
            {"x-signature": "value"},
            "X-Signature",
        )

        assert result == "value"

    def test_get_header_missing(self):
        """Should return empty string for missing header."""
        from aragora.connectors.chat.webhook_security import HMACVerifier

        verifier = HMACVerifier(secret="test", source="test")

        result = verifier._get_header(
            {"Other-Header": "value"},
            "X-Signature",
        )

        assert result == ""


class TestAllExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Should export all documented items."""
        from aragora.connectors.chat import webhook_security

        expected = [
            "WebhookVerificationError",
            "WebhookVerificationResult",
            "is_webhook_verification_required",
            "should_allow_unverified",
            "is_production_environment",
            "log_verification_attempt",
            "verify_slack_signature",
            "WebhookVerifier",
            "HMACVerifier",
            "Ed25519Verifier",
        ]

        for name in expected:
            assert hasattr(webhook_security, name), f"Missing export: {name}"
