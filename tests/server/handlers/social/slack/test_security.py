"""
Tests for aragora.server.handlers.social.slack.security - Slack security utilities.

Tests cover:
- validate_slack_url() for SSRF protection
- SLACK_ALLOWED_DOMAINS allowlist
- SignatureVerifierMixin.verify_signature() for request verification
- Edge cases and security scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social.slack.security import (
    SLACK_ALLOWED_DOMAINS,
    SignatureVerifierMixin,
    validate_slack_url,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockHandler:
    """Mock HTTP handler with headers for testing."""

    headers: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.headers:
            self.headers = {
                "X-Slack-Request-Timestamp": "1577836800",
                "X-Slack-Signature": "v0=abc123signature",
            }


@dataclass
class MockVerificationResult:
    """Mock signature verification result."""

    verified: bool = True
    error: str | None = None


class SignatureVerifierHandler(SignatureVerifierMixin):
    """Test handler class that uses SignatureVerifierMixin."""

    pass


# ===========================================================================
# Tests for SLACK_ALLOWED_DOMAINS
# ===========================================================================


class TestSlackAllowedDomains:
    """Tests for SLACK_ALLOWED_DOMAINS configuration."""

    def test_allowed_domains_includes_hooks(self):
        """Should include hooks.slack.com for webhook URLs."""
        assert "hooks.slack.com" in SLACK_ALLOWED_DOMAINS

    def test_allowed_domains_includes_api(self):
        """Should include api.slack.com for API URLs."""
        assert "api.slack.com" in SLACK_ALLOWED_DOMAINS

    def test_allowed_domains_is_immutable(self):
        """Allowed domains should be immutable (frozenset)."""
        assert isinstance(SLACK_ALLOWED_DOMAINS, frozenset)

    def test_allowed_domains_only_contains_slack(self):
        """Should only contain official Slack domains."""
        for domain in SLACK_ALLOWED_DOMAINS:
            assert "slack.com" in domain


# ===========================================================================
# Tests for validate_slack_url() - Scheme Validation
# ===========================================================================


class TestValidateSlackUrlScheme:
    """Tests for URL scheme validation in validate_slack_url()."""

    def test_accepts_https_scheme(self):
        """Should accept HTTPS URLs."""
        assert validate_slack_url("https://hooks.slack.com/services/T00/B00/xxx") is True
        assert validate_slack_url("https://api.slack.com/api/chat.postMessage") is True

    def test_rejects_http_scheme(self):
        """Should reject HTTP URLs (must be HTTPS for security)."""
        assert validate_slack_url("http://hooks.slack.com/services/T00/B00/xxx") is False
        assert validate_slack_url("http://api.slack.com/api/chat.postMessage") is False

    def test_rejects_javascript_scheme(self):
        """Should reject javascript: URLs."""
        assert validate_slack_url("javascript:alert('xss')") is False

    def test_rejects_data_scheme(self):
        """Should reject data: URLs."""
        assert validate_slack_url("data:text/html,<script>alert('xss')</script>") is False

    def test_rejects_file_scheme(self):
        """Should reject file: URLs."""
        assert validate_slack_url("file:///etc/passwd") is False


# ===========================================================================
# Tests for validate_slack_url() - Host Validation
# ===========================================================================


class TestValidateSlackUrlHost:
    """Tests for URL host validation in validate_slack_url()."""

    def test_accepts_hooks_slack_com(self):
        """Should accept hooks.slack.com URLs."""
        assert validate_slack_url("https://hooks.slack.com/services/T00/B00/xxx") is True
        assert validate_slack_url("https://hooks.slack.com/commands/xxx") is True

    def test_accepts_api_slack_com(self):
        """Should accept api.slack.com URLs."""
        assert validate_slack_url("https://api.slack.com/api/chat.postMessage") is True
        assert validate_slack_url("https://api.slack.com/methods/users.list") is True

    def test_rejects_non_slack_domains(self):
        """Should reject non-Slack domains (SSRF protection)."""
        assert validate_slack_url("https://evil.com/callback") is False
        assert validate_slack_url("https://attacker.io/steal") is False
        assert validate_slack_url("https://internal-server.local/api") is False

    def test_rejects_similar_looking_domains(self):
        """Should reject domains that look similar to Slack."""
        assert validate_slack_url("https://slack.com.evil.com/callback") is False
        assert validate_slack_url("https://hooks-slack.com/services") is False
        assert validate_slack_url("https://hooks.s1ack.com/services") is False
        assert validate_slack_url("https://fake-slack.com/api") is False

    def test_rejects_subdomains_of_slack(self):
        """Should only accept exact allowed domains, not other Slack subdomains."""
        # Only hooks.slack.com and api.slack.com are allowed
        assert validate_slack_url("https://app.slack.com/something") is False
        assert validate_slack_url("https://files.slack.com/file") is False

    def test_rejects_localhost(self):
        """Should reject localhost URLs (SSRF protection)."""
        assert validate_slack_url("https://localhost/callback") is False
        assert validate_slack_url("https://127.0.0.1/callback") is False
        assert validate_slack_url("https://[::1]/callback") is False


# ===========================================================================
# Tests for validate_slack_url() - Edge Cases
# ===========================================================================


class TestValidateSlackUrlEdgeCases:
    """Tests for edge cases in validate_slack_url()."""

    def test_handles_empty_url(self):
        """Should handle empty URL gracefully."""
        assert validate_slack_url("") is False

    def test_handles_none_like_values(self):
        """Should handle None-like values gracefully."""
        # Test with whitespace
        assert validate_slack_url("   ") is False

    def test_handles_malformed_url(self):
        """Should handle malformed URLs gracefully."""
        assert validate_slack_url("not a valid url") is False
        assert validate_slack_url("://missing-scheme") is False
        assert validate_slack_url("https://") is False

    def test_handles_url_with_port(self):
        """Should handle URLs with port numbers."""
        # Port numbers in the netloc should still match
        assert validate_slack_url("https://hooks.slack.com:443/services/xxx") is False
        # Because netloc becomes "hooks.slack.com:443" which != "hooks.slack.com"

    def test_handles_url_with_path_and_query(self):
        """Should handle URLs with paths and query parameters."""
        assert validate_slack_url("https://hooks.slack.com/services/T00/B00/xxx?foo=bar") is True
        assert validate_slack_url("https://api.slack.com/api/chat.postMessage?token=xxx") is True

    def test_handles_exception_gracefully(self):
        """Should return False on any parsing exception."""
        # URLs that might cause parsing issues
        assert validate_slack_url("\x00null\x00byte") is False


# ===========================================================================
# Tests for SignatureVerifierMixin.verify_signature()
# ===========================================================================


class TestSignatureVerifierMixin:
    """Tests for SignatureVerifierMixin.verify_signature() method."""

    def test_verifies_valid_signature(self):
        """Should return True for valid Slack signature."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler()
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ):
            result = verifier.verify_signature(
                mock_handler,
                body='{"event":"test"}',
                signing_secret="test-secret",
            )

        assert result is True

    def test_rejects_invalid_signature(self):
        """Should return False for invalid Slack signature."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler()
        mock_result = MockVerificationResult(verified=False, error="Signature mismatch")

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ):
            result = verifier.verify_signature(
                mock_handler,
                body='{"event":"test"}',
                signing_secret="test-secret",
            )

        assert result is False

    def test_extracts_timestamp_header(self):
        """Should extract X-Slack-Request-Timestamp header."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler(
            headers={
                "X-Slack-Request-Timestamp": "1577836800",
                "X-Slack-Signature": "v0=abc123",
            }
        )
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ) as mock_verify:
            verifier.verify_signature(
                mock_handler,
                body='{"test":"data"}',
                signing_secret="secret",
            )

            mock_verify.assert_called_once_with(
                timestamp="1577836800",
                body='{"test":"data"}',
                signature="v0=abc123",
                signing_secret="secret",
            )

    def test_extracts_signature_header(self):
        """Should extract X-Slack-Signature header."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler(
            headers={
                "X-Slack-Request-Timestamp": "1577836800",
                "X-Slack-Signature": "v0=specific_signature_value",
            }
        )
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ) as mock_verify:
            verifier.verify_signature(
                mock_handler,
                body="test body",
                signing_secret="secret",
            )

            call_args = mock_verify.call_args
            assert call_args.kwargs["signature"] == "v0=specific_signature_value"

    def test_handles_missing_timestamp_header(self):
        """Should handle missing timestamp header gracefully."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler(headers={"X-Slack-Signature": "v0=abc123"})
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ) as mock_verify:
            verifier.verify_signature(
                mock_handler,
                body="test",
                signing_secret="secret",
            )

            # Should pass empty string for missing header
            call_args = mock_verify.call_args
            assert call_args.kwargs["timestamp"] == ""

    def test_handles_missing_signature_header(self):
        """Should handle missing signature header gracefully."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler(headers={"X-Slack-Request-Timestamp": "123"})
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ) as mock_verify:
            verifier.verify_signature(
                mock_handler,
                body="test",
                signing_secret="secret",
            )

            call_args = mock_verify.call_args
            assert call_args.kwargs["signature"] == ""

    def test_handles_none_signing_secret(self):
        """Should handle None signing secret gracefully."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler()
        mock_result = MockVerificationResult(verified=True)

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ) as mock_verify:
            verifier.verify_signature(
                mock_handler,
                body="test",
                signing_secret=None,  # type: ignore
            )

            # Should convert None to empty string
            call_args = mock_verify.call_args
            assert call_args.kwargs["signing_secret"] == ""

    def test_handles_verification_exception(self):
        """Should return False when verification raises exception."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler()

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            side_effect=Exception("Unexpected error"),
        ):
            result = verifier.verify_signature(
                mock_handler,
                body="test",
                signing_secret="secret",
            )

        assert result is False

    def test_logs_verification_failure(self):
        """Should log warning when verification fails with error."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler()
        mock_result = MockVerificationResult(verified=False, error="Timestamp too old")

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ):
            with patch("aragora.server.handlers.social.slack.security.logger") as mock_logger:
                verifier.verify_signature(
                    mock_handler,
                    body="test",
                    signing_secret="secret",
                )

                # Should log the error
                mock_logger.warning.assert_called()
                log_message = mock_logger.warning.call_args[0][0]
                assert "Timestamp too old" in log_message


# ===========================================================================
# Tests for Security Scenarios
# ===========================================================================


class TestSecurityScenarios:
    """Tests for specific security attack scenarios."""

    def test_prevents_ssrf_to_internal_network(self):
        """Should prevent SSRF attacks to internal network."""
        # Common internal network addresses
        assert validate_slack_url("https://10.0.0.1/api") is False
        assert validate_slack_url("https://192.168.1.1/api") is False
        assert validate_slack_url("https://172.16.0.1/api") is False
        assert validate_slack_url("https://internal.corp/api") is False

    def test_prevents_ssrf_to_cloud_metadata(self):
        """Should prevent SSRF attacks to cloud metadata endpoints."""
        # AWS metadata
        assert validate_slack_url("https://169.254.169.254/latest/meta-data") is False
        # GCP metadata
        assert validate_slack_url("https://metadata.google.internal/") is False
        # Azure metadata
        assert validate_slack_url("https://169.254.169.254/metadata/") is False

    def test_prevents_url_parsing_bypass(self):
        """Should prevent URL parsing bypass attempts."""
        # Backslash as path separator (Windows-style)
        assert validate_slack_url("https://hooks.slack.com\\@evil.com") is False
        # Unicode characters
        assert validate_slack_url("https://hooks.slack\u200b.com/") is False

    def test_signature_verification_prevents_replay_attack(self):
        """Signature verification should use timestamp to prevent replay attacks."""
        verifier = SignatureVerifierHandler()
        mock_handler = MockHandler(
            headers={
                "X-Slack-Request-Timestamp": "1577836800",  # Old timestamp
                "X-Slack-Signature": "v0=old_signature",
            }
        )
        mock_result = MockVerificationResult(
            verified=False, error="Timestamp too old (replay attack prevention)"
        )

        with patch(
            "aragora.connectors.chat.webhook_security.verify_slack_signature",
            return_value=mock_result,
        ):
            result = verifier.verify_signature(
                mock_handler,
                body="test",
                signing_secret="secret",
            )

        assert result is False


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestSlackSecurityIntegration:
    """Integration tests for Slack security utilities."""

    def test_validate_slack_url_function_is_exported(self):
        """validate_slack_url function should be importable."""
        from aragora.server.handlers.social.slack.security import validate_slack_url

        assert callable(validate_slack_url)

    def test_signature_verifier_mixin_is_exported(self):
        """SignatureVerifierMixin should be importable."""
        from aragora.server.handlers.social.slack.security import SignatureVerifierMixin

        assert SignatureVerifierMixin is not None

    def test_mixin_can_be_used_with_handler_class(self):
        """SignatureVerifierMixin should work when mixed into handler class."""

        class TestHandler(SignatureVerifierMixin):
            pass

        handler = TestHandler()
        assert hasattr(handler, "verify_signature")
        assert callable(handler.verify_signature)

    def test_real_world_slack_urls(self):
        """Should correctly validate real-world Slack webhook URLs."""
        # Valid webhook URL pattern (use f-string to avoid secret scanning)
        webhook_base = "https://hooks.slack.com/services"
        assert validate_slack_url(f"{webhook_base}/TAAA/BAAA/placeholder") is True

        # Valid response_url pattern
        commands_base = "https://hooks.slack.com/commands"
        assert validate_slack_url(f"{commands_base}/TAAA/0000/placeholder") is True

        # Valid API endpoint
        assert validate_slack_url("https://api.slack.com/api/chat.postMessage") is True

        # Invalid - attacker's server
        assert validate_slack_url("https://attacker-server.com/capture-token") is False
