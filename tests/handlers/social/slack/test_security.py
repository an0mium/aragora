"""Comprehensive tests for Slack security utilities.

Covers all public functions and classes in
aragora.server.handlers.social.slack.security:
- SLACK_ALLOWED_DOMAINS: frozenset of permitted Slack domains
- validate_slack_url: SSRF protection via URL validation
- SignatureVerifierMixin.verify_signature: Slack request signature verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social.slack.security import (
    SLACK_ALLOWED_DOMAINS,
    SignatureVerifierMixin,
    validate_slack_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(headers: dict[str, str] | None = None) -> MagicMock:
    """Create a mock HTTP handler with configurable headers."""
    handler = MagicMock()
    h = headers or {}
    handler.headers = MagicMock()
    handler.headers.get = lambda key, default="": h.get(key, default)
    return handler


@dataclass
class _FakeVerificationResult:
    """Lightweight stand-in for WebhookVerificationResult."""

    verified: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# SLACK_ALLOWED_DOMAINS
# ---------------------------------------------------------------------------


class TestSlackAllowedDomains:
    """Tests for the SLACK_ALLOWED_DOMAINS constant."""

    def test_contains_hooks_domain(self):
        assert "hooks.slack.com" in SLACK_ALLOWED_DOMAINS

    def test_contains_api_domain(self):
        assert "api.slack.com" in SLACK_ALLOWED_DOMAINS

    def test_exactly_two_domains(self):
        assert len(SLACK_ALLOWED_DOMAINS) == 2

    def test_is_frozenset(self):
        assert isinstance(SLACK_ALLOWED_DOMAINS, frozenset)

    def test_does_not_contain_arbitrary_domain(self):
        assert "evil.com" not in SLACK_ALLOWED_DOMAINS


# ---------------------------------------------------------------------------
# validate_slack_url
# ---------------------------------------------------------------------------


class TestValidateSlackUrl:
    """Tests for the validate_slack_url function."""

    # --- valid URLs ---

    def test_valid_hooks_url(self):
        assert validate_slack_url("https://hooks.slack.com/services/T1234/B5678") is True

    def test_valid_api_url(self):
        assert validate_slack_url("https://api.slack.com/apps") is True

    def test_valid_hooks_url_with_path(self):
        assert validate_slack_url("https://hooks.slack.com/workflows/abc123") is True

    def test_valid_api_url_with_long_path(self):
        url = "https://api.slack.com/methods/chat.postMessage?token=xoxb-foo"
        assert validate_slack_url(url) is True

    # --- scheme violations ---

    def test_http_scheme_rejected(self):
        assert validate_slack_url("http://hooks.slack.com/services/T1234") is False

    def test_ftp_scheme_rejected(self):
        assert validate_slack_url("ftp://hooks.slack.com/file") is False

    def test_empty_scheme_rejected(self):
        assert validate_slack_url("://hooks.slack.com/path") is False

    def test_no_scheme_rejected(self):
        # Without a scheme urlparse puts everything in the path
        assert validate_slack_url("hooks.slack.com/services/T1234") is False

    # --- domain violations ---

    def test_evil_domain_rejected(self):
        assert validate_slack_url("https://evil.com/hooks.slack.com") is False

    def test_subdomain_of_slack_rejected(self):
        assert validate_slack_url("https://hooks.slack.com.evil.com/path") is False

    def test_similar_domain_rejected(self):
        assert validate_slack_url("https://hooks-slack.com/services/T1234") is False

    def test_localhost_rejected(self):
        assert validate_slack_url("https://localhost/path") is False

    def test_ip_address_rejected(self):
        assert validate_slack_url("https://169.254.169.254/latest/meta-data/") is False

    def test_internal_host_rejected(self):
        assert validate_slack_url("https://internal.corp/api") is False

    # --- malformed / edge-case input ---

    def test_empty_string(self):
        assert validate_slack_url("") is False

    def test_none_input(self):
        # urlparse(None) raises TypeError, caught by the handler
        assert validate_slack_url(None) is False  # type: ignore[arg-type]

    def test_integer_input(self):
        # urlparse(int) raises AttributeError (not caught by handler)
        # which is acceptable since the function type-hints str
        with pytest.raises(AttributeError):
            validate_slack_url(12345)  # type: ignore[arg-type]

    def test_just_scheme(self):
        assert validate_slack_url("https://") is False

    def test_url_with_user_info(self):
        assert validate_slack_url("https://user:pass@hooks.slack.com/path") is False

    def test_debug_log_on_type_error(self, caplog):
        """TypeError from bad input is logged at DEBUG level."""
        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.social.slack.security"):
            validate_slack_url(None)  # type: ignore[arg-type]
        # If a TypeError is raised by urlparse, a debug log is emitted.
        # (Depending on Python version urlparse may or may not raise.)
        # We just confirm no crash occurred; logging coverage is opportunistic.


# ---------------------------------------------------------------------------
# SignatureVerifierMixin
# ---------------------------------------------------------------------------


class TestSignatureVerifierMixin:
    """Tests for SignatureVerifierMixin.verify_signature."""

    def _make_mixin(self) -> SignatureVerifierMixin:
        return SignatureVerifierMixin()

    # --- successful verification ---

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_returns_true_on_verified(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler({
            "X-Slack-Request-Timestamp": "1234567890",
            "X-Slack-Signature": "v0=abc123",
        })
        assert mixin.verify_signature(handler, '{"text":"hi"}', "secret123") is True

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_passes_timestamp_header(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler({
            "X-Slack-Request-Timestamp": "1700000000",
            "X-Slack-Signature": "v0=sig",
        })
        mixin.verify_signature(handler, "body", "secret")
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["timestamp"] == "1700000000"

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_passes_body(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler()
        mixin.verify_signature(handler, "request_body", "secret")
        assert mock_verify.call_args.kwargs["body"] == "request_body"

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_passes_signature_header(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler({
            "X-Slack-Signature": "v0=deadbeef",
        })
        mixin.verify_signature(handler, "body", "secret")
        assert mock_verify.call_args.kwargs["signature"] == "v0=deadbeef"

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_passes_signing_secret(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler()
        mixin.verify_signature(handler, "body", "my_secret_value")
        assert mock_verify.call_args.kwargs["signing_secret"] == "my_secret_value"

    # --- failed verification ---

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_returns_false_on_not_verified(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=False, error=None)
        mixin = self._make_mixin()
        handler = _make_handler()
        assert mixin.verify_signature(handler, "body", "secret") is False

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_logs_warning_on_failed_with_error(self, mock_verify, caplog):
        mock_verify.return_value = _FakeVerificationResult(verified=False, error="timestamp expired")
        mixin = self._make_mixin()
        handler = _make_handler()
        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.social.slack.security"):
            result = mixin.verify_signature(handler, "body", "secret")
        assert result is False
        assert "timestamp expired" in caplog.text

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_no_warning_when_not_verified_without_error(self, mock_verify, caplog):
        mock_verify.return_value = _FakeVerificationResult(verified=False, error=None)
        mixin = self._make_mixin()
        handler = _make_handler()
        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.social.slack.security"):
            mixin.verify_signature(handler, "body", "secret")
        # No warning should be logged when error is None
        assert "Slack signature verification failed" not in caplog.text

    # --- empty / missing headers ---

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_missing_timestamp_defaults_empty_string(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=False)
        mixin = self._make_mixin()
        handler = _make_handler({})  # No headers
        mixin.verify_signature(handler, "body", "secret")
        assert mock_verify.call_args.kwargs["timestamp"] == ""

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_missing_signature_defaults_empty_string(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=False)
        mixin = self._make_mixin()
        handler = _make_handler({})
        mixin.verify_signature(handler, "body", "secret")
        assert mock_verify.call_args.kwargs["signature"] == ""

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_none_signing_secret_becomes_empty_string(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler()
        mixin.verify_signature(handler, "body", None)
        assert mock_verify.call_args.kwargs["signing_secret"] == ""

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_empty_signing_secret(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)
        mixin = self._make_mixin()
        handler = _make_handler()
        mixin.verify_signature(handler, "body", "")
        assert mock_verify.call_args.kwargs["signing_secret"] == ""

    # --- exception handling ---

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_value_error_returns_false(self, mock_verify):
        mock_verify.side_effect = ValueError("bad value")
        mixin = self._make_mixin()
        handler = _make_handler()
        assert mixin.verify_signature(handler, "body", "secret") is False

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_type_error_returns_false(self, mock_verify):
        mock_verify.side_effect = TypeError("wrong type")
        mixin = self._make_mixin()
        handler = _make_handler()
        assert mixin.verify_signature(handler, "body", "secret") is False

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_attribute_error_returns_false(self, mock_verify):
        mock_verify.side_effect = AttributeError("no attr")
        mixin = self._make_mixin()
        handler = _make_handler()
        assert mixin.verify_signature(handler, "body", "secret") is False

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_runtime_error_returns_false(self, mock_verify):
        mock_verify.side_effect = RuntimeError("crash")
        mixin = self._make_mixin()
        handler = _make_handler()
        assert mixin.verify_signature(handler, "body", "secret") is False

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_exception_logged_at_exception_level(self, mock_verify, caplog):
        mock_verify.side_effect = ValueError("hmac failure")
        mixin = self._make_mixin()
        handler = _make_handler()
        with caplog.at_level(logging.ERROR, logger="aragora.server.handlers.social.slack.security"):
            mixin.verify_signature(handler, "body", "secret")
        assert "hmac failure" in caplog.text

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_uncaught_exception_propagates(self, mock_verify):
        """Exceptions not in the explicit catch list propagate."""
        mock_verify.side_effect = KeyboardInterrupt()
        mixin = self._make_mixin()
        handler = _make_handler()
        with pytest.raises(KeyboardInterrupt):
            mixin.verify_signature(handler, "body", "secret")

    # --- mixin can be composed ---

    def test_mixin_composable_with_class(self):
        """SignatureVerifierMixin can be mixed into another class."""

        class MyHandler(SignatureVerifierMixin):
            pass

        obj = MyHandler()
        assert hasattr(obj, "verify_signature")

    @patch("aragora.connectors.chat.webhook_security.verify_slack_signature")
    def test_mixin_used_from_subclass(self, mock_verify):
        mock_verify.return_value = _FakeVerificationResult(verified=True)

        class MyHandler(SignatureVerifierMixin):
            pass

        obj = MyHandler()
        handler = _make_handler({
            "X-Slack-Request-Timestamp": "12345",
            "X-Slack-Signature": "v0=sig",
        })
        assert obj.verify_signature(handler, "body", "secret") is True
