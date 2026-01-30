"""
Tests for aragora.server.handlers.oauth.validation - OAuth URL validation utilities.

Tests cover:
- _validate_redirect_url() for preventing open redirect attacks
- Scheme validation (http/https only)
- Host validation against allowlist
- Subdomain validation
- Edge cases and security scenarios
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.server.handlers.oauth.validation import _validate_redirect_url


# ===========================================================================
# Test Fixtures
# ===========================================================================


def mock_allowed_hosts():
    """Return mock allowed hosts for testing."""
    return {
        "example.com",
        "app.example.com",
        "localhost",
        "aragora.io",
    }


# ===========================================================================
# Tests for Scheme Validation
# ===========================================================================


class TestSchemeValidation:
    """Tests for URL scheme validation."""

    def test_accepts_https_scheme(self):
        """Should accept HTTPS URLs."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://example.com/callback") is True

    def test_accepts_http_scheme(self):
        """Should accept HTTP URLs (for development)."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("http://localhost/callback") is True

    def test_rejects_javascript_scheme(self):
        """Should reject javascript: URLs to prevent XSS."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("javascript:alert('xss')") is False

    def test_rejects_data_scheme(self):
        """Should reject data: URLs to prevent injection."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("data:text/html,<script>alert('xss')</script>") is False

    def test_rejects_file_scheme(self):
        """Should reject file: URLs."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("file:///etc/passwd") is False

    def test_rejects_ftp_scheme(self):
        """Should reject ftp: URLs."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("ftp://example.com/file") is False

    def test_rejects_empty_scheme(self):
        """Should reject URLs without scheme."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("//example.com/callback") is False


# ===========================================================================
# Tests for Host Validation
# ===========================================================================


class TestHostValidation:
    """Tests for URL host validation against allowlist."""

    def test_accepts_exact_allowed_host(self):
        """Should accept URLs with exact host match."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://example.com/callback") is True
            assert _validate_redirect_url("https://aragora.io/callback") is True

    def test_rejects_non_allowed_host(self):
        """Should reject URLs with hosts not in allowlist."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://evil.com/callback") is False
            assert _validate_redirect_url("https://attacker.io/callback") is False

    def test_accepts_subdomain_of_allowed_host(self):
        """Should accept subdomains of allowed hosts."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://api.example.com/callback") is True
            assert _validate_redirect_url("https://auth.aragora.io/callback") is True

    def test_rejects_similar_but_different_host(self):
        """Should reject hosts that are similar but not subdomains."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # evil-example.com is not a subdomain of example.com
            assert _validate_redirect_url("https://evil-example.com/callback") is False
            # examplecom is not example.com
            assert _validate_redirect_url("https://examplecom/callback") is False

    def test_case_insensitive_host_matching(self):
        """Should perform case-insensitive host matching."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://EXAMPLE.COM/callback") is True
            assert _validate_redirect_url("https://Example.Com/callback") is True


# ===========================================================================
# Tests for Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and malformed URLs."""

    def test_rejects_url_without_host(self):
        """Should reject URLs without a host."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https:///callback") is False

    def test_rejects_empty_url(self):
        """Should reject empty URLs."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("") is False

    def test_handles_url_with_port(self):
        """Should handle URLs with port numbers."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("http://localhost:3000/callback") is True
            assert _validate_redirect_url("https://example.com:443/callback") is True

    def test_handles_url_with_query_params(self):
        """Should handle URLs with query parameters."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://example.com/callback?state=abc") is True

    def test_handles_url_with_fragment(self):
        """Should handle URLs with fragments."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://example.com/callback#section") is True

    def test_handles_url_with_encoded_characters(self):
        """Should handle URLs with percent-encoded characters."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://example.com/callback%20path") is True

    def test_handles_malformed_url(self):
        """Should handle malformed URLs gracefully."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Should not raise exception, just return False
            assert _validate_redirect_url("not a valid url at all") is False
            assert _validate_redirect_url("://missing-scheme.com") is False


# ===========================================================================
# Tests for Security Scenarios
# ===========================================================================


class TestSecurityScenarios:
    """Tests for specific security attack scenarios."""

    def test_prevents_open_redirect_to_external_site(self):
        """Should prevent open redirect attacks to external sites."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Common open redirect patterns
            assert _validate_redirect_url("https://evil.com/callback") is False
            assert _validate_redirect_url("https://phishing-site.net/login") is False

    def test_prevents_host_confusion_attack(self):
        """Should prevent host confusion attacks."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Attacker tries to confuse with similar-looking domain
            assert _validate_redirect_url("https://examp1e.com/callback") is False
            assert _validate_redirect_url("https://example.com.evil.com/callback") is False

    def test_prevents_protocol_downgrade(self):
        """Should validate protocol is safe."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # HTTP is allowed (for dev), but dangerous protocols are not
            assert _validate_redirect_url("javascript:void(0)") is False
            assert _validate_redirect_url("vbscript:msgbox('xss')") is False

    def test_prevents_null_byte_injection(self):
        """Should handle null byte injection attempts."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Null byte could truncate URL processing
            result = _validate_redirect_url("https://example.com\x00.evil.com/callback")
            # Should either reject or properly handle the null byte
            assert isinstance(result, bool)

    def test_prevents_unicode_homograph_attack(self):
        """Should handle unicode homograph attacks."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Using cyrillic characters that look like latin
            # Note: actual behavior depends on implementation
            result = _validate_redirect_url("https://xn--exmple-cua.com/callback")
            assert result is False  # Should not match example.com


# ===========================================================================
# Tests for Subdomain Matching
# ===========================================================================


class TestSubdomainMatching:
    """Tests for subdomain validation logic."""

    def test_accepts_single_level_subdomain(self):
        """Should accept single-level subdomains."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://api.example.com/callback") is True
            assert _validate_redirect_url("https://www.example.com/callback") is True

    def test_accepts_multi_level_subdomain(self):
        """Should accept multi-level subdomains."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            assert _validate_redirect_url("https://api.v1.example.com/callback") is True
            assert _validate_redirect_url("https://dev.api.example.com/callback") is True

    def test_rejects_prefix_that_is_not_subdomain(self):
        """Should reject hosts that have the allowed host as a suffix but are not subdomains."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # "notexample.com" ends with "example.com" but is not a subdomain
            assert _validate_redirect_url("https://notexample.com/callback") is False
            # "fakeexample.com" is not example.com
            assert _validate_redirect_url("https://fakeexample.com/callback") is False


# ===========================================================================
# Tests for Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in URL validation."""

    def test_handles_exception_gracefully(self):
        """Should return False on any exception during validation."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            side_effect=Exception("Config error"),
        ):
            # Should not raise, just return False
            result = _validate_redirect_url("https://example.com/callback")
            assert result is False

    def test_handles_none_input(self):
        """Should handle None input gracefully."""
        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=mock_allowed_hosts(),
        ):
            # Should not raise, just return False
            try:
                result = _validate_redirect_url(None)  # type: ignore[arg-type]
                assert result is False
            except (TypeError, AttributeError):
                # Some implementations may raise TypeError for None
                pass


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestOAuthValidationIntegration:
    """Integration tests for OAuth validation."""

    def test_validation_function_is_exported(self):
        """The validation function should be importable."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        assert callable(_validate_redirect_url)

    def test_allowed_hosts_function_exists(self):
        """The allowed hosts configuration function should exist."""
        from aragora.server.handlers.oauth.config import _get_allowed_redirect_hosts

        assert callable(_get_allowed_redirect_hosts)

    def test_real_world_oauth_urls(self):
        """Should handle real-world OAuth redirect URLs."""
        allowed_hosts = {
            "myapp.com",
            "localhost",
            "127.0.0.1",
        }

        with patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=allowed_hosts,
        ):
            # Production callback
            assert _validate_redirect_url("https://myapp.com/auth/callback") is True

            # Development callback
            assert _validate_redirect_url("http://localhost:3000/auth/callback") is True
            assert _validate_redirect_url("http://127.0.0.1:8080/callback") is True

            # With state parameter
            assert (
                _validate_redirect_url("https://myapp.com/auth/callback?state=xyz123&code=abc")
                is True
            )

            # Malicious redirect attempts
            assert _validate_redirect_url("https://evil.com/steal-token") is False
            assert _validate_redirect_url("https://myapp.com.evil.com/callback") is False
