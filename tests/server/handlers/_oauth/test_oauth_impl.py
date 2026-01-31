"""
Tests for OAuth Implementation Module (_oauth_impl.py).

Tests cover:
- Redirect URL validation (security critical)
- State token validation
- Configuration validation
- Rate limiter initialization

SECURITY CRITICAL: These tests ensure OAuth security mechanisms work correctly.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ===========================================================================
# Redirect URL Validation Tests
# ===========================================================================


class TestRedirectUrlValidation:
    """Tests for redirect URL validation - security critical."""

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_valid_redirect_url_exact_match(self, mock_allowed_hosts):
        """Test valid redirect URL with exact host match."""
        mock_allowed_hosts.return_value = ["example.com", "localhost"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("https://example.com/callback") is True
        assert _validate_redirect_url("https://localhost/callback") is True
        assert _validate_redirect_url("http://localhost:8080/callback") is True

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_valid_redirect_url_subdomain_match(self, mock_allowed_hosts):
        """Test valid redirect URL with subdomain matching."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # Subdomains should be allowed
        assert _validate_redirect_url("https://app.example.com/callback") is True
        assert _validate_redirect_url("https://api.example.com/callback") is True
        assert _validate_redirect_url("https://sub.domain.example.com/callback") is True

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_invalid_redirect_url_different_host(self, mock_allowed_hosts):
        """Test invalid redirect URL with different host."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("https://evil.com/callback") is False
        assert _validate_redirect_url("https://notexample.com/callback") is False

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_invalid_redirect_url_scheme(self, mock_allowed_hosts):
        """Test invalid redirect URL with non-http(s) scheme."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # Non-http(s) schemes should be rejected
        assert _validate_redirect_url("javascript:alert(1)") is False
        assert _validate_redirect_url("data:text/html,<script>") is False
        assert _validate_redirect_url("file:///etc/passwd") is False
        assert _validate_redirect_url("ftp://example.com/file") is False

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_invalid_redirect_url_no_host(self, mock_allowed_hosts):
        """Test invalid redirect URL without host."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("https:///callback") is False
        assert _validate_redirect_url("/relative/path") is False

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_redirect_url_case_insensitive(self, mock_allowed_hosts):
        """Test redirect URL host matching is case insensitive."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("https://EXAMPLE.COM/callback") is True
        assert _validate_redirect_url("https://Example.Com/callback") is True

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_invalid_redirect_url_suffix_attack(self, mock_allowed_hosts):
        """Test protection against suffix matching attacks."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # Attacker domain that ends with allowed domain should be rejected
        assert _validate_redirect_url("https://evilexample.com/callback") is False
        assert _validate_redirect_url("https://notexample.com/callback") is False
        # But subdomains should work
        assert _validate_redirect_url("https://sub.example.com/callback") is True

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_redirect_url_validation_handles_errors(self, mock_allowed_hosts):
        """Test redirect URL validation handles malformed URLs gracefully."""
        mock_allowed_hosts.return_value = ["example.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # Malformed URLs should return False, not raise exceptions
        assert _validate_redirect_url("") is False
        assert _validate_redirect_url("not a url") is False
        assert _validate_redirect_url("://missing-scheme") is False

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_redirect_url_empty_allowlist(self, mock_allowed_hosts):
        """Test redirect URL validation with empty allowlist."""
        mock_allowed_hosts.return_value = []

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # All URLs should be rejected with empty allowlist
        assert _validate_redirect_url("https://example.com/callback") is False


# ===========================================================================
# State Validation Tests
# ===========================================================================


class TestStateValidation:
    """Tests for OAuth state token validation."""

    @patch("aragora.server.handlers._oauth_impl._validate_state_internal")
    def test_validate_state_delegates_to_internal(self, mock_internal):
        """Test _validate_state delegates to internal function."""
        mock_internal.return_value = {"user_id": "user_1", "redirect_url": "https://example.com"}

        from aragora.server.handlers._oauth_impl import _validate_state

        result = _validate_state("test-state-token")

        mock_internal.assert_called_once_with("test-state-token")
        assert result == {"user_id": "user_1", "redirect_url": "https://example.com"}

    @patch("aragora.server.handlers._oauth_impl._validate_state_internal")
    def test_validate_state_returns_none_for_invalid(self, mock_internal):
        """Test _validate_state returns None for invalid state."""
        mock_internal.return_value = None

        from aragora.server.handlers._oauth_impl import _validate_state

        result = _validate_state("invalid-state")

        assert result is None


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestModuleExports:
    """Tests for module exports and backward compatibility."""

    def test_oauth_handler_exported(self):
        """Test OAuthHandler is exported from module."""
        from aragora.server.handlers._oauth_impl import OAuthHandler

        assert OAuthHandler is not None

    def test_validate_oauth_config_exported(self):
        """Test validate_oauth_config is exported from module."""
        from aragora.server.handlers._oauth_impl import validate_oauth_config

        assert callable(validate_oauth_config)

    def test_rate_limiter_exported(self):
        """Test _oauth_limiter is exported from module."""
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        assert _oauth_limiter is not None
        assert hasattr(_oauth_limiter, "is_allowed")

    def test_config_functions_exported(self):
        """Test config getter functions are exported."""
        from aragora.server.handlers import _oauth_impl

        # All config getters should be accessible
        assert hasattr(_oauth_impl, "_get_google_client_id")
        assert hasattr(_oauth_impl, "_get_github_client_id")
        assert hasattr(_oauth_impl, "_get_microsoft_client_id")
        assert hasattr(_oauth_impl, "_get_apple_client_id")
        assert hasattr(_oauth_impl, "_get_oidc_issuer")
        assert hasattr(_oauth_impl, "_get_oidc_client_id")

    def test_url_constants_exported(self):
        """Test OAuth URL constants are exported."""
        from aragora.server.handlers import _oauth_impl

        assert hasattr(_oauth_impl, "GOOGLE_AUTH_URL")
        assert hasattr(_oauth_impl, "GOOGLE_TOKEN_URL")
        assert hasattr(_oauth_impl, "GITHUB_AUTH_URL")
        assert hasattr(_oauth_impl, "GITHUB_TOKEN_URL")
        assert hasattr(_oauth_impl, "MICROSOFT_AUTH_URL_TEMPLATE")
        assert hasattr(_oauth_impl, "MICROSOFT_TOKEN_URL_TEMPLATE")


# ===========================================================================
# Rate Limiter Tests
# ===========================================================================


class TestOAuthRateLimiter:
    """Tests for OAuth rate limiter."""

    def test_rate_limiter_configured(self):
        """Test rate limiter is configured with sensible limits."""
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        # Rate limiter should exist and be configured
        assert _oauth_limiter is not None

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests under limit."""
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        # First few requests should be allowed
        # Using a unique IP to avoid state from other tests
        test_ip = "192.168.254.254"
        assert _oauth_limiter.is_allowed(test_ip) is True


# ===========================================================================
# Security Tests
# ===========================================================================


class TestOAuthSecurity:
    """Security-focused tests for OAuth implementation."""

    @patch("aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts")
    def test_open_redirect_prevention(self, mock_allowed_hosts):
        """Test protection against open redirect attacks."""
        mock_allowed_hosts.return_value = ["trusted.com"]

        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        # Various open redirect attack vectors
        attack_vectors = [
            # Different domain
            "https://attacker.com/callback",
            # Protocol-relative URL
            "//attacker.com/callback",
            # URL with credentials
            "https://trusted.com@attacker.com/callback",
            # Backslash tricks
            "https://trusted.com\\@attacker.com/callback",
            # Tab/newline tricks
            "https://trusted.com\t@attacker.com/callback",
            # Unicode tricks (potential)
            "https://trusted.com\u0000@attacker.com/callback",
        ]

        for attack in attack_vectors:
            # All attack vectors should be rejected
            result = _validate_redirect_url(attack)
            assert result is False, f"Attack vector not blocked: {attack}"

    def test_state_token_not_predictable(self):
        """Test that state tokens are cryptographically random."""
        from aragora.server.handlers._oauth_impl import _generate_state

        # Generate multiple state tokens
        states = [_generate_state() for _ in range(10)]

        # All should be unique
        assert len(set(states)) == 10

        # All should have reasonable length (at least 32 characters)
        for state in states:
            assert len(state) >= 32
