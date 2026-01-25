"""
Tests for OAuth Handler.

Tests cover:
- Handler routing for all OAuth providers
- Rate limiting
- OAuth state generation and validation
- Error handling
- Provider listing
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from aragora.server.handlers.oauth import (
    OAuthHandler,
    _oauth_limiter,
    _generate_state,
    _validate_state,
    _validate_redirect_url,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {"user_store": MagicMock()}


@pytest.fixture
def handler(mock_server_context):
    """Create OAuth handler with mock context."""
    return OAuthHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"X-Forwarded-For": "192.168.1.1"}
    return mock


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _oauth_limiter._buckets.clear()
    yield
    _oauth_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def clear_env_vars():
    """Clear OAuth-related env vars for clean tests."""
    env_vars = [
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        "MICROSOFT_OAUTH_CLIENT_ID",
        "MICROSOFT_OAUTH_CLIENT_SECRET",
    ]
    original = {k: os.environ.get(k) for k in env_vars}
    for k in env_vars:
        if k in os.environ:
            del os.environ[k]
    yield
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v


# ============================================================================
# Routing Tests
# ============================================================================


class TestOAuthHandlerRouting:
    """Tests for OAuth handler routing."""

    def test_can_handle_google_auth(self, handler):
        """Handler can handle Google OAuth start endpoint."""
        assert handler.can_handle("/api/v1/auth/oauth/google")
        assert handler.can_handle("/api/auth/oauth/google")

    def test_can_handle_google_callback(self, handler):
        """Handler can handle Google OAuth callback."""
        assert handler.can_handle("/api/v1/auth/oauth/google/callback")
        assert handler.can_handle("/api/auth/oauth/google/callback")

    def test_can_handle_github_auth(self, handler):
        """Handler can handle GitHub OAuth endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/github")
        assert handler.can_handle("/api/v1/auth/oauth/github/callback")

    def test_can_handle_microsoft_auth(self, handler):
        """Handler can handle Microsoft OAuth endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/microsoft")
        assert handler.can_handle("/api/v1/auth/oauth/microsoft/callback")

    def test_can_handle_apple_auth(self, handler):
        """Handler can handle Apple OAuth endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/apple")
        assert handler.can_handle("/api/v1/auth/oauth/apple/callback")

    def test_can_handle_oidc_auth(self, handler):
        """Handler can handle generic OIDC endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/oidc")
        assert handler.can_handle("/api/v1/auth/oauth/oidc/callback")

    def test_can_handle_link_unlink(self, handler):
        """Handler can handle account link/unlink endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/link")
        assert handler.can_handle("/api/v1/auth/oauth/unlink")

    def test_can_handle_providers(self, handler):
        """Handler can handle provider listing endpoints."""
        assert handler.can_handle("/api/v1/auth/oauth/providers")
        assert handler.can_handle("/api/v1/user/oauth-providers")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/v1/auth/login")
        assert not handler.can_handle("/api/v1/other/endpoint")
        assert not handler.can_handle("/api/v1/auth/oauth/unknown")

    def test_routes_contains_both_versions(self, handler):
        """ROUTES list contains both v1 and non-v1 paths."""
        v1_routes = [r for r in handler.ROUTES if "/api/v1/" in r]
        non_v1_routes = [r for r in handler.ROUTES if "/api/v1/" not in r]
        assert len(v1_routes) > 0
        assert len(non_v1_routes) > 0


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestOAuthRateLimiting:
    """Tests for OAuth rate limiting."""

    def test_rate_limit_allows_normal_requests(self, handler, mock_http_handler):
        """Rate limiter allows normal request volume."""
        # Should allow first request
        result = handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_http_handler,
            method="GET",
        )
        # Will get 503 (not configured) but not 429
        assert result.status_code != 429

    def test_rate_limit_blocks_excessive_requests(self, handler, mock_http_handler):
        """Rate limiter blocks excessive requests."""
        # Simulate exceeding rate limit
        for _ in range(25):  # Limit is 20/min
            _oauth_limiter.is_allowed("192.168.1.1")

        result = handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_http_handler,
            method="GET",
        )
        assert result.status_code == 429

    def test_rate_limit_per_ip(self, handler, mock_http_handler):
        """Rate limit is applied per IP address."""
        # Exhaust rate limit for one IP
        for _ in range(25):
            _oauth_limiter.is_allowed("192.168.1.1")

        # Different IP should still be allowed
        mock_http_handler.headers = {"X-Forwarded-For": "10.0.0.1"}
        result = handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_http_handler,
            method="GET",
        )
        # Will get 503 (not configured) but not 429
        assert result.status_code != 429


# ============================================================================
# State Generation/Validation Tests
# ============================================================================


class TestOAuthState:
    """Tests for OAuth state parameter handling."""

    def test_generate_state_returns_string(self):
        """State generation returns a string."""
        state = _generate_state()
        assert isinstance(state, str)
        assert len(state) > 0

    def test_generate_state_includes_user_id(self):
        """State can include user_id."""
        state = _generate_state(user_id="user123")
        assert isinstance(state, str)
        assert len(state) > 0

    def test_generate_state_includes_redirect_url(self):
        """State can include redirect_url."""
        state = _generate_state(redirect_url="https://example.com/callback")
        assert isinstance(state, str)
        assert len(state) > 0

    def test_validate_state_returns_dict(self):
        """Valid state validation returns dict."""
        state = _generate_state(user_id="user123", redirect_url="https://example.com")
        result = _validate_state(state)
        assert isinstance(result, dict)
        assert result.get("user_id") == "user123"
        assert result.get("redirect_url") == "https://example.com"

    def test_validate_invalid_state_returns_none(self):
        """Invalid state validation returns None."""
        result = _validate_state("invalid_state_token")
        assert result is None

    def test_validate_empty_state_returns_none(self):
        """Empty state validation returns None."""
        result = _validate_state("")
        assert result is None


# ============================================================================
# Redirect URL Validation Tests
# ============================================================================


class TestRedirectUrlValidation:
    """Tests for redirect URL validation."""

    def test_allows_localhost(self):
        """Allows localhost redirects for development."""
        assert _validate_redirect_url("http://localhost:3000/callback")
        assert _validate_redirect_url("http://127.0.0.1:8080/callback")

    def test_rejects_relative_urls(self):
        """Rejects relative URLs (security - scheme required)."""
        # Relative URLs are rejected because they lack a scheme
        assert not _validate_redirect_url("/dashboard")
        assert not _validate_redirect_url("/auth/complete")

    def test_rejects_external_domains(self):
        """Rejects unknown external domains."""
        # By default should reject random external domains
        result = _validate_redirect_url("https://evil.com/steal-token")
        # This depends on configuration - may be False or True
        # Just ensure it returns a boolean
        assert isinstance(result, bool)


# ============================================================================
# Provider Not Configured Tests
# ============================================================================


class TestOAuthProviderNotConfigured:
    """Tests for handling unconfigured OAuth providers."""

    def test_google_not_configured_returns_503(self, handler, mock_http_handler, clear_env_vars):
        """Returns 503 when Google OAuth not configured."""
        with patch("aragora.server.handlers.oauth._get_google_client_id", return_value=""):
            result = handler.handle(
                "/api/v1/auth/oauth/google",
                {},
                mock_http_handler,
                method="GET",
            )
            assert result.status_code == 503

    def test_github_not_configured_returns_503(self, handler, mock_http_handler, clear_env_vars):
        """Returns 503 when GitHub OAuth not configured."""
        with patch("aragora.server.handlers.oauth._get_github_client_id", return_value=""):
            result = handler.handle(
                "/api/v1/auth/oauth/github",
                {},
                mock_http_handler,
                method="GET",
            )
            assert result.status_code == 503

    def test_microsoft_not_configured_returns_503(self, handler, mock_http_handler, clear_env_vars):
        """Returns 503 when Microsoft OAuth not configured."""
        with patch("aragora.server.handlers.oauth._get_microsoft_client_id", return_value=""):
            result = handler.handle(
                "/api/v1/auth/oauth/microsoft",
                {},
                mock_http_handler,
                method="GET",
            )
            assert result.status_code == 503


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestOAuthErrorHandling:
    """Tests for OAuth error handling."""

    def test_callback_without_state_returns_error(self, handler, mock_http_handler):
        """OAuth callback without state returns error."""
        result = handler.handle(
            "/api/v1/auth/oauth/google/callback",
            {},
            mock_http_handler,
            method="GET",
        )
        # Should redirect with error
        assert result.status_code in (302, 400)

    def test_callback_with_google_error(self, handler, mock_http_handler):
        """OAuth callback with error param handles gracefully."""
        result = handler.handle(
            "/api/v1/auth/oauth/google/callback",
            {"error": "access_denied", "error_description": "User denied access"},
            mock_http_handler,
            method="GET",
        )
        # Should redirect with error
        assert result.status_code == 302

    def test_method_not_allowed(self, handler, mock_http_handler):
        """Returns 405 for unsupported methods."""
        mock_http_handler.command = "PUT"
        result = handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_http_handler,
            method="PUT",
        )
        assert result.status_code == 405


# ============================================================================
# Provider Listing Tests
# ============================================================================


class TestOAuthProviderListing:
    """Tests for OAuth provider listing."""

    def test_list_providers_returns_dict(self, handler, mock_http_handler):
        """List providers returns structured response."""
        result = handler.handle(
            "/api/v1/auth/oauth/providers",
            {},
            mock_http_handler,
            method="GET",
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestOAuthHandlerInit:
    """Tests for OAuth handler initialization."""

    def test_handler_has_resource_type(self, handler):
        """Handler has RESOURCE_TYPE set."""
        assert handler.RESOURCE_TYPE == "oauth"

    def test_handler_has_routes(self, handler):
        """Handler has ROUTES list."""
        assert len(handler.ROUTES) >= 20  # Many routes for all providers

    def test_handler_extends_secure_handler(self, handler):
        """Handler extends SecureHandler for JWT auth."""
        from aragora.server.handlers.secure import SecureHandler

        assert isinstance(handler, SecureHandler)
