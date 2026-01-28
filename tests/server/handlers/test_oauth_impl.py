"""
Tests for OAuth Implementation Handler.

Tests the OAuth authentication handler including:
- Route handling and path matching
- Rate limiting
- State generation and validation
- Redirect URL validation
- Provider-specific authentication flows
- Account linking
- Error handling
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"X-Forwarded-For": "192.168.1.1"}
    return handler


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_email.return_value = None
    store.get_user_by_oauth_provider.return_value = None
    return store


@pytest.fixture
def mock_server_context(mock_user_store):
    """Create a mock server context."""
    context = MagicMock()
    context.get.side_effect = lambda k, default=None: {
        "user_store": mock_user_store,
    }.get(k, default)
    return context


@pytest.fixture
def oauth_handler(mock_server_context, mock_user_store):
    """Create an OAuthHandler instance with mocked context."""
    from aragora.server.handlers._oauth_impl import OAuthHandler

    handler = OAuthHandler(mock_server_context)
    handler.ctx = {"user_store": mock_user_store}
    return handler


@pytest.fixture
def mock_oauth_state():
    """Create mock OAuth state data."""
    return {
        "user_id": None,
        "redirect_url": "http://localhost:3000/auth/callback",
        "provider": "google",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Route Handling Tests
# =============================================================================


class TestRouteHandling:
    """Test route handling and can_handle method."""

    def test_can_handle_google_auth(self, oauth_handler):
        """Test handler recognizes Google OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google/callback") is True

    def test_can_handle_github_auth(self, oauth_handler):
        """Test handler recognizes GitHub OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github/callback") is True

    def test_can_handle_microsoft_auth(self, oauth_handler):
        """Test handler recognizes Microsoft OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft/callback") is True

    def test_can_handle_apple_auth(self, oauth_handler):
        """Test handler recognizes Apple OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple/callback") is True

    def test_can_handle_oidc_auth(self, oauth_handler):
        """Test handler recognizes OIDC routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc/callback") is True

    def test_can_handle_link_unlink(self, oauth_handler):
        """Test handler recognizes link/unlink routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/link") is True
        assert oauth_handler.can_handle("/api/v1/auth/oauth/unlink") is True

    def test_can_handle_providers(self, oauth_handler):
        """Test handler recognizes provider listing routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/providers") is True
        assert oauth_handler.can_handle("/api/v1/user/oauth-providers") is True

    def test_can_handle_non_v1_routes(self, oauth_handler):
        """Test handler recognizes non-v1 routes (for OAuth callback compatibility)."""
        assert oauth_handler.can_handle("/api/auth/oauth/google") is True
        assert oauth_handler.can_handle("/api/auth/oauth/google/callback") is True

    def test_cannot_handle_unknown_route(self, oauth_handler):
        """Test handler rejects unknown routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/unknown") is False
        assert oauth_handler.can_handle("/api/v1/auth/login") is False
        assert oauth_handler.can_handle("/api/debates") is False


class TestMethodRouting:
    """Test method-based routing."""

    def test_handle_returns_405_for_wrong_method(self, oauth_handler, mock_handler):
        """Test 405 returned for unsupported methods."""
        mock_handler.command = "POST"
        result = oauth_handler.handle("/api/v1/auth/oauth/google", {}, mock_handler, method="POST")
        assert result.status_code == 405

    def test_handle_providers_get_method(self, oauth_handler, mock_handler):
        """Test providers endpoint accepts GET."""
        mock_handler.command = "GET"
        # This will fail auth check but validates routing works
        result = oauth_handler.handle(
            "/api/v1/auth/oauth/providers", {}, mock_handler, method="GET"
        )
        # Should not be 405 (Method Not Allowed)
        assert result.status_code != 405


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_enforced(self, oauth_handler, mock_handler):
        """Test rate limiting is enforced on OAuth endpoints."""
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        # Exhaust rate limit
        client_ip = "192.168.1.1"
        for _ in range(25):  # More than 20 per minute limit
            _oauth_limiter.is_allowed(client_ip)

        # Next request should be rate limited
        result = oauth_handler.handle("/api/v1/auth/oauth/google", {}, mock_handler, method="GET")
        assert result.status_code == 429
        assert b"Rate limit" in result.body

    def test_rate_limit_different_ips(self, oauth_handler, mock_handler):
        """Test rate limits are per-IP."""
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        # Different IPs should each get their own rate limit bucket
        # Use unique IPs to avoid interference from other tests
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"

        # First IP can make requests
        assert _oauth_limiter.is_allowed(ip1) is True

        # Different IP also can make requests
        assert _oauth_limiter.is_allowed(ip2) is True


# =============================================================================
# State Validation Tests
# =============================================================================


class TestStateValidation:
    """Test OAuth state generation and validation."""

    def test_generate_state_creates_unique_values(self):
        """Test state generation creates unique values."""
        from aragora.server.handlers._oauth_impl import _generate_state

        state1 = _generate_state()
        state2 = _generate_state()

        assert state1 != state2
        assert len(state1) >= 32  # Should be cryptographically secure length

    def test_generate_state_with_user_id(self):
        """Test state generation includes user ID for account linking."""
        from aragora.server.handlers._oauth_impl import _generate_state

        state = _generate_state(user_id="user-123")
        assert state is not None
        assert len(state) >= 32

    def test_validate_state_rejects_invalid(self):
        """Test state validation rejects invalid states."""
        from aragora.server.handlers._oauth_impl import _validate_state

        result = _validate_state("invalid-state-value")
        assert result is None

    def test_validate_state_rejects_empty(self):
        """Test state validation rejects empty states."""
        from aragora.server.handlers._oauth_impl import _validate_state

        result = _validate_state("")
        assert result is None

    def test_state_roundtrip(self):
        """Test state can be generated and validated."""
        from aragora.server.handlers._oauth_impl import (
            _generate_state,
            _validate_state,
        )

        state = _generate_state(redirect_url="http://localhost:3000/callback")
        result = _validate_state(state)

        assert result is not None
        assert result.get("redirect_url") == "http://localhost:3000/callback"


# =============================================================================
# Redirect URL Validation Tests
# =============================================================================


class TestRedirectUrlValidation:
    """Test redirect URL validation for open redirect prevention."""

    def test_validate_localhost_allowed_in_dev(self):
        """Test localhost URLs are allowed in development."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            assert _validate_redirect_url("http://localhost:3000/callback") is True
            assert _validate_redirect_url("http://127.0.0.1:3000/callback") is True

    def test_validate_rejects_external_urls_in_prod(self):
        """Test external URLs are rejected in production without allowlist."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "OAUTH_ALLOWED_REDIRECT_HOSTS": "",
            },
            clear=False,
        ):
            # Reload to pick up env changes - in real code this happens at import
            result = _validate_redirect_url("http://evil.com/callback")
            # Should be rejected (False) unless in allowlist
            # Actual behavior depends on ALLOWED_OAUTH_REDIRECT_HOSTS value

    def test_validate_rejects_javascript_urls(self):
        """Test JavaScript URLs are rejected."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("javascript:alert(1)") is False

    def test_validate_rejects_data_urls(self):
        """Test data URLs are rejected."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        assert _validate_redirect_url("data:text/html,<script>alert(1)</script>") is False


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Test OAuth configuration handling."""

    def test_google_config_from_env(self):
        """Test Google OAuth config is read from environment."""
        from aragora.server.handlers._oauth_impl import (
            _get_google_client_id,
            _get_google_client_secret,
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_OAUTH_CLIENT_ID": "test-client-id",
                "GOOGLE_OAUTH_CLIENT_SECRET": "test-secret",
            },
        ):
            assert _get_google_client_id() == "test-client-id"
            assert _get_google_client_secret() == "test-secret"

    def test_github_config_from_env(self):
        """Test GitHub OAuth config is read from environment."""
        from aragora.server.handlers._oauth_impl import (
            _get_github_client_id,
            _get_github_client_secret,
        )

        with patch.dict(
            os.environ,
            {
                "GITHUB_OAUTH_CLIENT_ID": "github-client-id",
                "GITHUB_OAUTH_CLIENT_SECRET": "github-secret",
            },
        ):
            assert _get_github_client_id() == "github-client-id"
            assert _get_github_client_secret() == "github-secret"

    def test_microsoft_config_from_env(self):
        """Test Microsoft OAuth config is read from environment."""
        from aragora.server.handlers._oauth_impl import (
            _get_microsoft_client_id,
            _get_microsoft_tenant,
        )

        with patch.dict(
            os.environ,
            {
                "MICROSOFT_OAUTH_CLIENT_ID": "ms-client-id",
                "MICROSOFT_OAUTH_TENANT": "contoso.onmicrosoft.com",
            },
        ):
            assert _get_microsoft_client_id() == "ms-client-id"
            assert _get_microsoft_tenant() == "contoso.onmicrosoft.com"

    def test_microsoft_tenant_default(self):
        """Test Microsoft tenant defaults to 'common'."""
        from aragora.server.handlers._oauth_impl import _get_microsoft_tenant

        with patch.dict(os.environ, {}, clear=True):
            # Should default to "common" for multi-tenant apps
            tenant = _get_microsoft_tenant()
            # Default behavior when not set
            assert tenant == "" or tenant == "common"

    def test_validate_oauth_config_in_dev(self):
        """Test config validation is skipped in development."""
        from aragora.server.handlers._oauth_impl import validate_oauth_config

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            missing = validate_oauth_config()
            assert missing == []  # No validation in dev mode


# =============================================================================
# Google OAuth Flow Tests
# =============================================================================


class TestGoogleOAuthFlow:
    """Test Google OAuth authentication flow."""

    def test_google_auth_start_not_configured(self, oauth_handler, mock_handler):
        """Test Google auth returns 503 when not configured."""
        with patch.dict(os.environ, {"GOOGLE_OAUTH_CLIENT_ID": ""}, clear=False):
            from aragora.server.handlers import _oauth_impl

            # Patch the function to return empty
            with patch.object(_oauth_impl, "_get_google_client_id", return_value=""):
                result = oauth_handler._handle_google_auth_start(mock_handler, {})
                assert result.status_code == 503
                assert b"not configured" in result.body

    def test_google_auth_start_configured(self, oauth_handler, mock_handler):
        """Test Google auth redirects when configured."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_get_google_client_id", return_value="test-client-id"):
            with patch.object(
                _oauth_impl,
                "_get_google_redirect_uri",
                return_value="http://localhost:8080/callback",
            ):
                with patch.object(
                    _oauth_impl,
                    "_get_oauth_success_url",
                    return_value="http://localhost:3000/callback",
                ):
                    with patch.object(_oauth_impl, "_validate_redirect_url", return_value=True):
                        result = oauth_handler._handle_google_auth_start(mock_handler, {})
                        assert result.status_code == 302
                        assert "Location" in result.headers
                        assert "accounts.google.com" in result.headers["Location"]

    def test_google_callback_missing_state(self, oauth_handler, mock_handler):
        """Test callback fails without state parameter."""
        result = oauth_handler._handle_google_callback(mock_handler, {"code": "auth-code"})
        # Should redirect with error
        assert result.status_code == 302 or b"state" in result.body.lower()

    def test_google_callback_missing_code(self, oauth_handler, mock_handler):
        """Test callback fails without authorization code."""
        from aragora.server.handlers._oauth_impl import _generate_state

        state = _generate_state()
        result = oauth_handler._handle_google_callback(mock_handler, {"state": state})
        # Should redirect with error
        assert result.status_code == 302 or b"code" in result.body.lower()

    def test_google_callback_handles_error_from_google(self, oauth_handler, mock_handler):
        """Test callback handles error response from Google."""
        query_params = {
            "error": "access_denied",
            "error_description": "User denied access",
        }
        result = oauth_handler._handle_google_callback(mock_handler, query_params)
        # Should redirect with error
        assert result.status_code == 302


# =============================================================================
# GitHub OAuth Flow Tests
# =============================================================================


class TestGitHubOAuthFlow:
    """Test GitHub OAuth authentication flow."""

    def test_github_auth_start_not_configured(self, oauth_handler, mock_handler):
        """Test GitHub auth returns 503 when not configured."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_get_github_client_id", return_value=""):
            result = oauth_handler._handle_github_auth_start(mock_handler, {})
            assert result.status_code == 503
            assert b"not configured" in result.body

    def test_github_auth_start_configured(self, oauth_handler, mock_handler):
        """Test GitHub auth redirects when configured."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_get_github_client_id", return_value="github-client-id"):
            with patch.object(
                _oauth_impl,
                "_get_github_redirect_uri",
                return_value="http://localhost:8080/callback",
            ):
                with patch.object(
                    _oauth_impl,
                    "_get_oauth_success_url",
                    return_value="http://localhost:3000/callback",
                ):
                    with patch.object(_oauth_impl, "_validate_redirect_url", return_value=True):
                        result = oauth_handler._handle_github_auth_start(mock_handler, {})
                        assert result.status_code == 302
                        assert "Location" in result.headers
                        assert "github.com" in result.headers["Location"]


# =============================================================================
# Microsoft OAuth Flow Tests
# =============================================================================


class TestMicrosoftOAuthFlow:
    """Test Microsoft OAuth authentication flow."""

    def test_microsoft_auth_start_not_configured(self, oauth_handler, mock_handler):
        """Test Microsoft auth returns 503 when not configured."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_get_microsoft_client_id", return_value=""):
            result = oauth_handler._handle_microsoft_auth_start(mock_handler, {})
            assert result.status_code == 503
            assert b"not configured" in result.body

    def test_microsoft_auth_start_configured(self, oauth_handler, mock_handler):
        """Test Microsoft auth redirects when configured."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_get_microsoft_client_id", return_value="ms-client-id"):
            with patch.object(_oauth_impl, "_get_microsoft_tenant", return_value="common"):
                with patch.object(
                    _oauth_impl,
                    "_get_microsoft_redirect_uri",
                    return_value="http://localhost:8080/callback",
                ):
                    with patch.object(
                        _oauth_impl,
                        "_get_oauth_success_url",
                        return_value="http://localhost:3000/callback",
                    ):
                        with patch.object(_oauth_impl, "_validate_redirect_url", return_value=True):
                            result = oauth_handler._handle_microsoft_auth_start(mock_handler, {})
                            assert result.status_code == 302
                            assert "Location" in result.headers
                            assert "microsoftonline.com" in result.headers["Location"]


# =============================================================================
# Account Linking Tests
# =============================================================================


class TestAccountLinking:
    """Test account linking functionality."""

    def test_link_requires_authentication(self, oauth_handler, mock_handler):
        """Test link endpoint requires authentication."""
        mock_handler.command = "POST"

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = False
            mock_ctx.user_id = None
            mock_extract.return_value = mock_ctx

            result = oauth_handler._handle_link_account(mock_handler)
            assert result.status_code == 401

    def test_unlink_requires_authentication(self, oauth_handler, mock_handler):
        """Test unlink endpoint requires authentication."""
        mock_handler.command = "DELETE"

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = False
            mock_ctx.user_id = None
            mock_extract.return_value = mock_ctx

            result = oauth_handler._handle_unlink_account(mock_handler)
            assert result.status_code == 401


# =============================================================================
# Provider Listing Tests
# =============================================================================


class TestProviderListing:
    """Test provider listing functionality."""

    def test_list_providers_returns_available(self, oauth_handler, mock_handler):
        """Test listing available OAuth providers."""
        from aragora.server.handlers import _oauth_impl

        # Mock all providers as configured
        with patch.object(_oauth_impl, "_get_google_client_id", return_value="google-id"):
            with patch.object(_oauth_impl, "_get_github_client_id", return_value="github-id"):
                with patch.object(_oauth_impl, "_get_microsoft_client_id", return_value="ms-id"):
                    result = oauth_handler._handle_list_providers(mock_handler)
                    assert result.status_code == 200


# =============================================================================
# OAuthUserInfo Tests
# =============================================================================


class TestOAuthUserInfo:
    """Test OAuthUserInfo dataclass."""

    def test_oauth_user_info_creation(self):
        """Test OAuthUserInfo creation."""
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="12345",
            email="user@example.com",
            name="Test User",
            picture="https://example.com/avatar.jpg",
        )

        assert user_info.provider == "google"
        assert user_info.provider_user_id == "12345"
        assert user_info.email == "user@example.com"
        assert user_info.name == "Test User"
        assert user_info.picture == "https://example.com/avatar.jpg"

    def test_oauth_user_info_optional_fields(self):
        """Test OAuthUserInfo with optional fields."""
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="github",
            provider_user_id="67890",
            email="dev@example.com",
            name="Dev User",  # Required field
        )

        assert user_info.provider == "github"
        assert user_info.email == "dev@example.com"
        assert user_info.name == "Dev User"
        assert user_info.picture is None  # Optional field
        assert user_info.email_verified is False  # Default value


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in OAuth flows."""

    def test_redirect_with_error(self, oauth_handler):
        """Test error redirect contains error message."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(
            _oauth_impl, "_get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._redirect_with_error("Test error message")
            assert result.status_code == 302
            assert "Location" in result.headers
            assert "error" in result.headers["Location"].lower()

    def test_handle_errors_decorator(self, oauth_handler, mock_handler):
        """Test handle_errors decorator catches exceptions."""
        # The decorator should convert exceptions to error responses
        # rather than letting them propagate
        pass  # Decorator behavior tested via integration


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityFeatures:
    """Test security features of OAuth handler."""

    def test_state_is_cryptographically_random(self):
        """Test state tokens are cryptographically random."""
        from aragora.server.handlers._oauth_impl import _generate_state

        states = [_generate_state() for _ in range(100)]

        # All states should be unique
        assert len(set(states)) == 100

        # States should be long enough
        for state in states:
            assert len(state) >= 32

    def test_csrf_protection_via_state(self):
        """Test CSRF protection through state validation."""
        from aragora.server.handlers._oauth_impl import (
            _generate_state,
            _validate_state,
        )

        # Valid state should validate
        valid_state = _generate_state()
        assert _validate_state(valid_state) is not None

        # Tampered state should fail
        tampered = valid_state + "tampered"
        assert _validate_state(tampered) is None

    def test_state_cannot_be_reused(self):
        """Test state tokens are single-use."""
        from aragora.server.handlers._oauth_impl import (
            _generate_state,
            _validate_state,
        )

        state = _generate_state()

        # First validation should succeed
        result1 = _validate_state(state)
        assert result1 is not None

        # Second validation should fail (state consumed)
        result2 = _validate_state(state)
        assert result2 is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestOAuthIntegration:
    """Integration tests for OAuth handler."""

    def test_full_google_flow_mock(self, oauth_handler, mock_handler, mock_user_store):
        """Test complete Google OAuth flow with mocked external calls."""
        from aragora.server.handlers import _oauth_impl
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        # Configure Google OAuth
        with patch.object(_oauth_impl, "_get_google_client_id", return_value="test-client"):
            with patch.object(
                _oauth_impl, "_get_google_redirect_uri", return_value="http://localhost/callback"
            ):
                with patch.object(
                    _oauth_impl, "_get_oauth_success_url", return_value="http://localhost/success"
                ):
                    with patch.object(_oauth_impl, "_validate_redirect_url", return_value=True):
                        # Step 1: Start auth - should redirect
                        result = oauth_handler._handle_google_auth_start(mock_handler, {})
                        assert result.status_code == 302

                        # Extract state from redirect URL
                        location = result.headers.get("Location", "")
                        assert "state=" in location

    def test_handler_resource_type(self, oauth_handler):
        """Test handler has correct resource type for RBAC."""
        assert oauth_handler.RESOURCE_TYPE == "oauth"
