"""
Tests for aragora.server.handlers._oauth_impl - OAuth authentication handler.

Tests cover:
- can_handle() route matching (v1 and non-v1 routes)
- handle() route dispatching for all providers
- Rate limiting
- _handle_google_auth_start() - Google OAuth flow initiation
- _handle_google_callback() - Google OAuth callback handling
- _handle_github_auth_start() - GitHub OAuth flow initiation
- _handle_github_callback() - GitHub OAuth callback handling
- _handle_microsoft_auth_start() - Microsoft OAuth flow initiation
- _handle_list_providers() - List available providers
- _handle_get_user_providers() - Get user's linked providers (with RBAC)
- _handle_link_account() - Link OAuth provider (with RBAC)
- _handle_unlink_account() - Unlink OAuth provider (with RBAC)
- _handle_oauth_url() - Get OAuth authorization URL as JSON
- _handle_oauth_callback_api() - Complete OAuth via API (POST)
- _validate_redirect_url() - Open redirect prevention
- validate_oauth_config() - Configuration validation
- OAuthUserInfo dataclass
- _get_param() helper
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Rate Limiter Reset Fixture
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_oauth_rate_limiter():
    """Reset OAuth rate limiter state before each test.

    This prevents rate limiting from carrying over between tests.
    """
    from aragora.server.middleware.rate_limit.oauth_limiter import (
        reset_oauth_limiter,
        reset_backoff_tracker,
    )

    reset_oauth_limiter()
    reset_backoff_tracker()
    yield
    # Also reset after test to ensure clean state for next test
    reset_oauth_limiter()
    reset_backoff_tracker()


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = None
    role: str = "member"
    is_active: bool = True
    password_hash: str = "hashed"
    password_salt: str = "salt"
    oauth_providers: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict = None,
        client_address: tuple = None,
        body: bytes = None,
        method: str = "GET",
    ):
        self.headers = headers or {}
        self.client_address = client_address or ("127.0.0.1", 12345)
        self._body = body or b""
        self.command = method
        self.rfile = MagicMock()
        self.rfile.read.return_value = self._body


class MockAuthContext:
    """Mock auth context for testing."""

    def __init__(
        self,
        is_authenticated: bool = False,
        user_id: str = None,
        role: str = "admin",
        org_id: str = "org-123",
        client_ip: str = "127.0.0.1",
    ):
        self.is_authenticated = is_authenticated
        self.authenticated = is_authenticated
        self.user_id = user_id
        self.role = role
        self.org_id = org_id
        self.client_ip = client_ip
        self.permissions = {
            "*",
            "admin",
            "authentication.read",
            "authentication.write",
            "authentication.update",
        }
        self.roles = {"admin", "owner"}


@dataclass
class MockTokenPair:
    """Mock token pair returned by create_token_pair."""

    access_token: str = "mock-access-token"
    refresh_token: str = "mock-refresh-token"
    token_type: str = "Bearer"
    expires_in: int = 3600


def create_oauth_handler():
    """Create an OAuthHandler with empty context."""
    from aragora.server.handlers.oauth import OAuthHandler

    return OAuthHandler({})


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        try:
            return json.loads(result.body.decode("utf-8"))
        except json.JSONDecodeError:
            return {"raw": result.body.decode("utf-8")}
    return result


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result


# ===========================================================================
# Test can_handle() Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_google_auth_v1(self):
        """Should handle /api/v1/auth/oauth/google."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/google") is True

    def test_handles_google_callback_v1(self):
        """Should handle /api/v1/auth/oauth/google/callback."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/google/callback") is True

    def test_handles_github_auth_v1(self):
        """Should handle /api/v1/auth/oauth/github."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/github") is True

    def test_handles_github_callback_v1(self):
        """Should handle /api/v1/auth/oauth/github/callback."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/github/callback") is True

    def test_handles_microsoft_auth_v1(self):
        """Should handle /api/v1/auth/oauth/microsoft."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/microsoft") is True

    def test_handles_apple_auth_v1(self):
        """Should handle /api/v1/auth/oauth/apple."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/apple") is True

    def test_handles_oidc_auth_v1(self):
        """Should handle /api/v1/auth/oauth/oidc."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/oidc") is True

    def test_handles_link(self):
        """Should handle /api/v1/auth/oauth/link."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/link") is True

    def test_handles_unlink(self):
        """Should handle /api/v1/auth/oauth/unlink."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/unlink") is True

    def test_handles_providers(self):
        """Should handle /api/v1/auth/oauth/providers."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/providers") is True

    def test_handles_user_providers(self):
        """Should handle /api/v1/user/oauth-providers."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/user/oauth-providers") is True

    def test_handles_non_v1_google(self):
        """Should handle non-v1 path /api/auth/oauth/google."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/auth/oauth/google") is True

    def test_handles_oauth_url(self):
        """Should handle /api/v1/auth/oauth/url."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/url") is True

    def test_handles_oauth_authorize(self):
        """Should handle /api/v1/auth/oauth/authorize."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/authorize") is True

    def test_rejects_unknown_routes(self):
        """Should reject unknown routes."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/facebook") is False
        assert handler.can_handle("/api/v1/auth/login") is False
        assert handler.can_handle("/api/v1/oauth") is False


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting in handle()."""

    def test_rate_limit_exceeded_returns_429(self):
        """Should return 429 when rate limit exceeded."""
        from aragora.server.handlers.oauth import _oauth_limiter

        handler = create_oauth_handler()
        mock_http = MockHandler(client_address=("192.168.1.100", 12345))

        with patch.object(_oauth_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/auth/oauth/google", {}, mock_http, "GET")

        assert get_status(result) == 429
        # Accept either old or new error message
        error_msg = get_body(result)["error"]
        assert "Rate limit" in error_msg or "Too many" in error_msg


# ===========================================================================
# Test Configuration Validation
# ===========================================================================


class TestValidateOAuthConfig:
    """Tests for validate_oauth_config() function."""

    def test_returns_empty_in_dev_mode(self):
        """Should return empty list in development mode."""
        from aragora.server.handlers import _oauth_impl

        with patch.object(_oauth_impl, "_IS_PRODUCTION", False):
            missing = _oauth_impl.validate_oauth_config()
            assert missing == []

    def test_returns_missing_vars_in_production(self):
        """Should return missing vars in production mode."""
        from aragora.server.handlers.oauth import config as oauth_config

        with (
            patch.object(oauth_config, "_IS_PRODUCTION", True),
            patch.object(oauth_config, "GOOGLE_CLIENT_ID", "test-client-id"),
            patch.object(oauth_config, "_get_google_client_secret", return_value=""),
            patch.object(oauth_config, "_get_google_redirect_uri", return_value=""),
            patch.object(oauth_config, "_get_oauth_success_url", return_value=""),
            patch.object(oauth_config, "_get_oauth_error_url", return_value=""),
            patch.object(oauth_config, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset()),
        ):
            missing = oauth_config.validate_oauth_config()
            assert "GOOGLE_OAUTH_CLIENT_SECRET" in missing
            assert "GOOGLE_OAUTH_REDIRECT_URI" in missing


# ===========================================================================
# Test _validate_redirect_url()
# ===========================================================================


class TestValidateRedirectUrl:
    """Tests for _validate_redirect_url() security function."""

    def test_rejects_javascript_scheme(self):
        """Should reject javascript: scheme."""
        from aragora.server.handlers.oauth import _validate_redirect_url

        assert _validate_redirect_url("javascript:alert(1)") is False

    def test_rejects_data_scheme(self):
        """Should reject data: scheme."""
        from aragora.server.handlers.oauth import _validate_redirect_url

        assert _validate_redirect_url("data:text/html,<script>alert(1)</script>") is False

    def test_allows_localhost_in_dev(self, monkeypatch):
        """Should allow localhost when in allowed hosts."""
        from aragora.server.handlers import oauth
        from aragora.server.handlers.oauth import _validate_redirect_url

        monkeypatch.setattr(oauth, "_get_allowed_redirect_hosts", lambda: frozenset(["localhost"]))
        assert _validate_redirect_url("http://localhost:3000/callback") is True

    def test_rejects_unknown_host(self):
        """Should reject hosts not in allowlist."""
        from aragora.server.handlers import _oauth_impl
        from aragora.server.handlers.oauth import _validate_redirect_url

        with patch.object(
            _oauth_impl, "_get_allowed_redirect_hosts", return_value=frozenset(["example.com"])
        ):
            assert _validate_redirect_url("https://evil.com/callback") is False

    def test_allows_subdomain_of_allowed_host(self):
        """Should allow subdomains of allowed hosts."""
        from aragora.server.handlers import _oauth_impl
        from aragora.server.handlers.oauth import _validate_redirect_url

        with patch.object(
            _oauth_impl, "_get_allowed_redirect_hosts", return_value=frozenset(["example.com"])
        ):
            assert _validate_redirect_url("https://app.example.com/callback") is True

    def test_handles_invalid_url(self):
        """Should return False for invalid URLs."""
        from aragora.server.handlers.oauth import _validate_redirect_url

        assert _validate_redirect_url("not-a-valid-url") is False
        assert _validate_redirect_url("") is False

    def test_rejects_ftp_scheme(self):
        """Should reject ftp: scheme."""
        from aragora.server.handlers.oauth import _validate_redirect_url

        assert _validate_redirect_url("ftp://localhost/callback") is False


# ===========================================================================
# Test OAuthUserInfo Dataclass
# ===========================================================================


class TestOAuthUserInfo:
    """Tests for OAuthUserInfo dataclass."""

    def test_creates_with_required_fields(self):
        """Should create with required fields."""
        from aragora.server.handlers.oauth import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="123456",
            email="test@gmail.com",
            name="Test User",
        )

        assert user_info.provider == "google"
        assert user_info.provider_user_id == "123456"
        assert user_info.email == "test@gmail.com"
        assert user_info.name == "Test User"

    def test_default_values(self):
        """Should have correct default values."""
        from aragora.server.handlers.oauth import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="test@test.com",
            name="Test",
        )

        assert user_info.picture is None
        assert user_info.email_verified is False


# ===========================================================================
# Test _get_param() helper
# ===========================================================================


class TestGetParam:
    """Tests for _get_param() query parameter extraction."""

    def test_extracts_scalar_value(self):
        """Should extract scalar value from query params."""
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": "value"}, "key") == "value"

    def test_extracts_first_element_from_list(self):
        """Should extract first element when value is a list."""
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": ["first", "second"]}, "key") == "first"

    def test_returns_default_when_missing(self):
        """Should return default when key is missing."""
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({}, "key", "default") == "default"

    def test_returns_default_for_empty_list(self):
        """Should return default when value is an empty list."""
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": []}, "key", "fallback") == "fallback"

    def test_returns_none_when_no_default(self):
        """Should return None when key missing and no default."""
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({}, "key") is None


# ===========================================================================
# Test _handle_google_auth_start()
# ===========================================================================


class TestHandleGoogleAuthStart:
    """Tests for _handle_google_auth_start()."""

    def test_returns_503_when_not_configured(self, monkeypatch):
        """Should return 503 when Google OAuth not configured."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        monkeypatch.setattr(oauth, "_get_google_client_id", lambda: "")
        result = handler._handle_google_auth_start(mock_http, {})

        assert get_status(result) == 503
        assert "not configured" in get_body(result)["error"]

    def test_rejects_invalid_redirect_url(self):
        """Should reject invalid redirect URL."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_google_client_id", return_value="test-client-id"),
            patch.object(_oauth_impl, "_validate_redirect_url", return_value=False),
        ):
            result = handler._handle_google_auth_start(
                mock_http, {"redirect_url": ["https://evil.com/steal"]}
            )

        assert get_status(result) == 400
        assert "Invalid redirect URL" in get_body(result)["error"]

    def test_returns_redirect_to_google(self):
        """Should return redirect to Google OAuth."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_google_client_id", return_value="test-client-id"),
            patch.object(
                _oauth_impl,
                "_get_google_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
            patch.object(
                _oauth_impl, "_get_oauth_success_url", return_value="http://localhost:3000/success"
            ),
            patch.object(_oauth_impl, "_validate_redirect_url", return_value=True),
            patch("aragora.server.handlers._oauth_impl._generate_state", return_value="test-state"),
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=None),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_google_auth_start(mock_http, {})

        assert get_status(result) == 302
        assert "Location" in result.headers
        assert "accounts.google.com" in result.headers["Location"]
        assert "test-client-id" in result.headers["Location"]


# ===========================================================================
# Test _handle_google_callback()
# ===========================================================================


class TestHandleGoogleCallback:
    """Tests for _handle_google_callback()."""

    @pytest.mark.asyncio
    async def test_handles_error_from_google(self):
        """Should handle error from Google OAuth."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_google_callback(
                mock_http, {"error": ["access_denied"], "error_description": ["User cancelled"]}
            )

            mock_redirect.assert_called()

    @pytest.mark.asyncio
    async def test_missing_state_returns_error(self):
        """Should return error when state is missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_google_callback(mock_http, {})

            mock_redirect.assert_called()
            assert "Missing state" in str(mock_redirect.call_args)

    @pytest.mark.asyncio
    async def test_invalid_state_returns_error(self):
        """Should return error when state is invalid."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch("aragora.server.handlers._oauth_impl._validate_state", return_value=None),
            patch.object(handler, "_redirect_with_error") as mock_redirect,
        ):
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_google_callback(mock_http, {"state": ["invalid-state"]})

            mock_redirect.assert_called()
            assert "Invalid or expired" in str(mock_redirect.call_args)

    @pytest.mark.asyncio
    async def test_missing_code_returns_error(self):
        """Should return error when authorization code is missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch(
                "aragora.server.handlers._oauth_impl._validate_state",
                return_value={"user_id": None},
            ),
            patch.object(handler, "_redirect_with_error") as mock_redirect,
        ):
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_google_callback(mock_http, {"state": ["valid-state"]})

            mock_redirect.assert_called()
            assert "Missing authorization code" in str(mock_redirect.call_args)


# ===========================================================================
# Test _handle_github_auth_start()
# ===========================================================================


class TestHandleGitHubAuthStart:
    """Tests for _handle_github_auth_start()."""

    def test_returns_503_when_not_configured(self):
        """Should return 503 when GitHub OAuth not configured."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(_oauth_impl, "_get_github_client_id", return_value=""):
            result = handler._handle_github_auth_start(mock_http, {})

        assert get_status(result) == 503
        assert "not configured" in get_body(result)["error"]

    def test_returns_redirect_to_github(self):
        """Should return redirect to GitHub OAuth."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_github_client_id", return_value="gh-client-id"),
            patch.object(
                _oauth_impl,
                "_get_github_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
            patch.object(
                _oauth_impl, "_get_oauth_success_url", return_value="http://localhost:3000/success"
            ),
            patch.object(_oauth_impl, "_validate_redirect_url", return_value=True),
            patch("aragora.server.handlers._oauth_impl._generate_state", return_value="gh-state"),
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=None),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_github_auth_start(mock_http, {})

        assert get_status(result) == 302
        assert "Location" in result.headers
        assert "github.com" in result.headers["Location"]
        assert "gh-client-id" in result.headers["Location"]

    def test_rejects_invalid_redirect_url(self):
        """Should reject invalid redirect URL for GitHub."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_github_client_id", return_value="gh-client-id"),
            patch.object(_oauth_impl, "_validate_redirect_url", return_value=False),
        ):
            result = handler._handle_github_auth_start(
                mock_http, {"redirect_url": "https://evil.com"}
            )

        assert get_status(result) == 400


# ===========================================================================
# Test _handle_github_callback()
# ===========================================================================


class TestHandleGitHubCallback:
    """Tests for _handle_github_callback()."""

    @pytest.mark.asyncio
    async def test_handles_error_from_github(self):
        """Should redirect with error when GitHub returns error."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_github_callback(
                mock_http, {"error": "access_denied", "error_description": "Cancelled"}
            )
            mock_redirect.assert_called()

    @pytest.mark.asyncio
    async def test_missing_state_returns_error(self):
        """Should return error when state is missing in GitHub callback."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            await handler._handle_github_callback(mock_http, {})
            mock_redirect.assert_called()
            assert "Missing state" in str(mock_redirect.call_args)


# ===========================================================================
# Test _handle_microsoft_auth_start()
# ===========================================================================


class TestHandleMicrosoftAuthStart:
    """Tests for _handle_microsoft_auth_start()."""

    def test_returns_503_when_not_configured(self):
        """Should return 503 when Microsoft OAuth not configured."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(_oauth_impl, "_get_microsoft_client_id", return_value=""):
            result = handler._handle_microsoft_auth_start(mock_http, {})

        assert get_status(result) == 503
        assert "not configured" in get_body(result)["error"]

    def test_returns_redirect_to_microsoft(self):
        """Should return redirect to Microsoft OAuth."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_microsoft_client_id", return_value="ms-client-id"),
            patch.object(_oauth_impl, "_get_microsoft_tenant", return_value="common"),
            patch.object(
                _oauth_impl,
                "_get_microsoft_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
            patch.object(
                _oauth_impl, "_get_oauth_success_url", return_value="http://localhost:3000/success"
            ),
            patch.object(_oauth_impl, "_validate_redirect_url", return_value=True),
            patch("aragora.server.handlers._oauth_impl._generate_state", return_value="ms-state"),
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=None),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_microsoft_auth_start(mock_http, {})

        assert get_status(result) == 302
        assert "Location" in result.headers
        assert "login.microsoftonline.com" in result.headers["Location"]


# ===========================================================================
# Test _handle_list_providers()
# ===========================================================================


class TestHandleListProviders:
    """Tests for _handle_list_providers()."""

    def test_returns_google_when_configured(self):
        """Should include Google when client ID is set."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_google_client_id", return_value="g-id"),
            patch.object(_oauth_impl, "_get_github_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_microsoft_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_apple_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_issuer", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_client_id", return_value=""),
        ):
            result = handler._handle_list_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert len(body["providers"]) == 1
        assert body["providers"][0]["id"] == "google"

    def test_returns_empty_when_none_configured(self):
        """Should return empty providers list when nothing is configured."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_google_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_github_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_microsoft_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_apple_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_issuer", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_client_id", return_value=""),
        ):
            result = handler._handle_list_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert body["providers"] == []

    def test_returns_multiple_providers(self):
        """Should list all configured providers."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(_oauth_impl, "_get_google_client_id", return_value="g-id"),
            patch.object(_oauth_impl, "_get_github_client_id", return_value="gh-id"),
            patch.object(_oauth_impl, "_get_microsoft_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_apple_client_id", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_issuer", return_value=""),
            patch.object(_oauth_impl, "_get_oidc_client_id", return_value=""),
        ):
            result = handler._handle_list_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        ids = [p["id"] for p in body["providers"]]
        assert "google" in ids
        assert "github" in ids


# ===========================================================================
# Test _handle_get_user_providers()
# ===========================================================================


class TestHandleGetUserProviders:
    """Tests for _handle_get_user_providers()."""

    def test_unauthenticated_returns_401(self):
        """Should return 401 when not authenticated."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_get_user_providers(mock_http)

        assert get_status(result) == 401

    def test_returns_user_providers(self):
        """Should return user's linked OAuth providers."""
        handler = create_oauth_handler()
        mock_http = MockHandler()
        mock_user_store = MagicMock()
        mock_user_store.get_oauth_providers.return_value = [
            {"provider": "google", "email": "test@gmail.com"}
        ]

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_get_user_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "providers" in body


# ===========================================================================
# Test Route Dispatching
# ===========================================================================


class TestRouteDispatching:
    """Tests for handle() route dispatching."""

    def test_get_google_auth_routes_correctly(self):
        """GET /api/v1/auth/oauth/google should route to google auth start."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_google_auth_start") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/google", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_get_callback_routes_correctly(self):
        """GET /api/v1/auth/oauth/google/callback should route to google callback."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_google_callback") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/google/callback", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_get_github_auth_routes_correctly(self):
        """GET /api/v1/auth/oauth/github should route to github auth start."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_github_auth_start") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/github", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_get_microsoft_auth_routes_correctly(self):
        """GET /api/v1/auth/oauth/microsoft should route to microsoft auth start."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_microsoft_auth_start") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/microsoft", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_post_link_routes_correctly(self):
        """POST /api/v1/auth/oauth/link should route to link account."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="POST")

        with patch.object(handler, "_handle_link_account") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/link", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_delete_unlink_routes_correctly(self):
        """DELETE /api/v1/auth/oauth/unlink should route to unlink account."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")

        with patch.object(handler, "_handle_unlink_account") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/unlink", {}, mock_http, "DELETE")
            mock_method.assert_called_once()

    def test_get_providers_routes_correctly(self):
        """GET /api/v1/auth/oauth/providers should route to list providers."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_list_providers") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/providers", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_get_oauth_url_routes_correctly(self):
        """GET /api/v1/auth/oauth/url should route to oauth URL handler."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_oauth_url") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/url", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_unsupported_method_returns_405(self):
        """Unsupported method should return 405."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="PATCH")

        result = handler.handle("/api/v1/auth/oauth/google", {}, mock_http, "PATCH")

        assert get_status(result) == 405

    def test_non_v1_route_dispatches_correctly(self):
        """Non-v1 routes should also dispatch correctly."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_google_auth_start") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/auth/oauth/google", {}, mock_http, "GET")
            mock_method.assert_called_once()


# ===========================================================================
# Test _handle_oauth_url()
# ===========================================================================


class TestHandleOAuthUrl:
    """Tests for _handle_oauth_url() - get auth URL as JSON."""

    def test_missing_provider_returns_400(self):
        """Should return 400 when provider is missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        result = handler._handle_oauth_url(mock_http, {})

        assert get_status(result) == 400
        assert "required" in get_body(result)["error"].lower()

    def test_unsupported_provider_returns_400(self):
        """Should return 400 for unsupported provider."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        result = handler._handle_oauth_url(mock_http, {"provider": "facebook"})

        assert get_status(result) == 400
        assert "Unsupported" in get_body(result)["error"]

    def test_returns_auth_url_for_google(self):
        """Should return auth_url JSON for a valid provider."""
        from aragora.server.handlers.base import HandlerResult

        handler = create_oauth_handler()
        mock_http = MockHandler()

        fake_location = "https://accounts.google.com/o/oauth2/v2/auth?state=abc&client_id=gid"
        fake_result = HandlerResult(
            status_code=302,
            content_type="text/html",
            body=b"",
            headers={"Location": fake_location},
        )

        with patch.object(handler, "_handle_google_auth_start", return_value=fake_result):
            result = handler._handle_oauth_url(mock_http, {"provider": "google"})

        body = get_body(result)
        assert get_status(result) == 200
        assert "auth_url" in body
        assert "accounts.google.com" in body["auth_url"]


# ===========================================================================
# Test _handle_oauth_callback_api()
# ===========================================================================


class TestHandleOAuthCallbackApi:
    """Tests for _handle_oauth_callback_api() - POST-based callback."""

    def test_missing_body_returns_400(self):
        """Should return 400 when JSON body is invalid."""
        handler = create_oauth_handler()
        mock_http = MockHandler(body=b"not json")

        with patch.object(handler, "read_json_body", return_value=None):
            result = handler._handle_oauth_callback_api(mock_http)

        assert get_status(result) == 400

    def test_missing_fields_returns_400(self):
        """Should return 400 when required fields are missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "read_json_body", return_value={"provider": "google"}):
            result = handler._handle_oauth_callback_api(mock_http)

        assert get_status(result) == 400
        assert "required" in get_body(result)["error"]

    def test_unsupported_provider_returns_400(self):
        """Should return 400 for unsupported provider in callback API."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(
            handler,
            "read_json_body",
            return_value={"provider": "facebook", "code": "abc", "state": "xyz"},
        ):
            result = handler._handle_oauth_callback_api(mock_http)

        assert get_status(result) == 400
        assert "Unsupported" in get_body(result)["error"]


# ===========================================================================
# Test _handle_link_account()
# ===========================================================================


class TestHandleLinkAccount:
    """Tests for _handle_link_account()."""

    def test_unauthenticated_returns_401(self):
        """Should return 401 when not authenticated."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="POST")

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_link_account(mock_http)

        assert get_status(result) == 401

    def test_unsupported_provider_returns_400(self):
        """Should return 400 for unsupported provider in link request."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="POST")

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
            patch.object(handler, "read_json_body", return_value={"provider": "facebook"}),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_link_account(mock_http)

        assert get_status(result) == 400
        assert "Unsupported" in get_body(result)["error"]

    def test_invalid_body_returns_400(self):
        """Should return 400 when JSON body is invalid."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="POST")

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
            patch.object(handler, "read_json_body", return_value=None),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_link_account(mock_http)

        assert get_status(result) == 400


# ===========================================================================
# Test _handle_unlink_account()
# ===========================================================================


class TestHandleUnlinkAccount:
    """Tests for _handle_unlink_account()."""

    def test_unauthenticated_returns_401(self):
        """Should return 401 when not authenticated."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_unlink_account(mock_http)

        assert get_status(result) == 401

    def test_unsupported_provider_returns_400(self):
        """Should return 400 for unsupported provider in unlink request."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=MagicMock()),
            patch.object(handler, "read_json_body", return_value={"provider": "facebook"}),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_unlink_account(mock_http)

        assert get_status(result) == 400

    def test_no_password_returns_400(self):
        """Should return 400 when user has no password set (cannot unlink all auth)."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")
        mock_user = MockUser(password_hash="")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = mock_user

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"provider": "google"}),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_unlink_account(mock_http)

        assert get_status(result) == 400
        assert "password" in get_body(result)["error"].lower()

    def test_successful_unlink(self):
        """Should successfully unlink provider when conditions are met."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")
        mock_user = MockUser(password_hash="hashed_pw")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_user_store.unlink_oauth_provider.return_value = True

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"provider": "google"}),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_unlink_account(mock_http)

        assert get_status(result) == 200
        body = get_body(result)
        assert "Unlinked" in body["message"]

    def test_user_not_found_returns_404(self):
        """Should return 404 when user is not found."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = None

        with (
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"provider": "google"}),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user-123")
            result = handler._handle_unlink_account(mock_http)

        assert get_status(result) == 404


# ===========================================================================
# Test _redirect_with_error()
# ===========================================================================


class TestRedirectWithError:
    """Tests for _redirect_with_error() helper."""

    def test_redirects_to_error_url(self):
        """Should redirect to error URL with error message."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()

        with patch.object(
            _oauth_impl, "_get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = handler._redirect_with_error("Something went wrong")

        assert get_status(result) == 302
        assert "Location" in result.headers
        assert "error=" in result.headers["Location"]
        assert "Something" in result.headers["Location"]

    def test_includes_no_cache_headers(self):
        """Should include cache-control headers to prevent caching."""
        from aragora.server.handlers import _oauth_impl

        handler = create_oauth_handler()

        with patch.object(
            _oauth_impl, "_get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = handler._redirect_with_error("error")

        assert "Cache-Control" in result.headers
        assert "no-store" in result.headers["Cache-Control"]


# ===========================================================================
# Test _redirect_with_tokens()
# ===========================================================================


class TestRedirectWithTokens:
    """Tests for _redirect_with_tokens() helper."""

    def test_redirects_with_token_params(self):
        """Should redirect with tokens in query parameters."""
        handler = create_oauth_handler()
        tokens = MockTokenPair()

        result = handler._redirect_with_tokens("http://localhost:3000/callback", tokens)

        assert get_status(result) == 302
        location = result.headers["Location"]
        assert "access_token=mock-access-token" in location
        assert "refresh_token=mock-refresh-token" in location
        assert "token_type=Bearer" in location
        assert "expires_in=3600" in location

    def test_includes_no_cache_headers(self):
        """Should include no-cache headers on token redirect."""
        handler = create_oauth_handler()
        tokens = MockTokenPair()

        result = handler._redirect_with_tokens("http://localhost:3000/callback", tokens)

        assert "Cache-Control" in result.headers
        assert "no-store" in result.headers["Cache-Control"]
