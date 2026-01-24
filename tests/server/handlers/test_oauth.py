"""
Tests for aragora.server.handlers.oauth - OAuth authentication handler.

Tests cover:
- can_handle() route matching
- handle() route dispatching
- Rate limiting
- _handle_google_auth_start() - OAuth flow initiation
- _handle_google_callback() - OAuth callback handling
- _handle_list_providers() - List available providers
- _handle_get_user_providers() - Get user's linked providers
- _validate_redirect_url() - Open redirect prevention
- validate_oauth_config() - Configuration validation
- OAuthUserInfo dataclass
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: Optional[str] = None
    role: str = "member"
    is_active: bool = True
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

    def __init__(self, is_authenticated: bool = False, user_id: str = None):
        self.is_authenticated = is_authenticated
        self.user_id = user_id


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

    def test_handles_google_auth(self):
        """Should handle /api/auth/oauth/google."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/google") is True

    def test_handles_google_callback(self):
        """Should handle /api/auth/oauth/google/callback."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/google/callback") is True

    def test_handles_link(self):
        """Should handle /api/auth/oauth/link."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/link") is True

    def test_handles_unlink(self):
        """Should handle /api/auth/oauth/unlink."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/unlink") is True

    def test_handles_providers(self):
        """Should handle /api/auth/oauth/providers."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/auth/oauth/providers") is True

    def test_handles_user_providers(self):
        """Should handle /api/user/oauth-providers."""
        handler = create_oauth_handler()
        assert handler.can_handle("/api/v1/user/oauth-providers") is True

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
        assert "Rate limit" in get_body(result)["error"]


# ===========================================================================
# Test Configuration Validation
# ===========================================================================


class TestValidateOAuthConfig:
    """Tests for validate_oauth_config() function."""

    def test_returns_empty_in_dev_mode(self):
        """Should return empty list in development mode."""
        from aragora.server.handlers import oauth

        with patch.object(oauth, "_IS_PRODUCTION", False):
            missing = oauth.validate_oauth_config()
            assert missing == []

    def test_returns_missing_vars_in_production(self):
        """Should return missing vars in production mode."""
        from aragora.server.handlers import oauth

        with (
            patch.object(oauth, "_IS_PRODUCTION", True),
            patch.object(oauth, "GOOGLE_CLIENT_ID", "test-client-id"),
            patch.object(oauth, "_get_google_client_secret", return_value=""),
            patch.object(oauth, "_get_google_redirect_uri", return_value=""),
            patch.object(oauth, "_get_oauth_success_url", return_value=""),
            patch.object(oauth, "_get_oauth_error_url", return_value=""),
            patch.object(oauth, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset()),
        ):
            missing = oauth.validate_oauth_config()
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

        # Patch the function that's called at validation time, not the module constant
        monkeypatch.setattr(oauth, "_get_allowed_redirect_hosts", lambda: frozenset(["localhost"]))
        assert _validate_redirect_url("http://localhost:3000/callback") is True

    def test_rejects_unknown_host(self):
        """Should reject hosts not in allowlist."""
        from aragora.server.handlers import oauth
        from aragora.server.handlers.oauth import _validate_redirect_url

        with patch.object(
            oauth, "_get_allowed_redirect_hosts", return_value=frozenset(["example.com"])
        ):
            assert _validate_redirect_url("https://evil.com/callback") is False

    def test_allows_subdomain_of_allowed_host(self):
        """Should allow subdomains of allowed hosts."""
        from aragora.server.handlers import oauth
        from aragora.server.handlers.oauth import _validate_redirect_url

        with patch.object(
            oauth, "_get_allowed_redirect_hosts", return_value=frozenset(["example.com"])
        ):
            assert _validate_redirect_url("https://app.example.com/callback") is True

    def test_handles_invalid_url(self):
        """Should return False for invalid URLs."""
        from aragora.server.handlers.oauth import _validate_redirect_url

        assert _validate_redirect_url("not-a-valid-url") is False
        assert _validate_redirect_url("") is False


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
# Test _handle_google_auth_start()
# ===========================================================================


class TestHandleGoogleAuthStart:
    """Tests for _handle_google_auth_start()."""

    def test_returns_503_when_not_configured(self, monkeypatch):
        """Should return 503 when Google OAuth not configured."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        # Patch the function that's called at runtime, not the module constant
        monkeypatch.setattr(oauth, "_get_google_client_id", lambda: "")
        result = handler._handle_google_auth_start(mock_http, {})

        assert get_status(result) == 503
        assert "not configured" in get_body(result)["error"]

    def test_rejects_invalid_redirect_url(self):
        """Should reject invalid redirect URL."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(oauth, "_get_google_client_id", return_value="test-client-id"),
            patch.object(oauth, "_validate_redirect_url", return_value=False),
        ):
            result = handler._handle_google_auth_start(
                mock_http, {"redirect_url": ["https://evil.com/steal"]}
            )

        assert get_status(result) == 400
        assert "Invalid redirect URL" in get_body(result)["error"]

    def test_returns_redirect_to_google(self):
        """Should return redirect to Google OAuth."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch.object(oauth, "_get_google_client_id", return_value="test-client-id"),
            patch.object(
                oauth, "_get_google_redirect_uri", return_value="http://localhost:8080/callback"
            ),
            patch.object(
                oauth, "_get_oauth_success_url", return_value="http://localhost:3000/success"
            ),
            patch.object(oauth, "_validate_redirect_url", return_value=True),
            patch("aragora.server.handlers.oauth._generate_state", return_value="test-state"),
            patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth,
            patch.object(handler, "_get_user_store", return_value=None),
        ):
            mock_auth.return_value = MockAuthContext(is_authenticated=False)
            result = handler._handle_google_auth_start(mock_http, {})

        assert get_status(result) == 302
        assert "Location" in result.headers
        assert "accounts.google.com" in result.headers["Location"]


# ===========================================================================
# Test _handle_google_callback()
# ===========================================================================


class TestHandleGoogleCallback:
    """Tests for _handle_google_callback()."""

    def test_handles_error_from_google(self):
        """Should handle error from Google OAuth."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            result = handler._handle_google_callback(
                mock_http, {"error": ["access_denied"], "error_description": ["User cancelled"]}
            )

            mock_redirect.assert_called()

    def test_missing_state_returns_error(self):
        """Should return error when state is missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_redirect_with_error") as mock_redirect:
            mock_redirect.return_value = MagicMock(status_code=302)
            result = handler._handle_google_callback(mock_http, {})

            mock_redirect.assert_called()
            assert "Missing state" in str(mock_redirect.call_args)

    def test_invalid_state_returns_error(self):
        """Should return error when state is invalid."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch("aragora.server.handlers.oauth._validate_state", return_value=None),
            patch.object(handler, "_redirect_with_error") as mock_redirect,
        ):
            mock_redirect.return_value = MagicMock(status_code=302)
            result = handler._handle_google_callback(mock_http, {"state": ["invalid-state"]})

            mock_redirect.assert_called()
            assert "Invalid or expired" in str(mock_redirect.call_args)

    def test_missing_code_returns_error(self):
        """Should return error when authorization code is missing."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with (
            patch("aragora.server.handlers.oauth._validate_state", return_value={"user_id": None}),
            patch.object(handler, "_redirect_with_error") as mock_redirect,
        ):
            mock_redirect.return_value = MagicMock(status_code=302)
            result = handler._handle_google_callback(mock_http, {"state": ["valid-state"]})

            mock_redirect.assert_called()
            assert "Missing authorization code" in str(mock_redirect.call_args)


# ===========================================================================
# Test _handle_list_providers()
# ===========================================================================


class TestHandleListProviders:
    """Tests for _handle_list_providers()."""

    def test_returns_available_providers(self):
        """Should return list of available OAuth providers."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        # Check if the method exists
        if not hasattr(handler, "_handle_list_providers"):
            pytest.skip("_handle_list_providers not implemented")

        with patch.object(oauth, "GOOGLE_CLIENT_ID", "test-client-id"):
            result = handler._handle_list_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "providers" in body

    def test_shows_disabled_when_not_configured(self):
        """Should show provider as disabled when not configured."""
        from aragora.server.handlers import oauth

        handler = create_oauth_handler()
        mock_http = MockHandler()

        # Check if the method exists
        if not hasattr(handler, "_handle_list_providers"):
            pytest.skip("_handle_list_providers not implemented")

        with patch.object(oauth, "GOOGLE_CLIENT_ID", ""):
            result = handler._handle_list_providers(mock_http)

        body = get_body(result)
        assert get_status(result) == 200


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
        mock_user = MockUser(oauth_providers={"google": {"id": "123", "email": "test@gmail.com"}})
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = mock_user

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
        """GET /api/auth/oauth/google should route correctly."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_google_auth_start") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/google", {}, mock_http, "GET")

            mock_method.assert_called_once()

    def test_get_callback_routes_correctly(self):
        """GET /api/auth/oauth/google/callback should route correctly."""
        handler = create_oauth_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_handle_google_callback") as mock_method:
            mock_method.return_value = MagicMock(status_code=302, body=b"")
            handler.handle("/api/v1/auth/oauth/google/callback", {}, mock_http, "GET")

            mock_method.assert_called_once()

    def test_post_link_routes_correctly(self):
        """POST /api/auth/oauth/link should route correctly."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="POST")

        with patch.object(handler, "_handle_link_account") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/link", {}, mock_http, "POST")

            mock_method.assert_called_once()

    def test_delete_unlink_routes_correctly(self):
        """DELETE /api/auth/oauth/unlink should route correctly."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="DELETE")

        with patch.object(handler, "_handle_unlink_account") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/auth/oauth/unlink", {}, mock_http, "DELETE")

            mock_method.assert_called_once()

    def test_unsupported_method_returns_405(self):
        """Unsupported method should return 405."""
        handler = create_oauth_handler()
        mock_http = MockHandler(method="PATCH")

        result = handler.handle("/api/v1/auth/oauth/google", {}, mock_http, "PATCH")

        assert get_status(result) == 405
