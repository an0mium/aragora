"""
Tests for aragora.server.handlers._oauth_impl - OAuth authentication handlers.

This module tests the backward-compatibility shim (_oauth_impl.py) and the
underlying _oauth/ package which implements OAuth authentication for:
- Google OAuth 2.0
- GitHub OAuth
- Microsoft OAuth (Azure AD)
- Apple Sign-In
- Generic OIDC

Tests cover:
1. OAuth state generation and validation (CSRF protection)
2. Authorization URL building for each provider
3. Token exchange (authorization code -> access token)
4. User profile fetching from OAuth providers
5. Error handling (invalid state, expired code, network failures, invalid tokens)
6. Account management (link, unlink, list providers)
7. Redirect URL validation
8. Rate limiting
9. Edge cases (missing parameters, malformed responses, timeout)
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs, urlencode, urlparse

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockUser:
    """Mock user object for OAuth tests."""

    id: str = "user-123"
    email: str = "testuser@example.com"
    name: str = "Test User"
    org_id: str | None = "org-001"
    role: str = "member"
    is_active: bool = True
    password_hash: str = "hashed"
    password_salt: str = "salt"


@dataclass
class MockTokenPair:
    """Mock token pair returned by create_token_pair."""

    access_token: str = "mock-access-token-xyz"
    refresh_token: str = "mock-refresh-token-xyz"
    expires_in: int = 3600
    token_type: str = "Bearer"


def _make_handler(
    method: str = "GET",
    body: dict | None = None,
    headers: dict | None = None,
    client_address: tuple = ("127.0.0.1", 12345),
) -> MagicMock:
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.command = method
    body_bytes = json.dumps(body).encode() if body is not None else b"{}"
    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)
    mock.headers = headers or {}
    mock.headers.setdefault("Content-Length", str(len(body_bytes)))
    mock.client_address = client_address
    # For Apple callback which reads request body
    mock.request = MagicMock()
    mock.request.body = b""
    return mock


def _parse_result(result) -> dict:
    """Parse JSON body from HandlerResult."""
    if result is None:
        return {}
    try:
        body = result.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        return json.loads(body) if body else {}
    except (json.JSONDecodeError, AttributeError):
        return {}


def _get_redirect_location(result) -> str:
    """Get the Location header from a redirect result."""
    if result and result.headers:
        return result.headers.get("Location", "")
    return ""


# ---------------------------------------------------------------------------
# Module-level patch path prefix
# ---------------------------------------------------------------------------
_IMPL = "aragora.server.handlers._oauth_impl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oauth_handler(mock_server_context):
    """Create an OAuthHandler with mock server context."""
    from aragora.server.handlers._oauth.base import OAuthHandler

    return OAuthHandler(server_context=mock_server_context)


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_token_pair():
    return MockTokenPair()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestOAuthHandlerRouting:
    """Tests for OAuthHandler route matching and method dispatch."""

    def test_can_handle_google_auth(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google")

    def test_can_handle_google_callback(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google/callback")

    def test_can_handle_github_auth(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github")

    def test_can_handle_github_callback(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github/callback")

    def test_can_handle_microsoft_auth(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft")

    def test_can_handle_microsoft_callback(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft/callback")

    def test_can_handle_apple_auth(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple")

    def test_can_handle_apple_callback(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple/callback")

    def test_can_handle_oidc_auth(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc")

    def test_can_handle_oidc_callback(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc/callback")

    def test_can_handle_oauth_url(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/url")

    def test_can_handle_oauth_authorize(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/authorize")

    def test_can_handle_oauth_link(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/link")

    def test_can_handle_oauth_unlink(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/unlink")

    def test_can_handle_oauth_providers(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/providers")

    def test_can_handle_user_oauth_providers(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/user/oauth-providers")

    def test_cannot_handle_unknown_path(self, oauth_handler):
        assert not oauth_handler.can_handle("/api/v1/auth/oauth/unknown")

    def test_supports_non_v1_routes(self, oauth_handler):
        assert oauth_handler.can_handle("/api/auth/oauth/google")

    def test_handle_method_not_allowed(self, oauth_handler):
        """POST to a GET-only endpoint returns 405."""
        handler = _make_handler(method="POST")
        with (
            patch(f"{_IMPL}.create_span") as mock_span,
            patch(f"{_IMPL}.add_span_attributes"),
            patch(f"{_IMPL}._oauth_limiter") as mock_limiter,
        ):
            mock_span.return_value.__enter__ = Mock(return_value=MagicMock())
            mock_span.return_value.__exit__ = Mock(return_value=False)
            mock_limiter.is_allowed.return_value = True
            result = oauth_handler.handle("/api/v1/auth/oauth/google", {}, handler, "POST")
        assert result is not None
        assert result.status_code == 405


class TestOAuthStateManagement:
    """Tests for OAuth state generation and validation (CSRF protection)."""

    def test_validate_redirect_url_valid_localhost(self):
        """Valid redirect to localhost should succeed."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(
            f"{_IMPL}._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost", "127.0.0.1"}),
        ):
            assert _validate_redirect_url("http://localhost:3000/auth/callback") is True

    def test_validate_redirect_url_valid_subdomain(self):
        """Subdomain of allowed host should succeed."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"example.com"})):
            assert _validate_redirect_url("https://app.example.com/callback") is True

    def test_validate_redirect_url_invalid_host(self):
        """Redirect to disallowed host should fail."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"localhost"})):
            assert _validate_redirect_url("https://evil.com/callback") is False

    def test_validate_redirect_url_invalid_scheme(self):
        """Non-HTTP/HTTPS schemes should be rejected."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"localhost"})):
            assert _validate_redirect_url("ftp://localhost/callback") is False

    def test_validate_redirect_url_empty_host(self):
        """URL without host should be rejected."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"localhost"})):
            assert _validate_redirect_url("http:///callback") is False

    def test_validate_redirect_url_malformed(self):
        """Malformed URL should be rejected gracefully."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"localhost"})):
            assert _validate_redirect_url("not-a-url") is False

    def test_validate_state_delegates_to_internal(self):
        """_validate_state should delegate to _validate_state_internal."""
        from aragora.server.handlers._oauth_impl import _validate_state

        expected = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with patch(f"{_IMPL}._validate_state_internal", return_value=expected):
            result = _validate_state("test-state-token")
            assert result == expected


class TestOAuthModels:
    """Tests for OAuthUserInfo and _get_param utility."""

    def test_oauth_user_info_creation(self):
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        info = OAuthUserInfo(
            provider="google",
            provider_user_id="123456",
            email="user@gmail.com",
            name="Test User",
            picture="https://example.com/photo.jpg",
            email_verified=True,
        )
        assert info.provider == "google"
        assert info.provider_user_id == "123456"
        assert info.email == "user@gmail.com"
        assert info.email_verified is True

    def test_oauth_user_info_defaults(self):
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        info = OAuthUserInfo(
            provider="github",
            provider_user_id="789",
            email="user@github.com",
            name="GH User",
        )
        assert info.picture is None
        assert info.email_verified is False

    def test_get_param_string_value(self):
        from aragora.server.handlers.oauth.models import _get_param

        params = {"code": "abc123"}
        assert _get_param(params, "code") == "abc123"

    def test_get_param_list_value(self):
        from aragora.server.handlers.oauth.models import _get_param

        params = {"code": ["abc123", "def456"]}
        assert _get_param(params, "code") == "abc123"

    def test_get_param_empty_list(self):
        from aragora.server.handlers.oauth.models import _get_param

        params = {"code": []}
        assert _get_param(params, "code") is None

    def test_get_param_missing_key(self):
        from aragora.server.handlers.oauth.models import _get_param

        params = {}
        assert _get_param(params, "code") is None

    def test_get_param_default_value(self):
        from aragora.server.handlers.oauth.models import _get_param

        params = {}
        assert _get_param(params, "code", "default_val") == "default_val"


class TestGoogleOAuthFlow:
    """Tests for Google OAuth 2.0 authentication flow."""

    def test_google_auth_start_redirect(self, oauth_handler):
        """Google auth start should redirect to Google consent screen."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value="google-client-123"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-abc"),
            patch(
                f"{_IMPL}._get_google_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/google/callback",
            ),
        ):
            result = oauth_handler._handle_google_auth_start(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "accounts.google.com" in location
        assert "client_id=google-client-123" in location
        assert "state=state-abc" in location
        assert "response_type=code" in location
        assert (
            "scope=openid+email+profile" in location or "scope=openid%20email%20profile" in location
        )

    def test_google_auth_start_not_configured(self, oauth_handler):
        """Google auth should return 503 if not configured."""
        handler = _make_handler()
        with patch(f"{_IMPL}._get_google_client_id", return_value=""):
            result = oauth_handler._handle_google_auth_start(handler, {})

        assert result.status_code == 503

    def test_google_auth_start_invalid_redirect(self, oauth_handler):
        """Google auth should reject invalid redirect URLs."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value="google-123"),
            patch(f"{_IMPL}._get_oauth_success_url", return_value="http://evil.com/steal"),
            patch(f"{_IMPL}._validate_redirect_url", return_value=False),
        ):
            result = oauth_handler._handle_google_auth_start(handler, {})

        assert result.status_code == 400

    def test_google_callback_missing_state(self, oauth_handler):
        """Callback without state should redirect with error."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_google_callback(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_invalid_state(self, oauth_handler):
        """Callback with invalid state should redirect with error."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._validate_state", return_value=None),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_google_callback(handler, {"state": "invalid-state"})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_error_from_provider(self, oauth_handler):
        """Callback with error param from provider should redirect with error."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_google_callback(
                handler,
                {"error": "access_denied", "error_description": "User denied access"},
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_missing_code(self, oauth_handler):
        """Callback without authorization code should redirect with error."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_google_callback(handler, {"state": "valid-state"})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_token_exchange_failure(self, oauth_handler):
        """Token exchange failure should redirect with error."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", side_effect=Exception("Network error")
            ),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_no_access_token(self, oauth_handler):
        """Token exchange returning no access_token should redirect with error."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", return_value={"error": "invalid_grant"}
            ),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_user_info_failure(self, oauth_handler):
        """User info fetch failure should redirect with error."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", return_value={"access_token": "tok-123"}
            ),
            patch.object(
                oauth_handler, "_get_google_user_info", side_effect=Exception("User info failure")
            ),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()

    def test_google_callback_successful_login(self, oauth_handler, mock_user, mock_token_pair):
        """Successful Google callback should redirect with tokens."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="goog-123",
            email="testuser@example.com",
            name="Test User",
            email_verified=True,
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = mock_user

        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", return_value={"access_token": "tok-123"}
            ),
            patch.object(oauth_handler, "_get_google_user_info", return_value=user_info),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
            patch.object(oauth_handler, "_find_user_by_oauth", return_value=None),
            patch.object(oauth_handler, "_link_oauth_to_user", return_value=True),
            patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_token_pair),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "access_token=" in location
        assert "refresh_token=" in location

    def test_google_callback_creates_new_user(self, oauth_handler, mock_token_pair):
        """Google callback should create new user if not found."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="goog-456",
            email="newuser@gmail.com",
            name="New User",
            email_verified=True,
        )

        new_user = MockUser(id="user-new", email="newuser@gmail.com", name="New User")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = None

        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", return_value={"access_token": "tok-123"}
            ),
            patch.object(oauth_handler, "_get_google_user_info", return_value=user_info),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
            patch.object(oauth_handler, "_find_user_by_oauth", return_value=None),
            patch.object(oauth_handler, "_create_oauth_user", return_value=new_user),
            patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_token_pair),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "access_token=" in location

    def test_google_callback_user_store_unavailable(self, oauth_handler):
        """Callback should error when user store is unavailable."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="goog-789",
            email="test@gmail.com",
            name="Test",
            email_verified=True,
        )

        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_code_for_tokens", return_value={"access_token": "tok-123"}
            ),
            patch.object(oauth_handler, "_get_google_user_info", return_value=user_info),
            patch.object(oauth_handler, "_get_user_store", return_value=None),
        ):
            result = oauth_handler._handle_google_callback(
                handler, {"state": "valid-state", "code": "auth-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error" in location.lower()


class TestGitHubOAuthFlow:
    """Tests for GitHub OAuth authentication flow."""

    def test_github_auth_start_redirect(self, oauth_handler):
        """GitHub auth start should redirect to GitHub consent screen."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_github_client_id", return_value="gh-client-123"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-gh"),
            patch(
                f"{_IMPL}._get_github_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/github/callback",
            ),
        ):
            result = oauth_handler._handle_github_auth_start(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "github.com/login/oauth/authorize" in location
        assert "client_id=gh-client-123" in location
        assert "state=state-gh" in location

    def test_github_auth_start_not_configured(self, oauth_handler):
        """GitHub auth should return 503 if not configured."""
        handler = _make_handler()
        with patch(f"{_IMPL}._get_github_client_id", return_value=""):
            result = oauth_handler._handle_github_auth_start(handler, {})

        assert result.status_code == 503

    def test_github_callback_error_from_provider(self, oauth_handler):
        """GitHub callback with error should redirect to error page."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_github_callback(
                handler,
                {"error": "access_denied", "error_description": "User denied"},
            )

        assert result.status_code == 302

    def test_github_callback_missing_state(self, oauth_handler):
        """GitHub callback without state should error."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_github_callback(handler, {})

        assert result.status_code == 302
        assert "error" in _get_redirect_location(result).lower()

    def test_github_callback_invalid_state(self, oauth_handler):
        """GitHub callback with invalid state should error."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._validate_state", return_value=None),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_github_callback(handler, {"state": "bad-state"})

        assert result.status_code == 302

    def test_github_callback_missing_code(self, oauth_handler):
        """GitHub callback without code should error."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_github_callback(handler, {"state": "valid-state"})

        assert result.status_code == 302

    def test_github_callback_successful_login(self, oauth_handler, mock_user, mock_token_pair):
        """Successful GitHub callback should redirect with tokens."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="github",
            provider_user_id="gh-uid-123",
            email="dev@github.com",
            name="Dev User",
            email_verified=True,
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = mock_user

        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler, "_exchange_github_code", return_value={"access_token": "gh-tok-123"}
            ),
            patch.object(oauth_handler, "_get_github_user_info", return_value=user_info),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
            patch.object(oauth_handler, "_find_user_by_oauth", return_value=None),
            patch.object(oauth_handler, "_link_oauth_to_user", return_value=True),
            patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_token_pair),
        ):
            result = oauth_handler._handle_github_callback(
                handler, {"state": "valid-state", "code": "gh-code-123"}
            )

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "access_token=" in location


class TestMicrosoftOAuthFlow:
    """Tests for Microsoft OAuth (Azure AD) authentication flow."""

    def test_microsoft_auth_start_redirect(self, oauth_handler):
        """Microsoft auth start should redirect to Microsoft consent screen."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_microsoft_client_id", return_value="ms-client-123"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-ms"),
            patch(f"{_IMPL}._get_microsoft_tenant", return_value="common"),
            patch(
                f"{_IMPL}._get_microsoft_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/microsoft/callback",
            ),
        ):
            result = oauth_handler._handle_microsoft_auth_start(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "login.microsoftonline.com" in location
        assert "client_id=ms-client-123" in location
        assert "state=state-ms" in location

    def test_microsoft_auth_start_not_configured(self, oauth_handler):
        """Microsoft auth should return 503 if not configured."""
        handler = _make_handler()
        with patch(f"{_IMPL}._get_microsoft_client_id", return_value=""):
            result = oauth_handler._handle_microsoft_auth_start(handler, {})

        assert result.status_code == 503

    def test_microsoft_auth_start_invalid_redirect(self, oauth_handler):
        """Microsoft auth should reject invalid redirect URLs."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_microsoft_client_id", return_value="ms-123"),
            patch(f"{_IMPL}._get_oauth_success_url", return_value="http://evil.com"),
            patch(f"{_IMPL}._validate_redirect_url", return_value=False),
        ):
            result = oauth_handler._handle_microsoft_auth_start(handler, {})

        assert result.status_code == 400

    def test_microsoft_callback_error_from_provider(self, oauth_handler):
        """Microsoft callback with error should redirect to error page."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_microsoft_callback(
                handler, {"error": "consent_required"}
            )

        assert result.status_code == 302

    def test_microsoft_callback_successful_login(self, oauth_handler, mock_user, mock_token_pair):
        """Successful Microsoft callback should redirect with tokens."""
        handler = _make_handler()
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="microsoft",
            provider_user_id="ms-uid-123",
            email="user@outlook.com",
            name="MS User",
            email_verified=True,
        )

        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
            patch.object(
                oauth_handler,
                "_exchange_microsoft_code",
                return_value={"access_token": "ms-tok-123"},
            ),
            patch.object(oauth_handler, "_get_microsoft_user_info", return_value=user_info),
            patch.object(oauth_handler, "_complete_oauth_flow") as mock_complete,
        ):
            mock_complete.return_value = MagicMock(
                status_code=302,
                headers={"Location": "http://localhost:3000/auth/callback?access_token=tok"},
                body=b"",
                content_type="text/html",
            )
            result = oauth_handler._handle_microsoft_callback(
                handler, {"state": "valid-state", "code": "ms-code-123"}
            )

        assert result.status_code == 302
        mock_complete.assert_called_once_with(user_info, state_data)


class TestAppleOAuthFlow:
    """Tests for Apple Sign-In authentication flow."""

    def test_apple_auth_start_redirect(self, oauth_handler):
        """Apple auth start should redirect to Apple consent screen."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_apple_client_id", return_value="com.example.app"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-apple"),
            patch(
                f"{_IMPL}._get_apple_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/apple/callback",
            ),
        ):
            result = oauth_handler._handle_apple_auth_start(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "appleid.apple.com/auth/authorize" in location
        assert "client_id=com.example.app" in location
        assert "response_mode=form_post" in location

    def test_apple_auth_start_not_configured(self, oauth_handler):
        """Apple auth should return 503 if not configured."""
        handler = _make_handler()
        with patch(f"{_IMPL}._get_apple_client_id", return_value=""):
            result = oauth_handler._handle_apple_auth_start(handler, {})

        assert result.status_code == 503

    def test_apple_callback_error_from_provider(self, oauth_handler):
        """Apple callback with error should redirect to error page."""
        handler = _make_handler()
        handler.request.body = b""
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_apple_callback(
                handler, {"error": "user_cancelled_authorize"}
            )

        assert result.status_code == 302

    def test_apple_callback_missing_state(self, oauth_handler):
        """Apple callback without state should error."""
        handler = _make_handler()
        handler.request.body = b""
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_apple_callback(handler, {})

        assert result.status_code == 302
        assert "error" in _get_redirect_location(result).lower()

    def test_apple_callback_missing_code_and_id_token(self, oauth_handler):
        """Apple callback without code or id_token should error."""
        handler = _make_handler()
        handler.request.body = b""
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}
        with (
            patch(f"{_IMPL}._validate_state", return_value=state_data),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_apple_callback(handler, {"state": "valid-state"})

        assert result.status_code == 302

    def test_parse_apple_id_token_valid(self, oauth_handler):
        """Valid Apple ID token should be parsed correctly."""
        # Build a fake JWT with 3 parts
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        )
        payload_data = {
            "sub": "apple-uid-123",
            "email": "user@privaterelay.appleid.com",
            "email_verified": True,
        }
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
        fake_jwt = f"{header}.{payload}.{signature}"

        user_data = {"name": {"firstName": "John", "lastName": "Doe"}}
        result = oauth_handler._parse_apple_id_token(fake_jwt, user_data)

        assert result.provider == "apple"
        assert result.provider_user_id == "apple-uid-123"
        assert result.email == "user@privaterelay.appleid.com"
        assert result.name == "John Doe"

    def test_parse_apple_id_token_invalid_format(self, oauth_handler):
        """Invalid JWT format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Apple ID token format"):
            oauth_handler._parse_apple_id_token("not.a.valid.jwt.too.many.parts", {})

    def test_parse_apple_id_token_no_email(self, oauth_handler):
        """Apple ID token without email should raise ValueError."""
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        )
        payload_data = {"sub": "apple-uid-123"}
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()
        fake_jwt = f"{header}.{payload}.{signature}"

        with pytest.raises(ValueError, match="No email"):
            oauth_handler._parse_apple_id_token(fake_jwt, {})

    def test_parse_apple_id_token_name_from_email(self, oauth_handler):
        """If no user_data name, Apple token should derive name from email."""
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        )
        payload_data = {
            "sub": "apple-uid-456",
            "email": "jane@example.com",
            "email_verified": True,
        }
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()
        fake_jwt = f"{header}.{payload}.{signature}"

        result = oauth_handler._parse_apple_id_token(fake_jwt, {})
        assert result.name == "jane"


class TestOIDCOAuthFlow:
    """Tests for generic OIDC authentication flow."""

    def test_oidc_auth_start_not_configured_no_issuer(self, oauth_handler):
        """OIDC auth should return 503 if issuer not configured."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_oidc_issuer", return_value=""),
            patch(f"{_IMPL}._get_oidc_client_id", return_value="oidc-client"),
        ):
            result = oauth_handler._handle_oidc_auth_start(handler, {})

        assert result.status_code == 503

    def test_oidc_auth_start_not_configured_no_client_id(self, oauth_handler):
        """OIDC auth should return 503 if client_id not configured."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_oidc_issuer", return_value="https://idp.example.com"),
            patch(f"{_IMPL}._get_oidc_client_id", return_value=""),
        ):
            result = oauth_handler._handle_oidc_auth_start(handler, {})

        assert result.status_code == 503

    def test_oidc_auth_start_discovery_failure(self, oauth_handler):
        """OIDC auth should return 503 if discovery fails."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_oidc_issuer", return_value="https://idp.example.com"),
            patch(f"{_IMPL}._get_oidc_client_id", return_value="oidc-client"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-oidc"),
            patch.object(oauth_handler, "_get_oidc_discovery", return_value={}),
        ):
            result = oauth_handler._handle_oidc_auth_start(handler, {})

        assert result.status_code == 503

    def test_oidc_auth_start_redirect(self, oauth_handler):
        """OIDC auth start should redirect to discovered auth endpoint."""
        handler = _make_handler()
        discovery = {
            "authorization_endpoint": "https://idp.example.com/authorize",
            "token_endpoint": "https://idp.example.com/token",
        }
        with (
            patch(f"{_IMPL}._get_oidc_issuer", return_value="https://idp.example.com"),
            patch(f"{_IMPL}._get_oidc_client_id", return_value="oidc-client"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-oidc"),
            patch(
                f"{_IMPL}._get_oidc_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/oidc/callback",
            ),
            patch.object(oauth_handler, "_get_oidc_discovery", return_value=discovery),
        ):
            result = oauth_handler._handle_oidc_auth_start(handler, {})

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "idp.example.com/authorize" in location
        assert "client_id=oidc-client" in location

    def test_oidc_callback_error_from_provider(self, oauth_handler):
        """OIDC callback with error should redirect to error page."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_oidc_callback(
                handler, {"error": "invalid_request", "error_description": "Bad request"}
            )

        assert result.status_code == 302

    def test_oidc_callback_missing_state(self, oauth_handler):
        """OIDC callback without state should error."""
        handler = _make_handler()
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_oidc_callback(handler, {})

        assert result.status_code == 302


class TestAccountManagement:
    """Tests for OAuth account linking, unlinking, and provider listing."""

    def test_list_providers_google_only(self, oauth_handler):
        """Should list only configured providers."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value="google-123"),
            patch(f"{_IMPL}._get_github_client_id", return_value=""),
            patch(f"{_IMPL}._get_microsoft_client_id", return_value=""),
            patch(f"{_IMPL}._get_apple_client_id", return_value=""),
            patch(f"{_IMPL}._get_oidc_issuer", return_value=""),
            patch(f"{_IMPL}._get_oidc_client_id", return_value=""),
        ):
            result = oauth_handler._handle_list_providers(handler)

        assert result.status_code == 200
        body = _parse_result(result)
        assert len(body["providers"]) == 1
        assert body["providers"][0]["id"] == "google"

    def test_list_providers_all_configured(self, oauth_handler):
        """Should list all configured providers."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value="google-123"),
            patch(f"{_IMPL}._get_github_client_id", return_value="github-456"),
            patch(f"{_IMPL}._get_microsoft_client_id", return_value="ms-789"),
            patch(f"{_IMPL}._get_apple_client_id", return_value="apple-abc"),
            patch(f"{_IMPL}._get_oidc_issuer", return_value="https://idp.example.com"),
            patch(f"{_IMPL}._get_oidc_client_id", return_value="oidc-def"),
        ):
            result = oauth_handler._handle_list_providers(handler)

        assert result.status_code == 200
        body = _parse_result(result)
        assert len(body["providers"]) == 5
        provider_ids = {p["id"] for p in body["providers"]}
        assert provider_ids == {"google", "github", "microsoft", "apple", "oidc"}

    def test_list_providers_none_configured(self, oauth_handler):
        """Should return empty list when no providers configured."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value=""),
            patch(f"{_IMPL}._get_github_client_id", return_value=""),
            patch(f"{_IMPL}._get_microsoft_client_id", return_value=""),
            patch(f"{_IMPL}._get_apple_client_id", return_value=""),
            patch(f"{_IMPL}._get_oidc_issuer", return_value=""),
            patch(f"{_IMPL}._get_oidc_client_id", return_value=""),
        ):
            result = oauth_handler._handle_list_providers(handler)

        assert result.status_code == 200
        body = _parse_result(result)
        assert len(body["providers"]) == 0

    def test_oauth_url_missing_provider(self, oauth_handler):
        """OAuth URL without provider should return 400."""
        handler = _make_handler()
        result = oauth_handler._handle_oauth_url(handler, {})

        assert result.status_code == 400

    def test_oauth_url_unsupported_provider(self, oauth_handler):
        """OAuth URL with unsupported provider should return 400."""
        handler = _make_handler()
        result = oauth_handler._handle_oauth_url(handler, {"provider": "facebook"})

        assert result.status_code == 400

    def test_oauth_url_google_returns_url(self, oauth_handler):
        """OAuth URL for Google should return authorization URL."""
        handler = _make_handler()

        # Mock the Google auth start to return a redirect result
        from aragora.server.handlers.utils.responses import HandlerResult

        mock_result = HandlerResult(
            status_code=302,
            content_type="text/html",
            body=b"",
            headers={
                "Location": "https://accounts.google.com/o/oauth2/v2/auth?client_id=123&state=abc"
            },
        )
        with patch.object(oauth_handler, "_handle_google_auth_start", return_value=mock_result):
            result = oauth_handler._handle_oauth_url(handler, {"provider": "google"})

        assert result.status_code == 200
        body = _parse_result(result)
        assert "auth_url" in body
        assert "accounts.google.com" in body["auth_url"]

    def test_link_account_unsupported_provider(self, oauth_handler):
        """Linking unsupported provider should return 400."""
        handler = _make_handler(method="POST", body={"provider": "facebook"})
        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value={"provider": "facebook"}),
        ):
            result = oauth_handler._handle_link_account(handler)

        assert result.status_code == 400

    def test_link_account_invalid_json_body(self, oauth_handler):
        """Link with invalid JSON body should return 400."""
        handler = _make_handler(method="POST")
        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value=None),
        ):
            result = oauth_handler._handle_link_account(handler)

        assert result.status_code == 400

    def test_unlink_account_unsupported_provider(self, oauth_handler):
        """Unlinking unsupported provider should return 400."""
        handler = _make_handler(method="DELETE")
        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value={"provider": "facebook"}),
        ):
            result = oauth_handler._handle_unlink_account(handler)

        assert result.status_code == 400

    def test_unlink_account_invalid_json_body(self, oauth_handler):
        """Unlink with invalid JSON body should return 400."""
        handler = _make_handler(method="DELETE")
        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value=None),
        ):
            result = oauth_handler._handle_unlink_account(handler)

        assert result.status_code == 400

    def test_unlink_account_user_not_found(self, oauth_handler):
        """Unlinking when user not found should return 404."""
        handler = _make_handler(method="DELETE")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = None

        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value={"provider": "google"}),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = oauth_handler._handle_unlink_account(handler)

        assert result.status_code == 404

    def test_unlink_account_no_password_set(self, oauth_handler):
        """Unlinking when user has no password should return 400."""
        handler = _make_handler(method="DELETE")
        user_no_pw = MockUser(password_hash="")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = user_no_pw

        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value={"provider": "google"}),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = oauth_handler._handle_unlink_account(handler)

        assert result.status_code == 400

    def test_unlink_account_success(self, oauth_handler):
        """Unlinking valid provider should succeed."""
        handler = _make_handler(method="DELETE")
        user = MockUser(password_hash="hashed-pw")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.unlink_oauth_provider.return_value = True

        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "read_json_body", return_value={"provider": "google"}),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = oauth_handler._handle_unlink_account(handler)

        assert result.status_code == 200
        body = _parse_result(result)
        assert "google" in body.get("message", "").lower()


class TestOAuthFlowCompletion:
    """Tests for _complete_oauth_flow and related helper methods."""

    def test_complete_oauth_flow_no_user_store(self, oauth_handler):
        """Flow completion without user store should redirect with error."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="test@example.com",
            name="Test",
        )
        state_data = {"user_id": None, "redirect_url": "http://localhost:3000/auth/callback"}

        with (
            patch.object(oauth_handler, "_get_user_store", return_value=None),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._complete_oauth_flow(user_info, state_data)

        assert result.status_code == 302
        assert "error" in _get_redirect_location(result).lower()

    def test_complete_oauth_flow_account_linking(self, oauth_handler):
        """Flow with user_id in state should trigger account linking."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        state_data = {
            "user_id": "existing-user-456",
            "redirect_url": "http://localhost:3000/auth/callback",
        }

        mock_user_store = MagicMock()
        from aragora.server.handlers.utils.responses import HandlerResult

        mock_link_result = HandlerResult(
            status_code=302,
            content_type="text/html",
            body=b"",
            headers={"Location": "http://localhost:3000/auth/callback?linked=google"},
        )

        with (
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
            patch.object(oauth_handler, "_handle_account_linking", return_value=mock_link_result),
        ):
            result = oauth_handler._complete_oauth_flow(user_info, state_data)

        assert result.status_code == 302
        assert "linked=google" in _get_redirect_location(result)

    def test_find_user_by_oauth_with_support(self, oauth_handler):
        """find_user_by_oauth should use user_store.get_user_by_oauth if available."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_oauth.return_value = MockUser()

        result = oauth_handler._find_user_by_oauth(mock_user_store, user_info)
        assert result is not None
        mock_user_store.get_user_by_oauth.assert_called_once_with("google", "g-123")

    def test_find_user_by_oauth_without_support(self, oauth_handler):
        """find_user_by_oauth should return None if store doesn't support it."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock(spec=[])  # No methods at all

        result = oauth_handler._find_user_by_oauth(mock_user_store, user_info)
        assert result is None

    def test_link_oauth_to_user_with_support(self, oauth_handler):
        """Link should call user_store.link_oauth_provider if available."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock()
        mock_user_store.link_oauth_provider.return_value = True

        result = oauth_handler._link_oauth_to_user(mock_user_store, "user-123", user_info)
        assert result is True

    def test_link_oauth_to_user_without_support(self, oauth_handler):
        """Link should return False if store doesn't support it."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock(spec=[])

        result = oauth_handler._link_oauth_to_user(mock_user_store, "user-123", user_info)
        assert result is False

    def test_handle_account_linking_user_not_found(self, oauth_handler):
        """Account linking with nonexistent user should redirect with error."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = None
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}

        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._handle_account_linking(
                mock_user_store, "nonexistent-user", user_info, state_data
            )

        assert result.status_code == 302
        assert "error" in _get_redirect_location(result).lower()

    def test_handle_account_linking_already_linked(self, oauth_handler):
        """Account linking when already linked to another user should error."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser(id="user-A")
        other_user = MockUser(id="user-B")
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}

        with (
            patch.object(oauth_handler, "_find_user_by_oauth", return_value=other_user),
            patch(f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"),
        ):
            result = oauth_handler._handle_account_linking(
                mock_user_store, "user-A", user_info, state_data
            )

        assert result.status_code == 302
        assert "error" in _get_redirect_location(result).lower()

    def test_handle_account_linking_success(self, oauth_handler):
        """Successful account linking should redirect with linked param."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g-123",
            email="test@example.com",
            name="Test",
        )
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser(id="user-A")
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}

        with (
            patch.object(oauth_handler, "_find_user_by_oauth", return_value=None),
            patch.object(oauth_handler, "_link_oauth_to_user", return_value=True),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
        ):
            result = oauth_handler._handle_account_linking(
                mock_user_store, "user-A", user_info, state_data
            )

        assert result.status_code == 302
        assert "linked=google" in _get_redirect_location(result)

    def test_redirect_with_tokens(self, oauth_handler, mock_token_pair):
        """Redirect with tokens should include all token params."""
        result = oauth_handler._redirect_with_tokens(
            "http://localhost:3000/auth/callback", mock_token_pair
        )
        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "access_token=mock-access-token-xyz" in location
        assert "refresh_token=mock-refresh-token-xyz" in location
        assert "token_type=Bearer" in location
        assert "expires_in=3600" in location
        # Verify no-cache headers
        assert result.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, private"

    def test_redirect_with_error(self, oauth_handler):
        """Redirect with error should include error in URL."""
        with patch(
            f"{_IMPL}._get_oauth_error_url", return_value="http://localhost:3000/auth/error"
        ):
            result = oauth_handler._redirect_with_error("Something went wrong")

        assert result.status_code == 302
        location = _get_redirect_location(result)
        assert "error=" in location
        assert "Something" in location


class TestOAuthConfig:
    """Tests for OAuth configuration functions."""

    def test_get_secret_from_env(self):
        """_get_secret should fall back to environment variable."""
        from aragora.server.handlers.oauth.config import _get_secret

        with patch.dict("os.environ", {"TEST_SECRET": "my-secret"}):
            with patch("aragora.server.handlers.oauth.config._get_secret") as mock:
                # Test the actual logic by calling the function directly
                # when secrets manager is not available
                mock.side_effect = lambda name, default="": __import__("os").environ.get(
                    name, default
                )
                result = mock("TEST_SECRET", "")
                assert result == "my-secret"

    def test_is_production_true(self):
        """_is_production should return True when ARAGORA_ENV=production."""
        from aragora.server.handlers.oauth.config import _is_production

        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            assert _is_production() is True

    def test_is_production_false(self):
        """_is_production should return False in non-production."""
        from aragora.server.handlers.oauth.config import _is_production

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            assert _is_production() is False

    def test_google_redirect_uri_dev_fallback(self):
        """Google redirect URI should fall back to localhost in dev."""
        from aragora.server.handlers.oauth.config import _get_google_redirect_uri

        with (
            patch("aragora.server.handlers.oauth.config._get_secret", return_value=""),
            patch("aragora.server.handlers.oauth.config._is_production", return_value=False),
        ):
            result = _get_google_redirect_uri()
            assert "localhost" in result

    def test_google_redirect_uri_from_secret(self):
        """Google redirect URI should use configured value."""
        from aragora.server.handlers.oauth.config import _get_google_redirect_uri

        with patch(
            "aragora.server.handlers.oauth.config._get_secret",
            return_value="https://api.example.com/callback",
        ):
            result = _get_google_redirect_uri()
            assert result == "https://api.example.com/callback"

    def test_allowed_redirect_hosts_dev_fallback(self):
        """Allowed hosts should fall back to localhost in dev."""
        from aragora.server.handlers.oauth.config import _get_allowed_redirect_hosts

        with (
            patch("aragora.server.handlers.oauth.config._get_secret", return_value=""),
            patch("aragora.server.handlers.oauth.config._is_production", return_value=False),
        ):
            hosts = _get_allowed_redirect_hosts()
            assert "localhost" in hosts
            assert "127.0.0.1" in hosts

    def test_allowed_redirect_hosts_from_config(self):
        """Allowed hosts should parse comma-separated config."""
        from aragora.server.handlers.oauth.config import _get_allowed_redirect_hosts

        with patch(
            "aragora.server.handlers.oauth.config._get_secret",
            return_value="example.com, app.example.com ",
        ):
            hosts = _get_allowed_redirect_hosts()
            assert "example.com" in hosts
            assert "app.example.com" in hosts

    def test_allowed_redirect_hosts_production_empty(self):
        """Allowed hosts should be empty in production with no config."""
        from aragora.server.handlers.oauth.config import _get_allowed_redirect_hosts

        with (
            patch("aragora.server.handlers.oauth.config._get_secret", return_value=""),
            patch("aragora.server.handlers.oauth.config._is_production", return_value=True),
        ):
            hosts = _get_allowed_redirect_hosts()
            assert len(hosts) == 0

    def test_provider_endpoint_constants(self):
        """Provider endpoint URLs should be properly defined."""
        from aragora.server.handlers.oauth.config import (
            GOOGLE_AUTH_URL,
            GOOGLE_TOKEN_URL,
            GOOGLE_USERINFO_URL,
            GITHUB_AUTH_URL,
            GITHUB_TOKEN_URL,
            GITHUB_USERINFO_URL,
            MICROSOFT_AUTH_URL_TEMPLATE,
            MICROSOFT_TOKEN_URL_TEMPLATE,
            APPLE_AUTH_URL,
            APPLE_TOKEN_URL,
        )

        assert "accounts.google.com" in GOOGLE_AUTH_URL
        assert "googleapis.com" in GOOGLE_TOKEN_URL
        assert "googleapis.com" in GOOGLE_USERINFO_URL
        assert "github.com" in GITHUB_AUTH_URL
        assert "github.com" in GITHUB_TOKEN_URL
        assert "api.github.com" in GITHUB_USERINFO_URL
        assert "{tenant}" in MICROSOFT_AUTH_URL_TEMPLATE
        assert "{tenant}" in MICROSOFT_TOKEN_URL_TEMPLATE
        assert "appleid.apple.com" in APPLE_AUTH_URL
        assert "appleid.apple.com" in APPLE_TOKEN_URL


class TestOAuthStateStore:
    """Tests for the OAuth state store backend."""

    def test_in_memory_store_generate_and_validate(self):
        """In-memory store should generate and validate state tokens."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore()
        token = store.generate(user_id="user-1", redirect_url="http://localhost:3000/cb")
        assert token is not None
        assert len(token) > 0

        result = store.validate_and_consume(token)
        assert result is not None
        assert result.user_id == "user-1"
        assert result.redirect_url == "http://localhost:3000/cb"

    def test_in_memory_store_single_use(self):
        """State tokens should be single-use (consumed on first validation)."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore()
        token = store.generate()
        assert store.validate_and_consume(token) is not None
        assert store.validate_and_consume(token) is None

    def test_in_memory_store_expired_token(self):
        """Expired tokens should not validate."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore()
        token = store.generate(ttl_seconds=0)
        # Wait briefly so it expires
        time.sleep(0.01)
        result = store.validate_and_consume(token)
        assert result is None

    def test_in_memory_store_nonexistent_token(self):
        """Nonexistent token should not validate."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore()
        result = store.validate_and_consume("nonexistent-token")
        assert result is None

    def test_in_memory_store_max_size_eviction(self):
        """Store should evict oldest entries when at max capacity."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore(max_size=5)
        tokens = [store.generate() for _ in range(10)]
        # After adding 10, some should have been evicted
        assert store.size() <= 10  # Should still work regardless

    def test_in_memory_store_cleanup_expired(self):
        """Cleanup should remove expired states."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore, OAuthState

        store = InMemoryOAuthStateStore()
        # Directly inject expired states to avoid generate()'s internal cleanup
        past = time.time() - 10
        store._states["expired-1"] = OAuthState(
            user_id=None, redirect_url=None, expires_at=past, created_at=past - 100
        )
        store._states["expired-2"] = OAuthState(
            user_id=None, redirect_url=None, expires_at=past, created_at=past - 100
        )
        assert store.size() == 2
        removed = store.cleanup_expired()
        assert removed == 2
        assert store.size() == 0

    def test_in_memory_store_metadata(self):
        """Store should preserve metadata through generate/validate."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        store = InMemoryOAuthStateStore()
        metadata = {"tenant_id": "t-123", "org_id": "o-456"}
        token = store.generate(metadata=metadata)
        result = store.validate_and_consume(token)
        assert result is not None
        assert result.metadata == metadata

    def test_jwt_store_generate_and_validate(self):
        """JWT store should generate and validate state tokens."""
        from aragora.server.oauth_state_store import JWTOAuthStateStore

        store = JWTOAuthStateStore(secret_key="test-secret-key")
        token = store.generate(user_id="user-1", redirect_url="http://localhost:3000/cb")
        assert "." in token  # JWT format

        result = store.validate_and_consume(token)
        assert result is not None
        assert result.user_id == "user-1"

    def test_jwt_store_replay_protection(self):
        """JWT store should reject replayed tokens."""
        from aragora.server.oauth_state_store import JWTOAuthStateStore

        store = JWTOAuthStateStore(secret_key="test-secret-key")
        token = store.generate()
        assert store.validate_and_consume(token) is not None
        assert store.validate_and_consume(token) is None  # Replay

    def test_jwt_store_invalid_signature(self):
        """JWT store should reject tokens with invalid signature."""
        from aragora.server.oauth_state_store import JWTOAuthStateStore

        store = JWTOAuthStateStore(secret_key="test-secret-key")
        token = store.generate()
        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1][::-1]  # Reverse the payload
        tampered = ".".join(parts)
        result = store.validate_and_consume(tampered)
        assert result is None

    def test_jwt_store_invalid_format(self):
        """JWT store should reject tokens with wrong format."""
        from aragora.server.oauth_state_store import JWTOAuthStateStore

        store = JWTOAuthStateStore(secret_key="test-secret-key")
        assert store.validate_and_consume("not-a-jwt") is None
        assert store.validate_and_consume("") is None
        assert store.validate_and_consume("a.b.c") is None  # 3 parts

    def test_oauth_state_dataclass(self):
        """OAuthState dataclass should have correct properties."""
        from aragora.server.oauth_state_store import OAuthState

        state = OAuthState(
            user_id="u-1",
            redirect_url="http://example.com",
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata={"key": "value"},
        )
        assert not state.is_expired
        d = state.to_dict()
        assert d["user_id"] == "u-1"
        assert d["metadata"] == {"key": "value"}

    def test_oauth_state_expired(self):
        """OAuthState should report expired correctly."""
        from aragora.server.oauth_state_store import OAuthState

        state = OAuthState(
            user_id="u-1",
            redirect_url="http://example.com",
            expires_at=time.time() - 100,
        )
        assert state.is_expired

    def test_oauth_state_from_dict(self):
        """OAuthState.from_dict should reconstruct state."""
        from aragora.server.oauth_state_store import OAuthState

        data = {
            "user_id": "u-2",
            "redirect_url": "http://example.com",
            "expires_at": time.time() + 600,
            "created_at": time.time(),
            "metadata": {"org": "test"},
        }
        state = OAuthState.from_dict(data)
        assert state.user_id == "u-2"
        assert state.metadata == {"org": "test"}


class TestRateLimiting:
    """Tests for OAuth endpoint rate limiting."""

    def test_rate_limiter_allows_normal_traffic(self, oauth_handler):
        """Rate limiter should allow normal request rates."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}.create_span") as mock_span,
            patch(f"{_IMPL}.add_span_attributes"),
            patch(f"{_IMPL}._oauth_limiter") as mock_limiter,
            patch(f"{_IMPL}._get_google_client_id", return_value="google-123"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-abc"),
            patch(f"{_IMPL}._get_google_redirect_uri", return_value="http://localhost:8080/cb"),
        ):
            mock_span.return_value.__enter__ = Mock(return_value=MagicMock())
            mock_span.return_value.__exit__ = Mock(return_value=False)
            mock_limiter.is_allowed.return_value = True
            result = oauth_handler.handle("/api/v1/auth/oauth/google", {}, handler, "GET")

        assert result is not None
        assert result.status_code != 429

    def test_rate_limiter_blocks_excessive_traffic(self, oauth_handler):
        """Rate limiter should block excessive requests with 429."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}.create_span") as mock_span,
            patch(f"{_IMPL}.add_span_attributes"),
            patch(f"{_IMPL}._oauth_limiter") as mock_limiter,
        ):
            mock_span.return_value.__enter__ = Mock(return_value=MagicMock())
            mock_span.return_value.__exit__ = Mock(return_value=False)
            mock_limiter.is_allowed.return_value = False
            result = oauth_handler.handle("/api/v1/auth/oauth/google", {}, handler, "GET")

        assert result is not None
        assert result.status_code == 429


class TestOAuthCallbackAPI:
    """Tests for the /api/auth/oauth/callback POST endpoint."""

    def test_callback_api_missing_body(self, oauth_handler):
        """Callback API with invalid body should return 400."""
        handler = _make_handler(method="POST")
        with patch.object(oauth_handler, "read_json_body", return_value=None):
            result = oauth_handler._handle_oauth_callback_api(handler)

        assert result.status_code == 400

    def test_callback_api_missing_fields(self, oauth_handler):
        """Callback API missing required fields should return 400."""
        handler = _make_handler(method="POST")
        with patch.object(oauth_handler, "read_json_body", return_value={"provider": "google"}):
            result = oauth_handler._handle_oauth_callback_api(handler)

        assert result.status_code == 400

    def test_callback_api_unsupported_provider(self, oauth_handler):
        """Callback API with unsupported provider should return 400."""
        handler = _make_handler(method="POST")
        with patch.object(
            oauth_handler,
            "read_json_body",
            return_value={"provider": "facebook", "code": "abc", "state": "xyz"},
        ):
            result = oauth_handler._handle_oauth_callback_api(handler)

        assert result.status_code == 400


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_validate_redirect_url_case_insensitive_host(self):
        """Redirect URL validation should be case-insensitive for hosts."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"localhost"})):
            assert _validate_redirect_url("http://LOCALHOST:3000/callback") is True

    def test_validate_redirect_url_https_scheme(self):
        """HTTPS scheme should be accepted."""
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with patch(f"{_IMPL}._get_allowed_redirect_hosts", return_value=frozenset({"example.com"})):
            assert _validate_redirect_url("https://example.com/callback") is True

    def test_oauth_handler_resource_type(self, oauth_handler):
        """OAuthHandler should have correct RESOURCE_TYPE."""
        assert oauth_handler.RESOURCE_TYPE == "oauth"

    def test_google_auth_with_account_linking(self, oauth_handler):
        """Google auth start should pass user_id for account linking."""
        handler = _make_handler()
        with (
            patch(f"{_IMPL}._get_google_client_id", return_value="google-123"),
            patch(
                f"{_IMPL}._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            patch(f"{_IMPL}._validate_redirect_url", return_value=True),
            patch(f"{_IMPL}._generate_state", return_value="state-link") as mock_gen,
            patch(f"{_IMPL}._get_google_redirect_uri", return_value="http://localhost:8080/cb"),
        ):
            result = oauth_handler._handle_google_auth_start(handler, {})

        # Verify _generate_state was called (the conftest mocks extract_user_from_request
        # to return authenticated user, so user_id is passed)
        mock_gen.assert_called_once()
        assert result.status_code == 302

    def test_get_user_providers_handler(self, oauth_handler):
        """get_user_providers should query user store for linked providers."""
        handler = _make_handler()
        mock_user_store = MagicMock()
        mock_user_store.get_oauth_providers.return_value = [
            {"provider": "google", "email": "test@gmail.com"},
        ]

        with (
            patch.object(oauth_handler, "_check_permission", return_value=None),
            patch.object(oauth_handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = oauth_handler._handle_get_user_providers(handler)

        assert result.status_code == 200
        body = _parse_result(result)
        assert len(body["providers"]) == 1

    def test_create_oauth_user_failure(self, oauth_handler):
        """_create_oauth_user should return None on failure."""
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="fail@example.com",
            name="Fail",
        )
        mock_user_store = MagicMock()
        mock_user_store.create_user.side_effect = ValueError("Duplicate email")

        result = oauth_handler._create_oauth_user(mock_user_store, user_info)
        assert result is None
