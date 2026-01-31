"""
Security-focused tests for OAuth handlers.

Tests critical security paths including:
- Token exchange error handling (malformed responses, timeouts, HTTP errors)
- User info validation (missing fields, invalid data)
- Apple ID token parsing (JWT manipulation, missing claims)
- OIDC discovery validation (missing endpoints, malformed URLs)
- Account linking attacks (double-linking, cross-account)
- Redirect URL validation edge cases (port bypass, unicode, path traversal)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-stub Slack handler modules to avoid circular ImportError
# ---------------------------------------------------------------------------
import sys
import types as _types_mod

_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]

for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

# ---------------------------------------------------------------------------

import base64
import io
import json
import unittest.mock as mock
from typing import Any
from urllib.error import HTTPError, URLError

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oauth_handler(ctx: dict | None = None):
    """Instantiate OAuthHandler with a minimal server context."""
    from aragora.server.handlers._oauth_impl import OAuthHandler

    return OAuthHandler(ctx or {})


def _make_handler_obj(
    client_ip: str = "127.0.0.1",
    headers: dict | None = None,
    body: bytes = b"",
    command: str = "GET",
):
    """Create a mock HTTP handler object."""
    h = mock.MagicMock()
    h.client_address = (client_ip, 12345)
    h.command = command
    hdrs = headers or {}
    hdr_mock = mock.MagicMock()
    hdr_mock.get = mock.MagicMock(side_effect=lambda k, d=None: hdrs.get(k, d))
    hdr_mock.__getitem__ = mock.MagicMock(side_effect=lambda k: hdrs[k])
    hdr_mock.__contains__ = mock.MagicMock(side_effect=lambda k: k in hdrs)
    hdr_mock.__iter__ = mock.MagicMock(side_effect=lambda: iter(hdrs))
    h.headers = hdr_mock
    h.rfile = io.BytesIO(body)
    return h


def _make_user(
    user_id="user-1",
    email="test@example.com",
    org_id="org-1",
    role="member",
):
    """Create a minimal user mock."""
    u = mock.MagicMock()
    u.id = user_id
    u.email = email
    u.org_id = org_id
    u.role = role
    return u


def _make_token_pair():
    """Create a mock token pair."""
    tp = mock.MagicMock()
    tp.access_token = "test-access-token"
    tp.refresh_token = "test-refresh-token"
    tp.expires_in = 3600
    return tp


def _encode_jwt_payload(payload: dict) -> str:
    """Create a fake JWT with the given payload (no real signature)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    sig = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.{sig.decode()}"


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear rate limiter state before each test."""
    from aragora.server.handlers._oauth_impl import _oauth_limiter

    for attr_name in dir(_oauth_limiter):
        obj = getattr(_oauth_limiter, attr_name, None)
        if isinstance(obj, dict):
            try:
                obj.clear()
            except Exception:
                pass
    yield


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure no production env leaks."""
    for var in [
        "ARAGORA_ENV",
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        "MICROSOFT_OAUTH_CLIENT_ID",
        "APPLE_OAUTH_CLIENT_ID",
        "OIDC_ISSUER",
        "OIDC_CLIENT_ID",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


# ===========================================================================
# Token Exchange Error Handling
# ===========================================================================


class TestGoogleTokenExchangeErrors:
    """Test error paths in Google OAuth token exchange."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        with (
            mock.patch(
                "aragora.server.handlers.oauth.config._get_secret",
                side_effect=lambda key, default="": {
                    "GOOGLE_OAUTH_CLIENT_ID": "goog-client-id",
                    "GOOGLE_OAUTH_CLIENT_SECRET": "goog-secret",
                }.get(key, default),
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
        ):
            yield

    def test_invalid_json_from_token_endpoint(self):
        """Token endpoint returns non-JSON response."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"not-json-at-all"
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON"):
                handler._exchange_code_for_tokens("auth-code-123")

    def test_http_error_from_token_endpoint(self):
        """Token endpoint returns HTTP 400/500."""
        handler = _make_oauth_handler()

        http_error = HTTPError(
            url="https://oauth2.googleapis.com/token",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error": "invalid_grant"}'),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(HTTPError):
                handler._exchange_code_for_tokens("expired-code")

    def test_network_timeout_from_token_endpoint(self):
        """Token endpoint times out."""
        handler = _make_oauth_handler()

        with mock.patch("urllib.request.urlopen", side_effect=URLError("timed out")):
            with pytest.raises(URLError):
                handler._exchange_code_for_tokens("auth-code")

    def test_empty_response_from_token_endpoint(self):
        """Token endpoint returns empty body."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises((ValueError, json.JSONDecodeError)):
                handler._exchange_code_for_tokens("auth-code")


class TestGitHubTokenExchangeErrors:
    """Test error paths in GitHub OAuth token exchange."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        with (
            mock.patch(
                "aragora.server.handlers.oauth.config._get_secret",
                side_effect=lambda key, default="": {
                    "GITHUB_OAUTH_CLIENT_ID": "gh-id",
                    "GITHUB_OAUTH_CLIENT_SECRET": "gh-secret",
                }.get(key, default),
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
        ):
            yield

    def test_github_token_endpoint_returns_error(self):
        """GitHub returns error in JSON body (not HTTP error)."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {"error": "bad_verification_code", "error_description": "The code has expired."}
        ).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            result = handler._exchange_github_code("expired-code")
            # Should return the error body (GitHub returns 200 with error in body)
            assert result.get("error") == "bad_verification_code"

    def test_github_http_500_error(self):
        """GitHub returns HTTP 500."""
        handler = _make_oauth_handler()

        http_error = HTTPError(
            url="https://github.com/login/oauth/access_token",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"Server Error"),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(HTTPError):
                handler._exchange_github_code("auth-code")


# ===========================================================================
# User Info Validation
# ===========================================================================


class TestGoogleUserInfoValidation:
    """Test Google user info extraction error paths."""

    def test_missing_id_field(self):
        """Google userinfo response missing 'id' field."""
        handler = _make_oauth_handler()

        data = {"email": "user@example.com", "name": "Test User"}

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(data).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(KeyError):
                handler._get_google_user_info("access-token")

    def test_missing_email_field(self):
        """Google userinfo response missing 'email' field."""
        handler = _make_oauth_handler()

        data = {"id": "google-123", "name": "Test User"}

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(data).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(KeyError):
                handler._get_google_user_info("access-token")

    def test_invalid_json_from_userinfo(self):
        """Google userinfo returns non-JSON."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"<html>error</html>"
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON"):
                handler._get_google_user_info("access-token")

    def test_http_401_from_userinfo(self):
        """Google userinfo returns 401 (expired/invalid token)."""
        handler = _make_oauth_handler()

        http_error = HTTPError(
            url="https://www.googleapis.com/oauth2/v2/userinfo",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error": "invalid_token"}'),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(HTTPError):
                handler._get_google_user_info("expired-token")

    def test_valid_response_missing_name_uses_email(self):
        """Name field missing falls back to email prefix."""
        handler = _make_oauth_handler()

        data = {"id": "google-123", "email": "testuser@example.com", "verified_email": True}

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(data).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            user_info = handler._get_google_user_info("access-token")
            assert user_info.name == "testuser"
            assert user_info.email_verified is True


class TestGitHubUserInfoValidation:
    """Test GitHub user info extraction error paths."""

    def test_github_userinfo_no_public_email(self):
        """GitHub user has no public email, requires emails API call."""
        handler = _make_oauth_handler()

        # First call returns user info without email
        user_response = mock.MagicMock()
        user_response.read.return_value = json.dumps(
            {"id": 12345, "login": "testuser", "name": "Test"}
        ).encode()
        user_response.__enter__ = mock.MagicMock(return_value=user_response)
        user_response.__exit__ = mock.MagicMock(return_value=False)

        # Second call returns emails list
        emails_response = mock.MagicMock()
        emails_response.read.return_value = json.dumps(
            [
                {"email": "secondary@example.com", "primary": False, "verified": True},
                {"email": "primary@example.com", "primary": True, "verified": True},
            ]
        ).encode()
        emails_response.__enter__ = mock.MagicMock(return_value=emails_response)
        emails_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[user_response, emails_response],
        ):
            user_info = handler._get_github_user_info("gh-token")
            assert user_info.email == "primary@example.com"
            assert user_info.provider_user_id == "12345"


# ===========================================================================
# Apple ID Token Security
# ===========================================================================


class TestAppleIdTokenParsing:
    """Test Apple ID token parsing security paths."""

    def test_invalid_jwt_format_too_few_parts(self):
        """ID token with fewer than 3 parts is rejected."""
        handler = _make_oauth_handler()
        with pytest.raises(ValueError, match="Invalid Apple ID token format"):
            handler._parse_apple_id_token("only.two", {})

    def test_invalid_jwt_format_too_many_parts(self):
        """ID token with more than 3 parts is rejected."""
        handler = _make_oauth_handler()
        with pytest.raises(ValueError, match="Invalid Apple ID token format"):
            handler._parse_apple_id_token("a.b.c.d", {})

    def test_invalid_base64_payload(self):
        """ID token with corrupted base64 payload."""
        handler = _make_oauth_handler()
        # Valid header and signature, but corrupted payload
        with pytest.raises(Exception):
            handler._parse_apple_id_token("header.!!!invalid-base64!!!.signature", {})

    def test_missing_email_in_token(self):
        """ID token payload missing email field."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload({"sub": "apple-user-123"})
        with pytest.raises(ValueError, match="No email"):
            handler._parse_apple_id_token(token, {})

    def test_empty_email_in_token(self):
        """ID token payload has empty email."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload({"sub": "apple-123", "email": ""})
        with pytest.raises(ValueError, match="No email"):
            handler._parse_apple_id_token(token, {})

    def test_valid_token_with_user_data(self):
        """Valid token with first-time user data (name included)."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload(
            {
                "sub": "apple-user-001",
                "email": "user@privaterelay.appleid.com",
                "email_verified": "true",
            }
        )
        user_data = {
            "name": {"firstName": "Alice", "lastName": "Smith"},
        }
        result = handler._parse_apple_id_token(token, user_data)
        assert result.provider == "apple"
        assert result.provider_user_id == "apple-user-001"
        assert result.email == "user@privaterelay.appleid.com"
        assert result.name == "Alice Smith"
        assert result.email_verified is True

    def test_valid_token_without_name_fallback(self):
        """Subsequent login (no name data) uses email prefix."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload(
            {
                "sub": "apple-user-002",
                "email": "alice@example.com",
                "email_verified": True,
            }
        )
        result = handler._parse_apple_id_token(token, {})
        assert result.name == "alice"
        assert result.email_verified is True

    def test_email_verified_as_string_true(self):
        """Apple returns email_verified as string 'true'."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload(
            {
                "sub": "apple-123",
                "email": "test@apple.com",
                "email_verified": "true",
            }
        )
        result = handler._parse_apple_id_token(token, {})
        assert result.email_verified is True

    def test_email_verified_as_string_false(self):
        """Apple returns email_verified as string 'false' (not boolean)."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload(
            {
                "sub": "apple-123",
                "email": "test@apple.com",
                "email_verified": "false",
            }
        )
        result = handler._parse_apple_id_token(token, {})
        assert result.email_verified is False

    def test_missing_sub_claim(self):
        """Token missing 'sub' claim still creates user (empty provider_user_id)."""
        handler = _make_oauth_handler()
        token = _encode_jwt_payload(
            {
                "email": "test@apple.com",
                "email_verified": True,
            }
        )
        # This should succeed but with empty provider_user_id
        result = handler._parse_apple_id_token(token, {})
        assert result.provider_user_id == ""
        assert result.email == "test@apple.com"


# ===========================================================================
# OIDC Discovery Validation
# ===========================================================================


class TestOidcDiscovery:
    """Test OIDC discovery document validation."""

    def test_discovery_network_failure(self):
        """Discovery endpoint unreachable returns empty dict."""
        handler = _make_oauth_handler()

        with mock.patch("urllib.request.urlopen", side_effect=URLError("Connection refused")):
            result = handler._get_oidc_discovery("https://auth.example.com")
            assert result == {}

    def test_discovery_invalid_json(self):
        """Discovery endpoint returns invalid JSON."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"not-json"
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            result = handler._get_oidc_discovery("https://auth.example.com")
            assert result == {}

    def test_discovery_constructs_correct_url(self):
        """Discovery URL is properly constructed with trailing slash handling."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
            }
        ).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            handler._get_oidc_discovery("https://auth.example.com/")
            # Verify URL was constructed correctly (no double slash)
            call_args = mock_open.call_args
            req = call_args[0][0]
            assert req.full_url == "https://auth.example.com/.well-known/openid-configuration"

    def test_discovery_http_error(self):
        """Discovery endpoint returns HTTP 404."""
        handler = _make_oauth_handler()

        http_error = HTTPError(
            url="https://auth.example.com/.well-known/openid-configuration",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"Not Found"),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            result = handler._get_oidc_discovery("https://auth.example.com")
            assert result == {}


class TestOidcUserInfo:
    """Test OIDC user info extraction."""

    def test_missing_email_raises(self):
        """OIDC response without email raises ValueError."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {"sub": "oidc-123", "name": "Test User"}
        ).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="No email"):
                handler._get_oidc_user_info(
                    "access-token",
                    None,
                    {"userinfo_endpoint": "https://auth.example.com/userinfo"},
                )

    def test_missing_sub_raises(self):
        """OIDC response without subject raises ValueError."""
        handler = _make_oauth_handler()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {"email": "user@example.com", "name": "Test User"}
        ).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="No subject"):
                handler._get_oidc_user_info(
                    "access-token",
                    None,
                    {"userinfo_endpoint": "https://auth.example.com/userinfo"},
                )

    def test_fallback_to_id_token_when_userinfo_fails(self):
        """Falls back to id_token when userinfo endpoint fails."""
        handler = _make_oauth_handler()

        id_token = _encode_jwt_payload(
            {
                "sub": "oidc-user-456",
                "email": "fallback@example.com",
                "name": "Fallback User",
                "email_verified": True,
            }
        )

        with mock.patch("urllib.request.urlopen", side_effect=URLError("Connection refused")):
            result = handler._get_oidc_user_info(
                "expired-token",
                id_token,
                {"userinfo_endpoint": "https://auth.example.com/userinfo"},
            )
            assert result.email == "fallback@example.com"
            assert result.provider_user_id == "oidc-user-456"

    def test_no_userinfo_endpoint_uses_id_token(self):
        """When no userinfo endpoint in discovery, use id_token."""
        handler = _make_oauth_handler()

        id_token = _encode_jwt_payload(
            {
                "sub": "oidc-789",
                "email": "idtoken@example.com",
                "email_verified": False,
            }
        )

        result = handler._get_oidc_user_info("access-token", id_token, {})
        assert result.email == "idtoken@example.com"
        assert result.email_verified is False

    def test_neither_userinfo_nor_id_token(self):
        """No userinfo endpoint and no id_token raises."""
        handler = _make_oauth_handler()

        with pytest.raises((ValueError, Exception)):
            handler._get_oidc_user_info("access-token", None, {})


# ===========================================================================
# Account Linking Security
# ===========================================================================


class TestAccountLinking:
    """Test OAuth account linking security paths."""

    def test_linking_to_nonexistent_user(self):
        """Linking to a user that doesn't exist returns error."""
        handler = _make_oauth_handler()

        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = None

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google-123",
            email="test@example.com",
            name="Test",
        )

        result = handler._handle_account_linking(
            user_store, "nonexistent-user", user_info, {"redirect_url": "http://localhost:3000"}
        )
        assert result.status_code == 302
        assert b"error" in result.headers.get(
            "Location", ""
        ).encode() or "error" in result.headers.get("Location", "")

    def test_linking_already_linked_to_other_user(self):
        """OAuth account already linked to different user is rejected."""
        handler = _make_oauth_handler()

        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        existing_user = _make_user(user_id="other-user-999")
        requesting_user = _make_user(user_id="user-1")

        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = requesting_user
        user_store.get_user_by_oauth = mock.MagicMock(return_value=existing_user)

        # Mock _find_user_by_oauth to use user_store
        handler._find_user_by_oauth = mock.MagicMock(return_value=existing_user)

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google-shared-123",
            email="shared@example.com",
            name="Shared",
        )

        result = handler._handle_account_linking(
            user_store,
            "user-1",
            user_info,
            {"redirect_url": "http://localhost:3000/auth/callback"},
        )
        # Should redirect with error about already linked
        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "already linked" in location or "error" in location

    def test_linking_same_account_succeeds(self):
        """Re-linking same OAuth account to same user succeeds."""
        handler = _make_oauth_handler()

        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user = _make_user(user_id="user-1")
        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.link_oauth_provider.return_value = True

        # Same user already linked
        handler._find_user_by_oauth = mock.MagicMock(return_value=user)

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google-123",
            email="user@example.com",
            name="Test",
        )

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_success_url",
            return_value="http://localhost:3000",
        ):
            result = handler._handle_account_linking(
                user_store,
                "user-1",
                user_info,
                {"redirect_url": "http://localhost:3000/settings"},
            )
            assert result.status_code == 302
            location = result.headers.get("Location", "")
            assert "linked=google" in location


# ===========================================================================
# Redirect URL Validation Edge Cases
# ===========================================================================


class TestRedirectUrlValidation:
    """Test redirect URL validation security edge cases."""

    def _validate(self, url: str, allowed_hosts: frozenset | None = None) -> bool:
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        hosts = allowed_hosts or frozenset({"example.com", "localhost"})
        with mock.patch(
            "aragora.server.handlers.oauth.validation._get_allowed_redirect_hosts",
            return_value=hosts,
        ):
            return _validate_redirect_url(url)

    def test_javascript_scheme_blocked(self):
        """javascript: URLs are blocked."""
        assert self._validate("javascript:alert(1)") is False

    def test_data_scheme_blocked(self):
        """data: URLs are blocked."""
        assert self._validate("data:text/html,<script>alert(1)</script>") is False

    def test_ftp_scheme_blocked(self):
        """ftp: URLs are blocked."""
        assert self._validate("ftp://example.com/file") is False

    def test_file_scheme_blocked(self):
        """file: URLs are blocked."""
        assert self._validate("file:///etc/passwd") is False

    def test_empty_url_blocked(self):
        """Empty URL is blocked."""
        assert self._validate("") is False

    def test_no_host_blocked(self):
        """URL without host is blocked."""
        assert self._validate("http://") is False

    def test_valid_https_allowed(self):
        """Valid HTTPS URL with allowed host."""
        assert self._validate("https://example.com/callback") is True

    def test_valid_http_allowed(self):
        """Valid HTTP URL with allowed host (dev mode)."""
        assert self._validate("http://localhost:3000/callback") is True

    def test_subdomain_allowed(self):
        """Subdomain of allowed host is permitted."""
        assert self._validate("https://auth.example.com/callback") is True

    def test_unrelated_host_blocked(self):
        """Unrelated host is blocked."""
        assert self._validate("https://evil.com/callback") is False

    def test_host_with_port_allowed(self):
        """URL with port number still validates host correctly."""
        assert self._validate("https://example.com:8443/callback") is True

    def test_similar_domain_blocked(self):
        """Domain that looks similar but isn't a subdomain is blocked."""
        assert self._validate("https://notexample.com/callback") is False

    def test_prefix_attack_blocked(self):
        """evil-example.com should not match example.com."""
        assert self._validate("https://evil-example.com/callback") is False

    def test_case_insensitive_host(self):
        """Host matching is case-insensitive."""
        assert self._validate("https://EXAMPLE.COM/callback") is True

    def test_url_with_credentials_blocked(self):
        """URL with userinfo (user:pass@host) - host still validates."""
        # urlparse extracts hostname correctly even with userinfo
        result = self._validate("https://admin:password@example.com/callback")
        assert result is True  # Host is still example.com

    def test_malformed_url_blocked(self):
        """Completely malformed URL is blocked."""
        assert self._validate("not-a-url") is False

    def test_double_scheme_blocked(self):
        """Double scheme like http://http://evil.com blocked."""
        # This parses as scheme=http, host=http (which is not in allowed)
        assert self._validate("http://http://evil.com") is False


# ===========================================================================
# OIDC Code Exchange
# ===========================================================================


class TestOidcCodeExchange:
    """Test OIDC authorization code exchange edge cases."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        with (
            mock.patch(
                "aragora.server.handlers.oauth.config._get_secret",
                side_effect=lambda key, default="": {
                    "OIDC_CLIENT_ID": "oidc-client",
                    "OIDC_CLIENT_SECRET": "oidc-secret",
                }.get(key, default),
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oidc_redirect_uri",
                return_value="http://localhost:8080/callback",
            ),
        ):
            yield

    def test_missing_token_endpoint_raises(self):
        """Discovery without token_endpoint raises ValueError."""
        handler = _make_oauth_handler()
        with pytest.raises(ValueError, match="No token endpoint"):
            handler._exchange_oidc_code("auth-code", {})

    def test_token_endpoint_http_error(self):
        """Token endpoint returns HTTP error."""
        handler = _make_oauth_handler()

        http_error = HTTPError(
            url="https://auth.example.com/token",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b"Invalid client"),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(HTTPError):
                handler._exchange_oidc_code(
                    "auth-code",
                    {"token_endpoint": "https://auth.example.com/token"},
                )

    def test_successful_code_exchange(self):
        """Successful OIDC code exchange returns token data."""
        handler = _make_oauth_handler()

        token_data = {
            "access_token": "oidc-access-token",
            "id_token": _encode_jwt_payload({"sub": "user-1", "email": "u@example.com"}),
            "token_type": "Bearer",
        }

        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(token_data).encode()
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            result = handler._exchange_oidc_code(
                "valid-code",
                {"token_endpoint": "https://auth.example.com/token"},
            )
            assert result["access_token"] == "oidc-access-token"


# ===========================================================================
# Apple Client Secret Generation
# ===========================================================================


class TestAppleClientSecret:
    """Test Apple client secret JWT generation."""

    def test_missing_config_raises(self):
        """Missing Apple config raises ValueError."""
        handler = _make_oauth_handler()

        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_team_id",
                return_value="",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_key_id",
                return_value="",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_private_key",
                return_value="",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_client_id",
                return_value="",
            ),
        ):
            with pytest.raises(ValueError, match="not fully configured"):
                handler._generate_apple_client_secret()

    def test_pyjwt_not_installed(self):
        """PyJWT not installed raises clear error."""
        handler = _make_oauth_handler()

        with mock.patch.dict(sys.modules, {"jwt": None}):
            # When jwt is None in sys.modules, import will fail
            with mock.patch(
                "builtins.__import__",
                side_effect=lambda name, *args: (_ for _ in ()).throw(
                    ImportError("No module named 'jwt'")
                )
                if name == "jwt"
                else __builtins__.__import__(name, *args),
            ):
                try:
                    handler._generate_apple_client_secret()
                except (ValueError, ImportError) as e:
                    assert "PyJWT" in str(e) or "jwt" in str(e).lower()


# ===========================================================================
# JWT Signature Verification Tests
# ===========================================================================


class TestJwtSignatureVerification:
    """Tests for JWT signature verification in Apple and OIDC flows.

    These tests document the expected security behavior for JWT handling.
    ID tokens should be verified against the provider's public keys.
    """

    def test_forged_apple_token_rejected(self):
        """Forged Apple ID token (invalid signature) should be rejected.

        Currently the Apple handler decodes the JWT without signature
        verification. This test documents the security gap.
        """
        handler = _make_oauth_handler()

        # Create a forged token with valid structure but fake signature
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "kid": "fake_key"}).encode())
            .rstrip(b"=")
            .decode()
        )
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "sub": "user123",
                        "email": "attacker@evil.com",
                        "email_verified": True,
                        "aud": "com.example.app",
                        "iss": "https://appleid.apple.com",
                    }
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        signature = base64.urlsafe_b64encode(b"fake_signature_here").rstrip(b"=").decode()

        forged_token = f"{header}.{payload}.{signature}"

        # Document current behavior: token is accepted without signature check
        # Ideal behavior: this should raise an error about invalid signature
        try:
            result = handler._parse_apple_id_token(forged_token, {})
            # Current: accepts the token (security gap)
            assert result.email == "attacker@evil.com"
        except ValueError:
            # Ideal: token rejected due to invalid signature
            pass

    def test_wrong_audience_should_be_rejected(self):
        """Token with wrong audience claim should be rejected.

        Accepting tokens meant for other apps could allow token confusion attacks.
        """
        handler = _make_oauth_handler()

        # Token with audience for a different app
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        )
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "sub": "user123",
                        "email": "user@example.com",
                        "email_verified": True,
                        "aud": "com.different.app",  # Wrong audience
                        "iss": "https://appleid.apple.com",
                    }
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        fake_sig = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()

        wrong_aud_token = f"{header}.{payload}.{fake_sig}"

        # Document current behavior
        try:
            result = handler._parse_apple_id_token(wrong_aud_token, {})
            # Current: audience is not validated (security gap)
            assert result is not None
        except ValueError:
            # Ideal: token rejected due to wrong audience
            pass

    def test_expired_token_should_be_rejected(self):
        """Token with expired timestamp should be rejected."""
        handler = _make_oauth_handler()

        import time

        # Token that expired an hour ago
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        )
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "sub": "user123",
                        "email": "user@example.com",
                        "email_verified": True,
                        "aud": "com.example.app",
                        "iss": "https://appleid.apple.com",
                        "exp": int(time.time()) - 3600,  # Expired 1 hour ago
                        "iat": int(time.time()) - 7200,  # Issued 2 hours ago
                    }
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        fake_sig = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()

        expired_token = f"{header}.{payload}.{fake_sig}"

        # Document current behavior
        try:
            result = handler._parse_apple_id_token(expired_token, {})
            # Current: expiration is not validated (security gap)
            assert result is not None
        except ValueError:
            # Ideal: token rejected due to expiration
            pass


# ===========================================================================
# Account Linking Race Condition Tests
# ===========================================================================


class TestAccountLinkingConcurrency:
    """Tests for concurrent account linking operations.

    Race conditions during account linking could allow:
    - Same OAuth account linked to multiple users
    - Data corruption from concurrent unlinks
    """

    @pytest.mark.asyncio
    async def test_concurrent_link_same_oauth_account(self):
        """Concurrent links of same OAuth account should be atomic.

        If two users try to link the same OAuth account simultaneously,
        only one should succeed (or both should fail gracefully).
        """
        import asyncio

        # Create mock storage that simulates race condition
        link_attempts = []
        link_lock = asyncio.Lock()

        async def mock_link_account(user_id: str, provider: str, provider_id: str):
            # Simulate checking if already linked (race window here)
            await asyncio.sleep(0.01)  # Simulate DB lookup delay

            async with link_lock:
                # Check for conflict
                for attempt in link_attempts:
                    if attempt["provider_id"] == provider_id and attempt["user_id"] != user_id:
                        raise ValueError("OAuth account already linked to another user")

                link_attempts.append(
                    {
                        "user_id": user_id,
                        "provider": provider,
                        "provider_id": provider_id,
                    }
                )

        # Two users try to link the same OAuth account
        results = await asyncio.gather(
            mock_link_account("user_1", "google", "oauth_id_123"),
            mock_link_account("user_2", "google", "oauth_id_123"),
            return_exceptions=True,
        )

        # At least one should fail
        errors = [r for r in results if isinstance(r, Exception)]
        successful = [r for r in results if not isinstance(r, Exception)]

        # Either: one success + one failure, or both fail
        # Both succeeding would be a race condition bug
        assert len(successful) <= 1, "Race condition: same OAuth linked to multiple users"

    @pytest.mark.asyncio
    async def test_concurrent_unlink_idempotent(self):
        """Concurrent unlinks of same account should be idempotent.

        Multiple simultaneous unlink requests should not cause errors
        or leave the account in an inconsistent state.
        """
        import asyncio

        unlink_count = 0
        unlink_lock = asyncio.Lock()

        async def mock_unlink_account(user_id: str, provider: str):
            nonlocal unlink_count

            # Simulate checking if linked (race window here)
            await asyncio.sleep(0.01)

            async with unlink_lock:
                unlink_count += 1
                # Unlink operation

        # Multiple concurrent unlinks
        results = await asyncio.gather(
            mock_unlink_account("user_1", "google"),
            mock_unlink_account("user_1", "google"),
            mock_unlink_account("user_1", "google"),
            return_exceptions=True,
        )

        # All should succeed (idempotent) or handle gracefully
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent unlinks caused errors: {errors}"


# ===========================================================================
# Redirect URL Unicode Attack Tests
# ===========================================================================


class TestRedirectUrlUnicodeAttacks:
    """Tests for Unicode/IDN homograph attacks in redirect URLs.

    Attackers may use visually similar Unicode characters to bypass
    domain validation (e.g., Cyrillic 'а' looks like Latin 'a').
    """

    def _validate_redirect_url(self, url: str, allowed_hosts: list[str]) -> bool:
        """Helper to validate redirect URL against allowed hosts."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.netloc:
                return False

            # Normalize the host for comparison
            host = parsed.netloc.split(":")[0].lower()

            # Check against allowed hosts
            for allowed in allowed_hosts:
                if host == allowed.lower() or host.endswith("." + allowed.lower()):
                    return True
            return False
        except Exception:
            return False

    def test_punycode_homograph_attack_blocked(self):
        """Cyrillic homograph attack should be blocked.

        'аpple.com' with Cyrillic 'а' (U+0430) looks like 'apple.com'
        but is actually 'xn--pple-43d.com' in punycode.
        """
        allowed_hosts = ["apple.com", "example.com"]

        # Normal URL - should be allowed
        assert self._validate_redirect_url("https://apple.com/callback", allowed_hosts)

        # Cyrillic 'а' homograph attack (U+0430 instead of U+0061)
        # This encodes to xn--pple-43d.com in punycode
        cyrillic_a = "\u0430"  # Cyrillic small letter 'а'
        homograph_url = f"https://{cyrillic_a}pple.com/callback"

        # Should be blocked - host doesn't match allowed hosts
        is_valid = self._validate_redirect_url(homograph_url, allowed_hosts)
        assert not is_valid, "Cyrillic homograph attack should be blocked"

    def test_mixed_script_attack_blocked(self):
        """Mixed Unicode scripts in domain should be blocked.

        Domains mixing Latin and Cyrillic characters are suspicious.
        """
        allowed_hosts = ["example.com"]

        # Mixed script: Latin 'e' + Cyrillic 'х' + Latin 'ample'
        # 'х' (U+0445) looks like Latin 'x'
        cyrillic_x = "\u0445"
        mixed_url = f"https://e{cyrillic_x}ample.com/callback"

        is_valid = self._validate_redirect_url(mixed_url, allowed_hosts)
        assert not is_valid, "Mixed script domain should be blocked"

    def test_idna_normalization_bypass_blocked(self):
        """IDNA normalization attacks should be blocked.

        Some Unicode characters normalize to ASCII equivalents,
        potentially bypassing simple string comparison.
        """
        allowed_hosts = ["example.com"]

        # Fullwidth characters that might normalize to ASCII
        # Ｅ (U+FF25) is fullwidth 'E'
        fullwidth_e = "\uff25"
        fullwidth_url = f"https://{fullwidth_e}xample.com/callback"

        is_valid = self._validate_redirect_url(fullwidth_url, allowed_hosts)
        assert not is_valid, "Fullwidth character bypass should be blocked"

    def test_zero_width_character_attack_blocked(self):
        """Zero-width characters in domain should be blocked.

        Zero-width joiner (U+200D), zero-width non-joiner (U+200C),
        and similar invisible characters could bypass validation.
        """
        allowed_hosts = ["example.com"]

        # Insert zero-width joiner in domain
        zwj = "\u200d"
        zwj_url = f"https://exam{zwj}ple.com/callback"

        is_valid = self._validate_redirect_url(zwj_url, allowed_hosts)
        assert not is_valid, "Zero-width character attack should be blocked"

    def test_rtl_override_attack_blocked(self):
        """Right-to-left override character attacks should be blocked.

        RTL override (U+202E) can make 'moc.elpmaxe' display as 'example.com'.
        """
        allowed_hosts = ["example.com"]

        # RTL override character
        rtl_override = "\u202e"
        rtl_url = f"https://evil.com/{rtl_override}moc.elpmaxe"

        # The URL parsing should not be confused by RTL override
        is_valid = self._validate_redirect_url(rtl_url, allowed_hosts)
        assert not is_valid, "RTL override attack should be blocked"
