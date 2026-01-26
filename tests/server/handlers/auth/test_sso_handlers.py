"""
Tests for aragora.server.handlers.auth.sso_handlers - SSO authentication flow.

Tests cover:
- SSO login initiation (provider selection, state generation)
- OAuth callback (CSRF validation, token exchange)
- Token refresh
- Logout
- Provider listing and configuration

Security test categories:
- CSRF protection: state parameter validation
- Token security: session expiry, single-use codes
- Authorization: admin-only config endpoint
- Provider configuration: env var validation
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auth.sso_handlers import (
    handle_sso_login,
    handle_sso_callback,
    handle_sso_refresh,
    handle_sso_logout,
    handle_list_providers,
    handle_get_sso_config,
    get_sso_handlers,
    _auth_sessions,
    _auth_sessions_lock,
    _sso_providers,
    _sso_providers_lock,
    AUTH_SESSION_TTL,
    _get_sso_provider,
    _cleanup_expired_sessions,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_result(result: HandlerResult) -> tuple[int, Dict[str, Any]]:
    """Parse HandlerResult into (status_code, body_dict)."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


def get_data(result: HandlerResult) -> Dict[str, Any]:
    """Get the 'data' from a success response."""
    _, body = parse_result(result)
    return body.get("data", body)


def get_error(result: HandlerResult) -> str:
    """Get the error message from an error response."""
    _, body = parse_result(result)
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", "")
    return error


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_sso_state():
    """Clear all in-memory SSO stores before and after each test."""
    with _auth_sessions_lock:
        _auth_sessions.clear()
    with _sso_providers_lock:
        _sso_providers.clear()
    yield
    with _auth_sessions_lock:
        _auth_sessions.clear()
    with _sso_providers_lock:
        _sso_providers.clear()


@pytest.fixture
def mock_oidc_provider():
    """Create a mock OIDC provider."""
    provider = MagicMock()
    provider.get_authorization_url = AsyncMock(
        return_value="https://idp.example.com/authorize?state=test"
    )
    provider.authenticate = AsyncMock()
    provider.refresh_token = AsyncMock()
    provider.logout = AsyncMock(return_value="https://idp.example.com/logout")
    return provider


@pytest.fixture
def valid_auth_session():
    """Create a valid auth session."""
    state = "valid_state_token_123"
    session_data = {
        "provider_type": "oidc",
        "redirect_url": "/dashboard",
        "created_at": time.time(),
    }
    with _auth_sessions_lock:
        _auth_sessions[state] = session_data
    return state, session_data


@pytest.fixture
def expired_auth_session():
    """Create an expired auth session."""
    state = "expired_state_token_456"
    session_data = {
        "provider_type": "oidc",
        "redirect_url": "/dashboard",
        "created_at": time.time() - AUTH_SESSION_TTL - 100,  # Expired
    }
    with _auth_sessions_lock:
        _auth_sessions[state] = session_data
    return state, session_data


@pytest.fixture
def mock_sso_user():
    """Create a mock SSO user response."""
    user = MagicMock()
    user.id = "sso_user_123"
    user.email = "user@example.com"
    user.name = "SSO User"
    user.access_token = "sso_access_token_abc"
    user.refresh_token = "sso_refresh_token_xyz"
    user.token_expires_at = time.time() + 3600
    user.to_dict = MagicMock(
        return_value={
            "id": "sso_user_123",
            "email": "user@example.com",
            "name": "SSO User",
        }
    )
    return user


# ===========================================================================
# Test _cleanup_expired_sessions
# ===========================================================================


class TestCleanupExpiredSessions:
    """Tests for session cleanup."""

    def test_cleanup_removes_expired_sessions(self, expired_auth_session):
        """Expired sessions should be removed."""
        state, _ = expired_auth_session

        # Add a valid session
        valid_state = "valid_state"
        with _auth_sessions_lock:
            _auth_sessions[valid_state] = {
                "provider_type": "oidc",
                "redirect_url": "/",
                "created_at": time.time(),
            }

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert state not in _auth_sessions
            assert valid_state in _auth_sessions

    def test_cleanup_preserves_valid_sessions(self, valid_auth_session):
        """Valid sessions should be preserved."""
        state, _ = valid_auth_session

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert state in _auth_sessions


# ===========================================================================
# Test handle_sso_login
# ===========================================================================


class TestHandleSsoLogin:
    """Tests for handle_sso_login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success_with_provider(self, mock_oidc_provider):
        """Login should return authorization URL."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({"provider": "oidc"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "authorization_url" in body
        assert "state" in body
        assert body["provider"] == "oidc"
        assert body["expires_in"] == AUTH_SESSION_TTL

    @pytest.mark.asyncio
    async def test_login_stores_session(self, mock_oidc_provider):
        """Login should store session with state."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({"provider": "google", "redirect_url": "/home"})

        body = get_data(result)
        state = body["state"]

        with _auth_sessions_lock:
            assert state in _auth_sessions
            session = _auth_sessions[state]
            assert session["provider_type"] == "google"
            assert session["redirect_url"] == "/home"

    @pytest.mark.asyncio
    async def test_login_provider_not_configured(self):
        """Unconfigured provider should return 503."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None

            result = await handle_sso_login({"provider": "unconfigured"})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 503
        assert "not configured" in error.lower()

    @pytest.mark.asyncio
    async def test_login_default_provider_oidc(self, mock_oidc_provider):
        """Default provider should be OIDC."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({})

        body = get_data(result)
        assert body["provider"] == "oidc"
        mock_get.assert_called_with("oidc")

    @pytest.mark.asyncio
    async def test_login_default_redirect_url(self, mock_oidc_provider):
        """Default redirect URL should be /."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({})

        body = get_data(result)
        state = body["state"]

        with _auth_sessions_lock:
            assert _auth_sessions[state]["redirect_url"] == "/"

    @pytest.mark.asyncio
    async def test_login_state_is_unique(self, mock_oidc_provider):
        """Each login should generate unique state."""
        states = []
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            for _ in range(10):
                result = await handle_sso_login({})
                body = get_data(result)
                states.append(body["state"])

        assert len(set(states)) == 10

    @pytest.mark.asyncio
    async def test_login_multiple_providers(self, mock_oidc_provider):
        """Different providers should be supported."""
        for provider in ["oidc", "google", "github"]:
            with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
                mock_get.return_value = mock_oidc_provider

                result = await handle_sso_login({"provider": provider})

            body = get_data(result)
            assert body["provider"] == provider

    @pytest.mark.asyncio
    async def test_login_custom_redirect_url(self, mock_oidc_provider):
        """Custom redirect URL should be stored."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({"redirect_url": "/custom/path"})

        body = get_data(result)
        state = body["state"]

        with _auth_sessions_lock:
            assert _auth_sessions[state]["redirect_url"] == "/custom/path"

    @pytest.mark.asyncio
    async def test_login_state_length(self, mock_oidc_provider):
        """State should be sufficiently long for security."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({})

        body = get_data(result)
        assert len(body["state"]) >= 32


# ===========================================================================
# Test handle_sso_callback
# ===========================================================================


class TestHandleSsoCallback:
    """Tests for handle_sso_callback endpoint."""

    @pytest.mark.asyncio
    async def test_callback_success(self, valid_auth_session, mock_oidc_provider, mock_sso_user):
        """Valid callback should return JWT and user info."""
        state, _ = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "our_jwt_token"

                result = await handle_sso_callback(
                    {
                        "code": "auth_code_123",
                        "state": state,
                    }
                )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["access_token"] == "our_jwt_token"
        assert body["token_type"] == "bearer"
        assert "user" in body
        assert body["redirect_url"] == "/dashboard"

    @pytest.mark.asyncio
    async def test_callback_removes_session(
        self, valid_auth_session, mock_oidc_provider, mock_sso_user
    ):
        """Callback should remove the auth session."""
        state, _ = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "jwt"

                await handle_sso_callback({"code": "code", "state": state})

        with _auth_sessions_lock:
            assert state not in _auth_sessions

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, valid_auth_session):
        """Missing code should return 400."""
        state, _ = valid_auth_session
        result = await handle_sso_callback({"state": state})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "code" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_missing_state(self):
        """Missing state should return 400."""
        result = await handle_sso_callback({"code": "auth_code"})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "state" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self):
        """Invalid state should return 401."""
        result = await handle_sso_callback(
            {
                "code": "auth_code",
                "state": "invalid_state_xyz",
            }
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "invalid" in error.lower() or "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_expired_state(self, expired_auth_session):
        """Expired state should return 401."""
        state, _ = expired_auth_session
        result = await handle_sso_callback(
            {
                "code": "auth_code",
                "state": state,
            }
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_idp_error(self, valid_auth_session):
        """IdP error should return 401."""
        state, _ = valid_auth_session
        result = await handle_sso_callback(
            {
                "error": "access_denied",
                "error_description": "User cancelled login",
                "state": state,
            }
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "access_denied" in error.lower() or "cancelled" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_provider_unavailable(self, valid_auth_session):
        """Unavailable provider should return 503."""
        state, _ = valid_auth_session

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None

            result = await handle_sso_callback(
                {
                    "code": "auth_code",
                    "state": state,
                }
            )

        status, _ = parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_callback_state_single_use(
        self, valid_auth_session, mock_oidc_provider, mock_sso_user
    ):
        """State should only work once (CSRF protection)."""
        state, _ = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "jwt"

                result1 = await handle_sso_callback({"code": "code1", "state": state})
                result2 = await handle_sso_callback({"code": "code2", "state": state})

        status1, _ = parse_result(result1)
        status2, _ = parse_result(result2)
        assert status1 == 200
        assert status2 == 401

    @pytest.mark.asyncio
    async def test_callback_includes_sso_token(
        self, valid_auth_session, mock_oidc_provider, mock_sso_user
    ):
        """Callback should include SSO provider token for API calls."""
        state, _ = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "jwt"

                result = await handle_sso_callback({"code": "code", "state": state})

        body = get_data(result)
        assert body["sso_access_token"] == "sso_access_token_abc"

    @pytest.mark.asyncio
    async def test_callback_authentication_failure(self, valid_auth_session, mock_oidc_provider):
        """Authentication failure from provider should return 401."""
        state, _ = valid_auth_session
        mock_oidc_provider.authenticate.side_effect = Exception("Authentication failed")

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_callback({"code": "code", "state": state})

        status, _ = parse_result(result)
        assert status == 401


# ===========================================================================
# Test handle_sso_refresh
# ===========================================================================


class TestHandleSsoRefresh:
    """Tests for handle_sso_refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, mock_oidc_provider, mock_sso_user):
        """Valid refresh should return new tokens."""
        mock_oidc_provider.refresh_token.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "refresh_token_xyz",
                }
            )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "access_token" in body
        assert "refresh_token" in body

    @pytest.mark.asyncio
    async def test_refresh_missing_token(self):
        """Missing refresh token should return 400."""
        result = await handle_sso_refresh({"provider": "oidc"})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "refresh_token" in error.lower()

    @pytest.mark.asyncio
    async def test_refresh_provider_unavailable(self):
        """Unavailable provider should return 503."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None

            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "token",
                }
            )

        status, _ = parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_refresh_failure(self, mock_oidc_provider):
        """Refresh failure should return 401."""
        mock_oidc_provider.refresh_token.return_value = None

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "expired_token",
                }
            )

        status, _ = parse_result(result)
        assert status == 401

    @pytest.mark.asyncio
    async def test_refresh_default_provider(self, mock_oidc_provider, mock_sso_user):
        """Default provider should be OIDC."""
        mock_oidc_provider.refresh_token.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_refresh({"refresh_token": "token"})

        status, _ = parse_result(result)
        assert status == 200
        mock_get.assert_called_with("oidc")

    @pytest.mark.asyncio
    async def test_refresh_exception_handling(self, mock_oidc_provider):
        """Exceptions during refresh should return 401."""
        mock_oidc_provider.refresh_token.side_effect = Exception("Token expired")

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "token",
                }
            )

        status, _ = parse_result(result)
        assert status == 401


# ===========================================================================
# Test handle_sso_logout
# ===========================================================================


class TestHandleSsoLogout:
    """Tests for handle_sso_logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_success_without_idp_logout(self):
        """Logout without id_token should succeed."""
        result = await handle_sso_logout({"provider": "oidc"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["logged_out"] is True
        assert body["logout_url"] is None

    @pytest.mark.asyncio
    async def test_logout_with_idp_logout_url(self, mock_oidc_provider):
        """Logout with id_token should return IdP logout URL."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "id_token_abc",
                }
            )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["logged_out"] is True
        assert body["logout_url"] == "https://idp.example.com/logout"

    @pytest.mark.asyncio
    async def test_logout_provider_unavailable(self):
        """Logout should succeed even if provider unavailable."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None

            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "token",
                }
            )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["logged_out"] is True
        assert body["logout_url"] is None

    @pytest.mark.asyncio
    async def test_logout_default_provider(self):
        """Default provider should be OIDC."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None

            result = await handle_sso_logout({})

        mock_get.assert_called_with("oidc")


# ===========================================================================
# Test handle_list_providers
# ===========================================================================


class TestHandleListProviders:
    """Tests for handle_list_providers endpoint."""

    @pytest.mark.asyncio
    async def test_list_providers_returns_all(self):
        """List should return all known providers."""
        result = await handle_list_providers({})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "providers" in body
        assert "sso_enabled" in body

        provider_types = [p["type"] for p in body["providers"]]
        assert "oidc" in provider_types
        assert "google" in provider_types
        assert "github" in provider_types

    @pytest.mark.asyncio
    async def test_list_providers_shows_enabled_status(self):
        """Providers should show enabled/disabled status."""
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "test_client_id"}):
            result = await handle_list_providers({})

        body = get_data(result)
        google_provider = next(p for p in body["providers"] if p["type"] == "google")
        assert google_provider["enabled"] is True

    @pytest.mark.asyncio
    async def test_list_providers_sso_enabled_flag(self):
        """sso_enabled should be True if any provider enabled."""
        # By default, no providers are enabled
        result = await handle_list_providers({})
        body = get_data(result)

        # Check if any provider is enabled
        any_enabled = any(p["enabled"] for p in body["providers"])
        assert body["sso_enabled"] == any_enabled


# ===========================================================================
# Test handle_get_sso_config
# ===========================================================================


class TestHandleGetSsoConfig:
    """Tests for handle_get_sso_config endpoint (admin only)."""

    @pytest.mark.asyncio
    async def test_get_config_oidc(self):
        """OIDC config should return public settings."""
        with patch.dict(
            os.environ,
            {
                "OIDC_CLIENT_ID": "client123",
                "OIDC_ISSUER_URL": "https://idp.example.com",
                "OIDC_SCOPES": "openid,email,profile",
            },
        ):
            # Note: This handler has @require_permission("admin:system")
            # For testing, we patch the decorator
            from aragora.server.handlers.auth import sso_handlers

            original_func = sso_handlers.handle_get_sso_config.__wrapped__

            result = await original_func({"provider": "oidc"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["provider"] == "oidc"
        assert body["enabled"] is True
        assert body["issuer_url"] == "https://idp.example.com"
        assert "client_secret" not in body  # Should NOT expose secrets

    @pytest.mark.asyncio
    async def test_get_config_google(self):
        """Google config should return public settings."""
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "google_client"}):
            from aragora.server.handlers.auth import sso_handlers

            original_func = sso_handlers.handle_get_sso_config.__wrapped__

            result = await original_func({"provider": "google"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["provider"] == "google"
        assert body["enabled"] is True
        assert body["issuer_url"] == "https://accounts.google.com"

    @pytest.mark.asyncio
    async def test_get_config_github(self):
        """GitHub config should return public settings."""
        with patch.dict(os.environ, {"GITHUB_CLIENT_ID": "github_client"}):
            from aragora.server.handlers.auth import sso_handlers

            original_func = sso_handlers.handle_get_sso_config.__wrapped__

            result = await original_func({"provider": "github"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["provider"] == "github"
        assert body["enabled"] is True
        assert "authorization_endpoint" in body

    @pytest.mark.asyncio
    async def test_get_config_disabled_provider(self):
        """Unconfigured provider should show disabled."""
        with patch.dict(os.environ, {}, clear=True):
            from aragora.server.handlers.auth import sso_handlers

            original_func = sso_handlers.handle_get_sso_config.__wrapped__

            result = await original_func({"provider": "oidc"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_get_config_default_provider(self):
        """Default provider should be OIDC."""
        from aragora.server.handlers.auth import sso_handlers

        original_func = sso_handlers.handle_get_sso_config.__wrapped__

        result = await original_func({})

        body = get_data(result)
        assert body["provider"] == "oidc"


# ===========================================================================
# Test get_sso_handlers
# ===========================================================================


class TestGetSsoHandlers:
    """Tests for handler registration."""

    def test_all_handlers_registered(self):
        """All handlers should be registered."""
        handlers = get_sso_handlers()

        expected = [
            "sso_login",
            "sso_callback",
            "sso_refresh",
            "sso_logout",
            "sso_list_providers",
            "sso_get_config",
        ]

        for name in expected:
            assert name in handlers, f"Handler {name} should be registered"

    def test_handlers_are_callable(self):
        """All handlers should be callable."""
        handlers = get_sso_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} should be callable"
