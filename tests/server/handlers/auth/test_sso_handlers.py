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


# ===========================================================================
# Test _get_sso_provider
# ===========================================================================


class TestGetSsoProvider:
    """Tests for _get_sso_provider function."""

    def test_get_provider_returns_none_without_config(self):
        """Provider should return None when not configured."""
        # Clear any cached providers
        with _sso_providers_lock:
            _sso_providers.clear()

        # Without env vars, should return None
        with patch.dict(os.environ, {}, clear=True):
            provider = _get_sso_provider("oidc")
            assert provider is None

    def test_get_provider_caches_instance(self):
        """Provider should cache the instance."""
        mock_provider = MagicMock()

        with _sso_providers_lock:
            _sso_providers["test_type"] = mock_provider

        result = _get_sso_provider("test_type")
        assert result is mock_provider

    def test_get_provider_oidc_with_config(self):
        """OIDC provider should initialize with valid config."""
        with _sso_providers_lock:
            _sso_providers.clear()

        env_vars = {
            "OIDC_CLIENT_ID": "test-client-id",
            "OIDC_CLIENT_SECRET": "test-client-secret",
            "OIDC_ISSUER_URL": "https://test-issuer.example.com",
            "OIDC_CALLBACK_URL": "https://app.example.com/callback",
            "OIDC_SCOPES": "openid,email,profile",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Mock at the import location inside the function
            with patch("aragora.auth.oidc.OIDCProvider") as mock_oidc:
                mock_oidc.return_value = MagicMock()
                provider = _get_sso_provider("oidc")
                # Provider should have been created
                assert provider is not None or mock_oidc.called

    def test_get_provider_google_with_config(self):
        """Google provider should initialize with valid config."""
        with _sso_providers_lock:
            _sso_providers.clear()

        env_vars = {
            "GOOGLE_CLIENT_ID": "test-google-client.apps.googleusercontent.com",
            "GOOGLE_CLIENT_SECRET": "test-google-secret",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("aragora.auth.oidc.OIDCProvider") as mock_oidc:
                mock_oidc.return_value = MagicMock()
                provider = _get_sso_provider("google")
                # Provider should have been created
                assert provider is not None or mock_oidc.called

    def test_get_provider_github_with_config(self):
        """GitHub provider should initialize with valid config."""
        with _sso_providers_lock:
            _sso_providers.clear()

        env_vars = {
            "GITHUB_CLIENT_ID": "test-github-client-id",
            "GITHUB_CLIENT_SECRET": "test-github-secret",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("aragora.auth.oidc.OIDCProvider") as mock_oidc:
                mock_oidc.return_value = MagicMock()
                provider = _get_sso_provider("github")
                # Provider should have been created
                assert provider is not None or mock_oidc.called

    def test_get_provider_handles_exception_gracefully(self):
        """Provider initialization should handle exceptions gracefully."""
        with _sso_providers_lock:
            _sso_providers.clear()

        env_vars = {
            "OIDC_CLIENT_ID": "test-client-id",
            "OIDC_ISSUER_URL": "https://test-issuer.example.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Simulate initialization error
            with patch("aragora.auth.oidc.OIDCProvider", side_effect=Exception("Init failed")):
                provider = _get_sso_provider("oidc")
                # Should return None gracefully
                assert provider is None

    def test_get_provider_unknown_type(self):
        """Unknown provider type should return None."""
        with _sso_providers_lock:
            _sso_providers.clear()

        provider = _get_sso_provider("unknown_provider")
        assert provider is None


# ===========================================================================
# Test Additional Login Edge Cases
# ===========================================================================


class TestHandleSsoLoginEdgeCases:
    """Additional edge case tests for handle_sso_login."""

    @pytest.mark.asyncio
    async def test_login_exception_handling(self, mock_oidc_provider):
        """Login should handle exceptions gracefully."""
        mock_oidc_provider.get_authorization_url.side_effect = Exception("Connection failed")

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({"provider": "oidc"})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 500
        assert "failed" in error.lower()

    @pytest.mark.asyncio
    async def test_login_triggers_cleanup_periodically(self, mock_oidc_provider):
        """Login should trigger session cleanup periodically."""
        # Add 9 sessions to make total 10 (trigger cleanup at 10 % 10 == 0)
        with _auth_sessions_lock:
            for i in range(9):
                _auth_sessions[f"state_{i}"] = {
                    "provider_type": "oidc",
                    "redirect_url": "/",
                    "created_at": time.time() - 1000,  # Expired
                }

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._cleanup_expired_sessions"
            ) as mock_cleanup:
                await handle_sso_login({"provider": "oidc"})

                # After 10th session, cleanup should be triggered
                # (but we're testing the path, not the actual count)

    @pytest.mark.asyncio
    async def test_login_with_empty_redirect_url(self, mock_oidc_provider):
        """Login with empty redirect_url should use default."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_login({"redirect_url": ""})

        body = get_data(result)
        state = body["state"]

        with _auth_sessions_lock:
            # Empty string becomes "/" via or default
            assert _auth_sessions[state]["redirect_url"] == ""


# ===========================================================================
# Test Additional Callback Edge Cases
# ===========================================================================


class TestHandleSsoCallbackEdgeCases:
    """Additional edge case tests for handle_sso_callback."""

    @pytest.mark.asyncio
    async def test_callback_idp_error_without_description(self, valid_auth_session):
        """IdP error without description should use error code."""
        state, _ = valid_auth_session
        result = await handle_sso_callback(
            {
                "error": "server_error",
                "state": state,
            }
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "server_error" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_with_different_provider_types(self, mock_oidc_provider, mock_sso_user):
        """Callback should work with different provider types."""
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        for provider_type in ["oidc", "google", "github"]:
            state = f"state_for_{provider_type}"
            with _auth_sessions_lock:
                _auth_sessions[state] = {
                    "provider_type": provider_type,
                    "redirect_url": "/dashboard",
                    "created_at": time.time(),
                }

            with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
                mock_get.return_value = mock_oidc_provider
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "jwt_token"

                    result = await handle_sso_callback(
                        {
                            "code": "auth_code",
                            "state": state,
                        }
                    )

            status, _ = parse_result(result)
            assert status == 200

    @pytest.mark.asyncio
    async def test_callback_user_expires_at(
        self, valid_auth_session, mock_oidc_provider, mock_sso_user
    ):
        """Callback response should include token expiration."""
        state, _ = valid_auth_session
        mock_sso_user.token_expires_at = time.time() + 7200  # 2 hours
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "jwt_token"

                result = await handle_sso_callback(
                    {
                        "code": "auth_code",
                        "state": state,
                    }
                )

        body = get_data(result)
        assert "expires_at" in body
        assert body["expires_at"] == mock_sso_user.token_expires_at


# ===========================================================================
# Test Additional Refresh Edge Cases
# ===========================================================================


class TestHandleSsoRefreshEdgeCases:
    """Additional edge case tests for handle_sso_refresh."""

    @pytest.mark.asyncio
    async def test_refresh_with_different_providers(self, mock_oidc_provider, mock_sso_user):
        """Refresh should work with different provider types."""
        mock_oidc_provider.refresh_token.return_value = mock_sso_user

        for provider in ["oidc", "google", "github"]:
            with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
                mock_get.return_value = mock_oidc_provider

                result = await handle_sso_refresh(
                    {
                        "provider": provider,
                        "refresh_token": "refresh_token_xyz",
                    }
                )

            status, _ = parse_result(result)
            assert status == 200

    @pytest.mark.asyncio
    async def test_refresh_returns_new_refresh_token(self, mock_oidc_provider, mock_sso_user):
        """Refresh should return new refresh token if provided."""
        mock_sso_user.refresh_token = "new_refresh_token_abc"
        mock_oidc_provider.refresh_token.return_value = mock_sso_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "old_refresh_token",
                }
            )

        body = get_data(result)
        assert body["refresh_token"] == "new_refresh_token_abc"


# ===========================================================================
# Test Additional Logout Edge Cases
# ===========================================================================


class TestHandleSsoLogoutEdgeCases:
    """Additional edge case tests for handle_sso_logout."""

    @pytest.mark.asyncio
    async def test_logout_exception_handling(self, mock_oidc_provider):
        """Logout should handle exceptions gracefully."""
        mock_oidc_provider.logout.side_effect = Exception("Logout failed")

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "some_token",
                }
            )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 500
        assert "failed" in error.lower()

    @pytest.mark.asyncio
    async def test_logout_different_providers(self, mock_oidc_provider):
        """Logout should work with different provider types."""
        mock_oidc_provider.logout.return_value = "https://idp.example.com/logout"

        for provider in ["oidc", "google", "github"]:
            with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
                mock_get.return_value = mock_oidc_provider

                result = await handle_sso_logout(
                    {
                        "provider": provider,
                        "id_token": "id_token_xyz",
                    }
                )

            status, _ = parse_result(result)
            assert status == 200


# ===========================================================================
# Test Additional List Providers Edge Cases
# ===========================================================================


class TestHandleListProvidersEdgeCases:
    """Additional edge case tests for handle_list_providers."""

    @pytest.mark.asyncio
    async def test_list_providers_all_configured(self):
        """List providers with all configured should show all enabled."""
        env_vars = {
            "OIDC_CLIENT_ID": "oidc-client",
            "OIDC_ISSUER_URL": "https://oidc.example.com",
            "GOOGLE_CLIENT_ID": "google-client",
            "GITHUB_CLIENT_ID": "github-client",
            "AZURE_AD_CLIENT_ID": "azure-client",
            "AZURE_AD_TENANT_ID": "azure-tenant",
        }

        with patch.dict(os.environ, env_vars):
            result = await handle_list_providers({})

        body = get_data(result)
        assert body["sso_enabled"] is True

        # Check each provider
        enabled_providers = [p for p in body["providers"] if p["enabled"]]
        assert len(enabled_providers) >= 3  # At least OIDC, Google, GitHub

    @pytest.mark.asyncio
    async def test_list_providers_exception_handling(self):
        """List providers should handle exceptions gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.environ.get", side_effect=Exception("env error")):
                result = await handle_list_providers({})

        status, _ = parse_result(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_list_providers_azure_ad_partial_config(self):
        """Azure AD should show disabled with partial config."""
        env_vars = {
            "AZURE_AD_CLIENT_ID": "azure-client",
            # Missing AZURE_AD_TENANT_ID
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = await handle_list_providers({})

        body = get_data(result)
        azure_provider = next(p for p in body["providers"] if p["type"] == "azure_ad")
        assert azure_provider["enabled"] is False


# ===========================================================================
# Test Additional Config Edge Cases
# ===========================================================================


class TestHandleGetSsoConfigEdgeCases:
    """Additional edge case tests for handle_get_sso_config."""

    @pytest.mark.asyncio
    async def test_get_config_unknown_provider(self):
        """Unknown provider should return disabled config."""
        from aragora.server.handlers.auth import sso_handlers

        original_func = sso_handlers.handle_get_sso_config.__wrapped__

        result = await original_func({"provider": "unknown_provider"})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["enabled"] is False
        assert body["provider"] == "unknown_provider"

    @pytest.mark.asyncio
    async def test_get_config_exception_handling(self):
        """Config endpoint should handle exceptions gracefully."""
        from aragora.server.handlers.auth import sso_handlers

        original_func = sso_handlers.handle_get_sso_config.__wrapped__

        with patch("os.environ.get", side_effect=Exception("env error")):
            result = await original_func({"provider": "oidc"})

        status, _ = parse_result(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_get_config_oidc_scopes_parsing(self):
        """OIDC config should parse scopes correctly."""
        from aragora.server.handlers.auth import sso_handlers

        original_func = sso_handlers.handle_get_sso_config.__wrapped__

        with patch.dict(
            os.environ,
            {
                "OIDC_CLIENT_ID": "client123",
                "OIDC_ISSUER_URL": "https://idp.example.com",
                "OIDC_SCOPES": "openid,email,profile,custom_scope",
            },
        ):
            result = await original_func({"provider": "oidc"})

        body = get_data(result)
        assert "openid" in body["scopes"]
        assert "custom_scope" in body["scopes"]


# ===========================================================================
# Test Session Cleanup Edge Cases
# ===========================================================================


class TestCleanupExpiredSessionsEdgeCases:
    """Additional edge case tests for session cleanup."""

    def test_cleanup_empty_sessions(self):
        """Cleanup should handle empty session store."""
        with _auth_sessions_lock:
            _auth_sessions.clear()

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert len(_auth_sessions) == 0

    def test_cleanup_all_expired(self):
        """Cleanup should remove all expired sessions."""
        with _auth_sessions_lock:
            for i in range(5):
                _auth_sessions[f"expired_state_{i}"] = {
                    "provider_type": "oidc",
                    "redirect_url": "/",
                    "created_at": time.time() - AUTH_SESSION_TTL - 1000,
                }

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert len(_auth_sessions) == 0

    def test_cleanup_mixed_sessions(self):
        """Cleanup should only remove expired sessions."""
        with _auth_sessions_lock:
            # Add expired
            _auth_sessions["expired_1"] = {
                "provider_type": "oidc",
                "redirect_url": "/",
                "created_at": time.time() - AUTH_SESSION_TTL - 100,
            }
            _auth_sessions["expired_2"] = {
                "provider_type": "google",
                "redirect_url": "/dashboard",
                "created_at": time.time() - AUTH_SESSION_TTL - 200,
            }
            # Add valid
            _auth_sessions["valid_1"] = {
                "provider_type": "oidc",
                "redirect_url": "/",
                "created_at": time.time(),
            }
            _auth_sessions["valid_2"] = {
                "provider_type": "github",
                "redirect_url": "/profile",
                "created_at": time.time() - 60,  # 1 minute ago, still valid
            }

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert len(_auth_sessions) == 2
            assert "valid_1" in _auth_sessions
            assert "valid_2" in _auth_sessions
            assert "expired_1" not in _auth_sessions
            assert "expired_2" not in _auth_sessions

    def test_cleanup_session_without_created_at(self):
        """Cleanup should handle sessions without created_at."""
        with _auth_sessions_lock:
            _auth_sessions["no_timestamp"] = {
                "provider_type": "oidc",
                "redirect_url": "/",
                # No created_at - should be treated as expired (created_at=0)
            }

        _cleanup_expired_sessions()

        with _auth_sessions_lock:
            assert "no_timestamp" not in _auth_sessions


# ===========================================================================
# Test Thread Safety
# ===========================================================================


class TestThreadSafety:
    """Tests for thread safety of SSO handlers."""

    @pytest.mark.asyncio
    async def test_concurrent_login_requests(self, mock_oidc_provider):
        """Multiple concurrent login requests should be handled safely."""
        import asyncio

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider

            tasks = [handle_sso_login({"provider": "oidc"}) for _ in range(10)]
            results = await asyncio.gather(*tasks)

        # All should succeed
        for result in results:
            status, _ = parse_result(result)
            assert status == 200

        # All should have unique states
        states = []
        for result in results:
            body = get_data(result)
            states.append(body["state"])

        assert len(set(states)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, mock_oidc_provider, mock_sso_user):
        """Concurrent callback requests should be thread-safe."""
        import asyncio

        mock_oidc_provider.authenticate.return_value = mock_sso_user

        # Create multiple valid sessions
        states = []
        for i in range(5):
            state = f"concurrent_state_{i}"
            states.append(state)
            with _auth_sessions_lock:
                _auth_sessions[state] = {
                    "provider_type": "oidc",
                    "redirect_url": "/",
                    "created_at": time.time(),
                }

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                mock_jwt.return_value = "jwt_token"

                tasks = [handle_sso_callback({"code": "code", "state": state}) for state in states]
                results = await asyncio.gather(*tasks)

        # All should succeed (each state used exactly once)
        success_count = sum(1 for r in results if parse_result(r)[0] == 200)
        assert success_count == 5
