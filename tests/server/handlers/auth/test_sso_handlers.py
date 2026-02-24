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

import asyncio
import json
import os
import time
from typing import Any
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
    _sso_providers,
    _sso_providers_lock,
    AUTH_SESSION_TTL,
    _get_sso_provider,
    _sso_state_store,
)
from aragora.server.oauth_state_store import (
    OAuthState,
    InMemoryOAuthStateStore,
    reset_oauth_state_store,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_result(result: HandlerResult) -> tuple[int, dict[str, Any]]:
    """Parse HandlerResult into (status_code, body_dict)."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


def get_data(result: HandlerResult) -> dict[str, Any]:
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
    """Clear all SSO stores before and after each test."""
    reset_oauth_state_store()
    with _sso_providers_lock:
        _sso_providers.clear()
    yield
    reset_oauth_state_store()
    with _sso_providers_lock:
        _sso_providers.clear()


@pytest.fixture
def mock_state_store():
    """Create a mock OAuth state store for testing."""
    return InMemoryOAuthStateStore()


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
def valid_auth_session(mock_state_store):
    """Create a valid auth session using the state store."""
    state = mock_state_store.generate(
        redirect_url="/dashboard",
        metadata={"provider_type": "oidc"},
        ttl_seconds=3600,
    )
    return state, mock_state_store


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
# Test OAuth State Store cleanup
# ===========================================================================


class TestOAuthStateStoreCleanup:
    """Tests for session cleanup via OAuth state store."""

    def test_cleanup_removes_expired_sessions(self, mock_state_store):
        """Expired sessions should be removed during cleanup."""
        # Generate a state with very short TTL
        expired_state = mock_state_store.generate(ttl_seconds=0)  # Immediately expired

        # Add a valid session
        valid_state = mock_state_store.generate(ttl_seconds=3600)

        # Wait a moment and cleanup
        time.sleep(0.01)
        mock_state_store.cleanup_expired()

        # Expired state should not validate
        assert mock_state_store.validate_and_consume(expired_state) is None
        # Valid state should still work
        assert mock_state_store.validate_and_consume(valid_state) is not None

    def test_cleanup_preserves_valid_sessions(self, mock_state_store):
        """Valid sessions should be preserved."""
        state = mock_state_store.generate(ttl_seconds=3600)

        mock_state_store.cleanup_expired()

        # Valid state should still work
        assert mock_state_store.validate_and_consume(state) is not None


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
    async def test_login_stores_session(self, mock_oidc_provider, mock_state_store):
        """Login should store session with state."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store:
                mock_store.return_value = mock_state_store

                result = await handle_sso_login({"provider": "google", "redirect_url": "/home"})

        body = get_data(result)
        state = body["state"]

        # Verify state was generated (state store tracks it internally)
        assert len(state) >= 32
        # The state store can validate the generated state
        oauth_state = mock_state_store.validate_and_consume(state)
        assert oauth_state is not None
        assert oauth_state.metadata["provider_type"] == "google"
        assert oauth_state.redirect_url == "/home"

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
    async def test_login_default_redirect_url(self, mock_oidc_provider, mock_state_store):
        """Default redirect URL should be /."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store:
                mock_store.return_value = mock_state_store

                result = await handle_sso_login({})

        body = get_data(result)
        state = body["state"]

        # Validate through the state store
        oauth_state = mock_state_store.validate_and_consume(state)
        assert oauth_state is not None
        assert oauth_state.redirect_url == "/"

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
    async def test_login_custom_redirect_url(self, mock_oidc_provider, mock_state_store):
        """Custom redirect URL should be stored."""
        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store:
                mock_store.return_value = mock_state_store

                result = await handle_sso_login({"redirect_url": "/custom/path"})

        body = get_data(result)
        state = body["state"]

        oauth_state = mock_state_store.validate_and_consume(state)
        assert oauth_state is not None
        assert oauth_state.redirect_url == "/custom/path"

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
        """Valid callback should return JWT token pair, create user in DB, and use Aragora user ID."""
        state, mock_store = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        # Mock user store to verify user creation
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = None  # New user
        mock_aragora_user = MagicMock()
        mock_aragora_user.id = "aragora_user_456"
        mock_aragora_user.email = "user@example.com"
        mock_aragora_user.name = "SSO User"
        mock_aragora_user.role = "member"
        mock_aragora_user.org_id = None
        mock_aragora_user.created_at = None
        mock_user_store.create_user.return_value = mock_aragora_user
        # get_user_by_id is called after auto-org creation attempt
        mock_user_store.get_user_by_id.return_value = mock_aragora_user

        # Mock token pair
        mock_tokens = MagicMock()
        mock_tokens.access_token = "our_jwt_token"
        mock_tokens.refresh_token = "our_refresh_token"
        mock_tokens.expires_in = 3600

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store_fn:
                mock_store_fn.return_value = mock_store
                with patch(
                    "aragora.server.handlers.auth.sso_handlers.create_token_pair"
                ) as mock_jwt:
                    mock_jwt.return_value = mock_tokens
                    with patch(
                        "aragora.storage.user_store.singleton.get_user_store"
                    ) as mock_get_store:
                        mock_get_store.return_value = mock_user_store

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
        assert body["refresh_token"] == "our_refresh_token"
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 3600
        assert "user" in body
        assert body["user"]["org_id"] is None
        assert body["redirect_url"] == "/dashboard"
        # Verify user was created in Aragora DB with SSO email
        mock_user_store.create_user.assert_called_once()
        call_kwargs = mock_user_store.create_user.call_args
        assert (
            call_kwargs[1]["email"] == "user@example.com" or call_kwargs[0][0] == "user@example.com"
        )
        # Verify JWT uses Aragora user ID, not SSO provider ID
        mock_jwt.assert_called_once_with(
            user_id="aragora_user_456",
            email="user@example.com",
            org_id=None,
            role="member",
        )

    @pytest.mark.asyncio
    async def test_callback_removes_session(
        self, valid_auth_session, mock_oidc_provider, mock_sso_user
    ):
        """Callback should remove the auth session (single-use)."""
        state, mock_store = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = None
        mock_user = MagicMock(id="u1", email="user@example.com", name="SSO User", role="member")
        mock_user_store.create_user.return_value = mock_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store_fn:
                mock_store_fn.return_value = mock_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "jwt"
                    with patch(
                        "aragora.storage.user_store.singleton.get_user_store"
                    ) as mock_get_store:
                        mock_get_store.return_value = mock_user_store

                        await handle_sso_callback({"code": "code", "state": state})

        # State should be consumed (can't use twice)
        assert mock_store.validate_and_consume(state) is None

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
    async def test_callback_expired_state(self, mock_state_store):
        """Expired state should return 401."""
        # Create an immediately expired state
        state = mock_state_store.generate(ttl_seconds=0)
        await asyncio.sleep(0.01)

        with patch(
            "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
        ) as mock_store_fn:
            mock_store_fn.return_value = mock_state_store

            result = await handle_sso_callback(
                {
                    "code": "auth_code",
                    "state": state,
                }
            )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "invalid" in error.lower() or "expired" in error.lower()

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
        state, mock_store = valid_auth_session

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = None
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store_fn:
                mock_store_fn.return_value = mock_store

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
        state, mock_store = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = None
        mock_user = MagicMock(id="u1", email="user@example.com", name="SSO User", role="member")
        mock_user_store.create_user.return_value = mock_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store_fn:
                mock_store_fn.return_value = mock_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "jwt"
                    with patch(
                        "aragora.storage.user_store.singleton.get_user_store"
                    ) as mock_get_store:
                        mock_get_store.return_value = mock_user_store

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
        state, mock_store = valid_auth_session
        mock_oidc_provider.authenticate.return_value = mock_sso_user

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_email.return_value = None
        mock_user = MagicMock(id="u1", email="user@example.com", name="SSO User", role="member")
        mock_user_store.create_user.return_value = mock_user

        with patch("aragora.server.handlers.auth.sso_handlers._get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch(
                "aragora.server.handlers.auth.sso_handlers._sso_state_store.get"
            ) as mock_store_fn:
                mock_store_fn.return_value = mock_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "jwt"
                    with patch(
                        "aragora.storage.user_store.singleton.get_user_store"
                    ) as mock_get_store:
                        mock_get_store.return_value = mock_user_store

                        result = await handle_sso_callback({"code": "code", "state": state})

        body = get_data(result)
        assert body["sso_access_token"] == "sso_access_token_abc"


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

    def test_get_provider_unknown_type(self):
        """Unknown provider type should return None."""
        with _sso_providers_lock:
            _sso_providers.clear()

        provider = _get_sso_provider("unknown_provider")
        assert provider is None


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
