"""Tests for SSO handlers (aragora/server/handlers/auth/sso_handlers.py).

Covers all 6 handler functions:
- handle_sso_login - Initiate SSO login flow
- handle_sso_callback - Handle OAuth callback from IdP
- handle_sso_refresh - Refresh SSO access token
- handle_sso_logout - Logout from SSO
- handle_list_providers - List available SSO providers
- handle_get_sso_config - Get provider configuration
- get_sso_handlers - Handler registry
- _cleanup_expired_sessions - Session expiry cleanup
- _get_sso_provider - Provider factory
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aragora.server.handlers.auth.sso_handlers import (
    AUTH_SESSION_TTL,
    _auth_sessions,
    _cleanup_expired_sessions,
    _get_sso_provider,
    _sso_providers,
    _sso_providers_lock,
    get_sso_handlers,
    handle_get_sso_config,
    handle_list_providers,
    handle_sso_callback,
    handle_sso_login,
    handle_sso_logout,
    handle_sso_refresh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _data(result) -> dict:
    """Extract the 'data' envelope from a success response."""
    body = _body(result)
    return body.get("data", body)


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


@dataclass
class MockSSOUser:
    """Mock SSO user returned by provider.authenticate()."""

    id: str = "sso-user-001"
    email: str = "sso-user@example.com"
    name: str = "SSO User"
    access_token: str | None = "sso-access-token-abc"
    refresh_token: str | None = "sso-refresh-token-xyz"
    id_token: str | None = "sso-id-token-123"
    token_expires_at: float | None = None

    def __post_init__(self):
        if self.token_expires_at is None:
            self.token_expires_at = time.time() + 3600


@dataclass
class MockTokenPair:
    """Mock token pair returned by create_token_pair."""

    access_token: str = "jwt-token-123"
    refresh_token: str = "jwt-refresh-456"
    expires_in: int = 86400


@dataclass
class MockUser:
    """Mock user from the user store."""

    id: str = "user-001"
    email: str = "sso-user@example.com"
    name: str = "SSO User"
    role: str = "member"
    org_id: str | None = None


@dataclass
class MockOAuthState:
    """Mock OAuth state from the state store."""

    redirect_url: str | None = "/"
    metadata: dict[str, Any] | None = field(default_factory=lambda: {"provider_type": "oidc"})
    expires_at: float = 0.0
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_sso_state():
    """Clear module-level SSO state between tests."""
    _auth_sessions.clear()
    with _sso_providers_lock:
        _sso_providers.clear()
    yield
    _auth_sessions.clear()
    with _sso_providers_lock:
        _sso_providers.clear()


@pytest.fixture
def mock_provider():
    """Create a mock SSO provider."""
    provider = AsyncMock()
    provider.get_authorization_url = AsyncMock(
        return_value="https://idp.example.com/authorize?state=abc"
    )
    provider.authenticate = AsyncMock(return_value=MockSSOUser())
    provider.refresh_token = AsyncMock(
        return_value=MockSSOUser(
            access_token="refreshed-access-token",
            refresh_token="refreshed-refresh-token",
        )
    )
    provider.logout = AsyncMock(return_value="https://idp.example.com/logout?post_logout=/")
    return provider


@pytest.fixture
def mock_state_store():
    """Create a mock OAuth state store."""
    store = MagicMock()
    store.generate.return_value = "test-state-token-abc"
    store.validate_and_consume.return_value = MockOAuthState()
    return store


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_email.return_value = MockUser()
    store.get_user_by_id.return_value = MockUser()
    store.create_user.return_value = MockUser()
    store.update_user.return_value = None
    return store


@pytest.fixture(autouse=True)
def _patch_sso_state_store(mock_state_store):
    """Patch the lazy SSO state store singleton."""
    with patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_lazy:
        mock_lazy.get.return_value = mock_state_store
        yield mock_lazy


# ---------------------------------------------------------------------------
# _cleanup_expired_sessions tests
# ---------------------------------------------------------------------------


class TestCleanupExpiredSessions:
    """Tests for _cleanup_expired_sessions."""

    def test_removes_expired_sessions(self):
        _auth_sessions["expired"] = {
            "created_at": time.time() - AUTH_SESSION_TTL - 100,
            "redirect_url": "/old",
            "provider_type": "oidc",
        }
        _auth_sessions["fresh"] = {
            "created_at": time.time(),
            "redirect_url": "/new",
            "provider_type": "oidc",
        }

        _cleanup_expired_sessions()

        assert "expired" not in _auth_sessions
        assert "fresh" in _auth_sessions

    def test_no_crash_on_empty(self):
        _cleanup_expired_sessions()
        assert len(_auth_sessions) == 0

    def test_removes_session_missing_created_at(self):
        # created_at defaults to 0, which is always expired
        _auth_sessions["no_ts"] = {
            "redirect_url": "/",
            "provider_type": "oidc",
        }
        _cleanup_expired_sessions()
        assert "no_ts" not in _auth_sessions

    def test_keeps_all_fresh_sessions(self):
        for i in range(5):
            _auth_sessions[f"s{i}"] = {
                "created_at": time.time(),
                "redirect_url": f"/{i}",
                "provider_type": "oidc",
            }
        _cleanup_expired_sessions()
        assert len(_auth_sessions) == 5


# ---------------------------------------------------------------------------
# _get_sso_provider tests
# ---------------------------------------------------------------------------


class TestGetSSOProvider:
    """Tests for _get_sso_provider factory."""

    def test_returns_cached_provider(self, mock_provider):
        with _sso_providers_lock:
            _sso_providers["oidc"] = mock_provider

        result = _get_sso_provider("oidc")
        assert result is mock_provider

    def test_returns_none_for_unconfigured_oidc(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_sso_provider("oidc")
            assert result is None

    def test_returns_none_for_unconfigured_google(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_sso_provider("google")
            assert result is None

    def test_returns_none_for_unconfigured_github(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_sso_provider("github")
            assert result is None

    def test_returns_none_for_unknown_provider(self):
        result = _get_sso_provider("unknown_provider")
        assert result is None

    def test_oidc_provider_created_when_configured(self):
        mock_oidc_provider = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {
                    "OIDC_CLIENT_ID": "test-client",
                    "OIDC_CLIENT_SECRET": "test-secret",
                    "OIDC_ISSUER_URL": "https://issuer.example.com",
                },
            ),
            patch(
                "aragora.auth.oidc.OIDCProvider",
                return_value=mock_oidc_provider,
            ),
            patch(
                "aragora.auth.oidc.OIDCConfig",
            ),
        ):
            result = _get_sso_provider("oidc")
            assert result is mock_oidc_provider

    def test_handles_import_error_gracefully(self):
        with patch.dict(
            "os.environ",
            {"OIDC_CLIENT_ID": "test", "OIDC_ISSUER_URL": "https://example.com"},
        ):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("no module"),
            ):
                # Import error is caught and None returned
                result = _get_sso_provider("oidc")
                assert result is None

    def test_caches_created_provider(self, mock_provider):
        # Simulate a provider being created and cached
        with _sso_providers_lock:
            _sso_providers["test_type"] = mock_provider

        assert _get_sso_provider("test_type") is mock_provider
        # Call again - should return cached
        assert _get_sso_provider("test_type") is mock_provider


# ---------------------------------------------------------------------------
# handle_sso_login tests
# ---------------------------------------------------------------------------


class TestHandleSSOLogin:
    """Tests for handle_sso_login."""

    @pytest.mark.asyncio
    async def test_successful_login(self, mock_provider, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "oidc", "redirect_url": "/dashboard"})

        assert _status(result) == 200
        data = _data(result)
        assert data["authorization_url"] == "https://idp.example.com/authorize?state=abc"
        assert data["state"] == "test-state-token-abc"
        assert data["provider"] == "oidc"
        assert data["expires_in"] == AUTH_SESSION_TTL

    @pytest.mark.asyncio
    async def test_login_default_provider(self, mock_provider, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "oidc"

    @pytest.mark.asyncio
    async def test_login_stores_session(self, mock_provider, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"redirect_url": "/after-auth"})

        assert _status(result) == 200
        state = _data(result)["state"]
        assert state in _auth_sessions
        session = _auth_sessions[state]
        assert session["redirect_url"] == "/after-auth"
        assert session["provider_type"] == "oidc"
        assert "created_at" in session

    @pytest.mark.asyncio
    async def test_login_provider_not_configured(self):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=None,
        ):
            result = await handle_sso_login({"provider": "saml"})

        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_login_connection_error(self, mock_provider, mock_state_store):
        mock_provider.get_authorization_url.side_effect = ConnectionError("IdP unreachable")
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "oidc"})

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_login_timeout_error(self, mock_provider, mock_state_store):
        mock_provider.get_authorization_url.side_effect = TimeoutError("timed out")
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "oidc"})

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_login_default_redirect_url(self, mock_provider, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({})

        assert _status(result) == 200
        # Session should have default redirect
        state = _data(result)["state"]
        assert _auth_sessions[state]["redirect_url"] == "/"

    @pytest.mark.asyncio
    async def test_login_cleans_up_expired_sessions_before_storing(
        self, mock_provider, mock_state_store
    ):
        # Add an expired session
        _auth_sessions["expired-state"] = {
            "created_at": time.time() - AUTH_SESSION_TTL - 100,
            "redirect_url": "/old",
            "provider_type": "oidc",
        }

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({})

        assert _status(result) == 200
        assert "expired-state" not in _auth_sessions

    @pytest.mark.asyncio
    async def test_login_generates_state_via_store(self, mock_provider, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "google", "redirect_url": "/home"})

        assert _status(result) == 200
        mock_state_store.generate.assert_called_once_with(
            redirect_url="/home",
            metadata={"provider_type": "google"},
        )

    @pytest.mark.asyncio
    async def test_login_value_error(self, mock_provider, mock_state_store):
        mock_state_store.generate.side_effect = ValueError("bad state")
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "oidc"})

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# handle_sso_callback tests
# ---------------------------------------------------------------------------


class TestHandleSSOCallback:
    """Tests for handle_sso_callback."""

    @pytest.mark.asyncio
    async def test_successful_callback_existing_user(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=MockTokenPair(),
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "auth-code-abc",
                    "state": "valid-state",
                }
            )

        assert _status(result) == 200
        data = _data(result)
        assert data["access_token"] == "jwt-token-123"
        assert data["token_type"] == "bearer"
        assert data["user"]["id"] == "user-001"
        assert data["user"]["email"] == "sso-user@example.com"
        assert data["redirect_url"] == "/"
        assert data["sso_access_token"] == "sso-access-token-abc"

    @pytest.mark.asyncio
    async def test_callback_creates_new_user(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        mock_user_store.get_user_by_email.return_value = None
        new_user = MockUser(id="new-user-002", email="new@example.com")
        mock_user_store.create_user.return_value = new_user

        mock_provider.authenticate.return_value = MockSSOUser(
            email="new@example.com", name="New User"
        )

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-new-user",
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "auth-code",
                    "state": "state-token",
                }
            )

        assert _status(result) == 200
        mock_user_store.create_user.assert_called_once_with(
            email="new@example.com",
            password_hash="sso",
            password_salt="",
            name="New User",
        )

    @pytest.mark.asyncio
    async def test_callback_new_user_name_fallback_to_email(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = MockUser(email="noname@corp.com", name="noname")
        mock_provider.authenticate.return_value = MockSSOUser(email="noname@corp.com", name="")

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-tok",
            ),
        ):
            result = await handle_sso_callback({"code": "c", "state": "s"})

        assert _status(result) == 200
        # When name is empty, handler should use email prefix
        call_kwargs = mock_user_store.create_user.call_args
        assert call_kwargs.kwargs["name"] == "noname"  # email.split("@")[0]

    @pytest.mark.asyncio
    async def test_callback_idp_error(self, mock_state_store):
        result = await handle_sso_callback(
            {
                "error": "access_denied",
                "error_description": "User cancelled authentication",
            }
        )

        assert _status(result) == 401
        assert "User cancelled authentication" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_callback_idp_error_no_description(self, mock_state_store):
        result = await handle_sso_callback(
            {
                "error": "server_error",
            }
        )

        assert _status(result) == 401
        assert "server_error" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_callback_no_code(self, mock_state_store):
        result = await handle_sso_callback({"state": "valid-state"})

        assert _status(result) == 400
        assert "code" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_no_state(self, mock_state_store):
        result = await handle_sso_callback({"code": "auth-code"})

        assert _status(result) == 400
        assert "state" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, mock_state_store):
        mock_state_store.validate_and_consume.return_value = None
        # No in-memory session either
        result = await handle_sso_callback(
            {
                "code": "auth-code",
                "state": "invalid-state",
            }
        )

        assert _status(result) == 401
        assert (
            "invalid" in _body(result).get("error", "").lower()
            or "expired" in _body(result).get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_callback_fallback_to_in_memory_session(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        """When state store validation fails, fall back to in-memory sessions."""
        mock_state_store.validate_and_consume.return_value = None

        _auth_sessions["fallback-state"] = {
            "created_at": time.time(),
            "redirect_url": "/fallback",
            "provider_type": "google",
        }

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-fallback",
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "auth-code",
                    "state": "fallback-state",
                }
            )

        assert _status(result) == 200
        data = _data(result)
        assert data["redirect_url"] == "/fallback"
        # Session should be consumed (popped)
        assert "fallback-state" not in _auth_sessions

    @pytest.mark.asyncio
    async def test_callback_provider_unavailable(self, mock_state_store):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=None,
        ):
            result = await handle_sso_callback(
                {
                    "code": "auth-code",
                    "state": "state",
                }
            )

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_user_store_unavailable(self, mock_provider, mock_state_store):
        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=None,
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "auth-code",
                    "state": "state",
                }
            )

        assert _status(result) == 503
        assert "unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_authenticate_failure(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        mock_provider.authenticate.side_effect = ConnectionError("IdP error")

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "bad-code",
                    "state": "state",
                }
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_updates_existing_user_name(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        existing = MockUser(id="u1", email="user@example.com", name="Old Name")
        mock_user_store.get_user_by_email.return_value = existing
        mock_user_store.get_user_by_id.return_value = MockUser(
            id="u1", email="user@example.com", name="SSO Updated"
        )

        mock_provider.authenticate.return_value = MockSSOUser(
            email="user@example.com", name="SSO Updated"
        )

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=MockTokenPair(),
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 200
        # Handler calls update_user for name change and again for last_login_at
        name_call = call("u1", name="SSO Updated")
        assert name_call in mock_user_store.update_user.call_args_list

    @pytest.mark.asyncio
    async def test_callback_missing_code_and_state(self, mock_state_store):
        result = await handle_sso_callback({})

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_callback_uses_state_redirect_url(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        mock_state_store.validate_and_consume.return_value = MockOAuthState(
            redirect_url="/custom-redirect",
            metadata={"provider_type": "github"},
        )

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-tok",
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 200
        assert _data(result)["redirect_url"] == "/custom-redirect"

    @pytest.mark.asyncio
    async def test_callback_state_with_no_metadata(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        mock_state_store.validate_and_consume.return_value = MockOAuthState(
            redirect_url=None,
            metadata=None,
        )

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-tok",
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 200
        # Defaults to "/" and "oidc"
        assert _data(result)["redirect_url"] == "/"

    @pytest.mark.asyncio
    async def test_callback_existing_user_without_update_user_method(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        """When user store lacks update_user, it should not crash."""
        existing = MockUser(id="u1")
        mock_user_store.get_user_by_email.return_value = existing
        mock_user_store.get_user_by_id.return_value = existing
        del mock_user_store.update_user  # Remove the method

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_access_token",
                return_value="jwt-tok",
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# handle_sso_refresh tests
# ---------------------------------------------------------------------------


class TestHandleSSORefresh:
    """Tests for handle_sso_refresh."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "old-refresh-token",
                }
            )

        assert _status(result) == 200
        data = _data(result)
        assert data["access_token"] == "refreshed-access-token"
        assert data["refresh_token"] == "refreshed-refresh-token"
        assert "expires_at" in data

    @pytest.mark.asyncio
    async def test_refresh_default_provider(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh({"refresh_token": "tok"})

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_refresh_no_token(self):
        result = await handle_sso_refresh({"provider": "oidc"})

        assert _status(result) == 400
        assert "refresh_token" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_provider_unavailable(self):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=None,
        ):
            result = await handle_sso_refresh(
                {
                    "provider": "oidc",
                    "refresh_token": "tok",
                }
            )

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_returns_none(self, mock_provider):
        mock_provider.refresh_token.return_value = None

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh(
                {
                    "refresh_token": "expired-token",
                }
            )

        assert _status(result) == 401
        assert "refresh failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_connection_error(self, mock_provider):
        mock_provider.refresh_token.side_effect = ConnectionError("IdP down")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh(
                {
                    "refresh_token": "tok",
                }
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_refresh_creates_temp_sso_user(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh(
                {
                    "refresh_token": "my-refresh",
                },
                user_id="u123",
            )

        assert _status(result) == 200
        # Verify the provider was called with a temp SSOUser having our refresh token
        call_args = mock_provider.refresh_token.call_args
        temp_user = call_args[0][0]  # First positional arg
        assert temp_user.refresh_token == "my-refresh"
        assert temp_user.id == "u123"

    @pytest.mark.asyncio
    async def test_refresh_empty_token(self):
        result = await handle_sso_refresh(
            {
                "provider": "oidc",
                "refresh_token": "",
            }
        )

        assert _status(result) == 400


# ---------------------------------------------------------------------------
# handle_sso_logout tests
# ---------------------------------------------------------------------------


class TestHandleSSOLogout:
    """Tests for handle_sso_logout."""

    @pytest.mark.asyncio
    async def test_successful_logout_with_id_token(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "my-id-token",
                }
            )

        assert _status(result) == 200
        data = _data(result)
        assert data["logged_out"] is True
        assert data["logout_url"] == "https://idp.example.com/logout?post_logout=/"

    @pytest.mark.asyncio
    async def test_logout_without_id_token(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout({"provider": "oidc"})

        assert _status(result) == 200
        data = _data(result)
        assert data["logged_out"] is True
        assert data["logout_url"] is None

    @pytest.mark.asyncio
    async def test_logout_without_provider(self):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=None,
        ):
            result = await handle_sso_logout({"id_token": "tok"})

        assert _status(result) == 200
        data = _data(result)
        assert data["logged_out"] is True
        assert data["logout_url"] is None

    @pytest.mark.asyncio
    async def test_logout_default_provider(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout({})

        assert _status(result) == 200
        data = _data(result)
        assert data["logged_out"] is True

    @pytest.mark.asyncio
    async def test_logout_connection_error(self, mock_provider):
        mock_provider.logout.side_effect = ConnectionError("IdP unreachable")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "tok",
                }
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_logout_creates_temp_sso_user_for_idp(self, mock_provider):
        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout(
                {"provider": "oidc", "id_token": "my-id"},
                user_id="u456",
            )

        assert _status(result) == 200
        call_args = mock_provider.logout.call_args
        temp_user = call_args[0][0]
        assert temp_user.id_token == "my-id"
        assert temp_user.id == "u456"


# ---------------------------------------------------------------------------
# handle_list_providers tests
# ---------------------------------------------------------------------------


class TestHandleListProviders:
    """Tests for handle_list_providers."""

    @pytest.mark.asyncio
    async def test_no_providers_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        assert data["sso_enabled"] is False
        providers = data["providers"]
        assert len(providers) == 4
        assert all(not p["enabled"] for p in providers)

    @pytest.mark.asyncio
    async def test_oidc_provider_configured(self):
        with patch.dict(
            "os.environ",
            {"OIDC_CLIENT_ID": "test", "OIDC_ISSUER_URL": "https://issuer.example.com"},
            clear=True,
        ):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        assert data["sso_enabled"] is True
        oidc = next(p for p in data["providers"] if p["type"] == "oidc")
        assert oidc["enabled"] is True
        assert oidc["name"] == "OIDC/OAuth 2.0"

    @pytest.mark.asyncio
    async def test_google_provider_configured(self):
        with patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "goog-client"}, clear=True):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        assert data["sso_enabled"] is True
        google = next(p for p in data["providers"] if p["type"] == "google")
        assert google["enabled"] is True

    @pytest.mark.asyncio
    async def test_github_provider_configured(self):
        with patch.dict("os.environ", {"GITHUB_CLIENT_ID": "gh-client"}, clear=True):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        github = next(p for p in data["providers"] if p["type"] == "github")
        assert github["enabled"] is True

    @pytest.mark.asyncio
    async def test_azure_ad_provider_configured(self):
        with patch.dict(
            "os.environ",
            {"AZURE_AD_CLIENT_ID": "azure-id", "AZURE_AD_TENANT_ID": "tenant-id"},
            clear=True,
        ):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        azure = next(p for p in data["providers"] if p["type"] == "azure_ad")
        assert azure["enabled"] is True

    @pytest.mark.asyncio
    async def test_azure_ad_partial_config_not_enabled(self):
        with patch.dict(
            "os.environ",
            {"AZURE_AD_CLIENT_ID": "azure-id"},
            clear=True,
        ):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        azure = next(p for p in data["providers"] if p["type"] == "azure_ad")
        assert azure["enabled"] is False

    @pytest.mark.asyncio
    async def test_multiple_providers_configured(self):
        with patch.dict(
            "os.environ",
            {
                "OIDC_CLIENT_ID": "oidc-id",
                "OIDC_ISSUER_URL": "https://issuer.com",
                "GOOGLE_CLIENT_ID": "google-id",
                "GITHUB_CLIENT_ID": "github-id",
            },
            clear=True,
        ):
            result = await handle_list_providers({})

        assert _status(result) == 200
        data = _data(result)
        assert data["sso_enabled"] is True
        enabled = [p for p in data["providers"] if p["enabled"]]
        assert len(enabled) == 3

    @pytest.mark.asyncio
    async def test_provider_types_in_response(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_list_providers({})

        data = _data(result)
        types = {p["type"] for p in data["providers"]}
        assert types == {"oidc", "google", "github", "azure_ad"}

    @pytest.mark.asyncio
    async def test_provider_names_in_response(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_list_providers({})

        data = _data(result)
        names = {p["name"] for p in data["providers"]}
        assert "OIDC/OAuth 2.0" in names
        assert "Google" in names
        assert "GitHub" in names
        assert "Azure AD" in names


# ---------------------------------------------------------------------------
# handle_get_sso_config tests
# ---------------------------------------------------------------------------


class TestHandleGetSSOConfig:
    """Tests for handle_get_sso_config."""

    @pytest.mark.asyncio
    async def test_oidc_config_enabled(self):
        with patch.dict(
            "os.environ",
            {
                "OIDC_CLIENT_ID": "client-123",
                "OIDC_ISSUER_URL": "https://issuer.example.com",
                "OIDC_SCOPES": "openid,email,profile,groups",
            },
            clear=True,
        ):
            result = await handle_get_sso_config({"provider": "oidc"})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "oidc"
        assert data["enabled"] is True
        assert data["issuer_url"] == "https://issuer.example.com"
        assert data["scopes"] == ["openid", "email", "profile", "groups"]

    @pytest.mark.asyncio
    async def test_oidc_config_default_scopes(self):
        with patch.dict(
            "os.environ",
            {
                "OIDC_CLIENT_ID": "client-123",
                "OIDC_ISSUER_URL": "https://issuer.example.com",
            },
            clear=True,
        ):
            result = await handle_get_sso_config({"provider": "oidc"})

        assert _status(result) == 200
        data = _data(result)
        assert data["scopes"] == ["openid", "email", "profile"]

    @pytest.mark.asyncio
    async def test_oidc_config_not_enabled(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_get_sso_config({"provider": "oidc"})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "oidc"
        assert data["enabled"] is False
        assert "issuer_url" not in data

    @pytest.mark.asyncio
    async def test_google_config_enabled(self):
        with patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "goog-123"}, clear=True):
            result = await handle_get_sso_config({"provider": "google"})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "google"
        assert data["enabled"] is True
        assert data["issuer_url"] == "https://accounts.google.com"
        assert data["scopes"] == ["openid", "email", "profile"]

    @pytest.mark.asyncio
    async def test_google_config_not_enabled(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_get_sso_config({"provider": "google"})

        assert _status(result) == 200
        data = _data(result)
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_github_config_enabled(self):
        with patch.dict("os.environ", {"GITHUB_CLIENT_ID": "gh-123"}, clear=True):
            result = await handle_get_sso_config({"provider": "github"})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "github"
        assert data["enabled"] is True
        assert data["authorization_endpoint"] == "https://github.com/login/oauth/authorize"
        assert data["scopes"] == ["user:email", "read:user"]

    @pytest.mark.asyncio
    async def test_github_config_not_enabled(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_get_sso_config({"provider": "github"})

        assert _status(result) == 200
        data = _data(result)
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_get_sso_config({"provider": "saml"})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "saml"
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_default_provider_is_oidc(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await handle_get_sso_config({})

        assert _status(result) == 200
        data = _data(result)
        assert data["provider"] == "oidc"

    @pytest.mark.asyncio
    async def test_config_does_not_expose_secrets(self):
        with patch.dict(
            "os.environ",
            {
                "OIDC_CLIENT_ID": "client-123",
                "OIDC_CLIENT_SECRET": "super-secret",
                "OIDC_ISSUER_URL": "https://issuer.example.com",
            },
            clear=True,
        ):
            result = await handle_get_sso_config({"provider": "oidc"})

        assert _status(result) == 200
        data = _data(result)
        response_str = json.dumps(data)
        assert "super-secret" not in response_str
        assert "client-123" not in response_str  # client_id not in config response


# ---------------------------------------------------------------------------
# get_sso_handlers tests
# ---------------------------------------------------------------------------


class TestGetSSOHandlers:
    """Tests for get_sso_handlers registry function."""

    def test_returns_dict_of_handlers(self):
        handlers = get_sso_handlers()
        assert isinstance(handlers, dict)

    def test_contains_all_handler_keys(self):
        handlers = get_sso_handlers()
        expected_keys = {
            "sso_login",
            "sso_callback",
            "sso_refresh",
            "sso_logout",
            "sso_list_providers",
            "sso_get_config",
        }
        assert set(handlers.keys()) == expected_keys

    def test_all_handlers_are_callable(self):
        handlers = get_sso_handlers()
        for name, handler in handlers.items():
            assert callable(handler), f"Handler '{name}' is not callable"

    def test_handler_function_references(self):
        handlers = get_sso_handlers()
        # The handlers may be wrapped by decorators, so check they exist
        assert handlers["sso_login"] is not None
        assert handlers["sso_callback"] is not None
        assert handlers["sso_refresh"] is not None
        assert handlers["sso_logout"] is not None
        assert handlers["sso_list_providers"] is not None
        assert handlers["sso_get_config"] is not None


# ---------------------------------------------------------------------------
# AUTH_SESSION_TTL tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_session_ttl_is_positive(self):
        assert AUTH_SESSION_TTL > 0

    def test_session_ttl_matches_oauth_state_ttl(self):
        from aragora.server.oauth_state_store import OAUTH_STATE_TTL_SECONDS

        assert AUTH_SESSION_TTL == OAUTH_STATE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Edge cases and integration patterns
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for SSO handlers."""

    @pytest.mark.asyncio
    async def test_login_with_all_provider_types(self, mock_provider, mock_state_store):
        """Test login flow works for each supported provider type."""
        for provider_type in ["oidc", "google", "github"]:
            with patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ):
                result = await handle_sso_login({"provider": provider_type})
            assert _status(result) == 200
            assert _data(result)["provider"] == provider_type

    @pytest.mark.asyncio
    async def test_callback_with_empty_code(self, mock_state_store):
        result = await handle_sso_callback(
            {
                "code": "",
                "state": "state-token",
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_callback_with_empty_state(self, mock_state_store):
        result = await handle_sso_callback(
            {
                "code": "auth-code",
                "state": "",
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_refresh_with_none_token(self):
        result = await handle_sso_refresh(
            {
                "provider": "oidc",
                "refresh_token": None,
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_callback_attribute_error_caught(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        """AttributeError in the callback flow should be caught."""
        mock_provider.authenticate.side_effect = AttributeError("missing attr")

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_logout_value_error_caught(self, mock_provider):
        mock_provider.logout.side_effect = ValueError("bad token")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "bad",
                }
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_refresh_import_error_caught(self, mock_provider):
        """ImportError during SSOUser import should be caught."""
        mock_provider.refresh_token.side_effect = ImportError("no module")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_refresh(
                {
                    "refresh_token": "tok",
                }
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_key_error_caught(
        self, mock_provider, mock_state_store, mock_user_store
    ):
        """KeyError in callback flow should be caught."""
        mock_provider.authenticate.side_effect = KeyError("missing_key")

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=mock_user_store,
            ),
        ):
            result = await handle_sso_callback(
                {
                    "code": "code",
                    "state": "state",
                }
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_login_os_error_caught(self, mock_provider, mock_state_store):
        mock_provider.get_authorization_url.side_effect = OSError("disk error")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_login({"provider": "oidc"})

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_logout_os_error_caught(self, mock_provider):
        mock_provider.logout.side_effect = OSError("os error")

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
            return_value=mock_provider,
        ):
            result = await handle_sso_logout(
                {
                    "provider": "oidc",
                    "id_token": "tok",
                }
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_idp_error_takes_precedence(self, mock_state_store):
        """IdP error should be returned even if code and state are present."""
        result = await handle_sso_callback(
            {
                "code": "auth-code",
                "state": "state",
                "error": "invalid_scope",
                "error_description": "Requested scope not allowed",
            }
        )

        assert _status(result) == 401
        assert "scope" in _body(result).get("error", "").lower()
