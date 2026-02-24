"""
OAuth E2E Authentication and Registration Flow Tests.

Tests cover:
- Successful OAuth login for new user (auto-registration)
- Successful OAuth login for existing user
- Account linking (add new provider to existing account)
- Post-login redirect works correctly
- Invalid OAuth state handling
- Expired OAuth token handling
- Session creation and binding
- SSO callback auto-registration
- SSO callback account linking
- SSO callback session binding
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeUser:
    """Minimal User stand-in for tests."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    role: str = "member"
    org_id: str | None = None
    is_active: bool = True
    password_hash: str = "hashed"
    password_salt: str = "salt"
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    created_at: datetime | None = None
    last_login_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "org_id": self.org_id,
            "is_active": self.is_active,
        }

    def verify_password(self, password: str) -> bool:
        return password == "correct-password"

    def verify_api_key(self, key: str) -> bool:
        return False


@dataclass
class FakeOrg:
    """Minimal Organization stand-in for tests."""

    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: str = "free"
    owner_id: str = "user-123"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "tier": self.tier,
            "owner_id": self.owner_id,
        }


@dataclass
class FakeTokenPair:
    """Minimal TokenPair stand-in for tests."""

    access_token: str = "access-token-123"
    refresh_token: str = "refresh-token-456"
    expires_in: int = 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": "Bearer",
            "expires_in": self.expires_in,
        }


@dataclass
class FakeSSOUser:
    """Minimal SSOUser stand-in for tests."""

    id: str = "sso-user-id-123"
    email: str = "ssouser@example.com"
    name: str = "SSO User"
    access_token: str = "sso-access-token"
    refresh_token: str | None = None
    id_token: str | None = None
    token_expires_at: float = 0.0
    first_name: str = ""
    last_name: str = ""
    username: str = ""
    roles: list[str] | None = None
    groups: list[str] | None = None
    provider_type: str = "oidc"
    provider_id: str = ""
    raw_claims: dict[str, Any] | None = None


class FakeUserStore:
    """In-memory user store for testing."""

    def __init__(self) -> None:
        self.users: dict[str, FakeUser] = {}
        self.users_by_email: dict[str, str] = {}
        self.oauth_links: dict[str, list[dict[str, str]]] = {}

    def get_user_by_id(self, user_id: str) -> FakeUser | None:
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> FakeUser | None:
        uid = self.users_by_email.get(email.lower())
        return self.users.get(uid) if uid else None

    def create_user(
        self,
        email: str,
        password_hash: str = "",
        password_salt: str = "",
        name: str = "",
    ) -> FakeUser:
        import uuid

        user_id = f"user-{uuid.uuid4().hex[:8]}"
        user = FakeUser(
            id=user_id,
            email=email,
            name=name or email.split("@")[0],
            password_hash=password_hash,
            password_salt=password_salt,
        )
        self.users[user_id] = user
        self.users_by_email[email.lower()] = user_id
        return user

    def update_user(self, user_id: str, **kwargs: Any) -> None:
        user = self.users.get(user_id)
        if user:
            for k, v in kwargs.items():
                if hasattr(user, k):
                    setattr(user, k, v)

    def create_organization(self, name: str, owner_id: str) -> FakeOrg:
        org = FakeOrg(name=name, owner_id=owner_id)
        user = self.users.get(owner_id)
        if user:
            user.org_id = org.id
        return org

    def get_organization_by_id(self, org_id: str) -> FakeOrg | None:
        for user in self.users.values():
            if user.org_id == org_id:
                return FakeOrg(id=org_id, owner_id=user.id)
        return None

    def link_oauth_provider(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: str,
    ) -> bool:
        if user_id not in self.oauth_links:
            self.oauth_links[user_id] = []
        self.oauth_links[user_id].append(
            {
                "provider": provider,
                "provider_user_id": provider_user_id,
                "email": email,
            }
        )
        return True

    def get_oauth_providers(self, user_id: str) -> list[dict[str, str]]:
        return self.oauth_links.get(user_id, [])

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> FakeUser | None:
        for uid, links in self.oauth_links.items():
            for link in links:
                if link["provider"] == provider and link["provider_user_id"] == provider_user_id:
                    return self.users.get(uid)
        return None

    def increment_token_version(self, user_id: str) -> int:
        return 1


def _parse_success_body(result: Any) -> dict[str, Any]:
    """Parse a HandlerResult body, unwrapping the success_response envelope."""
    import json as _json

    body = result.body
    if isinstance(body, bytes):
        body = _json.loads(body.decode())
    # success_response wraps in {"success": true, "data": {...}}
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


@pytest.fixture
def user_store() -> FakeUserStore:
    return FakeUserStore()


@pytest.fixture
def existing_user(user_store: FakeUserStore) -> FakeUser:
    """Create an existing user in the store."""
    user = FakeUser(
        id="existing-user-1",
        email="existing@example.com",
        name="Existing User",
        org_id="org-existing",
    )
    user_store.users[user.id] = user
    user_store.users_by_email[user.email.lower()] = user.id
    return user


# ============================================================================
# SSO Callback Tests
# ============================================================================


class TestSSOCallbackNewUser:
    """Test SSO callback auto-registration for new users."""

    @pytest.mark.asyncio
    async def test_new_user_auto_registered(self, user_store):
        """New user is auto-created on first SSO login."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(email="newuser@example.com", name="New User")
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            # Configure state store
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/dashboard"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            # Configure provider
            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code-123", "state": "valid-state"},
                user_id="default",
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        # Verify user was created
        assert body["is_new_user"] is True
        assert body["access_token"] == fake_tokens.access_token
        assert body["user"]["email"] == "newuser@example.com"

    @pytest.mark.asyncio
    async def test_new_user_gets_default_organization(self, user_store):
        """New SSO user gets an auto-created organization."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(email="orguser@example.com", name="Org User")
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "google"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "state-token"},
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        # The user should have an org_id set
        assert body["user"]["org_id"] is not None


class TestSSOCallbackExistingUser:
    """Test SSO callback for existing users."""

    @pytest.mark.asyncio
    async def test_existing_user_login(self, user_store, existing_user):
        """Existing user can login via SSO."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(
            email=existing_user.email,
            name="Updated Name",
        )
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        assert body["is_new_user"] is False
        assert body["user"]["id"] == existing_user.id
        assert body["user"]["email"] == existing_user.email


class TestSSOCallbackAccountLinking:
    """Test SSO callback account linking (existing user + new provider)."""

    @pytest.mark.asyncio
    async def test_account_linking_on_sso_login(self, user_store, existing_user):
        """When existing user logs in via SSO, provider is linked to account."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(
            id="google-provider-id-456",
            email=existing_user.email,
            name="Existing User",
        )
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "google"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 200

        # Verify provider was linked
        providers = user_store.get_oauth_providers(existing_user.id)
        assert len(providers) == 1
        assert providers[0]["provider"] == "google"
        assert providers[0]["provider_user_id"] == "google-provider-id-456"


class TestSSOCallbackPostLoginRedirect:
    """Test post-login redirect from SSO callback."""

    @pytest.mark.asyncio
    async def test_redirect_url_preserved(self, user_store):
        """Redirect URL from state is included in response."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(email="redirect@example.com")
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/debates/123"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        assert body["redirect_url"] == "/debates/123"

    @pytest.mark.asyncio
    async def test_default_redirect_when_not_specified(self, user_store):
        """Default redirect URL is / when not specified in state."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(email="default@example.com")
        fake_tokens = FakeTokenPair()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = None  # No redirect URL
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        assert body["redirect_url"] == "/"


class TestSSOCallbackInvalidState:
    """Test invalid OAuth state handling."""

    @pytest.mark.asyncio
    async def test_missing_state_returns_400(self):
        """Missing state parameter returns 400."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        result = await handle_sso_callback(
            {"code": "auth-code"},  # No state
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_code_returns_400(self):
        """Missing authorization code returns 400."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        result = await handle_sso_callback(
            {"state": "some-state"},  # No code
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_state_returns_401(self):
        """Invalid/expired state returns 401."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        with patch(
            "aragora.server.handlers.auth.sso_handlers._sso_state_store"
        ) as mock_state_store:
            mock_store = MagicMock()
            mock_store.validate_and_consume.return_value = None
            mock_state_store.get.return_value = mock_store

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "expired-state"},
            )

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_idp_error_returns_401(self):
        """IdP error in callback returns 401."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        result = await handle_sso_callback(
            {
                "error": "access_denied",
                "error_description": "User denied access",
            },
        )
        assert result.status_code == 401


class TestSSOCallbackSessionBinding:
    """Test session creation after SSO callback."""

    @pytest.mark.asyncio
    async def test_session_created_after_login(self, user_store):
        """A session is created for the user after successful SSO login."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        sso_user = FakeSSOUser(email="session@example.com")
        fake_tokens = FakeTokenPair()
        mock_session_manager = MagicMock()

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
            patch(
                "aragora.billing.auth.sessions.get_session_manager",
                return_value=mock_session_manager,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = sso_user
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 200

        # Verify session was created
        mock_session_manager.create_session.assert_called_once()
        call_kwargs = mock_session_manager.create_session.call_args
        assert call_kwargs.kwargs["token_jti"] is not None
        # Verify JTI is derived from token hash
        expected_jti = hashlib.sha256(fake_tokens.access_token.encode()).hexdigest()[:32]
        assert call_kwargs.kwargs["token_jti"] == expected_jti


class TestSSOCallbackExpiredToken:
    """Test expired OAuth token handling."""

    @pytest.mark.asyncio
    async def test_provider_auth_failure_returns_401(self, user_store):
        """When the provider rejects the code (expired), return 401."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider"
            ) as mock_get_provider,
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            mock_provider = AsyncMock()
            mock_provider.authenticate.side_effect = ValueError("Token expired")
            mock_get_provider.return_value = mock_provider

            result = await handle_sso_callback(
                {"code": "expired-code", "state": "valid-state"},
            )

        assert result.status_code == 401


class TestSSOCallbackProviderUnavailable:
    """Test SSO callback when provider is unavailable."""

    @pytest.mark.asyncio
    async def test_unavailable_provider_returns_503(self):
        """When SSO provider is not configured, return 503."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        with (
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=None,
            ),
        ):
            mock_store = MagicMock()
            mock_state = MagicMock()
            mock_state.metadata = {"provider_type": "oidc"}
            mock_state.redirect_url = "/"
            mock_store.validate_and_consume.return_value = mock_state
            mock_state_store.get.return_value = mock_store

            result = await handle_sso_callback(
                {"code": "auth-code", "state": "valid-state"},
            )

        assert result.status_code == 503


# ============================================================================
# SSO Login Flow Tests
# ============================================================================


class TestSSOLoginFlow:
    """Test SSO login initiation."""

    @pytest.mark.asyncio
    async def test_login_returns_auth_url(self):
        """SSO login returns an authorization URL."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_login

        mock_provider = AsyncMock()
        mock_provider.get_authorization_url.return_value = "https://idp.example.com/auth?state=abc"

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
        ):
            mock_store = MagicMock()
            mock_store.generate.return_value = "state-token-abc"
            mock_state_store.get.return_value = mock_store

            result = await handle_sso_login(
                {"provider": "oidc", "redirect_url": "/debates"},
            )

        assert result.status_code == 200
        body = _parse_success_body(result)

        assert "authorization_url" in body
        assert body["state"] == "state-token-abc"

    @pytest.mark.asyncio
    async def test_login_stores_redirect_url(self):
        """SSO login stores the redirect URL in state."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_login

        mock_provider = AsyncMock()
        mock_provider.get_authorization_url.return_value = "https://idp.example.com/auth"

        with (
            patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=mock_provider,
            ),
            patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_state_store,
        ):
            mock_store = MagicMock()
            mock_store.generate.return_value = "state-token"
            mock_state_store.get.return_value = mock_store

            result = await handle_sso_login(
                {"provider": "google", "redirect_url": "/debate/456"},
            )

        assert result.status_code == 200

        # Verify the state store was called with the redirect_url
        mock_store.generate.assert_called_once()
        call_kwargs = mock_store.generate.call_args
        assert call_kwargs.kwargs.get("redirect_url") == "/debate/456" or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "/debate/456"
        )


# ============================================================================
# OAuth Handler Integration Tests
# ============================================================================


class TestOAuthHandlerNewUser:
    """Test OAuth handler auto-registration for new users."""

    def test_create_oauth_user_returns_user(self):
        """_create_oauth_user creates a new user from OAuth info."""
        from aragora.server.handlers._oauth.base import OAuthHandler
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        handler = OAuthHandler(ctx={"user_store": None})
        store = FakeUserStore()

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google-123",
            email="new-oauth@example.com",
            name="OAuth User",
            email_verified=True,
        )

        with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
            user = handler._create_oauth_user(store, user_info)

        assert user is not None
        assert user.email == "new-oauth@example.com"
        assert user.name == "OAuth User"

    def test_create_oauth_user_links_provider(self):
        """_create_oauth_user also links the OAuth provider."""
        from aragora.server.handlers._oauth.base import OAuthHandler
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        handler = OAuthHandler(ctx={"user_store": None})
        store = FakeUserStore()

        user_info = OAuthUserInfo(
            provider="github",
            provider_user_id="gh-456",
            email="github-user@example.com",
            name="GH User",
            email_verified=True,
        )

        with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
            user = handler._create_oauth_user(store, user_info)

        # Verify provider was linked
        providers = store.get_oauth_providers(user.id)
        assert len(providers) == 1
        assert providers[0]["provider"] == "github"
        assert providers[0]["provider_user_id"] == "gh-456"


class TestOAuthHandlerExistingUser:
    """Test OAuth handler for existing user login."""

    def test_find_user_by_oauth(self):
        """_find_user_by_oauth finds user by provider ID."""
        from aragora.server.handlers._oauth.base import OAuthHandler
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        handler = OAuthHandler(ctx={"user_store": None})
        store = FakeUserStore()

        # Create user and link OAuth
        user = store.create_user(email="linked@example.com", name="Linked")
        store.link_oauth_provider(user.id, "google", "google-linked-123", "linked@example.com")

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google-linked-123",
            email="linked@example.com",
            name="Linked",
            email_verified=True,
        )

        found = handler._find_user_by_oauth(store, user_info)
        assert found is not None
        assert found.id == user.id


class TestOAuthHandlerAccountLinking:
    """Test OAuth account linking to existing user."""

    def test_link_oauth_to_user(self):
        """_link_oauth_to_user links a new provider to existing user."""
        from aragora.server.handlers._oauth.base import OAuthHandler
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        handler = OAuthHandler(ctx={"user_store": None})
        store = FakeUserStore()

        user = store.create_user(email="linking@example.com", name="Linker")

        user_info = OAuthUserInfo(
            provider="microsoft",
            provider_user_id="ms-789",
            email="linking@example.com",
            name="Linker",
            email_verified=True,
        )

        result = handler._link_oauth_to_user(store, user.id, user_info)
        assert result is True

        providers = store.get_oauth_providers(user.id)
        assert len(providers) == 1
        assert providers[0]["provider"] == "microsoft"


class TestOAuthHandlerRedirect:
    """Test OAuth redirect with tokens."""

    def test_redirect_with_tokens_uses_fragment(self):
        """_redirect_with_tokens puts tokens in URL fragment (not query)."""
        from aragora.server.handlers._oauth.base import OAuthHandler

        handler = OAuthHandler(ctx={})
        tokens = FakeTokenPair(
            access_token="my-access-token",
            refresh_token="my-refresh-token",
        )

        result = handler._redirect_with_tokens("https://app.example.com/callback", tokens)
        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "#" in location
        assert "access_token=my-access-token" in location
        assert "refresh_token=my-refresh-token" in location
        # Tokens should NOT be in query params (security)
        assert "?" not in location.split("#")[0] or "access_token" not in location.split("#")[0]

    def test_redirect_with_error(self):
        """_redirect_with_error redirects to error URL."""
        from aragora.server.handlers._oauth.base import OAuthHandler

        handler = OAuthHandler(ctx={})

        with patch("aragora.server.handlers._oauth.base._impl") as mock_impl:
            mock_impl()._get_oauth_error_url.return_value = "https://app.example.com/auth/error"
            result = handler._redirect_with_error("Something went wrong")

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "error=" in location


class TestOAuthHandlerSessionBinding:
    """Test session binding in OAuth handler complete flow."""

    def test_session_created_in_complete_flow(self):
        """Session is created when _complete_oauth_flow_async runs."""
        from aragora.server.handlers._oauth.base import OAuthHandler
        from aragora.server.handlers.oauth.models import OAuthUserInfo

        store = FakeUserStore()
        user = store.create_user(email="flow@example.com", name="Flow User")
        store.link_oauth_provider(user.id, "google", "gid", "flow@example.com")

        handler = OAuthHandler(ctx={"user_store": store})
        mock_session_manager = MagicMock()

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="gid",
            email="flow@example.com",
            name="Flow User",
            email_verified=True,
        )

        fake_tokens = FakeTokenPair()

        with (
            patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=fake_tokens,
            ),
            patch(
                "aragora.billing.auth.sessions.get_session_manager",
                return_value=mock_session_manager,
            ),
            patch("aragora.server.handlers._oauth.base._impl") as mock_impl,
        ):
            mock_impl()._get_oauth_success_url.return_value = "/auth/callback"

            result = handler._complete_oauth_flow(user_info, {"redirect_url": "/auth/callback"})

        assert result.status_code == 302
        mock_session_manager.create_session.assert_called_once()


# ============================================================================
# OAuth State Validation Tests
# ============================================================================


class TestOAuthStateValidation:
    """Test OAuth state generation and validation."""

    def test_generate_and_validate_state(self):
        """Generated state can be validated."""
        from aragora.server.oauth_state_store import (
            generate_oauth_state,
            validate_oauth_state,
        )

        state = generate_oauth_state(user_id="user-1", redirect_url="/debates")
        result = validate_oauth_state(state)

        assert result is not None
        assert result.get("user_id") == "user-1"
        assert result.get("redirect_url") == "/debates"

    def test_state_consumed_on_validation(self):
        """State can only be validated once (consumed)."""
        from aragora.server.oauth_state_store import (
            generate_oauth_state,
            validate_oauth_state,
        )

        state = generate_oauth_state(user_id="user-2")
        result1 = validate_oauth_state(state)
        assert result1 is not None

        result2 = validate_oauth_state(state)
        assert result2 is None

    def test_invalid_state_returns_none(self):
        """Invalid state token returns None."""
        from aragora.server.oauth_state_store import validate_oauth_state

        result = validate_oauth_state("totally-bogus-state-token")
        assert result is None

    def test_empty_state_returns_none(self):
        """Empty state token returns None."""
        from aragora.server.oauth_state_store import validate_oauth_state

        result = validate_oauth_state("")
        assert result is None
