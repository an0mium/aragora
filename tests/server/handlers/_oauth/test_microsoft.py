"""
Tests for Microsoft OAuth Mixin (Azure AD).

Tests cover:
- Authorization URL generation (single and multi-tenant)
- OAuth callback handling
- Token exchange with Microsoft
- User info retrieval from Microsoft Graph
- State validation and CSRF protection
- Error handling for invalid tokens and missing parameters

SECURITY CRITICAL: These tests ensure Microsoft OAuth authentication is secure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers._oauth.microsoft import MicrosoftOAuthMixin
from aragora.server.handlers.oauth.models import OAuthUserInfo


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user object for testing."""

    id: str
    email: str
    name: str
    org_id: str | None = None
    role: str = "member"
    password_hash: str | None = "hashed"


@dataclass
class MockTokens:
    """Mock token pair for testing."""

    access_token: str
    refresh_token: str
    expires_in: int = 3600


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.oauth_links: dict[str, dict[str, str]] = {}
        self.created_users: list[MockUser] = []

    def get_user_by_email(self, email: str) -> MockUser | None:
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> MockUser | None:
        for user_id, links in self.oauth_links.items():
            if links.get(provider) == provider_user_id:
                return self.users.get(user_id)
        return None

    def create_user(
        self, email: str, password_hash: str, password_salt: str, name: str | None = None
    ) -> MockUser:
        user_id = f"user_{len(self.users) + 1}"
        user = MockUser(
            id=user_id, email=email, name=name or email.split("@")[0], password_hash=password_hash
        )
        self.users[user_id] = user
        self.created_users.append(user)
        return user

    def update_user(self, user_id: str, **kwargs) -> bool:
        return True

    def link_oauth_provider(
        self, user_id: str, provider: str, provider_user_id: str, email: str
    ) -> bool:
        if user_id not in self.oauth_links:
            self.oauth_links[user_id] = {}
        self.oauth_links[user_id][provider] = provider_user_id
        return True


class MicrosoftOAuthTestHandler(MicrosoftOAuthMixin):
    """Test handler that mixes in MicrosoftOAuthMixin."""

    def __init__(self, user_store: MockUserStore | None = None):
        self.user_store = user_store or MockUserStore()
        self.ctx = {"user_store": self.user_store}

    def _get_user_store(self):
        return self.user_store

    def _find_user_by_oauth(self, user_store, user_info):
        return user_store.get_user_by_oauth(user_info.provider, user_info.provider_user_id)

    def _link_oauth_to_user(self, user_store, user_id, user_info):
        return user_store.link_oauth_provider(
            user_id, user_info.provider, user_info.provider_user_id, user_info.email
        )

    def _create_oauth_user(self, user_store, user_info):
        return user_store.create_user(
            email=user_info.email,
            password_hash="hash",
            password_salt="salt",
            name=user_info.name,
        )

    def _complete_oauth_flow(self, user_info, state_data):
        return MagicMock(
            status_code=302,
            headers={"Location": f"{state_data.get('redirect_url', '')}?access_token=test_token"},
        )

    def _redirect_with_error(self, error):
        return MagicMock(status_code=302, headers={"Location": f"?error={error}"})


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    return MockUserStore()


@pytest.fixture
def microsoft_handler(mock_user_store):
    """Create a Microsoft OAuth test handler."""
    return MicrosoftOAuthTestHandler(mock_user_store)


@pytest.fixture
def mock_request_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Host": "localhost:8080", "Authorization": "Bearer test_token"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


# ===========================================================================
# Authorization URL Tests
# ===========================================================================


class TestMicrosoftAuthStart:
    """Tests for Microsoft OAuth authorization initiation."""

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_AUTH_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    )
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_auth_start_generates_authorization_url(
        self,
        mock_extract,
        mock_client_id,
        mock_tenant,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test auth start generates correct Microsoft authorization URL."""
        mock_client_id.return_value = "ms-client-id-guid"
        mock_tenant.return_value = "common"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state-token"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        result = microsoft_handler._handle_microsoft_auth_start(mock_request_handler, {})

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "login.microsoftonline.com" in location
        assert "/common/" in location
        assert "client_id=ms-client-id-guid" in location
        assert "response_type=code" in location
        assert "state=test-state-token" in location
        assert "response_mode=query" in location

    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    def test_auth_start_not_configured(
        self, mock_client_id, microsoft_handler, mock_request_handler
    ):
        """Test auth start returns 503 when not configured."""
        mock_client_id.return_value = None

        result = microsoft_handler._handle_microsoft_auth_start(mock_request_handler, {})

        assert result.status_code == 503

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    def test_auth_start_invalid_redirect_url(
        self,
        mock_client_id,
        mock_success_url,
        mock_validate_redirect,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test auth start rejects invalid redirect URL."""
        mock_client_id.return_value = "ms-client-id"
        mock_success_url.return_value = "https://example.com/success"
        mock_validate_redirect.return_value = False

        result = microsoft_handler._handle_microsoft_auth_start(
            mock_request_handler, {"redirect_url": "https://evil.com"}
        )

        assert result.status_code == 400
        assert b"Invalid redirect URL" in result.body

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_AUTH_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    )
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_auth_start_with_single_tenant(
        self,
        mock_extract,
        mock_client_id,
        mock_tenant,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test auth start uses specific tenant ID."""
        mock_client_id.return_value = "ms-client-id"
        mock_tenant.return_value = "12345678-1234-1234-1234-123456789abc"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        result = microsoft_handler._handle_microsoft_auth_start(mock_request_handler, {})

        location = result.headers.get("Location", "")
        assert "12345678-1234-1234-1234-123456789abc" in location
        assert "/common/" not in location


# ===========================================================================
# Callback Tests
# ===========================================================================


class TestMicrosoftCallback:
    """Tests for Microsoft OAuth callback handling."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_missing_state(
        self, mock_validate_state, microsoft_handler, mock_request_handler
    ):
        """Test callback fails without state parameter."""
        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"code": "auth-code"}
        )

        assert result.status_code == 302
        assert "Missing state" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_invalid_state(
        self, mock_validate_state, microsoft_handler, mock_request_handler
    ):
        """Test callback fails with invalid state."""
        mock_validate_state.return_value = None

        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"code": "auth-code", "state": "invalid-state"}
        )

        assert result.status_code == 302
        assert "Invalid or expired state" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_missing_code(
        self, mock_validate_state, microsoft_handler, mock_request_handler
    ):
        """Test callback fails without authorization code."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Missing authorization code" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    async def test_callback_with_oauth_error(self, microsoft_handler, mock_request_handler):
        """Test callback handles OAuth error from Microsoft."""
        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler,
            {
                "error": "access_denied",
                "error_description": "AADSTS65004: User declined consent.",
            },
        )

        assert result.status_code == 302
        assert "OAuth error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_success(
        self,
        mock_validate_state,
        mock_success_url,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test successful callback completes OAuth flow."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com/success"}
        mock_success_url.return_value = "https://example.com/success"

        async def mock_exchange(code):
            return {"access_token": "ms_access_token"}

        async def mock_user_info(token):
            return OAuthUserInfo(
                provider="microsoft",
                provider_user_id="ms-user-id-guid",
                email="user@example.onmicrosoft.com",
                name="Test User",
                picture=None,
                email_verified=True,
            )

        microsoft_handler._exchange_microsoft_code = mock_exchange
        microsoft_handler._get_microsoft_user_info = mock_user_info

        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"code": "auth-code", "state": "valid-state"}
        )

        # Should complete flow
        assert result.status_code == 302
        assert "access_token=" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_token_exchange_failure(
        self, mock_validate_state, microsoft_handler, mock_request_handler
    ):
        """Test callback handles token exchange failure."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        async def mock_exchange_failure(code):
            raise Exception("Token exchange failed")

        microsoft_handler._exchange_microsoft_code = mock_exchange_failure

        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"code": "bad-code", "state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Failed to exchange authorization code" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_no_access_token(
        self, mock_validate_state, microsoft_handler, mock_request_handler
    ):
        """Test callback handles missing access token in response."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        async def mock_exchange_no_token(code):
            return {"error": "invalid_grant"}

        microsoft_handler._exchange_microsoft_code = mock_exchange_no_token

        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler, {"code": "expired-code", "state": "valid-state"}
        )

        assert result.status_code == 302
        assert "No access token received" in result.headers.get("Location", "")


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestMicrosoftTokenExchange:
    """Tests for Microsoft token exchange."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_secret")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_TOKEN_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
    )
    async def test_exchange_code_success(
        self,
        mock_tenant,
        mock_client_id,
        mock_client_secret,
        mock_redirect_uri,
        mock_async_client,
        microsoft_handler,
    ):
        """Test successful token exchange."""
        mock_tenant.return_value = "common"
        mock_client_id.return_value = "ms-client-id"
        mock_client_secret.return_value = "ms-client-secret"
        mock_redirect_uri.return_value = "https://example.com/callback"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test",
            "refresh_token": "M.R3_BAY.test-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await microsoft_handler._exchange_microsoft_code("test-auth-code")

        assert "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9" in result["access_token"]
        assert result["token_type"] == "Bearer"


# ===========================================================================
# User Info Tests
# ===========================================================================


class TestMicrosoftUserInfo:
    """Tests for Microsoft Graph user info retrieval."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_USERINFO_URL",
        "https://graph.microsoft.com/v1.0/me",
    )
    async def test_get_user_info_with_mail(self, mock_async_client, microsoft_handler):
        """Test user info retrieval with mail field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "abc12345-def6-7890-ghij-klmnopqrstuv",
            "displayName": "Test User",
            "givenName": "Test",
            "surname": "User",
            "mail": "user@example.com",
            "userPrincipalName": "user@example.onmicrosoft.com",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await microsoft_handler._get_microsoft_user_info("test-access-token")

        assert result.provider == "microsoft"
        assert result.provider_user_id == "abc12345-def6-7890-ghij-klmnopqrstuv"
        assert result.email == "user@example.com"
        assert result.name == "Test User"
        assert result.email_verified is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_USERINFO_URL",
        "https://graph.microsoft.com/v1.0/me",
    )
    async def test_get_user_info_falls_back_to_upn(self, mock_async_client, microsoft_handler):
        """Test user info falls back to userPrincipalName when mail is null."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "user-guid",
            "displayName": "Test User",
            "mail": None,  # No mail
            "userPrincipalName": "user@example.onmicrosoft.com",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await microsoft_handler._get_microsoft_user_info("test-access-token")

        # Should fall back to userPrincipalName
        assert result.email == "user@example.onmicrosoft.com"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_USERINFO_URL",
        "https://graph.microsoft.com/v1.0/me",
    )
    async def test_get_user_info_no_email_raises(self, mock_async_client, microsoft_handler):
        """Test user info raises when no valid email can be found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "user-guid",
            "displayName": "Test User",
            "mail": None,
            "userPrincipalName": "invalid-no-at-symbol",  # No @ symbol
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        with pytest.raises(ValueError, match="Could not retrieve email"):
            await microsoft_handler._get_microsoft_user_info("test-access-token")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_USERINFO_URL",
        "https://graph.microsoft.com/v1.0/me",
    )
    async def test_get_user_info_uses_email_as_name_fallback(
        self, mock_async_client, microsoft_handler
    ):
        """Test user info uses email prefix as name when displayName is missing."""
        mock_response = MagicMock()
        # When displayName key is missing (not just None), the default is used
        mock_response.json.return_value = {
            "id": "user-guid",
            # No displayName key - .get() will use the default
            "mail": "user@example.com",
            "userPrincipalName": "user@example.onmicrosoft.com",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await microsoft_handler._get_microsoft_user_info("test-access-token")

        # Should use email prefix as name when displayName key is missing
        assert result.name == "user"


# ===========================================================================
# Tenant Configuration Tests
# ===========================================================================


class TestMicrosoftTenantConfiguration:
    """Tests for Microsoft OAuth tenant configuration."""

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_AUTH_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    )
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_organizations_tenant(
        self,
        mock_extract,
        mock_client_id,
        mock_tenant,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test organizations tenant for work accounts only."""
        mock_client_id.return_value = "ms-client-id"
        mock_tenant.return_value = "organizations"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        result = microsoft_handler._handle_microsoft_auth_start(mock_request_handler, {})

        location = result.headers.get("Location", "")
        assert "/organizations/" in location

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch(
        "aragora.server.handlers._oauth_impl.MICROSOFT_AUTH_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    )
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consumers_tenant(
        self,
        mock_extract,
        mock_client_id,
        mock_tenant,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        microsoft_handler,
        mock_request_handler,
    ):
        """Test consumers tenant for personal accounts only."""
        mock_client_id.return_value = "ms-client-id"
        mock_tenant.return_value = "consumers"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        result = microsoft_handler._handle_microsoft_auth_start(mock_request_handler, {})

        location = result.headers.get("Location", "")
        assert "/consumers/" in location


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestMicrosoftErrorHandling:
    """Tests for Microsoft OAuth error handling."""

    @pytest.mark.asyncio
    async def test_aadsts_access_denied_error(self, microsoft_handler, mock_request_handler):
        """Test handling of AADSTS access denied error."""
        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler,
            {
                "error": "access_denied",
                "error_description": "AADSTS65004: The user declined to consent.",
            },
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    async def test_invalid_grant_error(self, microsoft_handler, mock_request_handler):
        """Test handling of invalid_grant error."""
        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler,
            {
                "error": "invalid_grant",
                "error_description": "AADSTS70000: The provided authorization code is invalid.",
            },
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    async def test_tenant_not_found_error(self, microsoft_handler, mock_request_handler):
        """Test handling of tenant not found error."""
        result = await microsoft_handler._handle_microsoft_callback(
            mock_request_handler,
            {
                "error": "tenant_not_found",
                "error_description": "AADSTS90002: Tenant 'invalid' not found.",
            },
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")
