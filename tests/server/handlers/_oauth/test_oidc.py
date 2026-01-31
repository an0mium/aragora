"""
Tests for OIDC OAuth Mixin (Generic OpenID Connect).

Tests cover:
- Authorization URL generation with OIDC discovery
- OAuth callback handling
- Token exchange with OIDC provider
- User info retrieval from userinfo endpoint and ID token fallback
- State validation and CSRF protection
- Error handling for invalid tokens and missing parameters

SECURITY CRITICAL: These tests ensure OIDC authentication is secure.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers._oauth.oidc import OIDCOAuthMixin
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


class OIDCOAuthTestHandler(OIDCOAuthMixin):
    """Test handler that mixes in OIDCOAuthMixin."""

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
def oidc_handler(mock_user_store):
    """Create an OIDC OAuth test handler."""
    return OIDCOAuthTestHandler(mock_user_store)


@pytest.fixture
def mock_request_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Host": "localhost:8080", "Authorization": "Bearer test_token"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def mock_oidc_discovery():
    """Create mock OIDC discovery document."""
    return {
        "issuer": "https://example.com",
        "authorization_endpoint": "https://example.com/authorize",
        "token_endpoint": "https://example.com/token",
        "userinfo_endpoint": "https://example.com/userinfo",
        "jwks_uri": "https://example.com/.well-known/jwks.json",
        "scopes_supported": ["openid", "email", "profile"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
    }


# ===========================================================================
# OIDC Discovery Tests
# ===========================================================================


class TestOIDCDiscovery:
    """Tests for OIDC discovery document fetching."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_discovery_success(
        self, mock_async_client, oidc_handler, mock_oidc_discovery
    ):
        """Test successful OIDC discovery document fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_oidc_discovery

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await oidc_handler._get_oidc_discovery("https://example.com")

        assert result["authorization_endpoint"] == "https://example.com/authorize"
        assert result["token_endpoint"] == "https://example.com/token"
        assert result["userinfo_endpoint"] == "https://example.com/userinfo"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_discovery_failure(self, mock_async_client, oidc_handler):
        """Test OIDC discovery document fetch failure returns empty dict."""
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await oidc_handler._get_oidc_discovery("https://example.com")

        assert result == {}


# ===========================================================================
# Authorization URL Tests
# ===========================================================================


class TestOIDCAuthStart:
    """Tests for OIDC OAuth authorization initiation."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    async def test_auth_start_generates_authorization_url(
        self,
        mock_extract,
        mock_issuer,
        mock_client_id,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        oidc_handler,
        mock_request_handler,
        mock_oidc_discovery,
    ):
        """Test auth start generates correct OIDC authorization URL."""
        mock_issuer.return_value = "https://example.com"
        mock_client_id.return_value = "oidc-client-id"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state-token"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        # Mock discovery
        oidc_handler._get_oidc_discovery = AsyncMock(return_value=mock_oidc_discovery)

        result = await oidc_handler._handle_oidc_auth_start(mock_request_handler, {})

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "example.com/authorize" in location
        assert "client_id=oidc-client-id" in location
        assert "response_type=code" in location
        assert "state=test-state-token" in location
        assert "scope=" in location

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    async def test_auth_start_not_configured_no_issuer(
        self, mock_issuer, mock_client_id, oidc_handler, mock_request_handler
    ):
        """Test auth start returns 503 when issuer not configured."""
        mock_issuer.return_value = None
        mock_client_id.return_value = "oidc-client-id"

        result = await oidc_handler._handle_oidc_auth_start(mock_request_handler, {})

        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    async def test_auth_start_not_configured_no_client_id(
        self, mock_issuer, mock_client_id, oidc_handler, mock_request_handler
    ):
        """Test auth start returns 503 when client_id not configured."""
        mock_issuer.return_value = "https://example.com"
        mock_client_id.return_value = None

        result = await oidc_handler._handle_oidc_auth_start(mock_request_handler, {})

        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    async def test_auth_start_invalid_redirect_url(
        self,
        mock_issuer,
        mock_client_id,
        mock_success_url,
        mock_validate_redirect,
        oidc_handler,
        mock_request_handler,
    ):
        """Test auth start rejects invalid redirect URL."""
        mock_issuer.return_value = "https://example.com"
        mock_client_id.return_value = "oidc-client-id"
        mock_success_url.return_value = "https://example.com/success"
        mock_validate_redirect.return_value = False

        result = await oidc_handler._handle_oidc_auth_start(
            mock_request_handler, {"redirect_url": "https://evil.com"}
        )

        assert result.status_code == 400
        assert b"Invalid redirect URL" in result.body

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    async def test_auth_start_discovery_failed(
        self,
        mock_extract,
        mock_issuer,
        mock_client_id,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        oidc_handler,
        mock_request_handler,
    ):
        """Test auth start returns 503 when discovery fails."""
        mock_issuer.return_value = "https://example.com"
        mock_client_id.return_value = "oidc-client-id"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        # Mock discovery failure (no authorization_endpoint)
        oidc_handler._get_oidc_discovery = AsyncMock(return_value={})

        result = await oidc_handler._handle_oidc_auth_start(mock_request_handler, {})

        assert result.status_code == 503


# ===========================================================================
# Callback Tests
# ===========================================================================


class TestOIDCCallback:
    """Tests for OIDC OAuth callback handling."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_missing_state(
        self, mock_validate_state, oidc_handler, mock_request_handler
    ):
        """Test callback fails without state parameter."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "auth-code"}
        )

        assert result.status_code == 302
        assert "Missing state" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_invalid_state(
        self, mock_validate_state, oidc_handler, mock_request_handler
    ):
        """Test callback fails with invalid state."""
        mock_validate_state.return_value = None

        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "auth-code", "state": "invalid-state"}
        )

        assert result.status_code == 302
        assert "Invalid or expired state" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    async def test_callback_missing_code(
        self, mock_issuer, mock_validate_state, oidc_handler, mock_request_handler
    ):
        """Test callback fails without authorization code."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}
        mock_issuer.return_value = "https://example.com"

        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Missing authorization code" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    async def test_callback_with_oauth_error(self, oidc_handler, mock_request_handler):
        """Test callback handles OAuth error from provider."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler,
            {
                "error": "access_denied",
                "error_description": "User cancelled the request",
            },
        )

        assert result.status_code == 302
        assert "OIDC error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_success(
        self,
        mock_validate_state,
        mock_issuer,
        oidc_handler,
        mock_request_handler,
        mock_oidc_discovery,
    ):
        """Test successful callback completes OAuth flow."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com/success"}
        mock_issuer.return_value = "https://example.com"

        # Mock discovery
        oidc_handler._get_oidc_discovery = AsyncMock(return_value=mock_oidc_discovery)

        # Mock token exchange
        async def mock_exchange(code, discovery):
            return {"access_token": "oidc_access_token", "id_token": "mock_id_token"}

        oidc_handler._exchange_oidc_code = mock_exchange

        # Mock user info
        async def mock_user_info(access_token, id_token, discovery):
            return OAuthUserInfo(
                provider="oidc",
                provider_user_id="oidc-user-id",
                email="user@example.com",
                name="Test User",
                picture=None,
                email_verified=True,
            )

        oidc_handler._get_oidc_user_info = mock_user_info

        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "auth-code", "state": "valid-state"}
        )

        # Should complete flow
        assert result.status_code == 302
        assert "access_token=" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_token_exchange_failure(
        self,
        mock_validate_state,
        mock_issuer,
        oidc_handler,
        mock_request_handler,
        mock_oidc_discovery,
    ):
        """Test callback handles token exchange failure."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}
        mock_issuer.return_value = "https://example.com"

        oidc_handler._get_oidc_discovery = AsyncMock(return_value=mock_oidc_discovery)

        async def mock_exchange_failure(code, discovery):
            raise Exception("Token exchange failed")

        oidc_handler._exchange_oidc_code = mock_exchange_failure

        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "bad-code", "state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Failed to exchange authorization code" in result.headers.get("Location", "")


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestOIDCTokenExchange:
    """Tests for OIDC token exchange."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_secret")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    async def test_exchange_code_success(
        self,
        mock_client_id,
        mock_client_secret,
        mock_redirect_uri,
        mock_async_client,
        oidc_handler,
        mock_oidc_discovery,
    ):
        """Test successful token exchange."""
        mock_client_id.return_value = "oidc-client-id"
        mock_client_secret.return_value = "oidc-client-secret"
        mock_redirect_uri.return_value = "https://example.com/callback"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "oidc-access-token",
            "id_token": "eyJ.test.id_token",
            "refresh_token": "oidc-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await oidc_handler._exchange_oidc_code("test-auth-code", mock_oidc_discovery)

        assert result["access_token"] == "oidc-access-token"
        assert result["id_token"] == "eyJ.test.id_token"
        assert result["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_exchange_code_no_token_endpoint(self, oidc_handler):
        """Test token exchange fails when no token endpoint in discovery."""
        discovery = {}  # No token_endpoint

        with pytest.raises(ValueError, match="No token endpoint"):
            await oidc_handler._exchange_oidc_code("test-code", discovery)


# ===========================================================================
# User Info Tests
# ===========================================================================


class TestOIDCUserInfo:
    """Tests for OIDC user info retrieval."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_user_info_from_userinfo_endpoint(
        self, mock_async_client, oidc_handler, mock_oidc_discovery
    ):
        """Test user info retrieval from userinfo endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sub": "oidc-user-123",
            "email": "user@example.com",
            "email_verified": True,
            "name": "Test User",
            "picture": "https://example.com/photo.jpg",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await oidc_handler._get_oidc_user_info(
            "access-token", "id-token", mock_oidc_discovery
        )

        assert result.provider == "oidc"
        assert result.provider_user_id == "oidc-user-123"
        assert result.email == "user@example.com"
        assert result.name == "Test User"
        assert result.email_verified is True

    @pytest.mark.asyncio
    async def test_get_user_info_from_id_token_fallback(self, oidc_handler):
        """Test user info falls back to ID token when userinfo fails."""
        # Create a mock ID token payload
        payload = {
            "sub": "token-user-456",
            "email": "fallback@example.com",
            "email_verified": True,
            "name": "Fallback User",
        }
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")

        # Format: header.payload.signature
        id_token = f"eyJhbGciOiJSUzI1NiJ9.{payload_b64}.signature"

        # Discovery without userinfo endpoint
        discovery = {"token_endpoint": "https://example.com/token"}

        result = await oidc_handler._get_oidc_user_info(None, id_token, discovery)

        assert result.provider_user_id == "token-user-456"
        assert result.email == "fallback@example.com"
        assert result.name == "Fallback User"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_user_info_userinfo_fails_uses_id_token(
        self, mock_async_client, oidc_handler, mock_oidc_discovery
    ):
        """Test user info falls back to ID token when userinfo endpoint fails."""
        # Mock userinfo endpoint failure
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Userinfo failed")
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        # Create a mock ID token
        payload = {
            "sub": "id-token-user",
            "email": "id-token@example.com",
            "name": "ID Token User",
        }
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
        id_token = f"header.{payload_b64}.signature"

        result = await oidc_handler._get_oidc_user_info(
            "access-token", id_token, mock_oidc_discovery
        )

        assert result.provider_user_id == "id-token-user"
        assert result.email == "id-token@example.com"

    @pytest.mark.asyncio
    async def test_get_user_info_no_email_raises(self, oidc_handler):
        """Test user info raises when no email in response."""
        # ID token without email
        payload = {"sub": "user-id", "name": "No Email User"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
        id_token = f"header.{payload_b64}.signature"

        discovery = {}

        with pytest.raises(ValueError, match="No email"):
            await oidc_handler._get_oidc_user_info(None, id_token, discovery)

    @pytest.mark.asyncio
    async def test_get_user_info_no_subject_raises(self, oidc_handler):
        """Test user info raises when no subject in response."""
        # ID token without sub
        payload = {"email": "user@example.com", "name": "No Sub User"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
        id_token = f"header.{payload_b64}.signature"

        discovery = {}

        with pytest.raises(ValueError, match="No subject"):
            await oidc_handler._get_oidc_user_info(None, id_token, discovery)

    @pytest.mark.asyncio
    async def test_get_user_info_uses_email_as_name_fallback(self, oidc_handler):
        """Test user info uses email prefix as name when name is missing."""
        payload = {"sub": "user-id", "email": "testuser@example.com"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
        id_token = f"header.{payload_b64}.signature"

        discovery = {}

        result = await oidc_handler._get_oidc_user_info(None, id_token, discovery)

        # Should use email prefix as name
        assert result.name == "testuser"


# ===========================================================================
# CSRF Protection Tests
# ===========================================================================


class TestOIDCCSRFProtection:
    """Tests for CSRF protection in OIDC OAuth."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_state_parameter_required(
        self, mock_validate_state, oidc_handler, mock_request_handler
    ):
        """Test that state parameter is required for callback."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "auth-code"}
        )

        assert result.status_code == 302
        assert "Missing state" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_expired_state_rejected(
        self, mock_validate_state, oidc_handler, mock_request_handler
    ):
        """Test that expired state tokens are rejected."""
        mock_validate_state.return_value = None

        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler, {"code": "auth-code", "state": "expired-state"}
        )

        assert result.status_code == 302
        assert "Invalid or expired" in result.headers.get("Location", "")


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestOIDCErrorHandling:
    """Tests for OIDC OAuth error handling."""

    @pytest.mark.asyncio
    async def test_access_denied_error(self, oidc_handler, mock_request_handler):
        """Test handling of access_denied error."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler,
            {"error": "access_denied", "error_description": "User denied access"},
        )

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "OIDC error" in location or "error" in location

    @pytest.mark.asyncio
    async def test_invalid_scope_error(self, oidc_handler, mock_request_handler):
        """Test handling of invalid_scope error."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler,
            {"error": "invalid_scope", "error_description": "Unknown scope requested"},
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    async def test_server_error(self, oidc_handler, mock_request_handler):
        """Test handling of server_error."""
        result = await oidc_handler._handle_oidc_callback(
            mock_request_handler,
            {"error": "server_error", "error_description": "Internal server error"},
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")
