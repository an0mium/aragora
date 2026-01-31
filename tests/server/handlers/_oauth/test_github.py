"""
Tests for GitHub OAuth Mixin.

Tests cover:
- Authorization URL generation
- OAuth callback handling
- Token exchange with GitHub
- User info retrieval from GitHub (including email fallback)
- State validation and CSRF protection
- Error handling for invalid tokens and missing parameters

SECURITY CRITICAL: These tests ensure GitHub OAuth authentication is secure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers._oauth.github import GitHubOAuthMixin
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


class GitHubOAuthTestHandler(GitHubOAuthMixin):
    """Test handler that mixes in GitHubOAuthMixin."""

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

    def _handle_account_linking(self, user_store, linking_user_id, user_info, state_data):
        return MagicMock(status_code=302, headers={"Location": f"?linked={user_info.provider}"})

    def _redirect_with_tokens(self, redirect_url, tokens):
        return MagicMock(
            status_code=302,
            headers={"Location": f"{redirect_url}?access_token={tokens.access_token}"},
        )

    def _redirect_with_error(self, error):
        return MagicMock(status_code=302, headers={"Location": f"?error={error}"})


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    return MockUserStore()


@pytest.fixture
def github_handler(mock_user_store):
    """Create a GitHub OAuth test handler."""
    return GitHubOAuthTestHandler(mock_user_store)


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


class TestGitHubAuthStart:
    """Tests for GitHub OAuth authorization initiation."""

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_github_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_AUTH_URL",
        "https://github.com/login/oauth/authorize",
    )
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_auth_start_generates_authorization_url(
        self,
        mock_extract,
        mock_client_id,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        github_handler,
        mock_request_handler,
    ):
        """Test auth start generates correct GitHub authorization URL."""
        mock_client_id.return_value = "gh-client-id-12345"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state-token"
        mock_validate_redirect.return_value = True
        mock_extract.return_value = MagicMock(is_authenticated=False)

        result = github_handler._handle_github_auth_start(mock_request_handler, {})

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "github.com/login/oauth/authorize" in location
        assert "client_id=gh-client-id-12345" in location
        assert "scope=read%3Auser+user%3Aemail" in location or "scope=read:user" in location
        assert "state=test-state-token" in location

    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    def test_auth_start_not_configured(self, mock_client_id, github_handler, mock_request_handler):
        """Test auth start returns 503 when not configured."""
        mock_client_id.return_value = None

        result = github_handler._handle_github_auth_start(mock_request_handler, {})

        assert result.status_code == 503

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    def test_auth_start_invalid_redirect_url(
        self,
        mock_client_id,
        mock_success_url,
        mock_validate_redirect,
        github_handler,
        mock_request_handler,
    ):
        """Test auth start rejects invalid redirect URL."""
        mock_client_id.return_value = "gh-client-id"
        mock_success_url.return_value = "https://example.com/success"
        mock_validate_redirect.return_value = False

        result = github_handler._handle_github_auth_start(
            mock_request_handler, {"redirect_url": "https://evil.com"}
        )

        assert result.status_code == 400
        assert b"Invalid redirect URL" in result.body


# ===========================================================================
# Callback Tests
# ===========================================================================


class TestGitHubCallback:
    """Tests for GitHub OAuth callback handling."""

    @patch("aragora.server.handlers._oauth_impl._validate_state")
    def test_callback_missing_state(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test callback fails without state parameter."""
        result = github_handler._handle_github_callback(mock_request_handler, {"code": "auth-code"})

        assert result.status_code == 302
        assert "Missing state" in result.headers.get("Location", "")

    @patch("aragora.server.handlers._oauth_impl._validate_state")
    def test_callback_invalid_state(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test callback fails with invalid state."""
        mock_validate_state.return_value = None

        result = github_handler._handle_github_callback(
            mock_request_handler, {"code": "auth-code", "state": "invalid-state"}
        )

        assert result.status_code == 302
        assert "Invalid or expired state" in result.headers.get("Location", "")

    @patch("aragora.server.handlers._oauth_impl._validate_state")
    def test_callback_missing_code(self, mock_validate_state, github_handler, mock_request_handler):
        """Test callback fails without authorization code."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        result = github_handler._handle_github_callback(
            mock_request_handler, {"state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Missing authorization code" in result.headers.get("Location", "")

    def test_callback_with_oauth_error(self, github_handler, mock_request_handler):
        """Test callback handles OAuth error from GitHub."""
        result = github_handler._handle_github_callback(
            mock_request_handler,
            {
                "error": "access_denied",
                "error_description": "The user has denied your application access.",
            },
        )

        assert result.status_code == 302
        assert "OAuth error" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    async def test_callback_success_creates_user(
        self,
        mock_create_tokens,
        mock_validate_state,
        mock_success_url,
        github_handler,
        mock_user_store,
        mock_request_handler,
    ):
        """Test successful callback creates new user."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com/success"}
        mock_success_url.return_value = "https://example.com/success"
        mock_create_tokens.return_value = MockTokens(
            access_token="access_123", refresh_token="refresh_456"
        )

        async def mock_exchange(code):
            return {"access_token": "gho_github_access_token"}

        async def mock_user_info(token):
            return OAuthUserInfo(
                provider="github",
                provider_user_id="12345678",
                email="developer@github.com",
                name="Developer",
                picture="https://avatars.githubusercontent.com/u/12345678",
                email_verified=True,
            )

        github_handler._exchange_github_code = mock_exchange
        github_handler._get_github_user_info = mock_user_info

        result = await github_handler._handle_github_callback(
            mock_request_handler, {"code": "auth-code", "state": "valid-state"}
        )

        # Verify user was created
        assert len(mock_user_store.created_users) == 1
        assert mock_user_store.created_users[0].email == "developer@github.com"

        # Verify redirect with tokens
        assert result.status_code == 302
        assert "access_token=" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_token_exchange_failure(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test callback handles token exchange failure."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        async def mock_exchange_failure(code):
            raise Exception("Token exchange failed")

        github_handler._exchange_github_code = mock_exchange_failure

        result = await github_handler._handle_github_callback(
            mock_request_handler, {"code": "bad-code", "state": "valid-state"}
        )

        assert result.status_code == 302
        assert "Failed to exchange authorization code" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._validate_state")
    async def test_callback_error_in_token_response(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test callback handles error in token response."""
        mock_validate_state.return_value = {"redirect_url": "https://example.com"}

        async def mock_exchange_error(code):
            return {
                "error": "bad_verification_code",
                "error_description": "The code passed is incorrect or expired.",
            }

        github_handler._exchange_github_code = mock_exchange_error

        result = await github_handler._handle_github_callback(
            mock_request_handler, {"code": "expired-code", "state": "valid-state"}
        )

        assert result.status_code == 302
        assert "No access token received" in result.headers.get("Location", "")


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestGitHubTokenExchange:
    """Tests for GitHub token exchange."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl._get_github_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_secret")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_CLIENT_ID", "gh-client-id")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_TOKEN_URL",
        "https://github.com/login/oauth/access_token",
    )
    async def test_exchange_code_success(
        self,
        mock_client_secret,
        mock_redirect_uri,
        mock_async_client,
        github_handler,
    ):
        """Test successful token exchange."""
        mock_client_secret.return_value = "gh-client-secret"
        mock_redirect_uri.return_value = "https://example.com/callback"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "gho_test-access-token",
            "token_type": "bearer",
            "scope": "read:user,user:email",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await github_handler._exchange_github_code("test-auth-code")

        assert result["access_token"] == "gho_test-access-token"
        assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl._get_github_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_secret")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_CLIENT_ID", "gh-client-id")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_TOKEN_URL",
        "https://github.com/login/oauth/access_token",
    )
    async def test_exchange_code_invalid_json(
        self,
        mock_client_secret,
        mock_redirect_uri,
        mock_async_client,
        github_handler,
    ):
        """Test token exchange handles invalid JSON response."""
        mock_client_secret.return_value = "gh-client-secret"
        mock_redirect_uri.return_value = "https://example.com/callback"

        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        with pytest.raises(ValueError, match="Invalid JSON"):
            await github_handler._exchange_github_code("test-auth-code")


# ===========================================================================
# User Info Tests
# ===========================================================================


class TestGitHubUserInfo:
    """Tests for GitHub user info retrieval."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_USERINFO_URL", "https://api.github.com/user")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_EMAILS_URL",
        "https://api.github.com/user/emails",
    )
    async def test_get_user_info_with_public_email(self, mock_async_client, github_handler):
        """Test user info retrieval when email is public."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": 12345678,
            "login": "testuser",
            "name": "Test User",
            "email": "public@example.com",
            "avatar_url": "https://avatars.githubusercontent.com/u/12345678",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await github_handler._get_github_user_info("test-access-token")

        assert result.provider == "github"
        assert result.provider_user_id == "12345678"
        assert result.email == "public@example.com"
        assert result.name == "Test User"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_USERINFO_URL", "https://api.github.com/user")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_EMAILS_URL",
        "https://api.github.com/user/emails",
    )
    async def test_get_user_info_fetches_private_email(self, mock_async_client, github_handler):
        """Test user info fetches email from emails endpoint when not public."""
        user_response = MagicMock()
        user_response.json.return_value = {
            "id": 12345678,
            "login": "privateuser",
            "name": "Private User",
            "email": None,  # Email is private
            "avatar_url": "https://avatars.githubusercontent.com/u/12345678",
        }

        emails_response = MagicMock()
        emails_response.json.return_value = [
            {"email": "secondary@example.com", "primary": False, "verified": True},
            {"email": "primary@example.com", "primary": True, "verified": True},
        ]

        mock_client_instance = AsyncMock()
        # First call returns user info, second call returns emails
        mock_client_instance.get.side_effect = [user_response, emails_response]
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await github_handler._get_github_user_info("test-access-token")

        # Should use primary verified email
        assert result.email == "primary@example.com"
        assert result.email_verified is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_USERINFO_URL", "https://api.github.com/user")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_EMAILS_URL",
        "https://api.github.com/user/emails",
    )
    async def test_get_user_info_falls_back_to_verified_email(
        self, mock_async_client, github_handler
    ):
        """Test user info falls back to any verified email when no primary."""
        user_response = MagicMock()
        user_response.json.return_value = {
            "id": 12345678,
            "login": "testuser",
            "name": None,  # No name
            "email": None,
        }

        emails_response = MagicMock()
        emails_response.json.return_value = [
            {"email": "unverified@example.com", "primary": False, "verified": False},
            {"email": "verified@example.com", "primary": False, "verified": True},
        ]

        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = [user_response, emails_response]
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        result = await github_handler._get_github_user_info("test-access-token")

        # Should use the verified email
        assert result.email == "verified@example.com"
        # Should use login as name since name is null
        assert result.name == "testuser"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_USERINFO_URL", "https://api.github.com/user")
    @patch(
        "aragora.server.handlers._oauth_impl.GITHUB_EMAILS_URL",
        "https://api.github.com/user/emails",
    )
    async def test_get_user_info_no_email_raises(self, mock_async_client, github_handler):
        """Test user info raises when no email can be retrieved."""
        user_response = MagicMock()
        user_response.json.return_value = {
            "id": 12345678,
            "login": "noemail",
            "name": "No Email User",
            "email": None,
        }

        emails_response = MagicMock()
        emails_response.json.return_value = []  # No emails

        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = [user_response, emails_response]
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        with pytest.raises(ValueError, match="Could not retrieve email"):
            await github_handler._get_github_user_info("test-access-token")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_USERINFO_URL", "https://api.github.com/user")
    async def test_get_user_info_invalid_json(self, mock_async_client, github_handler):
        """Test user info handles invalid JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_async_client.return_value = mock_client_instance

        with pytest.raises(ValueError, match="Invalid JSON"):
            await github_handler._get_github_user_info("test-access-token")


# ===========================================================================
# CSRF Protection Tests
# ===========================================================================


class TestGitHubCSRFProtection:
    """Tests for CSRF protection in GitHub OAuth."""

    @patch("aragora.server.handlers._oauth_impl._validate_state")
    def test_state_parameter_required(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test that state parameter is required for callback."""
        result = github_handler._handle_github_callback(mock_request_handler, {"code": "auth-code"})

        assert result.status_code == 302
        assert "Missing state" in result.headers.get("Location", "")

    @patch("aragora.server.handlers._oauth_impl._validate_state")
    def test_expired_state_rejected(
        self, mock_validate_state, github_handler, mock_request_handler
    ):
        """Test that expired state tokens are rejected."""
        mock_validate_state.return_value = None

        result = github_handler._handle_github_callback(
            mock_request_handler, {"code": "auth-code", "state": "expired-state"}
        )

        assert result.status_code == 302
        assert "Invalid or expired" in result.headers.get("Location", "")


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestGitHubErrorHandling:
    """Tests for GitHub OAuth error handling."""

    def test_access_denied_error(self, github_handler, mock_request_handler):
        """Test handling of access_denied error."""
        result = github_handler._handle_github_callback(
            mock_request_handler,
            {"error": "access_denied", "error_description": "User denied access"},
        )

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "OAuth error" in location or "error" in location

    def test_application_suspended_error(self, github_handler, mock_request_handler):
        """Test handling of application_suspended error."""
        result = github_handler._handle_github_callback(
            mock_request_handler,
            {
                "error": "application_suspended",
                "error_description": "Your application has been suspended.",
            },
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")

    def test_redirect_uri_mismatch_error(self, github_handler, mock_request_handler):
        """Test handling of redirect_uri_mismatch error."""
        result = github_handler._handle_github_callback(
            mock_request_handler,
            {
                "error": "redirect_uri_mismatch",
                "error_description": "The redirect_uri does not match.",
            },
        )

        assert result.status_code == 302
        assert "error" in result.headers.get("Location", "")
