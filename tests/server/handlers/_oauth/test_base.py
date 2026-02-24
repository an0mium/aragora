"""
Tests for OAuth Handler Base Class.

Tests cover:
- Route handling and matching
- OAuth flow completion
- User creation and linking
- Token generation and redirect handling
- RBAC permission checks
- Rate limiting
- Error handling

SECURITY CRITICAL: These tests ensure OAuth authentication flows are secure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers._oauth.base import OAuthHandler
from aragora.server.handlers.oauth.models import OAuthUserInfo
from aragora.server.handlers.base import HandlerResult


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

    def __init__(self, users: dict[str, MockUser] | None = None):
        self.users = users or {}
        self.oauth_links: dict[str, dict[str, str]] = {}  # user_id -> {provider: provider_user_id}
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
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        name: str | None = None,
    ) -> MockUser:
        user_id = f"user_{len(self.users) + 1}"
        user = MockUser(
            id=user_id,
            email=email,
            name=name or email.split("@")[0],
            password_hash=password_hash,
        )
        self.users[user_id] = user
        self.created_users.append(user)
        return user

    def update_user(self, user_id: str, **kwargs) -> bool:
        user = self.users.get(user_id)
        if user:
            for key, value in kwargs.items():
                setattr(user, key, value)
            return True
        return False

    def link_oauth_provider(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: str,
    ) -> bool:
        if user_id not in self.oauth_links:
            self.oauth_links[user_id] = {}
        self.oauth_links[user_id][provider] = provider_user_id
        return True

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        if user_id in self.oauth_links and provider in self.oauth_links[user_id]:
            del self.oauth_links[user_id][provider]
            return True
        return False

    def get_oauth_providers(self, user_id: str) -> list[dict]:
        links = self.oauth_links.get(user_id, {})
        return [{"provider": p, "provider_user_id": uid} for p, uid in links.items()]


@pytest.fixture
def mock_user_store():
    """Create a mock user store with a test user."""
    store = MockUserStore()
    store.users["user_1"] = MockUser(
        id="user_1",
        email="existing@example.com",
        name="Existing User",
        org_id="org_1",
    )
    return store


@pytest.fixture
def oauth_handler(mock_user_store):
    """Create an OAuth handler with mock context."""
    ctx = {"user_store": mock_user_store}
    handler = OAuthHandler(ctx=ctx)
    return handler


@pytest.fixture
def mock_request_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Host": "localhost:8080"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def sample_user_info():
    """Create sample OAuth user info."""
    return OAuthUserInfo(
        provider="google",
        provider_user_id="google_123456",
        email="test@example.com",
        name="Test User",
        picture="https://example.com/photo.jpg",
        email_verified=True,
    )


# ===========================================================================
# Route Handling Tests
# ===========================================================================


class TestOAuthHandlerRoutes:
    """Tests for OAuth handler route matching."""

    def test_can_handle_google_routes(self, oauth_handler):
        """Test handler matches Google OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google/callback")
        assert oauth_handler.can_handle("/api/auth/oauth/google")
        assert oauth_handler.can_handle("/api/auth/oauth/google/callback")

    def test_can_handle_github_routes(self, oauth_handler):
        """Test handler matches GitHub OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/github/callback")
        assert oauth_handler.can_handle("/api/auth/oauth/github")
        assert oauth_handler.can_handle("/api/auth/oauth/github/callback")

    def test_can_handle_microsoft_routes(self, oauth_handler):
        """Test handler matches Microsoft OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/microsoft/callback")

    def test_can_handle_apple_routes(self, oauth_handler):
        """Test handler matches Apple OAuth routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/apple/callback")

    def test_can_handle_oidc_routes(self, oauth_handler):
        """Test handler matches OIDC routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/oidc/callback")

    def test_can_handle_account_management_routes(self, oauth_handler):
        """Test handler matches account management routes."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/link")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/unlink")
        assert oauth_handler.can_handle("/api/v1/auth/oauth/providers")
        assert oauth_handler.can_handle("/api/v1/user/oauth-providers")

    def test_does_not_handle_unknown_routes(self, oauth_handler):
        """Test handler does not match unknown routes."""
        assert not oauth_handler.can_handle("/api/v1/auth/login")
        assert not oauth_handler.can_handle("/api/v1/debates")
        assert not oauth_handler.can_handle("/api/v1/auth/oauth/unknown")


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestOAuthRateLimiting:
    """Tests for OAuth rate limiting."""

    @patch("aragora.server.handlers._oauth_impl._oauth_limiter")
    @patch("aragora.server.handlers._oauth_impl.create_span")
    @patch("aragora.server.handlers._oauth_impl.add_span_attributes")
    def test_rate_limit_exceeded_returns_429(
        self, mock_add_span, mock_create_span, mock_limiter, oauth_handler, mock_request_handler
    ):
        """Test that exceeding rate limit returns 429 error."""
        # Setup mock span context manager
        mock_span = MagicMock()
        mock_create_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_create_span.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate rate limit exceeded
        mock_limiter.is_allowed.return_value = False

        result = oauth_handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_request_handler,
            "GET",
        )

        assert result is not None
        assert result.status_code == 429
        assert b"Too many authentication attempts" in result.body

    @patch("aragora.server.handlers._oauth_impl._oauth_limiter")
    @patch("aragora.server.handlers._oauth_impl.create_span")
    @patch("aragora.server.handlers._oauth_impl.add_span_attributes")
    @patch("aragora.server.handlers._oauth_impl._get_google_client_id")
    def test_rate_limit_allowed_proceeds(
        self,
        mock_client_id,
        mock_add_span,
        mock_create_span,
        mock_limiter,
        oauth_handler,
        mock_request_handler,
    ):
        """Test that requests within rate limit proceed normally."""
        mock_span = MagicMock()
        mock_create_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_create_span.return_value.__exit__ = MagicMock(return_value=False)

        mock_limiter.is_allowed.return_value = True
        mock_client_id.return_value = None  # No OAuth configured

        result = oauth_handler.handle(
            "/api/v1/auth/oauth/google",
            {},
            mock_request_handler,
            "GET",
        )

        # Should get 503 (not configured) instead of 429 (rate limited)
        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# OAuth Flow Completion Tests
# ===========================================================================


class TestOAuthFlowCompletion:
    """Tests for OAuth flow completion and user creation."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    async def test_complete_flow_creates_new_user(
        self,
        mock_create_tokens,
        mock_success_url,
        oauth_handler,
        sample_user_info,
        mock_user_store,
    ):
        """Test OAuth flow creates new user when not found."""
        mock_success_url.return_value = "https://example.com/success"
        mock_create_tokens.return_value = MockTokens(
            access_token="access_token_123",
            refresh_token="refresh_token_456",
        )

        state_data = {"redirect_url": "https://example.com/callback"}

        result = await oauth_handler._complete_oauth_flow(sample_user_info, state_data)

        # Verify user was created
        assert len(mock_user_store.created_users) == 1
        created_user = mock_user_store.created_users[0]
        assert created_user.email == "test@example.com"
        assert created_user.name == "Test User"

        # Verify redirect with tokens (may use JS redirect with 200 or HTTP 302)
        assert result is not None
        assert result.status_code in (200, 302)
        # Check tokens in body (JS redirect) or Location header (HTTP redirect)
        body_str = result.body.decode() if isinstance(result.body, bytes) else str(result.body)
        location = result.headers.get("Location", "")
        assert (
            "access_token=access_token_123" in body_str
            or "access_token=access_token_123" in location
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    async def test_complete_flow_links_existing_user(
        self,
        mock_create_tokens,
        mock_success_url,
        oauth_handler,
        mock_user_store,
    ):
        """Test OAuth flow links to existing user with same email."""
        mock_success_url.return_value = "https://example.com/success"
        mock_create_tokens.return_value = MockTokens(
            access_token="access_token_123",
            refresh_token="refresh_token_456",
        )

        # Use email that matches existing user
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google_789",
            email="existing@example.com",  # Matches mock_user_store user
            name="Existing User",
            email_verified=True,
        )

        state_data = {}

        result = await oauth_handler._complete_oauth_flow(user_info, state_data)

        # No new user should be created
        assert len(mock_user_store.created_users) == 0

        # OAuth should be linked
        assert "user_1" in mock_user_store.oauth_links
        assert mock_user_store.oauth_links["user_1"]["google"] == "google_789"

        assert result.status_code in (200, 302)  # May use JS or HTTP redirect

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    async def test_complete_flow_account_linking(
        self,
        mock_success_url,
        oauth_handler,
        sample_user_info,
        mock_user_store,
    ):
        """Test OAuth flow handles account linking when user_id in state."""
        mock_success_url.return_value = "https://example.com/success"

        state_data = {
            "user_id": "user_1",
            "redirect_url": "https://example.com/settings",
        }

        result = await oauth_handler._complete_oauth_flow(sample_user_info, state_data)

        # OAuth should be linked to existing user
        assert "user_1" in mock_user_store.oauth_links
        assert mock_user_store.oauth_links["user_1"]["google"] == "google_123456"

        # Should redirect with 'linked' param (may use JS or HTTP redirect)
        assert result.status_code in (200, 302)
        body_str = result.body.decode() if isinstance(result.body, bytes) else str(result.body)
        location = result.headers.get("Location", "")
        assert "linked=google" in body_str or "linked=google" in location

    @pytest.mark.asyncio
    async def test_complete_flow_no_user_store(self, oauth_handler, sample_user_info):
        """Test OAuth flow fails gracefully without user store."""
        oauth_handler.ctx = {}  # No user_store

        result = await oauth_handler._complete_oauth_flow(sample_user_info, {})

        assert result.status_code == 302
        assert "error=" in result.headers.get("Location", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    async def test_account_linking_already_linked_different_user(
        self,
        mock_success_url,
        oauth_handler,
        mock_user_store,
    ):
        """Test account linking fails when OAuth already linked to another user."""
        mock_success_url.return_value = "https://example.com/success"

        # Create another user and link the OAuth account to them
        mock_user_store.users["user_2"] = MockUser(
            id="user_2",
            email="other@example.com",
            name="Other User",
        )
        mock_user_store.oauth_links["user_2"] = {"google": "google_123456"}

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google_123456",  # Already linked to user_2
            email="test@example.com",
            name="Test User",
            email_verified=True,
        )

        state_data = {
            "user_id": "user_1",  # Trying to link to user_1
        }

        result = await oauth_handler._complete_oauth_flow(user_info, state_data)

        # Should redirect with error
        assert result.status_code == 302
        assert "error=" in result.headers.get("Location", "")
        # URL encoding may use %20 or + for spaces
        location_lower = result.headers.get("Location", "").lower()
        assert "already" in location_lower and "linked" in location_lower


# ===========================================================================
# Redirect URL Tests
# ===========================================================================


class TestOAuthRedirects:
    """Tests for OAuth redirect handling."""

    def test_redirect_with_tokens(self, oauth_handler):
        """Test redirect includes tokens in query params."""
        tokens = MockTokens(
            access_token="test_access",
            refresh_token="test_refresh",
            expires_in=3600,
        )

        result = oauth_handler._redirect_with_tokens("https://example.com/callback", tokens)

        assert result.status_code in (200, 302)  # May use JS or HTTP redirect
        # Check tokens in body (JS redirect) or Location header (HTTP redirect)
        body_str = result.body.decode() if isinstance(result.body, bytes) else str(result.body)
        location = result.headers.get("Location", "")
        content = body_str + location
        assert "access_token=test_access" in content
        assert "refresh_token=test_refresh" in content
        assert "token_type=Bearer" in content
        assert "expires_in=3600" in content

    def test_redirect_with_tokens_has_no_cache_headers(self, oauth_handler):
        """Test redirect includes cache prevention headers."""
        tokens = MockTokens(access_token="test", refresh_token="test")

        result = oauth_handler._redirect_with_tokens("https://example.com/callback", tokens)

        assert result.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, private"
        assert result.headers.get("Pragma") == "no-cache"

    @patch("aragora.server.handlers._oauth_impl._get_oauth_error_url")
    def test_redirect_with_error(self, mock_error_url, oauth_handler):
        """Test error redirect includes URL-encoded error message."""
        mock_error_url.return_value = "https://example.com/error"

        result = oauth_handler._redirect_with_error("Invalid state token")

        assert result.status_code == 302
        location = result.headers.get("Location", "")
        assert "https://example.com/error" in location
        assert "error=" in location
        # URL-encoded error message
        assert "Invalid" in location

    @patch("aragora.server.handlers._oauth_impl._get_oauth_error_url")
    def test_redirect_with_error_has_no_cache_headers(self, mock_error_url, oauth_handler):
        """Test error redirect includes cache prevention headers."""
        mock_error_url.return_value = "https://example.com/error"

        result = oauth_handler._redirect_with_error("Error")

        assert result.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, private"


# ===========================================================================
# Permission Check Tests
# ===========================================================================


@pytest.mark.no_auto_auth
class TestOAuthPermissionChecks:
    """Tests for RBAC permission checks in OAuth handler."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_check_permission_unauthenticated(
        self, mock_extract, oauth_handler, mock_request_handler
    ):
        """Test permission check fails for unauthenticated users."""
        mock_extract.return_value = MagicMock(
            is_authenticated=False,
            user_id=None,
        )

        result = oauth_handler._check_permission(mock_request_handler, "authentication.read")

        assert result is not None
        assert result.status_code == 401

    @patch("aragora.rbac.check_permission")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_check_permission_denied(
        self, mock_extract, mock_check_perm, oauth_handler, mock_request_handler
    ):
        """Test permission check fails when permission denied."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
            client_ip="127.0.0.1",
        )
        mock_check_perm.return_value = MagicMock(
            allowed=False,
            reason="Insufficient privileges",
        )

        result = oauth_handler._check_permission(mock_request_handler, "admin.users.delete")

        assert result is not None
        assert result.status_code == 403

    @patch("aragora.rbac.check_permission")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_check_permission_allowed(
        self, mock_extract, mock_check_perm, oauth_handler, mock_request_handler
    ):
        """Test permission check passes when permission allowed."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="admin",
            client_ip="127.0.0.1",
        )
        mock_check_perm.return_value = MagicMock(
            allowed=True,
            reason=None,
        )

        result = oauth_handler._check_permission(mock_request_handler, "authentication.read")

        # None means allowed
        assert result is None


# ===========================================================================
# User Creation Tests
# ===========================================================================


class TestOAuthUserCreation:
    """Tests for OAuth user creation."""

    @pytest.mark.asyncio
    async def test_find_user_by_oauth(self, oauth_handler, mock_user_store):
        """Test finding user by OAuth provider ID."""
        # Link OAuth to user_1
        mock_user_store.oauth_links["user_1"] = {"google": "google_123"}

        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google_123",
            email="test@example.com",
            name="Test",
            email_verified=True,
        )

        user = await oauth_handler._find_user_by_oauth(mock_user_store, user_info)

        assert user is not None
        assert user.id == "user_1"

    @pytest.mark.asyncio
    async def test_find_user_by_oauth_not_found(self, oauth_handler, mock_user_store):
        """Test finding user by OAuth returns None when not found."""
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="unknown_id",
            email="test@example.com",
            name="Test",
            email_verified=True,
        )

        user = await oauth_handler._find_user_by_oauth(mock_user_store, user_info)

        assert user is None

    @pytest.mark.asyncio
    async def test_link_oauth_to_user(self, oauth_handler, mock_user_store):
        """Test linking OAuth provider to user."""
        user_info = OAuthUserInfo(
            provider="github",
            provider_user_id="github_456",
            email="test@example.com",
            name="Test",
            email_verified=True,
        )

        result = await oauth_handler._link_oauth_to_user(mock_user_store, "user_1", user_info)

        assert result is True
        assert mock_user_store.oauth_links["user_1"]["github"] == "github_456"

    @pytest.mark.asyncio
    async def test_create_oauth_user(self, oauth_handler, mock_user_store):
        """Test creating new user from OAuth info."""
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google_new",
            email="newuser@example.com",
            name="New User",
            email_verified=True,
        )

        user = await oauth_handler._create_oauth_user(mock_user_store, user_info)

        assert user is not None
        assert user.email == "newuser@example.com"
        assert user.name == "New User"
        assert len(mock_user_store.created_users) == 1

    @pytest.mark.asyncio
    async def test_create_oauth_user_trusted_provider_allows_unverified_email(
        self, oauth_handler, mock_user_store
    ):
        """Trusted providers should still auto-provision when email_verified is false."""
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="google_unverified",
            email="unverified-google@example.com",
            name="Unverified Google User",
            email_verified=False,
        )

        user = await oauth_handler._create_oauth_user(mock_user_store, user_info)

        assert user is not None
        assert user.email == "unverified-google@example.com"
        assert user.id in mock_user_store.oauth_links
        assert mock_user_store.oauth_links[user.id]["google"] == "google_unverified"

    @pytest.mark.asyncio
    async def test_create_oauth_user_untrusted_provider_requires_verified_email(
        self, oauth_handler, mock_user_store
    ):
        """Untrusted providers must not auto-provision with unverified email."""
        user_info = OAuthUserInfo(
            provider="oidc",
            provider_user_id="oidc_unverified",
            email="unverified-oidc@example.com",
            name="Unverified OIDC User",
            email_verified=False,
        )

        user = await oauth_handler._create_oauth_user(mock_user_store, user_info)

        assert user is None
        assert not any(
            u.email == "unverified-oidc@example.com" for u in mock_user_store.created_users
        )


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestOAuthMethodNotAllowed:
    """Tests for method not allowed responses."""

    @patch("aragora.server.handlers._oauth_impl._oauth_limiter")
    @patch("aragora.server.handlers._oauth_impl.create_span")
    @patch("aragora.server.handlers._oauth_impl.add_span_attributes")
    def test_post_to_get_only_route_returns_405(
        self, mock_add_span, mock_create_span, mock_limiter, oauth_handler, mock_request_handler
    ):
        """Test POST to GET-only route returns 405."""
        mock_span = MagicMock()
        mock_create_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_create_span.return_value.__exit__ = MagicMock(return_value=False)
        mock_limiter.is_allowed.return_value = True

        # Set the command on the handler (handle() uses handler.command if available)
        mock_request_handler.command = "POST"

        # POST to providers list (GET only)
        result = oauth_handler.handle(
            "/api/v1/auth/oauth/providers",
            {},
            mock_request_handler,
            "POST",
        )

        assert result is not None
        assert result.status_code == 405

    @patch("aragora.server.handlers._oauth_impl._oauth_limiter")
    @patch("aragora.server.handlers._oauth_impl.create_span")
    @patch("aragora.server.handlers._oauth_impl.add_span_attributes")
    def test_delete_to_link_route_returns_405(
        self, mock_add_span, mock_create_span, mock_limiter, oauth_handler, mock_request_handler
    ):
        """Test DELETE to link route returns 405."""
        mock_span = MagicMock()
        mock_create_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_create_span.return_value.__exit__ = MagicMock(return_value=False)
        mock_limiter.is_allowed.return_value = True

        # Set the command on the handler (handle() uses handler.command if available)
        mock_request_handler.command = "DELETE"

        result = oauth_handler.handle(
            "/api/v1/auth/oauth/link",
            {},
            mock_request_handler,
            "DELETE",  # Should be POST
        )

        assert result is not None
        assert result.status_code == 405
