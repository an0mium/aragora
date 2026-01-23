"""
Tests for OAuth Authentication Handlers.

Tests cover:
- OAuthUserInfo dataclass
- State management (generate, validate, cleanup)
- OAuthHandler routing
- Google OAuth flow (start, callback)
- Account linking and unlinking
- Provider listing
- Error handling and redirects
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
from urllib.parse import parse_qs, urlparse

import pytest


# ============================================================================
# Module-level patches for environment variables
# ============================================================================


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for OAuth."""
    env_vars = {
        "GOOGLE_OAUTH_CLIENT_ID": "test-client-id",
        "GOOGLE_OAUTH_CLIENT_SECRET": "test-client-secret",
        "GOOGLE_OAUTH_REDIRECT_URI": "http://localhost:8080/api/auth/oauth/google/callback",
        "OAUTH_SUCCESS_URL": "http://localhost:3000/auth/callback",
        "OAUTH_ERROR_URL": "http://localhost:3000/auth/error",
    }
    with patch.dict(os.environ, env_vars):
        # Need to reload the module to pick up env vars
        import importlib
        import aragora.server.handlers.oauth as oauth_module

        importlib.reload(oauth_module)
        yield oauth_module


@pytest.fixture
def oauth_module(mock_env_vars):
    """Return the mocked oauth module."""
    return mock_env_vars


@pytest.fixture(autouse=True)
def clear_oauth_states(oauth_module):
    """Clear OAuth states before and after each test."""
    oauth_module._OAUTH_STATES.clear()
    yield
    oauth_module._OAUTH_STATES.clear()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server_context():
    """Create mock server context for handler initialization."""
    return {
        "user_store": None,
        "storage": None,
        "elo_system": None,
    }


@pytest.fixture
def oauth_handler(oauth_module, server_context):
    """Create OAuthHandler with mock context."""
    return oauth_module.OAuthHandler(server_context)


# ============================================================================
# Mock Classes
# ============================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str
    email: str
    name: str = "Test User"
    role: str = "user"
    org_id: Optional[str] = None
    password_hash: Optional[str] = "hashed_password"


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.oauth_links: dict[tuple[str, str], str] = {}  # (provider, provider_id) -> user_id

    def create_user(
        self, email: str, password_hash: str, password_salt: str, name: str = ""
    ) -> MockUser:
        user_id = f"user_{len(self.users) + 1}"
        user = MockUser(id=user_id, email=email, name=name or email.split("@")[0])
        self.users[user_id] = user
        return user

    def get_user_by_id(self, user_id: str) -> Optional[MockUser]:
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[MockUser]:
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> Optional[MockUser]:
        user_id = self.oauth_links.get((provider, provider_user_id))
        if user_id:
            return self.users.get(user_id)
        return None

    def link_oauth_provider(
        self, user_id: str, provider: str, provider_user_id: str, email: str
    ) -> bool:
        self.oauth_links[(provider, provider_user_id)] = user_id
        return True

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        keys_to_remove = [
            k for k in self.oauth_links if k[0] == provider and self.oauth_links[k] == user_id
        ]
        for k in keys_to_remove:
            del self.oauth_links[k]
        return len(keys_to_remove) > 0

    def update_user(self, user_id: str, **kwargs) -> None:
        user = self.users.get(user_id)
        if user:
            for k, v in kwargs.items():
                if hasattr(user, k):
                    setattr(user, k, v)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, method: str = "GET", body: bytes = b""):
        self.command = method
        self._body = body
        self.rfile = MagicMock()
        self.rfile.read.return_value = body
        self.headers = {}

    def set_body(self, data: dict):
        """Set JSON body."""
        self._body = json.dumps(data).encode()
        self.rfile.read.return_value = self._body


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = False
    user_id: Optional[str] = None


@dataclass
class MockTokenPair:
    """Mock token pair."""

    access_token: str = "test_access_token"
    refresh_token: str = "test_refresh_token"
    expires_in: int = 3600


# ============================================================================
# OAuthUserInfo Tests
# ============================================================================


class TestOAuthUserInfo:
    """Tests for OAuthUserInfo dataclass."""

    def test_basic_creation(self, oauth_module):
        """Test basic OAuthUserInfo creation."""
        info = oauth_module.OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="test@example.com",
            name="Test User",
        )
        assert info.provider == "google"
        assert info.provider_user_id == "123"
        assert info.email == "test@example.com"
        assert info.name == "Test User"

    def test_default_values(self, oauth_module):
        """Test OAuthUserInfo default values."""
        info = oauth_module.OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="test@example.com",
            name="Test",
        )
        assert info.picture is None
        assert info.email_verified is False

    def test_with_picture_and_verified(self, oauth_module):
        """Test OAuthUserInfo with all fields."""
        info = oauth_module.OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="test@example.com",
            name="Test",
            picture="https://example.com/photo.jpg",
            email_verified=True,
        )
        assert info.picture == "https://example.com/photo.jpg"
        assert info.email_verified is True


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for OAuth state management functions."""

    def test_generate_state(self, oauth_module):
        """Test state generation."""
        state = oauth_module._generate_state()
        assert len(state) > 20  # Should be a secure token
        assert state in oauth_module._OAUTH_STATES

    def test_generate_state_with_user_id(self, oauth_module):
        """Test state generation with user ID."""
        state = oauth_module._generate_state(user_id="user_123")
        state_data = oauth_module._OAUTH_STATES[state]
        assert state_data["user_id"] == "user_123"

    def test_generate_state_with_redirect(self, oauth_module):
        """Test state generation with redirect URL."""
        state = oauth_module._generate_state(redirect_url="http://example.com")
        state_data = oauth_module._OAUTH_STATES[state]
        assert state_data["redirect_url"] == "http://example.com"

    def test_validate_state_success(self, oauth_module):
        """Test successful state validation."""
        state = oauth_module._generate_state(user_id="user_123")
        result = oauth_module._validate_state(state)

        assert result is not None
        assert result["user_id"] == "user_123"
        # State should be consumed
        assert state not in oauth_module._OAUTH_STATES

    def test_validate_state_invalid(self, oauth_module):
        """Test validation of invalid state."""
        result = oauth_module._validate_state("invalid_state")
        assert result is None

    def test_validate_state_consumed(self, oauth_module):
        """Test state can only be used once."""
        state = oauth_module._generate_state()

        # First validation succeeds
        result1 = oauth_module._validate_state(state)
        assert result1 is not None

        # Second validation fails (consumed)
        result2 = oauth_module._validate_state(state)
        assert result2 is None

    def test_cleanup_expired_states(self, oauth_module):
        """Test expired state cleanup."""
        # Create state with past expiration
        state = "expired_state"
        oauth_module._OAUTH_STATES[state] = {
            "expires_at": time.time() - 100,
            "user_id": None,
        }

        # Cleanup should remove it
        oauth_module._cleanup_expired_states()
        assert state not in oauth_module._OAUTH_STATES

    def test_state_expiration(self, oauth_module):
        """Test state includes expiration time."""
        state = oauth_module._generate_state()
        state_data = oauth_module._OAUTH_STATES[state]

        assert "expires_at" in state_data
        assert state_data["expires_at"] > time.time()


# ============================================================================
# OAuthHandler Routing Tests
# ============================================================================


class TestOAuthHandlerRouting:
    """Tests for OAuthHandler routing."""

    def test_can_handle_google_auth(self, oauth_handler):
        """Test handler can handle Google auth endpoint."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google")

    def test_can_handle_callback(self, oauth_handler):
        """Test handler can handle callback endpoint."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google/callback")

    def test_can_handle_link(self, oauth_handler):
        """Test handler can handle link endpoint."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/link")

    def test_can_handle_unlink(self, oauth_handler):
        """Test handler can handle unlink endpoint."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/unlink")

    def test_can_handle_providers(self, oauth_handler):
        """Test handler can handle providers endpoint."""
        assert oauth_handler.can_handle("/api/v1/auth/oauth/providers")

    def test_cannot_handle_unknown(self, oauth_handler):
        """Test handler rejects unknown paths."""
        assert not oauth_handler.can_handle("/api/v1/auth/unknown")

    def test_routes_list(self, oauth_handler):
        """Test handler has expected routes."""
        expected = [
            "/api/auth/oauth/google",
            "/api/auth/oauth/google/callback",
            "/api/auth/oauth/link",
            "/api/auth/oauth/unlink",
            "/api/auth/oauth/providers",
        ]
        assert oauth_handler.ROUTES == expected


# ============================================================================
# Google OAuth Start Tests
# ============================================================================


class TestGoogleAuthStart:
    """Tests for Google OAuth start endpoint."""

    def test_returns_redirect(self, oauth_handler):
        """Test OAuth start returns redirect response."""
        mock_handler = MockHandler()

        with patch.object(oauth_handler, "_get_user_store", return_value=None):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                result = oauth_handler._handle_google_auth_start(mock_handler, {})

        assert result.status_code == 302
        assert "Location" in result.headers

    def test_redirect_url_contains_state(self, oauth_handler):
        """Test redirect URL contains state parameter."""
        mock_handler = MockHandler()

        with patch.object(oauth_handler, "_get_user_store", return_value=None):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                result = oauth_handler._handle_google_auth_start(mock_handler, {})

        location = result.headers["Location"]
        assert "state=" in location

    def test_redirect_url_contains_client_id(self, oauth_handler):
        """Test redirect URL contains client ID."""
        mock_handler = MockHandler()

        with patch.object(oauth_handler, "_get_user_store", return_value=None):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                result = oauth_handler._handle_google_auth_start(mock_handler, {})

        location = result.headers["Location"]
        assert "client_id=test-client-id" in location

    def test_redirect_includes_scopes(self, oauth_handler):
        """Test redirect URL includes required scopes."""
        mock_handler = MockHandler()

        with patch.object(oauth_handler, "_get_user_store", return_value=None):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                result = oauth_handler._handle_google_auth_start(mock_handler, {})

        location = result.headers["Location"]
        assert "scope=" in location
        assert "openid" in location or "openid" in location.replace("%20", " ")

    def test_stores_user_id_for_linking(self, oauth_handler, oauth_module):
        """Test user ID is stored in state for account linking."""
        mock_handler = MockHandler()
        user_store = MockUserStore()

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(
                    is_authenticated=True, user_id="existing_user"
                )
                oauth_handler._handle_google_auth_start(mock_handler, {})

        # Find the state that was created
        states = list(oauth_module._OAUTH_STATES.values())
        assert len(states) == 1
        assert states[0]["user_id"] == "existing_user"


# ============================================================================
# Google OAuth Callback Tests
# ============================================================================


class TestGoogleCallback:
    """Tests for Google OAuth callback endpoint."""

    def test_error_from_google(self, oauth_handler):
        """Test handling of error from Google."""
        mock_handler = MockHandler()

        query_params = {
            "error": ["access_denied"],
            "error_description": ["User cancelled"],
        }

        result = oauth_handler._handle_google_callback(mock_handler, query_params)

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]

    def test_missing_state(self, oauth_handler):
        """Test handling of missing state parameter."""
        mock_handler = MockHandler()

        result = oauth_handler._handle_google_callback(mock_handler, {})

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]

    def test_invalid_state(self, oauth_handler):
        """Test handling of invalid state parameter."""
        mock_handler = MockHandler()

        result = oauth_handler._handle_google_callback(mock_handler, {"state": ["invalid"]})

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]

    def test_missing_code(self, oauth_handler, oauth_module):
        """Test handling of missing authorization code."""
        mock_handler = MockHandler()

        # Generate valid state
        state = oauth_module._generate_state()

        result = oauth_handler._handle_google_callback(mock_handler, {"state": [state]})

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]

    def test_successful_login_new_user(self, oauth_handler, oauth_module):
        """Test successful OAuth login creates new user."""
        mock_handler = MockHandler()
        user_store = MockUserStore()

        # Generate valid state
        state = oauth_module._generate_state()

        query_params = {
            "state": [state],
            "code": ["auth_code_123"],
        }

        # Mock token exchange and user info
        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch.object(oauth_handler, "_exchange_code_for_tokens") as mock_exchange:
                mock_exchange.return_value = {"access_token": "test_token"}
                with patch.object(oauth_handler, "_get_google_user_info") as mock_info:
                    mock_info.return_value = oauth_module.OAuthUserInfo(
                        provider="google",
                        provider_user_id="google_123",
                        email="new@example.com",
                        name="New User",
                        email_verified=True,
                    )
                    with patch("aragora.billing.jwt_auth.create_token_pair") as mock_tokens:
                        mock_tokens.return_value = MockTokenPair()
                        with patch("aragora.billing.models.hash_password") as mock_hash:
                            mock_hash.return_value = ("hash", "salt")
                            result = oauth_handler._handle_google_callback(
                                mock_handler, query_params
                            )

        assert result.status_code == 302
        # Should have created a user
        assert len(user_store.users) == 1

    def test_successful_login_existing_oauth_user(self, oauth_handler, oauth_module):
        """Test successful OAuth login for existing OAuth user."""
        mock_handler = MockHandler()
        user_store = MockUserStore()

        # Create existing user with OAuth link
        user = user_store.create_user("existing@example.com", "hash", "salt", "Existing")
        user_store.link_oauth_provider(user.id, "google", "google_123", "existing@example.com")

        state = oauth_module._generate_state()
        query_params = {"state": [state], "code": ["auth_code"]}

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch.object(oauth_handler, "_exchange_code_for_tokens") as mock_exchange:
                mock_exchange.return_value = {"access_token": "test_token"}
                with patch.object(oauth_handler, "_get_google_user_info") as mock_info:
                    mock_info.return_value = oauth_module.OAuthUserInfo(
                        provider="google",
                        provider_user_id="google_123",
                        email="existing@example.com",
                        name="Existing User",
                    )
                    with patch("aragora.billing.jwt_auth.create_token_pair") as mock_tokens:
                        mock_tokens.return_value = MockTokenPair()
                        result = oauth_handler._handle_google_callback(mock_handler, query_params)

        assert result.status_code == 302
        # Should NOT create new user
        assert len(user_store.users) == 1


# ============================================================================
# Account Linking Tests
# ============================================================================


class TestAccountLinking:
    """Tests for OAuth account linking."""

    def test_link_requires_auth(self, oauth_handler):
        """Test link endpoint requires authentication."""
        mock_handler = MockHandler(method="POST")
        mock_handler.set_body({"provider": "google"})
        user_store = MockUserStore()

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "google"}
                ):
                    result = oauth_handler._handle_link_account(mock_handler)

        assert result.status_code == 401

    def test_link_invalid_provider(self, oauth_handler):
        """Test link endpoint rejects invalid provider."""
        mock_handler = MockHandler(method="POST")
        user_store = MockUserStore()
        user_store.create_user("test@example.com", "hash", "salt")

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id="user_1")
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "facebook"}
                ):
                    result = oauth_handler._handle_link_account(mock_handler)

        assert result.status_code == 400

    def test_link_returns_auth_url(self, oauth_handler):
        """Test link endpoint returns auth URL for valid provider."""
        mock_handler = MockHandler(method="POST")
        user_store = MockUserStore()
        user = user_store.create_user("test@example.com", "hash", "salt")

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id=user.id)
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "google"}
                ):
                    result = oauth_handler._handle_link_account(mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "accounts.google.com" in body["auth_url"]


# ============================================================================
# Account Unlinking Tests
# ============================================================================


class TestAccountUnlinking:
    """Tests for OAuth account unlinking."""

    def test_unlink_requires_auth(self, oauth_handler):
        """Test unlink endpoint requires authentication."""
        mock_handler = MockHandler(method="DELETE")
        user_store = MockUserStore()

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "google"}
                ):
                    result = oauth_handler._handle_unlink_account(mock_handler)

        assert result.status_code == 401

    def test_unlink_requires_password(self, oauth_handler):
        """Test unlink fails if user has no password."""
        mock_handler = MockHandler(method="DELETE")
        user_store = MockUserStore()
        user = user_store.create_user("test@example.com", "hash", "salt")
        user.password_hash = None  # No password

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id=user.id)
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "google"}
                ):
                    result = oauth_handler._handle_unlink_account(mock_handler)

        assert result.status_code == 400
        assert b"password" in result.body.lower()

    def test_unlink_success(self, oauth_handler):
        """Test successful OAuth unlinking."""
        mock_handler = MockHandler(method="DELETE")
        user_store = MockUserStore()
        user = user_store.create_user("test@example.com", "hash", "salt")
        user_store.link_oauth_provider(user.id, "google", "google_123", "test@example.com")

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=True, user_id=user.id)
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "google"}
                ):
                    result = oauth_handler._handle_unlink_account(mock_handler)

        assert result.status_code == 200
        # OAuth link should be removed
        assert user_store.get_user_by_oauth("google", "google_123") is None


# ============================================================================
# Provider Listing Tests
# ============================================================================


class TestProviderListing:
    """Tests for OAuth provider listing."""

    def test_list_providers_with_google(self, oauth_handler):
        """Test listing providers when Google is configured."""
        mock_handler = MockHandler()

        result = oauth_handler._handle_list_providers(mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "providers" in body
        assert len(body["providers"]) == 1
        assert body["providers"][0]["id"] == "google"
        assert body["providers"][0]["enabled"] is True

    def test_provider_includes_auth_url(self, oauth_handler):
        """Test provider listing includes auth URL."""
        mock_handler = MockHandler()

        result = oauth_handler._handle_list_providers(mock_handler)

        body = json.loads(result.body.decode())
        provider = body["providers"][0]
        assert "auth_url" in provider
        assert provider["auth_url"] == "/api/auth/oauth/google"


# ============================================================================
# Redirect Helper Tests
# ============================================================================


class TestRedirectHelpers:
    """Tests for redirect helper methods."""

    def test_redirect_with_error(self, oauth_handler):
        """Test error redirect includes error message."""
        result = oauth_handler._redirect_with_error("Test error message")

        assert result.status_code == 302
        location = result.headers["Location"]
        assert "error=" in location
        assert "Test" in location or "error" in location

    def test_redirect_with_tokens(self, oauth_handler):
        """Test token redirect includes tokens in fragment."""
        tokens = MockTokenPair()

        result = oauth_handler._redirect_with_tokens("http://localhost:3000/callback", tokens)

        assert result.status_code == 302
        location = result.headers["Location"]
        assert "#" in location  # Fragment separator
        assert "access_token=" in location


# ============================================================================
# Token Exchange Tests
# ============================================================================


class TestTokenExchange:
    """Tests for token exchange functionality."""

    def test_exchange_makes_request(self, oauth_handler):
        """Test token exchange makes HTTP request."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(
                {
                    "access_token": "test_token",
                    "token_type": "Bearer",
                }
            ).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = oauth_handler._exchange_code_for_tokens("auth_code_123")

        assert result["access_token"] == "test_token"
        mock_urlopen.assert_called_once()


# ============================================================================
# User Info Retrieval Tests
# ============================================================================


class TestUserInfoRetrieval:
    """Tests for Google user info retrieval."""

    def test_get_user_info(self, oauth_handler):
        """Test getting user info from Google."""
        google_response = {
            "id": "google_123",
            "email": "user@gmail.com",
            "name": "Test User",
            "picture": "https://example.com/photo.jpg",
            "verified_email": True,
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(google_response).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = oauth_handler._get_google_user_info("access_token")

        assert result.provider == "google"
        assert result.provider_user_id == "google_123"
        assert result.email == "user@gmail.com"
        assert result.name == "Test User"
        assert result.picture == "https://example.com/photo.jpg"
        assert result.email_verified is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestOAuthIntegration:
    """Integration tests for OAuth flow."""

    def test_full_oauth_flow_new_user(self, oauth_handler, oauth_module):
        """Test complete OAuth flow for new user."""
        user_store = MockUserStore()

        # Step 1: Start OAuth flow
        mock_handler = MockHandler()
        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(is_authenticated=False)
                start_result = oauth_handler._handle_google_auth_start(mock_handler, {})

        # Extract state from redirect URL
        location = start_result.headers["Location"]
        parsed = urlparse(location)
        query = parse_qs(parsed.query)
        state = query["state"][0]

        # Step 2: Callback with authorization code
        callback_params = {"state": [state], "code": ["auth_code"]}

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch.object(oauth_handler, "_exchange_code_for_tokens") as mock_exchange:
                mock_exchange.return_value = {"access_token": "test_token"}
                with patch.object(oauth_handler, "_get_google_user_info") as mock_info:
                    mock_info.return_value = oauth_module.OAuthUserInfo(
                        provider="google",
                        provider_user_id="google_new_user",
                        email="newuser@gmail.com",
                        name="New User",
                        email_verified=True,
                    )
                    with patch("aragora.billing.jwt_auth.create_token_pair") as mock_tokens:
                        mock_tokens.return_value = MockTokenPair()
                        with patch("aragora.billing.models.hash_password") as mock_hash:
                            mock_hash.return_value = ("hash", "salt")
                            callback_result = oauth_handler._handle_google_callback(
                                mock_handler, callback_params
                            )

        # Verify user was created
        assert len(user_store.users) == 1
        user = list(user_store.users.values())[0]
        assert user.email == "newuser@gmail.com"

        # Verify OAuth link was created
        linked_user = user_store.get_user_by_oauth("google", "google_new_user")
        assert linked_user is not None
        assert linked_user.id == user.id

        # Verify redirect with tokens
        assert callback_result.status_code == 302
        assert "access_token" in callback_result.headers["Location"]

    def test_oauth_account_linking_flow(self, oauth_handler, oauth_module):
        """Test OAuth account linking for existing user."""
        user_store = MockUserStore()

        # Create existing user
        existing_user = user_store.create_user("existing@example.com", "hash", "salt", "Existing")

        # Step 1: Start OAuth flow with authenticated user (for linking)
        mock_handler = MockHandler()
        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext(
                    is_authenticated=True, user_id=existing_user.id
                )
                start_result = oauth_handler._handle_google_auth_start(mock_handler, {})

        # Extract state
        location = start_result.headers["Location"]
        parsed = urlparse(location)
        query = parse_qs(parsed.query)
        state = query["state"][0]

        # Verify state contains user_id for linking
        state_data = oauth_module._OAUTH_STATES.get(state)
        assert state_data["user_id"] == existing_user.id

        # Step 2: Callback links OAuth to existing user
        callback_params = {"state": [state], "code": ["auth_code"]}

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            with patch.object(oauth_handler, "_exchange_code_for_tokens") as mock_exchange:
                mock_exchange.return_value = {"access_token": "test_token"}
                with patch.object(oauth_handler, "_get_google_user_info") as mock_info:
                    mock_info.return_value = oauth_module.OAuthUserInfo(
                        provider="google",
                        provider_user_id="google_link_id",
                        email="different@gmail.com",  # Different email
                        name="Google User",
                    )
                    callback_result = oauth_handler._handle_google_callback(
                        mock_handler, callback_params
                    )

        # Verify no new user created
        assert len(user_store.users) == 1

        # Verify OAuth was linked to existing user
        linked_user = user_store.get_user_by_oauth("google", "google_link_id")
        assert linked_user is not None
        assert linked_user.id == existing_user.id

        # Verify redirect indicates linking
        assert callback_result.status_code == 302
        assert "linked=google" in callback_result.headers["Location"]


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_no_google_client_id(self, oauth_handler, oauth_module):
        """Test handling when Google OAuth is not configured."""
        mock_handler = MockHandler()

        # Temporarily clear client ID
        original = oauth_module.GOOGLE_CLIENT_ID
        oauth_module.GOOGLE_CLIENT_ID = ""

        try:
            with patch.object(oauth_handler, "_get_user_store", return_value=None):
                result = oauth_handler._handle_google_auth_start(mock_handler, {})
        finally:
            oauth_module.GOOGLE_CLIENT_ID = original

        assert result.status_code == 503

    def test_user_store_unavailable(self, oauth_handler, oauth_module):
        """Test handling when user store is unavailable."""
        mock_handler = MockHandler()

        state = oauth_module._generate_state()
        query_params = {"state": [state], "code": ["auth_code"]}

        with patch.object(oauth_handler, "_get_user_store", return_value=None):
            with patch.object(oauth_handler, "_exchange_code_for_tokens") as mock_exchange:
                mock_exchange.return_value = {"access_token": "test_token"}
                with patch.object(oauth_handler, "_get_google_user_info") as mock_info:
                    mock_info.return_value = oauth_module.OAuthUserInfo(
                        provider="google",
                        provider_user_id="123",
                        email="test@example.com",
                        name="Test",
                    )
                    result = oauth_handler._handle_google_callback(mock_handler, query_params)

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]

    def test_oauth_already_linked_to_other_user(self, oauth_handler, oauth_module):
        """Test error when OAuth is already linked to another user."""
        mock_handler = MockHandler()
        user_store = MockUserStore()

        # Create two users
        user1 = user_store.create_user("user1@example.com", "hash", "salt")
        user2 = user_store.create_user("user2@example.com", "hash", "salt")

        # Link OAuth to user1
        user_store.link_oauth_provider(user1.id, "google", "google_123", "oauth@gmail.com")

        # Try to link same OAuth to user2
        state = oauth_module._generate_state(user_id=user2.id)

        with patch.object(oauth_handler, "_get_user_store", return_value=user_store):
            result = oauth_handler._handle_account_linking(
                user_store,
                user2.id,
                oauth_module.OAuthUserInfo(
                    provider="google",
                    provider_user_id="google_123",
                    email="oauth@gmail.com",
                    name="OAuth User",
                ),
                {"user_id": user2.id, "redirect_url": "http://localhost"},
            )

        assert result.status_code == 302
        assert "error=" in result.headers["Location"]
        # Check for URL-encoded "already linked" message
        from urllib.parse import unquote

        assert "already linked" in unquote(result.headers["Location"])

    def test_method_not_allowed(self, oauth_handler):
        """Test method not allowed for wrong HTTP method."""
        mock_handler = MockHandler(method="PUT")

        result = oauth_handler.handle("/api/auth/oauth/google", {}, mock_handler, method="PUT")

        assert result.status_code == 405
