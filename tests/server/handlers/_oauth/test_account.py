"""
Tests for OAuth Account Management Mixin.

Tests cover:
- List configured OAuth providers
- Get OAuth authorization URL
- OAuth callback API endpoint
- Get user's linked OAuth providers
- Link OAuth account to user
- Unlink OAuth account from user
- RBAC permission checks

SECURITY CRITICAL: These tests ensure OAuth account management is secure.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch
import json

import pytest

from aragora.server.handlers._oauth.account import AccountManagementMixin
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


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.oauth_links: dict[str, dict[str, str]] = {}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_oauth_providers(self, user_id: str) -> list[dict]:
        links = self.oauth_links.get(user_id, {})
        return [{"provider": p, "provider_user_id": uid} for p, uid in links.items()]

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        if user_id in self.oauth_links and provider in self.oauth_links[user_id]:
            del self.oauth_links[user_id][provider]
            return True
        return False


class AccountManagementTestHandler(AccountManagementMixin):
    """Test handler that mixes in AccountManagementMixin."""

    def __init__(self, user_store: MockUserStore | None = None):
        self.user_store = user_store or MockUserStore()
        self.ctx = {"user_store": self.user_store}

    def _get_user_store(self):
        return self.user_store

    def _check_permission(self, handler, permission_key, resource_id=None):
        # Return None (allowed) by default for tests
        # Tests that need to check permissions will mock this
        return None

    def read_json_body(self, handler) -> dict | None:
        if hasattr(handler, "_json_body"):
            return handler._json_body
        return None

    # Mock provider auth start methods
    def _handle_google_auth_start(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://google.com/authorize?state=test-state"},
            body=b"",
            content_type="text/html",
        )

    def _handle_github_auth_start(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://github.com/login/oauth/authorize?state=test-state"},
            body=b"",
            content_type="text/html",
        )

    def _handle_microsoft_auth_start(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://microsoft.com/authorize?state=test-state"},
            body=b"",
            content_type="text/html",
        )

    def _handle_apple_auth_start(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://apple.com/authorize?state=test-state"},
            body=b"",
            content_type="text/html",
        )

    def _handle_oidc_auth_start(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/authorize?state=test-state"},
            body=b"",
            content_type="text/html",
        )

    # Mock provider callback methods (needed for callback_map building)
    def _handle_google_callback(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/success?access_token=test"},
            body=b"",
            content_type="text/html",
        )

    def _handle_github_callback(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/success?access_token=test"},
            body=b"",
            content_type="text/html",
        )

    def _handle_microsoft_callback(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/success?access_token=test"},
            body=b"",
            content_type="text/html",
        )

    def _handle_apple_callback(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/success?access_token=test"},
            body=b"",
            content_type="text/html",
        )

    def _handle_oidc_callback(self, handler, query_params):
        return HandlerResult(
            status_code=302,
            headers={"Location": "https://example.com/success?access_token=test"},
            body=b"",
            content_type="text/html",
        )


@pytest.fixture
def mock_user_store():
    """Create a mock user store with test user."""
    store = MockUserStore()
    store.users["user_1"] = MockUser(
        id="user_1",
        email="test@example.com",
        name="Test User",
        org_id="org_1",
    )
    store.oauth_links["user_1"] = {"google": "google_123", "github": "github_456"}
    return store


@pytest.fixture
def account_handler(mock_user_store):
    """Create an account management test handler."""
    return AccountManagementTestHandler(mock_user_store)


@pytest.fixture
def mock_request_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Host": "localhost:8080", "Authorization": "Bearer test_token"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


# ===========================================================================
# List Providers Tests
# ===========================================================================


class TestListProviders:
    """Tests for listing configured OAuth providers."""

    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._get_apple_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_google_client_id")
    def test_list_providers_all_configured(
        self,
        mock_google,
        mock_github,
        mock_microsoft,
        mock_apple,
        mock_oidc_issuer,
        mock_oidc_client,
        account_handler,
        mock_request_handler,
    ):
        """Test listing providers when all are configured."""
        mock_google.return_value = "google-client-id"
        mock_github.return_value = "github-client-id"
        mock_microsoft.return_value = "microsoft-client-id"
        mock_apple.return_value = "apple-client-id"
        mock_oidc_issuer.return_value = "https://example.com"
        mock_oidc_client.return_value = "oidc-client-id"

        result = account_handler._handle_list_providers(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        providers = body["providers"]

        assert len(providers) == 5
        provider_ids = [p["id"] for p in providers]
        assert "google" in provider_ids
        assert "github" in provider_ids
        assert "microsoft" in provider_ids
        assert "apple" in provider_ids
        assert "oidc" in provider_ids

    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._get_apple_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_google_client_id")
    def test_list_providers_only_google_configured(
        self,
        mock_google,
        mock_github,
        mock_microsoft,
        mock_apple,
        mock_oidc_issuer,
        mock_oidc_client,
        account_handler,
        mock_request_handler,
    ):
        """Test listing providers when only Google is configured."""
        mock_google.return_value = "google-client-id"
        mock_github.return_value = None
        mock_microsoft.return_value = None
        mock_apple.return_value = None
        mock_oidc_issuer.return_value = None
        mock_oidc_client.return_value = None

        result = account_handler._handle_list_providers(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        providers = body["providers"]

        assert len(providers) == 1
        assert providers[0]["id"] == "google"
        assert providers[0]["enabled"] is True
        assert providers[0]["auth_url"] == "/api/auth/oauth/google"

    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._get_apple_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_google_client_id")
    def test_list_providers_none_configured(
        self,
        mock_google,
        mock_github,
        mock_microsoft,
        mock_apple,
        mock_oidc_issuer,
        mock_oidc_client,
        account_handler,
        mock_request_handler,
    ):
        """Test listing providers when none are configured."""
        mock_google.return_value = None
        mock_github.return_value = None
        mock_microsoft.return_value = None
        mock_apple.return_value = None
        mock_oidc_issuer.return_value = None
        mock_oidc_client.return_value = None

        result = account_handler._handle_list_providers(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["providers"] == []

    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.server.handlers._oauth_impl._get_apple_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_github_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_google_client_id")
    def test_oidc_requires_both_issuer_and_client_id(
        self,
        mock_google,
        mock_github,
        mock_microsoft,
        mock_apple,
        mock_oidc_issuer,
        mock_oidc_client,
        account_handler,
        mock_request_handler,
    ):
        """Test OIDC provider requires both issuer and client_id."""
        mock_google.return_value = None
        mock_github.return_value = None
        mock_microsoft.return_value = None
        mock_apple.return_value = None
        # Only issuer, no client_id
        mock_oidc_issuer.return_value = "https://example.com"
        mock_oidc_client.return_value = None

        result = account_handler._handle_list_providers(mock_request_handler)

        body = json.loads(result.body.decode())
        # OIDC should not be in providers
        assert len(body["providers"]) == 0


# ===========================================================================
# OAuth URL Tests
# ===========================================================================


class TestOAuthUrl:
    """Tests for getting OAuth authorization URL."""

    def test_get_oauth_url_google(self, account_handler, mock_request_handler):
        """Test getting Google OAuth URL."""
        result = account_handler._handle_oauth_url(mock_request_handler, {"provider": "google"})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "google" in body["auth_url"]
        assert "state" in body

    def test_get_oauth_url_github(self, account_handler, mock_request_handler):
        """Test getting GitHub OAuth URL."""
        result = account_handler._handle_oauth_url(mock_request_handler, {"provider": "github"})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "github" in body["auth_url"]

    def test_get_oauth_url_missing_provider(self, account_handler, mock_request_handler):
        """Test error when provider is missing."""
        result = account_handler._handle_oauth_url(mock_request_handler, {})

        assert result.status_code == 400
        assert b"Provider is required" in result.body

    def test_get_oauth_url_unsupported_provider(self, account_handler, mock_request_handler):
        """Test error for unsupported provider."""
        result = account_handler._handle_oauth_url(
            mock_request_handler, {"provider": "unsupported"}
        )

        assert result.status_code == 400
        assert b"Unsupported provider" in result.body

    def test_get_oauth_url_provider_case_insensitive(self, account_handler, mock_request_handler):
        """Test provider name is case insensitive."""
        result = account_handler._handle_oauth_url(mock_request_handler, {"provider": "GOOGLE"})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "google" in body["auth_url"]


# ===========================================================================
# Get User Providers Tests
# ===========================================================================


class TestGetUserProviders:
    """Tests for getting user's linked OAuth providers."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_user_providers(
        self, mock_extract, account_handler, mock_user_store, mock_request_handler
    ):
        """Test getting user's linked providers."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )

        result = account_handler._handle_get_user_providers(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        providers = body["providers"]

        assert len(providers) == 2
        provider_names = [p["provider"] for p in providers]
        assert "google" in provider_names
        assert "github" in provider_names

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_get_user_providers_no_linked(
        self, mock_extract, account_handler, mock_user_store, mock_request_handler
    ):
        """Test getting providers when user has none linked."""
        # Add user without OAuth links
        mock_user_store.users["user_2"] = MockUser(
            id="user_2", email="noauth@example.com", name="No Auth"
        )

        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_2",
            org_id="org_1",
            role="member",
        )

        result = account_handler._handle_get_user_providers(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["providers"] == []

    def test_get_user_providers_permission_denied(self, account_handler, mock_request_handler):
        """Test permission denied for getting user providers."""

        def mock_permission_denied(*args, **kwargs):
            return HandlerResult(
                status_code=403,
                body=b'{"error": "Permission denied"}',
                content_type="application/json",
            )

        account_handler._check_permission = mock_permission_denied

        result = account_handler._handle_get_user_providers(mock_request_handler)

        assert result.status_code == 403


# ===========================================================================
# Link Account Tests
# ===========================================================================


class TestLinkAccount:
    """Tests for linking OAuth account."""

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_google_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl.GOOGLE_CLIENT_ID", "google-client-id")
    @patch("aragora.server.handlers._oauth_impl.GOOGLE_AUTH_URL", "https://google.com/authorize")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_google_account(
        self,
        mock_extract,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking Google account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )
        mock_success_url.return_value = "https://example.com/success"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True

        mock_request_handler._json_body = {"provider": "google"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "google" in body["auth_url"]

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_github_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_CLIENT_ID", "github-client-id")
    @patch("aragora.server.handlers._oauth_impl.GITHUB_AUTH_URL", "https://github.com/login/oauth/authorize")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_github_account(
        self,
        mock_extract,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking GitHub account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )
        mock_success_url.return_value = "https://example.com/success"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True

        mock_request_handler._json_body = {"provider": "github"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "github" in body["auth_url"]

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_tenant")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.server.handlers._oauth_impl.MICROSOFT_AUTH_URL_TEMPLATE", "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_microsoft_account(
        self,
        mock_extract,
        mock_ms_client_id,
        mock_ms_tenant,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking Microsoft account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )
        mock_success_url.return_value = "https://example.com/success"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_ms_client_id.return_value = "microsoft-client-id"
        mock_ms_tenant.return_value = "common"

        mock_request_handler._json_body = {"provider": "microsoft"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "microsoftonline.com" in body["auth_url"]
        assert "common" in body["auth_url"]

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_apple_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_apple_client_id")
    @patch("aragora.server.handlers._oauth_impl.APPLE_AUTH_URL", "https://appleid.apple.com/auth/authorize")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_apple_account(
        self,
        mock_extract,
        mock_apple_client_id,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking Apple account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )
        mock_success_url.return_value = "https://example.com/success"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_apple_client_id.return_value = "apple-client-id"

        mock_request_handler._json_body = {"provider": "apple"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "appleid.apple.com" in body["auth_url"]
        assert "form_post" in body["auth_url"]

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_redirect_uri")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_oidc_account(
        self,
        mock_extract,
        mock_oidc_issuer,
        mock_oidc_client_id,
        mock_redirect_uri,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking OIDC account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )
        mock_success_url.return_value = "https://example.com/success"
        mock_redirect_uri.return_value = "https://example.com/callback"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_oidc_issuer.return_value = "https://idp.example.com"
        mock_oidc_client_id.return_value = "oidc-client-id"

        # Mock the _get_oidc_discovery method on the handler instance
        account_handler._get_oidc_discovery = MagicMock(
            return_value={"authorization_endpoint": "https://idp.example.com/authorize"}
        )

        mock_request_handler._json_body = {"provider": "oidc"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "auth_url" in body
        assert "idp.example.com" in body["auth_url"]

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_microsoft_client_id")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_microsoft_not_configured(
        self,
        mock_extract,
        mock_ms_client_id,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking Microsoft account when not configured returns 503."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_ms_client_id.return_value = None

        mock_request_handler._json_body = {"provider": "microsoft"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 503
        assert b"Microsoft OAuth not configured" in result.body

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._generate_state")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_client_id")
    @patch("aragora.server.handlers._oauth_impl._get_oidc_issuer")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_oidc_not_configured(
        self,
        mock_extract,
        mock_oidc_issuer,
        mock_oidc_client_id,
        mock_success_url,
        mock_generate_state,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test linking OIDC account when not configured returns 503."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")
        mock_success_url.return_value = "https://example.com/success"
        mock_generate_state.return_value = "test-state"
        mock_validate_redirect.return_value = True
        mock_oidc_issuer.return_value = None
        mock_oidc_client_id.return_value = None

        mock_request_handler._json_body = {"provider": "oidc"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 503
        assert b"OIDC provider not configured" in result.body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_account_invalid_json(self, mock_extract, account_handler, mock_request_handler):
        """Test link account with invalid JSON body."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")

        # No JSON body
        mock_request_handler._json_body = None

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 400
        assert b"Invalid JSON body" in result.body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_account_unsupported_provider(
        self, mock_extract, account_handler, mock_request_handler
    ):
        """Test link account with unsupported provider."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")

        mock_request_handler._json_body = {"provider": "unsupported"}

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 400
        assert b"Unsupported provider" in result.body

    @patch("aragora.server.handlers._oauth_impl._validate_redirect_url")
    @patch("aragora.server.handlers._oauth_impl._get_oauth_success_url")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_link_account_invalid_redirect(
        self,
        mock_extract,
        mock_success_url,
        mock_validate_redirect,
        account_handler,
        mock_request_handler,
    ):
        """Test link account rejects invalid redirect URL."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")
        mock_success_url.return_value = "https://example.com/success"
        mock_validate_redirect.return_value = False

        mock_request_handler._json_body = {
            "provider": "google",
            "redirect_url": "https://evil.com",
        }

        result = account_handler._handle_link_account(mock_request_handler)

        assert result.status_code == 400
        assert b"Invalid redirect URL" in result.body


# ===========================================================================
# Unlink Account Tests
# ===========================================================================


class TestUnlinkAccount:
    """Tests for unlinking OAuth account."""

    @patch("aragora.server.handlers._oauth.account.audit_action")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unlink_oauth_account(
        self, mock_extract, mock_audit, account_handler, mock_user_store, mock_request_handler
    ):
        """Test unlinking OAuth account."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="user_1",
            org_id="org_1",
            role="member",
        )

        mock_request_handler._json_body = {"provider": "google"}

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "Unlinked google" in body["message"]

        # Verify Google was unlinked
        assert "google" not in mock_user_store.oauth_links.get("user_1", {})

        # Verify audit was called (patched at import location)
        mock_audit.assert_called_once()

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unlink_account_invalid_json(self, mock_extract, account_handler, mock_request_handler):
        """Test unlink account with invalid JSON body."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")

        mock_request_handler._json_body = None

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 400
        assert b"Invalid JSON body" in result.body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unlink_account_unsupported_provider(
        self, mock_extract, account_handler, mock_request_handler
    ):
        """Test unlink account with unsupported provider."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id="user_1")

        mock_request_handler._json_body = {"provider": "unsupported"}

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 400
        assert b"Unsupported provider" in result.body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unlink_account_user_not_found(
        self, mock_extract, account_handler, mock_request_handler
    ):
        """Test unlink account when user not found."""
        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="nonexistent",
            org_id="org_1",
            role="member",
        )

        mock_request_handler._json_body = {"provider": "google"}

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 404
        assert b"User not found" in result.body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unlink_account_no_password_set(
        self, mock_extract, account_handler, mock_user_store, mock_request_handler
    ):
        """Test unlink account fails when user has no password (OAuth-only user)."""
        # Create user without password
        mock_user_store.users["oauth_only"] = MockUser(
            id="oauth_only",
            email="oauthonly@example.com",
            name="OAuth Only",
            password_hash=None,  # No password
        )
        mock_user_store.oauth_links["oauth_only"] = {"google": "google_789"}

        mock_extract.return_value = MagicMock(
            is_authenticated=True,
            user_id="oauth_only",
            org_id="org_1",
            role="member",
        )

        mock_request_handler._json_body = {"provider": "google"}

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 400
        assert b"no password set" in result.body

    def test_unlink_account_permission_denied(self, account_handler, mock_request_handler):
        """Test permission denied for unlinking account."""

        def mock_permission_denied(*args, **kwargs):
            return HandlerResult(
                status_code=403,
                body=b'{"error": "Permission denied"}',
                content_type="application/json",
            )

        account_handler._check_permission = mock_permission_denied

        result = account_handler._handle_unlink_account(mock_request_handler)

        assert result.status_code == 403


# ===========================================================================
# OAuth Callback API Tests
# ===========================================================================


class TestOAuthCallbackApi:
    """Tests for OAuth callback API endpoint."""

    def test_callback_api_missing_fields(self, account_handler, mock_request_handler):
        """Test callback API with missing required fields."""
        mock_request_handler._json_body = {"provider": "google"}  # Missing code and state

        result = account_handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 400
        assert b"provider, code, and state are required" in result.body

    def test_callback_api_invalid_json(self, account_handler, mock_request_handler):
        """Test callback API with invalid JSON."""
        mock_request_handler._json_body = None

        result = account_handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 400
        assert b"Invalid JSON body" in result.body

    def test_callback_api_unsupported_provider(self, account_handler, mock_request_handler):
        """Test callback API with unsupported provider."""
        mock_request_handler._json_body = {
            "provider": "unsupported",
            "code": "test-code",
            "state": "test-state",
        }

        result = account_handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 400
        assert b"Unsupported provider" in result.body

    def test_callback_api_extracts_tokens_from_fragment(self, mock_request_handler):
        """Test callback API extracts tokens from URL fragment (#) - security best practice."""
        # Create handler with callback that returns tokens in fragment
        handler = AccountManagementTestHandler()

        def mock_google_callback(h, q):
            # Tokens in fragment (#) - this is the secure way
            return HandlerResult(
                status_code=302,
                headers={
                    "Location": "https://app.example.com/callback#access_token=frag-token&refresh_token=frag-refresh&token_type=Bearer&expires_in=3600"
                },
                body=b"",
                content_type="text/html",
            )

        handler._handle_google_callback = mock_google_callback

        mock_request_handler._json_body = {
            "provider": "google",
            "code": "test-code",
            "state": "test-state",
        }

        result = handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 200
        response = json.loads(result.body)
        assert response["access_token"] == "frag-token"
        assert response["refresh_token"] == "frag-refresh"
        assert response["token_type"] == "Bearer"
        assert response["expires_in"] == 3600

    def test_callback_api_falls_back_to_query_params(self, mock_request_handler):
        """Test callback API falls back to query params (?) for backward compatibility."""
        # Create handler with callback that returns tokens in query params
        handler = AccountManagementTestHandler()

        def mock_google_callback(h, q):
            # Tokens in query params (?) - legacy behavior
            return HandlerResult(
                status_code=302,
                headers={
                    "Location": "https://app.example.com/callback?access_token=query-token&refresh_token=query-refresh&token_type=Bearer&expires_in=7200"
                },
                body=b"",
                content_type="text/html",
            )

        handler._handle_google_callback = mock_google_callback

        mock_request_handler._json_body = {
            "provider": "google",
            "code": "test-code",
            "state": "test-state",
        }

        result = handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 200
        response = json.loads(result.body)
        assert response["access_token"] == "query-token"
        assert response["refresh_token"] == "query-refresh"

    def test_callback_api_prefers_fragment_over_query(self, mock_request_handler):
        """Test that fragment takes precedence over query params when both present."""
        handler = AccountManagementTestHandler()

        def mock_google_callback(h, q):
            # Both fragment and query params present - fragment should win
            return HandlerResult(
                status_code=302,
                headers={
                    "Location": "https://app.example.com/callback?access_token=query-token#access_token=fragment-token&refresh_token=fragment-refresh"
                },
                body=b"",
                content_type="text/html",
            )

        handler._handle_google_callback = mock_google_callback

        mock_request_handler._json_body = {
            "provider": "google",
            "code": "test-code",
            "state": "test-state",
        }

        result = handler._handle_oauth_callback_api(mock_request_handler)

        assert result.status_code == 200
        response = json.loads(result.body)
        # Fragment should take precedence
        assert response["access_token"] == "fragment-token"
