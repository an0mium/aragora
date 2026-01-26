"""
Tests for TeamsOAuthHandler - Microsoft Teams OAuth installation flow.

Tests cover:
- Install endpoint (redirect to Microsoft)
- OAuth callback (token exchange, tenant storage)
- Refresh endpoint (token refresh)
- State token CSRF protection
- Error handling
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.teams_oauth import (
    TeamsOAuthHandler,
    create_teams_oauth_handler,
)
from aragora.server.oauth_state_store import (
    InMemoryOAuthStateStore,
    OAuthState,
    reset_oauth_state_store as reset_global_oauth_state_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context():
    """Create a mock server context."""
    return MagicMock()


@pytest.fixture
def oauth_handler(mock_server_context):
    """Create an OAuth handler for testing."""
    return TeamsOAuthHandler(mock_server_context)


@pytest.fixture
def oauth_state_store():
    """Create an in-memory OAuth state store for tests."""
    store = InMemoryOAuthStateStore()
    with patch("aragora.server.handlers.social.teams_oauth._get_state_store", return_value=store):
        yield store


@pytest.fixture(autouse=True)
def reset_oauth_state_store(oauth_state_store):
    """Reset OAuth states after tests."""
    oauth_state_store._states.clear()
    yield
    oauth_state_store._states.clear()


def parse_handler_response(result) -> Dict[str, Any]:
    """Parse handler result body as JSON."""
    if hasattr(result, "body") and result.body:
        body = result.body
        if isinstance(body, bytes):
            try:
                return json.loads(body.decode())
            except json.JSONDecodeError:
                return {}
        return json.loads(body) if body else {}
    return {}


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestTeamsOAuthHandlerRouting:
    """Tests for request routing."""

    def test_can_handle_install(self, oauth_handler):
        """Test can_handle for install endpoint."""
        assert oauth_handler.can_handle("/api/integrations/teams/install") is True

    def test_can_handle_callback(self, oauth_handler):
        """Test can_handle for callback endpoint."""
        assert oauth_handler.can_handle("/api/integrations/teams/callback") is True

    def test_can_handle_refresh(self, oauth_handler):
        """Test can_handle for refresh endpoint."""
        assert oauth_handler.can_handle("/api/integrations/teams/refresh") is True

    def test_cannot_handle_other_paths(self, oauth_handler):
        """Test can_handle returns False for other paths."""
        assert oauth_handler.can_handle("/api/teams/install") is False
        assert oauth_handler.can_handle("/api/v2/teams/oauth") is False

    def test_routes_attribute(self, oauth_handler):
        """Test ROUTES includes all endpoints."""
        assert "/api/integrations/teams/install" in oauth_handler.ROUTES
        assert "/api/integrations/teams/callback" in oauth_handler.ROUTES
        assert "/api/integrations/teams/refresh" in oauth_handler.ROUTES


# ===========================================================================
# Install Endpoint Tests
# ===========================================================================


class TestTeamsOAuthInstall:
    """Tests for OAuth install endpoint."""

    @pytest.mark.asyncio
    async def test_install_no_client_id(self, oauth_handler):
        """Test install without TEAMS_CLIENT_ID configured."""
        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", ""):
            result = await oauth_handler.handle("GET", "/api/integrations/teams/install")

        assert result.status_code == 503
        data = parse_handler_response(result)
        assert "not configured" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_install_redirect(self, oauth_handler, oauth_state_store):
        """Test install redirects to Microsoft OAuth."""
        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle("GET", "/api/integrations/teams/install")

        assert result.status_code == 302
        assert "Location" in result.headers
        assert "login.microsoftonline.com" in result.headers["Location"]
        assert "client_id=test-client-id" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_generates_state(self, oauth_handler, oauth_state_store):
        """Test install generates state token."""
        initial_count = len(oauth_state_store._states)

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle("GET", "/api/integrations/teams/install")

        assert len(oauth_state_store._states) == initial_count + 1
        assert "state=" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_with_org_id(self, oauth_handler, oauth_state_store):
        """Test install stores org_id in state."""
        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle(
                "GET",
                "/api/integrations/teams/install",
                query_params={"org_id": "org-001"},
            )

        # Find the new state
        for state, data in oauth_state_store._states.items():
            if data.metadata and data.metadata.get("org_id") == "org-001":
                assert True
                return

        pytest.fail("org_id not stored in state")

    @pytest.mark.asyncio
    async def test_install_cleans_old_states(self, oauth_handler, oauth_state_store):
        """Test install cleans up expired states."""
        # Add old state
        old_state = "old-state-token"
        oauth_state_store._states[old_state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() - 10,
            created_at=time.time() - 700,
            metadata=None,
        )

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "test-client-id"):
            await oauth_handler.handle("GET", "/api/integrations/teams/install")

        assert old_state not in oauth_state_store._states

    @pytest.mark.asyncio
    async def test_install_method_not_allowed(self, oauth_handler):
        """Test install rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/teams/install")

        assert result.status_code == 405


# ===========================================================================
# Callback Endpoint Tests
# ===========================================================================


class TestTeamsOAuthCallback:
    """Tests for OAuth callback endpoint."""

    @pytest.mark.asyncio
    async def test_callback_error_from_microsoft(self, oauth_handler):
        """Test callback handles error from Microsoft."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/teams/callback",
            query_params={"error": "access_denied"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "denied" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, oauth_handler):
        """Test callback requires authorization code."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/teams/callback",
            query_params={"state": "some-state"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "code" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_missing_state(self, oauth_handler):
        """Test callback requires state parameter."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/teams/callback",
            query_params={"code": "auth-code"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "state" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, oauth_handler, oauth_state_store):
        """Test callback rejects invalid state token."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/teams/callback",
            query_params={"code": "auth-code", "state": "invalid-state"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert (
            "expired" in data.get("error", "").lower() or "invalid" in data.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_callback_no_client_secret(self, oauth_handler, oauth_state_store):
        """Test callback fails without client secret."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata=None,
        )

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", ""):
                result = await oauth_handler.handle(
                    "GET",
                    "/api/integrations/teams/callback",
                    query_params={"code": "auth-code", "state": state},
                )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_callback_token_exchange_error(self, oauth_handler, oauth_state_store):
        """Test callback handles token exchange errors."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata=None,
        )

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Network error")
                    )
                    result = await oauth_handler.handle(
                        "GET",
                        "/api/integrations/teams/callback",
                        query_params={"code": "auth-code", "state": state},
                    )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_callback_success(self, oauth_handler, oauth_state_store):
        """Test successful callback stores tenant."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata={"org_id": "org-001"},
        )

        mock_token_response = MagicMock()
        mock_token_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
            "scope": "https://graph.microsoft.com/.default",
        }
        mock_token_response.raise_for_status = MagicMock()

        mock_org_response = MagicMock()
        mock_org_response.status_code = 200
        mock_org_response.json.return_value = {
            "value": [{"id": "tenant-001", "displayName": "Test Tenant"}]
        }

        mock_me_response = MagicMock()
        mock_me_response.status_code = 200
        mock_me_response.json.return_value = {"id": "bot-001"}

        mock_store = MagicMock()
        mock_store.save.return_value = True

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_instance = MagicMock()
                    mock_instance.post = AsyncMock(return_value=mock_token_response)
                    mock_instance.get = AsyncMock(side_effect=[mock_org_response, mock_me_response])
                    mock_client.return_value.__aenter__.return_value = mock_instance
                    with patch(
                        "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                        return_value=mock_store,
                    ):
                        result = await oauth_handler.handle(
                            "GET",
                            "/api/integrations/teams/callback",
                            query_params={"code": "auth-code", "state": state},
                        )

        assert result.status_code == 200
        assert result.content_type == "text/html"
        assert b"Connected" in result.body
        assert b"Test Tenant" in result.body
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed(self, oauth_handler):
        """Test callback rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/teams/callback")

        assert result.status_code == 405


# ===========================================================================
# Refresh Endpoint Tests
# ===========================================================================


class TestTeamsOAuthRefresh:
    """Tests for token refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_missing_tenant_id(self, oauth_handler):
        """Test refresh requires tenant_id."""
        result = await oauth_handler.handle(
            "POST",
            "/api/integrations/teams/refresh",
            body={},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "tenant_id" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_tenant_not_found(self, oauth_handler):
        """Test refresh handles tenant not found."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch(
                    "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                    return_value=mock_store,
                ):
                    result = await oauth_handler.handle(
                        "POST",
                        "/api/integrations/teams/refresh",
                        body={"tenant_id": "unknown-tenant"},
                    )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_refresh_no_refresh_token(self, oauth_handler):
        """Test refresh handles missing refresh token."""
        mock_tenant = MagicMock()
        mock_tenant.refresh_token = None

        mock_store = MagicMock()
        mock_store.get.return_value = mock_tenant

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch(
                    "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                    return_value=mock_store,
                ):
                    result = await oauth_handler.handle(
                        "POST",
                        "/api/integrations/teams/refresh",
                        body={"tenant_id": "tenant-001"},
                    )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "refresh" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_success(self, oauth_handler):
        """Test successful token refresh."""
        mock_tenant = MagicMock()
        mock_tenant.refresh_token = "old-refresh-token"

        mock_store = MagicMock()
        mock_store.get.return_value = mock_tenant
        mock_store.update_tokens.return_value = True

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch(
                    "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                    return_value=mock_store,
                ):
                    with patch("httpx.AsyncClient") as mock_client:
                        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                            return_value=mock_response
                        )
                        result = await oauth_handler.handle(
                            "POST",
                            "/api/integrations/teams/refresh",
                            body={"tenant_id": "tenant-001"},
                        )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data.get("success") is True
        assert data.get("expires_in") == 3600
        mock_store.update_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_method_not_allowed(self, oauth_handler):
        """Test refresh rejects non-POST methods."""
        result = await oauth_handler.handle("GET", "/api/integrations/teams/refresh")

        assert result.status_code == 405


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestTeamsOAuthHandlerFactory:
    """Tests for handler factory function."""

    def test_create_teams_oauth_handler(self, mock_server_context):
        """Test factory creates handler."""
        handler = create_teams_oauth_handler(mock_server_context)

        assert isinstance(handler, TeamsOAuthHandler)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestTeamsOAuthHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_not_found(self, oauth_handler):
        """Test handle returns 404 for unknown path."""
        result = await oauth_handler.handle("GET", "/api/integrations/teams/unknown")

        assert result.status_code == 404


# ===========================================================================
# State Token Tests
# ===========================================================================


class TestTeamsOAuthState:
    """Tests for OAuth state token handling."""

    @pytest.mark.asyncio
    async def test_state_consumed_after_callback(self, oauth_handler, oauth_state_store):
        """Test state token is consumed after callback attempt."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata=None,
        )

        # Make callback fail early but still consume state
        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Error")
                    )
                    await oauth_handler.handle(
                        "GET",
                        "/api/integrations/teams/callback",
                        query_params={"code": "code", "state": state},
                    )

        # State should be consumed (removed)
        assert state not in oauth_state_store._states

    @pytest.mark.asyncio
    async def test_state_includes_timestamp(self, oauth_handler, oauth_state_store):
        """Test state includes creation timestamp."""
        with patch("aragora.server.handlers.social.teams_oauth.TEAMS_CLIENT_ID", "id"):
            before = time.time()
            await oauth_handler.handle("GET", "/api/integrations/teams/install")
            after = time.time()

        for state, data in oauth_state_store._states.items():
            assert data.created_at
            assert before <= data.created_at <= after
