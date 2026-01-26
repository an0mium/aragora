"""
Tests for SlackOAuthHandler - Slack OAuth installation flow.

Tests cover:
- Install endpoint (redirect to Slack)
- OAuth callback (token exchange, workspace storage)
- Uninstall webhook handling
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

from aragora.server.handlers.social.slack_oauth import (
    SlackOAuthHandler,
    create_slack_oauth_handler,
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
    return SlackOAuthHandler(mock_server_context)


@pytest.fixture
def oauth_state_store():
    """Create an in-memory OAuth state store for tests."""
    # Reset global state store to avoid Redis connection attempts
    reset_global_oauth_state_store()
    store = InMemoryOAuthStateStore()
    with patch("aragora.server.handlers.social.slack_oauth._get_state_store", return_value=store):
        with patch.dict("os.environ", {"REDIS_URL": ""}, clear=False):
            yield store
    # Clean up after test
    reset_global_oauth_state_store()


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


class TestSlackOAuthHandlerRouting:
    """Tests for request routing."""

    def test_can_handle_install(self, oauth_handler):
        """Test can_handle for install endpoint."""
        assert oauth_handler.can_handle("/api/integrations/slack/install") is True

    def test_can_handle_callback(self, oauth_handler):
        """Test can_handle for callback endpoint."""
        assert oauth_handler.can_handle("/api/integrations/slack/callback") is True

    def test_can_handle_uninstall(self, oauth_handler):
        """Test can_handle for uninstall endpoint."""
        assert oauth_handler.can_handle("/api/integrations/slack/uninstall") is True

    def test_cannot_handle_other_paths(self, oauth_handler):
        """Test can_handle returns False for other paths."""
        assert oauth_handler.can_handle("/api/slack/install") is False
        assert oauth_handler.can_handle("/api/v2/slack/oauth") is False

    def test_routes_attribute(self, oauth_handler):
        """Test ROUTES includes all endpoints."""
        assert "/api/integrations/slack/install" in oauth_handler.ROUTES
        assert "/api/integrations/slack/callback" in oauth_handler.ROUTES
        assert "/api/integrations/slack/uninstall" in oauth_handler.ROUTES


# ===========================================================================
# Install Endpoint Tests
# ===========================================================================


class TestSlackOAuthInstall:
    """Tests for OAuth install endpoint."""

    @pytest.mark.asyncio
    async def test_install_no_client_id(self, oauth_handler):
        """Test install without SLACK_CLIENT_ID configured."""
        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", ""):
            result = await oauth_handler.handle("GET", "/api/integrations/slack/install")

        assert result.status_code == 503
        data = parse_handler_response(result)
        assert "not configured" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_install_redirect(self, oauth_handler, oauth_state_store):
        """Test install redirects to Slack OAuth."""
        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle("GET", "/api/integrations/slack/install")

        assert result.status_code == 302
        assert "Location" in result.headers
        assert "slack.com/oauth" in result.headers["Location"]
        assert "client_id=test-client-id" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_generates_state(self, oauth_handler, oauth_state_store):
        """Test install generates state token."""
        initial_count = len(oauth_state_store._states)

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle("GET", "/api/integrations/slack/install")

        assert len(oauth_state_store._states) == initial_count + 1
        assert "state=" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_with_tenant_id(self, oauth_handler, oauth_state_store):
        """Test install stores tenant_id in state."""
        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "test-client-id"):
            result = await oauth_handler.handle(
                "GET",
                "/api/integrations/slack/install",
                query_params={"tenant_id": "tenant-001"},
            )

        # Find the new state
        for state, data in oauth_state_store._states.items():
            if data.metadata and data.metadata.get("tenant_id") == "tenant-001":
                assert True
                return

        pytest.fail("tenant_id not stored in state")

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

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "test-client-id"):
            await oauth_handler.handle("GET", "/api/integrations/slack/install")

        assert old_state not in oauth_state_store._states

    @pytest.mark.asyncio
    async def test_install_method_not_allowed(self, oauth_handler):
        """Test install rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/slack/install")

        assert result.status_code == 405


# ===========================================================================
# Callback Endpoint Tests
# ===========================================================================


class TestSlackOAuthCallback:
    """Tests for OAuth callback endpoint."""

    @pytest.mark.asyncio
    async def test_callback_error_from_slack(self, oauth_handler):
        """Test callback handles error from Slack."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/slack/callback",
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
            "/api/integrations/slack/callback",
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
            "/api/integrations/slack/callback",
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
            "/api/integrations/slack/callback",
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

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_SECRET", ""):
                result = await oauth_handler.handle(
                    "GET",
                    "/api/integrations/slack/callback",
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

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Network error")
                    )
                    result = await oauth_handler.handle(
                        "GET",
                        "/api/integrations/slack/callback",
                        query_params={"code": "auth-code", "state": state},
                    )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_callback_slack_oauth_error(self, oauth_handler, oauth_state_store):
        """Test callback handles Slack OAuth error response."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata=None,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "invalid_code"}
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        return_value=mock_response
                    )
                    result = await oauth_handler.handle(
                        "GET",
                        "/api/integrations/slack/callback",
                        query_params={"code": "auth-code", "state": state},
                    )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "invalid_code" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_callback_success(self, oauth_handler, oauth_state_store):
        """Test successful callback stores workspace."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata={"tenant_id": "tenant-001"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "access_token": "xoxb-test-token",
            "team": {"id": "T12345678", "name": "Test Workspace"},
            "bot_user_id": "U87654321",
            "authed_user": {"id": "U11111111"},
            "scope": "channels:history,chat:write",
        }
        mock_response.raise_for_status = MagicMock()

        mock_store = MagicMock()
        mock_store.save.return_value = True

        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        return_value=mock_response
                    )
                    with patch(
                        "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                        return_value=mock_store,
                    ):
                        result = await oauth_handler.handle(
                            "GET",
                            "/api/integrations/slack/callback",
                            query_params={"code": "auth-code", "state": state},
                        )

        assert result.status_code == 200
        assert result.content_type == "text/html"
        assert b"Connected" in result.body
        assert b"Test Workspace" in result.body
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed(self, oauth_handler):
        """Test callback rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/slack/callback")

        assert result.status_code == 405


# ===========================================================================
# Uninstall Endpoint Tests
# ===========================================================================


class TestSlackOAuthUninstall:
    """Tests for uninstall webhook endpoint."""

    @pytest.mark.asyncio
    async def test_uninstall_app_uninstalled_event(self, oauth_handler):
        """Test uninstall handles app_uninstalled event."""
        mock_store = MagicMock()

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
        ):
            result = await oauth_handler.handle(
                "POST",
                "/api/integrations/slack/uninstall",
                body={
                    "team_id": "T12345678",
                    "event": {"type": "app_uninstalled", "team_id": "T12345678"},
                },
            )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data.get("ok") is True
        mock_store.deactivate.assert_called_once_with("T12345678")

    @pytest.mark.asyncio
    async def test_uninstall_tokens_revoked_event(self, oauth_handler):
        """Test uninstall handles tokens_revoked event."""
        mock_store = MagicMock()

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
        ):
            result = await oauth_handler.handle(
                "POST",
                "/api/integrations/slack/uninstall",
                body={
                    "team_id": "T12345678",
                    "event": {
                        "type": "tokens_revoked",
                        "tokens": {"bot": ["xoxb-token"]},
                    },
                },
            )

        assert result.status_code == 200
        mock_store.deactivate.assert_called_once_with("T12345678")

    @pytest.mark.asyncio
    async def test_uninstall_unknown_event(self, oauth_handler):
        """Test uninstall handles unknown event type."""
        result = await oauth_handler.handle(
            "POST",
            "/api/integrations/slack/uninstall",
            body={"event": {"type": "unknown_event"}},
        )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data.get("ok") is True

    @pytest.mark.asyncio
    async def test_uninstall_store_unavailable(self, oauth_handler):
        """Test uninstall handles store import error gracefully."""
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            side_effect=ImportError("Store not available"),
        ):
            result = await oauth_handler.handle(
                "POST",
                "/api/integrations/slack/uninstall",
                body={
                    "team_id": "T12345678",
                    "event": {"type": "app_uninstalled"},
                },
            )

        assert result.status_code == 200  # Still acknowledges event

    @pytest.mark.asyncio
    async def test_uninstall_method_not_allowed(self, oauth_handler):
        """Test uninstall rejects non-POST methods."""
        result = await oauth_handler.handle("GET", "/api/integrations/slack/uninstall")

        assert result.status_code == 405


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestSlackOAuthHandlerFactory:
    """Tests for handler factory function."""

    def test_create_slack_oauth_handler(self, mock_server_context):
        """Test factory creates handler."""
        handler = create_slack_oauth_handler(mock_server_context)

        assert isinstance(handler, SlackOAuthHandler)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestSlackOAuthHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_not_found(self, oauth_handler):
        """Test handle returns 404 for unknown path."""
        result = await oauth_handler.handle("GET", "/api/integrations/slack/unknown")

        assert result.status_code == 404


# ===========================================================================
# State Token Tests
# ===========================================================================


class TestSlackOAuthState:
    """Tests for OAuth state token handling."""

    @pytest.mark.asyncio
    async def test_state_consumed_after_callback(self, oauth_handler, oauth_state_store):
        """Test state token is consumed after successful callback."""
        state = "valid-state"
        oauth_state_store._states[state] = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
            created_at=time.time(),
            metadata=None,
        )

        # Make callback fail early but still consume state
        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_SECRET", "secret"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Error")
                    )
                    await oauth_handler.handle(
                        "GET",
                        "/api/integrations/slack/callback",
                        query_params={"code": "code", "state": state},
                    )

        # State should be consumed (removed)
        assert state not in oauth_state_store._states

    @pytest.mark.asyncio
    async def test_state_includes_timestamp(self, oauth_handler, oauth_state_store):
        """Test state includes creation timestamp."""
        with patch("aragora.server.handlers.social.slack_oauth.SLACK_CLIENT_ID", "id"):
            before = time.time()
            await oauth_handler.handle("GET", "/api/integrations/slack/install")
            after = time.time()

        for state, data in oauth_state_store._states.items():
            assert data.created_at
            assert before <= data.created_at <= after
