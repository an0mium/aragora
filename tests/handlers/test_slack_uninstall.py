"""Tests for Slack workspace uninstall handler."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


class TestSlackUninstallHandler:
    """Tests for the Slack app_uninstalled event handler."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""

        class MockRequest:
            def __init__(self, data: dict):
                self._data = data

            async def body(self) -> bytes:
                return json.dumps(self._data).encode()

        return MockRequest

    @pytest.fixture
    def mock_workspace_store(self):
        """Create a mock workspace store."""
        store = MagicMock()
        store.revoke_token = MagicMock(return_value=True)
        store.deactivate = MagicMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_app_uninstalled_event_revokes_token(self, mock_request, mock_workspace_store):
        """Test that app_uninstalled event revokes workspace tokens."""
        from aragora.server.handlers.bots.slack import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "team_id": "T12345",
                "event": {"type": "app_uninstalled"},
            }
        )

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handle_slack_events(request)

            # Should return success - HandlerResult has status_code attribute
            assert result.status_code == 200

            # Should have called revoke_token
            mock_workspace_store.revoke_token.assert_called_once_with("T12345")

    @pytest.mark.asyncio
    async def test_tokens_revoked_event_revokes_token(self, mock_request, mock_workspace_store):
        """Test that tokens_revoked event revokes workspace tokens."""
        from aragora.server.handlers.bots.slack import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "team_id": "T12345",
                "event": {"type": "tokens_revoked"},
            }
        )

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handle_slack_events(request)

            # Should return success
            assert result.status_code == 200

            # Should have called revoke_token
            mock_workspace_store.revoke_token.assert_called_once_with("T12345")

    @pytest.mark.asyncio
    async def test_url_verification_challenge(self, mock_request):
        """Test URL verification challenge response."""
        from aragora.server.handlers.bots.slack import handle_slack_events

        request = mock_request(
            {
                "type": "url_verification",
                "challenge": "test-challenge-token",
            }
        )

        result = await handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["challenge"] == "test-challenge-token"


class TestSlackWorkspaceStoreRevokeToken:
    """Tests for the revoke_token method."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Ensure ARAGORA_ENV is not production (can be leaked by other tests)."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        return str(tmp_path / "test_slack.db")

    def test_revoke_token_clears_sensitive_data(self, temp_db):
        """Test that revoke_token clears sensitive token data."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create a workspace with tokens
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-real-token",
            bot_user_id="U12345",
            installed_at=time.time(),
            refresh_token="xoxr-refresh-token",
            signing_secret="secret123",
            token_expires_at=time.time() + 3600,
            is_active=True,
        )
        store.save(workspace)

        # Revoke tokens
        result = store.revoke_token("T12345")
        assert result is True

        # Verify tokens are cleared
        revoked = store.get("T12345")
        assert revoked is not None
        assert revoked.access_token == "[REVOKED]"
        assert revoked.refresh_token is None
        assert revoked.signing_secret is None
        assert revoked.token_expires_at is None
        assert revoked.is_active is False

        # Workspace metadata should be preserved
        assert revoked.workspace_name == "Test Workspace"
        assert revoked.workspace_id == "T12345"

    def test_revoke_token_nonexistent_workspace(self, temp_db):
        """Test revoking tokens for nonexistent workspace."""
        from aragora.storage.slack_workspace_store import SlackWorkspaceStore

        store = SlackWorkspaceStore(temp_db)

        # Should succeed even if workspace doesn't exist (no-op)
        result = store.revoke_token("T_NONEXISTENT")
        assert result is True

    def test_revoke_token_already_revoked(self, temp_db):
        """Test revoking already revoked tokens."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create and revoke
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-token",
            bot_user_id="U12345",
            installed_at=time.time(),
        )
        store.save(workspace)
        store.revoke_token("T12345")

        # Revoke again should succeed
        result = store.revoke_token("T12345")
        assert result is True
