"""Tests for Slack token refresh scheduler."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSlackTokenRefreshScheduler:
    """Tests for the Slack token refresh scheduler."""

    @pytest.fixture
    def mock_workspace_store(self):
        """Create a mock workspace store."""
        store = MagicMock()
        store.get_expiring_tokens = MagicMock(return_value=[])
        store.refresh_workspace_token = AsyncMock(return_value=None)
        return store

    @pytest.fixture
    def mock_workspace(self):
        """Create a mock workspace with expiring token."""
        workspace = MagicMock()
        workspace.workspace_id = "T12345"
        workspace.workspace_name = "Test Workspace"
        workspace.access_token = "xoxb-test-token"
        workspace.refresh_token = "xoxr-refresh-token"
        workspace.token_expires_at = time.time() + 1800  # 30 minutes from now
        return workspace

    @pytest.mark.asyncio
    async def test_scheduler_not_started_without_credentials(self):
        """Test scheduler doesn't start without Slack credentials."""
        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        with patch.dict("os.environ", {}, clear=True):
            task = await init_slack_token_refresh_scheduler()
            assert task is None

    @pytest.mark.asyncio
    async def test_scheduler_not_started_without_client_secret(self):
        """Test scheduler doesn't start without client secret."""
        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        with patch.dict("os.environ", {"SLACK_CLIENT_ID": "test-id"}, clear=True):
            task = await init_slack_token_refresh_scheduler()
            assert task is None

    @pytest.mark.asyncio
    async def test_scheduler_starts_with_credentials(self, mock_workspace_store):
        """Test scheduler starts when credentials are provided."""
        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        with (
            patch.dict(
                "os.environ",
                {"SLACK_CLIENT_ID": "test-id", "SLACK_CLIENT_SECRET": "test-secret"},
            ),
            patch(
                "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                return_value=mock_workspace_store,
            ),
        ):
            task = await init_slack_token_refresh_scheduler()

            # Task should be created
            assert task is not None
            assert isinstance(task, asyncio.Task)

            # Cancel to clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_scheduler_refreshes_expiring_tokens(self, mock_workspace_store, mock_workspace):
        """Test scheduler refreshes tokens that are expiring."""
        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        # Return the mock workspace as expiring
        mock_workspace_store.get_expiring_tokens.return_value = [mock_workspace]
        mock_workspace_store.refresh_workspace_token = AsyncMock(return_value=mock_workspace)

        with (
            patch.dict(
                "os.environ",
                {"SLACK_CLIENT_ID": "test-id", "SLACK_CLIENT_SECRET": "test-secret"},
            ),
            patch(
                "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                return_value=mock_workspace_store,
            ),
        ):
            task = await init_slack_token_refresh_scheduler()
            assert task is not None

            # Let the scheduler run one iteration
            await asyncio.sleep(0.1)

            # Verify get_expiring_tokens was called
            mock_workspace_store.get_expiring_tokens.assert_called()

            # Clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestSlackWorkspaceStoreExpiringTokens:
    """Tests for the get_expiring_tokens method."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        return str(tmp_path / "test_slack.db")

    def test_get_expiring_tokens_empty(self, temp_db):
        """Test getting expiring tokens from empty store."""
        from aragora.storage.slack_workspace_store import SlackWorkspaceStore

        store = SlackWorkspaceStore(temp_db)
        tokens = store.get_expiring_tokens(hours=1)
        assert tokens == []

    def test_get_expiring_tokens_finds_expiring(self, temp_db):
        """Test finding tokens that are about to expire."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create a workspace with token expiring in 30 minutes
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-test-token",
            bot_user_id="U12345",
            installed_at=time.time(),
            refresh_token="xoxr-refresh-token",
            token_expires_at=time.time() + 1800,  # 30 minutes from now
        )
        store.save(workspace)

        # Should find the token when looking 1 hour ahead
        expiring = store.get_expiring_tokens(hours=1)
        assert len(expiring) == 1
        assert expiring[0].workspace_id == "T12345"

    def test_get_expiring_tokens_ignores_non_expiring(self, temp_db):
        """Test that tokens not expiring soon are not returned."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create a workspace with token expiring in 3 hours
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-test-token",
            bot_user_id="U12345",
            installed_at=time.time(),
            refresh_token="xoxr-refresh-token",
            token_expires_at=time.time() + 10800,  # 3 hours from now
        )
        store.save(workspace)

        # Should not find when looking only 1 hour ahead
        expiring = store.get_expiring_tokens(hours=1)
        assert len(expiring) == 0

    def test_get_expiring_tokens_requires_refresh_token(self, temp_db):
        """Test that workspaces without refresh tokens are not returned."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create a workspace without refresh token
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-test-token",
            bot_user_id="U12345",
            installed_at=time.time(),
            refresh_token=None,  # No refresh token
            token_expires_at=time.time() + 1800,
        )
        store.save(workspace)

        # Should not find - can't refresh without refresh token
        expiring = store.get_expiring_tokens(hours=1)
        assert len(expiring) == 0

    def test_get_expiring_tokens_requires_active(self, temp_db):
        """Test that inactive workspaces are not returned."""
        from aragora.storage.slack_workspace_store import (
            SlackWorkspace,
            SlackWorkspaceStore,
        )

        store = SlackWorkspaceStore(temp_db)

        # Create an inactive workspace
        workspace = SlackWorkspace(
            workspace_id="T12345",
            workspace_name="Test Workspace",
            access_token="xoxb-test-token",
            bot_user_id="U12345",
            installed_at=time.time(),
            refresh_token="xoxr-refresh-token",
            token_expires_at=time.time() + 1800,
            is_active=False,  # Inactive
        )
        store.save(workspace)

        # Should not find - workspace is inactive
        expiring = store.get_expiring_tokens(hours=1)
        assert len(expiring) == 0
