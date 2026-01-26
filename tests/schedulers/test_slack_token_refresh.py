"""Tests for Slack token refresh scheduler."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.schedulers.slack_token_refresh import (
    RefreshResult,
    RefreshStats,
    SlackTokenRefreshScheduler,
)


@dataclass
class MockWorkspace:
    """Mock workspace for testing."""

    workspace_id: str
    workspace_name: str
    refresh_token: str = "xoxr-test-refresh-token"
    token_expires_at: float = 0.0


class MockWorkspaceStore:
    """Mock workspace store for testing."""

    def __init__(self, workspaces: Optional[List[MockWorkspace]] = None):
        self.workspaces = workspaces or []
        self.refresh_calls: List[str] = []
        self.should_fail_refresh: bool = False

    def get_expiring_tokens(self, hours: int = 2) -> List[MockWorkspace]:
        """Return workspaces with expiring tokens."""
        return self.workspaces

    def refresh_workspace_token(
        self, workspace_id: str, client_id: str, client_secret: str
    ) -> Optional[MockWorkspace]:
        """Mock token refresh."""
        self.refresh_calls.append(workspace_id)

        if self.should_fail_refresh:
            return None

        # Find and return the workspace
        for ws in self.workspaces:
            if ws.workspace_id == workspace_id:
                return ws
        return None


class TestSlackTokenRefreshScheduler:
    """Tests for SlackTokenRefreshScheduler."""

    def test_init_default_values(self):
        """Test scheduler initializes with default values."""
        store = MockWorkspaceStore()
        scheduler = SlackTokenRefreshScheduler(store)

        assert scheduler.interval_minutes == 30
        assert scheduler.expiry_window_hours == 2
        assert not scheduler.is_running
        assert scheduler.last_run is None
        assert scheduler.last_stats is None

    def test_init_custom_values(self):
        """Test scheduler initializes with custom values."""
        store = MockWorkspaceStore()
        scheduler = SlackTokenRefreshScheduler(
            store,
            interval_minutes=15,
            expiry_window_hours=4,
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

        assert scheduler.interval_minutes == 15
        assert scheduler.expiry_window_hours == 4
        assert scheduler.client_id == "test-client-id"
        assert scheduler.client_secret == "test-client-secret"

    @pytest.mark.asyncio
    async def test_start_without_credentials(self):
        """Test scheduler doesn't start without OAuth credentials."""
        store = MockWorkspaceStore()
        scheduler = SlackTokenRefreshScheduler(store, client_id="", client_secret="")

        await scheduler.start()

        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test scheduler starts and stops correctly."""
        store = MockWorkspaceStore()
        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
            interval_minutes=1,
        )

        await scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_refresh_now_no_expiring_tokens(self):
        """Test manual refresh with no expiring tokens."""
        store = MockWorkspaceStore(workspaces=[])
        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        stats = await scheduler.refresh_now()

        assert stats.total_checked == 0
        assert stats.refreshed == 0
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_refresh_now_with_expiring_tokens(self):
        """Test manual refresh with expiring tokens."""
        workspaces = [
            MockWorkspace("W001", "Workspace 1"),
            MockWorkspace("W002", "Workspace 2"),
        ]
        store = MockWorkspaceStore(workspaces=workspaces)
        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        stats = await scheduler.refresh_now()

        assert stats.total_checked == 2
        assert stats.refreshed == 2
        assert stats.failed == 0
        assert len(store.refresh_calls) == 2
        assert "W001" in store.refresh_calls
        assert "W002" in store.refresh_calls

    @pytest.mark.asyncio
    async def test_refresh_now_with_failures(self):
        """Test manual refresh when some tokens fail to refresh."""
        workspaces = [
            MockWorkspace("W001", "Workspace 1"),
            MockWorkspace("W002", "Workspace 2"),
        ]
        store = MockWorkspaceStore(workspaces=workspaces)
        store.should_fail_refresh = True

        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        stats = await scheduler.refresh_now()

        assert stats.total_checked == 2
        assert stats.refreshed == 0
        assert stats.failed == 2

    @pytest.mark.asyncio
    async def test_failure_callback_called(self):
        """Test that failure callback is called on refresh failures."""
        workspaces = [MockWorkspace("W001", "Workspace 1")]
        store = MockWorkspaceStore(workspaces=workspaces)
        store.should_fail_refresh = True

        failures: List[RefreshResult] = []

        def on_failure(result: RefreshResult):
            failures.append(result)

        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
            on_refresh_failure=on_failure,
        )

        await scheduler.refresh_now()

        assert len(failures) == 1
        assert failures[0].workspace_id == "W001"
        assert not failures[0].success

    def test_get_status(self):
        """Test get_status returns correct information."""
        store = MockWorkspaceStore()
        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
            interval_minutes=15,
            expiry_window_hours=4,
        )

        status = scheduler.get_status()

        assert status["running"] is False
        assert status["interval_minutes"] == 15
        assert status["expiry_window_hours"] == 4
        assert status["credentials_configured"] is True
        assert status["last_run"] is None
        assert status["last_stats"] is None

    @pytest.mark.asyncio
    async def test_get_status_after_refresh(self):
        """Test get_status after a refresh cycle."""
        workspaces = [MockWorkspace("W001", "Workspace 1")]
        store = MockWorkspaceStore(workspaces=workspaces)
        scheduler = SlackTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        await scheduler.refresh_now()

        status = scheduler.get_status()

        assert status["last_run"] is not None
        assert status["last_stats"]["total_checked"] == 1
        assert status["last_stats"]["refreshed"] == 1


class TestRefreshResult:
    """Tests for RefreshResult dataclass."""

    def test_success_result(self):
        """Test creating a successful refresh result."""
        result = RefreshResult(
            workspace_id="W001",
            workspace_name="Test Workspace",
            success=True,
        )

        assert result.workspace_id == "W001"
        assert result.workspace_name == "Test Workspace"
        assert result.success is True
        assert result.error is None
        assert result.timestamp is not None

    def test_failure_result(self):
        """Test creating a failed refresh result."""
        result = RefreshResult(
            workspace_id="W001",
            workspace_name="Test Workspace",
            success=False,
            error="Token revoked",
        )

        assert result.success is False
        assert result.error == "Token revoked"


class TestRefreshStats:
    """Tests for RefreshStats dataclass."""

    def test_empty_stats(self):
        """Test creating empty stats."""
        stats = RefreshStats()

        assert stats.total_checked == 0
        assert stats.refreshed == 0
        assert stats.failed == 0
        assert stats.skipped == 0
        assert len(stats.results) == 0

    def test_stats_with_results(self):
        """Test stats with results."""
        results = [
            RefreshResult("W001", "WS1", True),
            RefreshResult("W002", "WS2", False, "Error"),
        ]
        stats = RefreshStats(
            total_checked=2,
            refreshed=1,
            failed=1,
            results=results,
        )

        assert stats.total_checked == 2
        assert stats.refreshed == 1
        assert stats.failed == 1
        assert len(stats.results) == 2
