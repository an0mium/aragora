"""Tests for Teams token refresh scheduler."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.schedulers.teams_token_refresh import (
    RefreshResult,
    RefreshStats,
    TeamsTokenRefreshScheduler,
)


@dataclass
class MockTenant:
    """Mock tenant for testing."""

    tenant_id: str
    tenant_name: str
    refresh_token: str = "test-refresh-token"
    token_expires_at: float = 0.0


class MockTenantStore:
    """Mock tenant store for testing."""

    def __init__(self, tenants: Optional[List[MockTenant]] = None):
        self.tenants = tenants or []
        self.refresh_calls: List[str] = []
        self.should_fail_refresh: bool = False

    def get_expiring_tokens(self, hours: int = 2) -> List[MockTenant]:
        """Return tenants with expiring tokens."""
        return self.tenants

    def refresh_workspace_token(
        self, tenant_id: str, client_id: str, client_secret: str
    ) -> Optional[MockTenant]:
        """Mock token refresh."""
        self.refresh_calls.append(tenant_id)

        if self.should_fail_refresh:
            return None

        # Find and return the tenant
        for tenant in self.tenants:
            if tenant.tenant_id == tenant_id:
                return tenant
        return None


class TestTeamsTokenRefreshScheduler:
    """Tests for TeamsTokenRefreshScheduler."""

    def test_init_default_values(self):
        """Test scheduler initializes with default values."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(store)

        assert scheduler.interval_minutes == 30
        assert scheduler.expiry_window_hours == 2
        assert not scheduler.is_running
        assert scheduler.last_run is None
        assert scheduler.last_stats is None

    def test_init_custom_values(self):
        """Test scheduler initializes with custom values."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(
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
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(store, client_id="", client_secret="")

        await scheduler.start()

        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test scheduler starts and stops correctly."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(
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
        store = MockTenantStore(tenants=[])
        scheduler = TeamsTokenRefreshScheduler(
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
        tenants = [
            MockTenant("T001", "Tenant 1"),
            MockTenant("T002", "Tenant 2"),
        ]
        store = MockTenantStore(tenants=tenants)
        scheduler = TeamsTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        stats = await scheduler.refresh_now()

        assert stats.total_checked == 2
        assert stats.refreshed == 2
        assert stats.failed == 0
        assert len(store.refresh_calls) == 2
        assert "T001" in store.refresh_calls
        assert "T002" in store.refresh_calls

    @pytest.mark.asyncio
    async def test_refresh_now_with_failures(self):
        """Test manual refresh when some tokens fail to refresh."""
        tenants = [
            MockTenant("T001", "Tenant 1"),
            MockTenant("T002", "Tenant 2"),
        ]
        store = MockTenantStore(tenants=tenants)
        store.should_fail_refresh = True

        scheduler = TeamsTokenRefreshScheduler(
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
        tenants = [MockTenant("T001", "Tenant 1")]
        store = MockTenantStore(tenants=tenants)
        store.should_fail_refresh = True

        failures: List[RefreshResult] = []

        def on_failure(result: RefreshResult):
            failures.append(result)

        scheduler = TeamsTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
            on_refresh_failure=on_failure,
        )

        await scheduler.refresh_now()

        assert len(failures) == 1
        assert failures[0].tenant_id == "T001"
        assert not failures[0].success

    def test_get_status(self):
        """Test get_status returns correct information."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(
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
        tenants = [MockTenant("T001", "Tenant 1")]
        store = MockTenantStore(tenants=tenants)
        scheduler = TeamsTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        await scheduler.refresh_now()

        status = scheduler.get_status()

        assert status["last_run"] is not None
        assert status["last_stats"]["total_checked"] == 1
        assert status["last_stats"]["refreshed"] == 1

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test that starting an already running scheduler logs warning."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
            interval_minutes=1,
        )

        await scheduler.start()
        assert scheduler.is_running

        # Start again should not create duplicate task
        await scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test that stopping a non-running scheduler is safe."""
        store = MockTenantStore()
        scheduler = TeamsTokenRefreshScheduler(
            store,
            client_id="test-client",
            client_secret="test-secret",
        )

        # Should not raise
        await scheduler.stop()
        assert not scheduler.is_running


class TestRefreshResult:
    """Tests for RefreshResult dataclass."""

    def test_success_result(self):
        """Test creating a successful refresh result."""
        result = RefreshResult(
            tenant_id="T001",
            tenant_name="Test Tenant",
            success=True,
        )

        assert result.tenant_id == "T001"
        assert result.tenant_name == "Test Tenant"
        assert result.success is True
        assert result.error is None
        assert result.timestamp is not None

    def test_failure_result(self):
        """Test creating a failed refresh result."""
        result = RefreshResult(
            tenant_id="T001",
            tenant_name="Test Tenant",
            success=False,
            error="Token revoked",
        )

        assert result.success is False
        assert result.error == "Token revoked"

    def test_result_has_timestamp(self):
        """Test that result has a UTC timestamp."""
        result = RefreshResult(
            tenant_id="T001",
            tenant_name="Test Tenant",
            success=True,
        )

        assert result.timestamp is not None
        assert result.timestamp.tzinfo == timezone.utc


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
            RefreshResult("T001", "Tenant1", True),
            RefreshResult("T002", "Tenant2", False, "Error"),
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

    def test_stats_default_factory(self):
        """Test that results list is not shared between instances."""
        stats1 = RefreshStats()
        stats2 = RefreshStats()

        stats1.results.append(RefreshResult("T001", "Tenant1", True))

        assert len(stats1.results) == 1
        assert len(stats2.results) == 0
