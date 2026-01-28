"""
Teams Token Refresh Scheduler.

Proactively refreshes Microsoft Teams OAuth tokens before they expire to prevent
silent authentication failures in production.

Features:
- Background async task that runs at configurable intervals
- Batches token refreshes to avoid rate limiting
- Notifies on refresh failures for operational awareness
- Graceful shutdown support
- Prometheus metrics for observability
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Protocol

logger = logging.getLogger(__name__)

# Prometheus metrics (lazy initialization to avoid import errors)
_metrics_initialized = False
_teams_token_refresh_total: Any = None
_teams_token_refresh_failures: Any = None
_teams_tenants_active: Any = None
_teams_refresh_duration: Any = None


def _init_metrics() -> bool:
    """Initialize Prometheus metrics if available."""
    global _metrics_initialized
    global _teams_token_refresh_total, _teams_token_refresh_failures
    global _teams_tenants_active, _teams_refresh_duration

    if _metrics_initialized:
        return _teams_token_refresh_total is not None

    _metrics_initialized = True

    try:
        from prometheus_client import Counter, Gauge, Histogram

        _teams_token_refresh_total = Counter(
            "aragora_teams_token_refresh_total",
            "Total number of Teams token refresh attempts",
            ["status"],
        )
        _teams_token_refresh_failures = Counter(
            "aragora_teams_token_refresh_failures_total",
            "Total number of Teams token refresh failures",
            ["error_type"],
        )
        _teams_tenants_active = Gauge(
            "aragora_teams_tenants_active",
            "Number of active Teams tenants",
        )
        _teams_refresh_duration = Histogram(
            "aragora_teams_token_refresh_duration_seconds",
            "Duration of Teams token refresh operations",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        logger.debug("Prometheus metrics initialized for Teams token refresh")
        return True
    except ImportError:
        logger.debug("prometheus_client not available, metrics disabled")
        return False


def _record_refresh_success() -> None:
    """Record a successful token refresh."""
    if _teams_token_refresh_total:
        _teams_token_refresh_total.labels(status="success").inc()


def _record_refresh_failure(error_type: str = "unknown") -> None:
    """Record a failed token refresh."""
    if _teams_token_refresh_total:
        _teams_token_refresh_total.labels(status="failure").inc()
    if _teams_token_refresh_failures:
        _teams_token_refresh_failures.labels(error_type=error_type).inc()


def _update_active_tenants(count: int) -> None:
    """Update the active tenants gauge."""
    if _teams_tenants_active:
        _teams_tenants_active.set(count)


# Configuration from environment
TEAMS_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", os.environ.get("TEAMS_CLIENT_ID", ""))
TEAMS_CLIENT_SECRET = os.environ.get(
    "AZURE_CLIENT_SECRET", os.environ.get("TEAMS_CLIENT_SECRET", "")
)
DEFAULT_REFRESH_INTERVAL_MINUTES = int(os.environ.get("TEAMS_TOKEN_REFRESH_INTERVAL", "30"))
DEFAULT_EXPIRY_WINDOW_HOURS = int(os.environ.get("TEAMS_TOKEN_EXPIRY_WINDOW", "2"))


class TenantStoreProtocol(Protocol):
    """Protocol for tenant store to allow dependency injection."""

    def get_expiring_tokens(self, hours: int) -> list:
        """Get tenants with tokens expiring within the specified hours."""
        ...

    def refresh_workspace_token(
        self, tenant_id: str, client_id: str, client_secret: str
    ) -> object | None:
        """Refresh a tenant's access token."""
        ...


@dataclass
class RefreshResult:
    """Result of a token refresh attempt."""

    tenant_id: str
    tenant_name: str
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RefreshStats:
    """Statistics for a refresh cycle."""

    total_checked: int = 0
    refreshed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[RefreshResult] = field(default_factory=list)


class TeamsTokenRefreshScheduler:
    """
    Proactively refresh Microsoft Teams OAuth tokens before expiry.

    Runs as a background async task that:
    1. Periodically checks for tokens expiring soon
    2. Refreshes tokens with Microsoft's OAuth API
    3. Notifies operators on failures

    Usage:
        scheduler = TeamsTokenRefreshScheduler(tenant_store)
        await scheduler.start()
        # ... later ...
        await scheduler.stop()
    """

    def __init__(
        self,
        store: TenantStoreProtocol,
        interval_minutes: int = DEFAULT_REFRESH_INTERVAL_MINUTES,
        expiry_window_hours: int = DEFAULT_EXPIRY_WINDOW_HOURS,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        on_refresh_failure: Optional[Callable[[RefreshResult], None]] = None,
    ):
        """
        Initialize the token refresh scheduler.

        Args:
            store: Tenant store for token operations
            interval_minutes: How often to check for expiring tokens
            expiry_window_hours: Refresh tokens expiring within this window
            client_id: Azure AD client ID (defaults to env var)
            client_secret: Azure AD client secret (defaults to env var)
            on_refresh_failure: Callback when a refresh fails
        """
        self.store = store
        self.interval_minutes = interval_minutes
        self.expiry_window_hours = expiry_window_hours
        self.client_id = client_id or TEAMS_CLIENT_ID
        self.client_secret = client_secret or TEAMS_CLIENT_SECRET
        self.on_refresh_failure = on_refresh_failure

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_run: Optional[datetime] = None
        self._last_stats: Optional[RefreshStats] = None

        # Initialize Prometheus metrics
        _init_metrics()

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is currently running."""
        return self._running and self._task is not None and not self._task.done()

    @property
    def last_run(self) -> Optional[datetime]:
        """Get the timestamp of the last refresh cycle."""
        return self._last_run

    @property
    def last_stats(self) -> Optional[RefreshStats]:
        """Get statistics from the last refresh cycle."""
        return self._last_stats

    async def start(self) -> None:
        """Start the background refresh loop."""
        if self.is_running:
            logger.warning("Teams token refresh scheduler is already running")
            return

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Azure AD credentials not configured. "
                "Set AZURE_CLIENT_ID and AZURE_CLIENT_SECRET to enable token refresh."
            )
            return

        self._running = True
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"Started Teams token refresh scheduler "
            f"(interval={self.interval_minutes}min, window={self.expiry_window_hours}h)"
        )

    async def stop(self) -> None:
        """Stop the background refresh loop gracefully."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped Teams token refresh scheduler")

    async def _refresh_loop(self) -> None:
        """Main refresh loop that runs in the background."""
        while self._running:
            try:
                stats = await self._refresh_expiring_tokens()
                self._last_run = datetime.now(timezone.utc)
                self._last_stats = stats

                if stats.failed > 0:
                    logger.warning(
                        f"Teams token refresh cycle completed with errors: "
                        f"{stats.refreshed} refreshed, {stats.failed} failed"
                    )
                elif stats.refreshed > 0:
                    logger.info(f"Teams token refresh cycle completed: {stats.refreshed} refreshed")

            except Exception as e:
                logger.error(f"Error in Teams token refresh cycle: {e}", exc_info=True)

            # Wait for next cycle
            await asyncio.sleep(self.interval_minutes * 60)

    async def _refresh_expiring_tokens(self) -> RefreshStats:
        """Check and refresh tokens expiring soon."""
        import time

        stats = RefreshStats()
        start_time = time.time()

        # Get tenants with tokens expiring soon
        try:
            expiring = self.store.get_expiring_tokens(hours=self.expiry_window_hours)
            stats.total_checked = len(expiring)

            # Update active tenants metric
            _update_active_tenants(len(expiring))
        except Exception as e:
            logger.error(f"Failed to get expiring Teams tokens: {e}")
            _record_refresh_failure("store_error")
            return stats

        if not expiring:
            logger.debug("No Teams tokens expiring soon")
            return stats

        logger.info(f"Found {len(expiring)} Teams tokens to refresh")

        # Refresh each token with small delays to avoid rate limiting
        for tenant in expiring:
            result = await self._refresh_single_token(tenant)
            stats.results.append(result)

            if result.success:
                stats.refreshed += 1
                _record_refresh_success()
            else:
                stats.failed += 1
                error_type = "revoked" if "invalid_grant" in (result.error or "") else "api_error"
                _record_refresh_failure(error_type)
                if self.on_refresh_failure:
                    try:
                        self.on_refresh_failure(result)
                    except Exception as e:
                        logger.error(f"Error in refresh failure callback: {e}")

            # Small delay between refreshes to avoid rate limiting
            await asyncio.sleep(0.5)

        # Record duration metric
        duration = time.time() - start_time
        if _teams_refresh_duration:
            _teams_refresh_duration.observe(duration)

        return stats

    async def _refresh_single_token(self, tenant) -> RefreshResult:
        """Refresh a single tenant's token."""
        tenant_id = tenant.tenant_id
        tenant_name = getattr(tenant, "workspace_name", getattr(tenant, "tenant_name", tenant_id))

        try:
            # Run the synchronous refresh in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.store.refresh_workspace_token,
                tenant_id,
                self.client_id,
                self.client_secret,
            )

            if result:
                logger.info(f"Successfully refreshed Teams token for tenant: {tenant_name}")
                return RefreshResult(
                    tenant_id=tenant_id,
                    tenant_name=tenant_name,
                    success=True,
                )
            else:
                error = "Refresh returned None (token may be revoked)"
                logger.warning(f"Failed to refresh Teams token for {tenant_name}: {error}")
                return RefreshResult(
                    tenant_id=tenant_id,
                    tenant_name=tenant_name,
                    success=False,
                    error=error,
                )

        except Exception as e:
            error = str(e)
            logger.error(f"Exception refreshing Teams token for {tenant_name}: {error}")
            return RefreshResult(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                success=False,
                error=error,
            )

    async def refresh_now(self) -> RefreshStats:
        """Trigger an immediate refresh cycle (for testing or manual intervention)."""
        logger.info("Manual Teams token refresh triggered")
        stats = await self._refresh_expiring_tokens()
        self._last_run = datetime.now(timezone.utc)
        self._last_stats = stats
        return stats

    def get_status(self) -> dict:
        """Get current scheduler status for monitoring."""
        return {
            "running": self.is_running,
            "interval_minutes": self.interval_minutes,
            "expiry_window_hours": self.expiry_window_hours,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_stats": {
                "total_checked": self._last_stats.total_checked,
                "refreshed": self._last_stats.refreshed,
                "failed": self._last_stats.failed,
            }
            if self._last_stats
            else None,
            "credentials_configured": bool(self.client_id and self.client_secret),
        }


# Singleton scheduler instance
_scheduler: Optional[TeamsTokenRefreshScheduler] = None


def get_teams_token_scheduler() -> Optional[TeamsTokenRefreshScheduler]:
    """Get the global Teams token refresh scheduler instance."""
    global _scheduler
    if _scheduler is None:
        try:
            from aragora.storage.teams_workspace_store import get_teams_workspace_store

            store = get_teams_workspace_store()
            _scheduler = TeamsTokenRefreshScheduler(store)
        except ImportError:
            logger.debug("Teams workspace store not available")
            return None
    return _scheduler


async def start_teams_token_scheduler() -> bool:
    """Start the global Teams token refresh scheduler."""
    scheduler = get_teams_token_scheduler()
    if scheduler:
        await scheduler.start()
        return True
    return False


async def stop_teams_token_scheduler() -> None:
    """Stop the global Teams token refresh scheduler."""
    scheduler = get_teams_token_scheduler()
    if scheduler:
        await scheduler.stop()


__all__ = [
    "TeamsTokenRefreshScheduler",
    "RefreshResult",
    "RefreshStats",
    "get_teams_token_scheduler",
    "start_teams_token_scheduler",
    "stop_teams_token_scheduler",
]
