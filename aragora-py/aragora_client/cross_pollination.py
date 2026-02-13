"""Cross-pollination API for the Aragora SDK.

Provides access to cross-subsystem event observability and synchronization:
- Subscriber statistics and management
- Bridge status and metrics
- Knowledge Mound cross-pollination sync
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class CrossPollinationAPI:
    """API for cross-pollination operations.

    Cross-pollination enables knowledge and patterns to flow between
    subsystems (debates, memory, knowledge mound, etc.) for institutional
    learning and continuous improvement.
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Statistics & Monitoring
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get cross-subscriber statistics.

        Returns:
            Dictionary with:
            - summary: Total counts and status
            - subscribers: Per-subscriber statistics including events
              processed, failed, and last activity time

        Example:
            stats = await client.cross_pollination.get_stats()
            print(f"Total events: {stats['summary']['total_events_processed']}")
        """
        return await self._client._get("/api/v1/cross-pollination/stats")

    async def get_subscribers(self) -> dict[str, Any]:
        """List all registered cross-pollination subscribers.

        Returns:
            Dictionary with list of subscribers and their event types

        Example:
            subs = await client.cross_pollination.get_subscribers()
            for sub in subs['subscribers']:
                print(f"{sub['name']} listens to {sub['event_type']}")
        """
        return await self._client._get("/api/v1/cross-pollination/subscribers")

    async def get_bridge_status(self) -> dict[str, Any]:
        """Get Arena event bridge status.

        The event bridge connects Arena debates to cross-pollination
        subscribers, enabling real-time knowledge propagation.

        Returns:
            Dictionary with bridge health, connected subscribers,
            and event flow metrics

        Example:
            bridge = await client.cross_pollination.get_bridge_status()
            if bridge['status'] == 'healthy':
                print(f"Bridge connected to {bridge['subscriber_count']} subs")
        """
        return await self._client._get("/api/v1/cross-pollination/bridge")

    async def get_metrics(self) -> dict[str, Any]:
        """Get cross-pollination metrics summary.

        Returns detailed metrics about event flow, latencies, and
        processing rates across all subscribers.

        Returns:
            Dictionary with metrics including throughput, latency
            percentiles, and error rates
        """
        return await self._client._get("/api/v1/cross-pollination/metrics")

    # =========================================================================
    # Management Operations
    # =========================================================================

    async def reset_stats(self, subscriber: str | None = None) -> dict[str, Any]:
        """Reset subscriber statistics.

        Args:
            subscriber: Optional subscriber name. If None, resets all.

        Returns:
            Confirmation with reset count

        Example:
            # Reset all
            await client.cross_pollination.reset_stats()

            # Reset specific subscriber
            await client.cross_pollination.reset_stats("km_sync")
        """
        data: dict[str, Any] = {}
        if subscriber:
            data["subscriber"] = subscriber
        return await self._client._post("/api/v1/cross-pollination/reset", data)

    # =========================================================================
    # Knowledge Mound Integration
    # =========================================================================

    async def get_km_status(self) -> dict[str, Any]:
        """Get Knowledge Mound cross-pollination status.

        Returns the status of knowledge synchronization between
        debates and the Knowledge Mound.

        Returns:
            Dictionary with:
            - sync_enabled: Whether auto-sync is active
            - last_sync: Timestamp of last sync
            - pending_items: Items waiting to sync
            - adapters: Status of KM adapters
        """
        return await self._client._get("/api/v1/cross-pollination/km")

    async def trigger_km_sync(
        self,
        *,
        full: bool = False,
        adapters: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trigger Knowledge Mound synchronization.

        Initiates sync of debate outcomes, patterns, and insights
        to the Knowledge Mound.

        Args:
            full: If True, performs full sync instead of incremental
            adapters: Specific adapters to sync (default: all)

        Returns:
            Dictionary with sync job ID and status

        Example:
            # Incremental sync
            result = await client.cross_pollination.trigger_km_sync()

            # Full sync of specific adapters
            result = await client.cross_pollination.trigger_km_sync(
                full=True,
                adapters=["continuum", "consensus"]
            )
        """
        data: dict[str, Any] = {"full": full}
        if adapters:
            data["adapters"] = adapters
        return await self._client._post("/api/v1/cross-pollination/km/sync", data)

    async def get_km_staleness(self) -> dict[str, Any]:
        """Get Knowledge Mound staleness report.

        Returns information about stale knowledge that may need
        revalidation or refreshing.

        Returns:
            Dictionary with staleness metrics per knowledge category
        """
        return await self._client._get("/api/v1/cross-pollination/km/staleness")

    async def get_km_culture(self) -> dict[str, Any]:
        """Get Knowledge Mound culture patterns.

        Culture patterns represent organizational decision-making
        preferences learned from historical debates.

        Returns:
            Dictionary with culture pattern statistics and top patterns
        """
        return await self._client._get("/api/v1/cross-pollination/km/culture")

    async def refresh_km_culture(
        self,
        *,
        since_days: int | None = None,
    ) -> dict[str, Any]:
        """Refresh Knowledge Mound culture patterns.

        Re-analyzes recent debates to update culture patterns.

        Args:
            since_days: Only analyze debates from last N days (default: 30)

        Returns:
            Dictionary with refresh job status
        """
        data: dict[str, Any] = {}
        if since_days is not None:
            data["since_days"] = since_days
        return await self._client._post(
            "/api/v1/cross-pollination/km/culture/refresh", data
        )
