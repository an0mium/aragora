"""
Health Namespace API

Provides methods for checking system health and readiness:
- Overall health check
- Deep health check with all subsystems
- Component-level health (database, stores, encryption)
- Worker and job queue status
- Circuit breaker status
- Knowledge Mound health
- Platform health overview
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class HealthAPI:
    """
    Synchronous Health API.

    Provides comprehensive health checking across all system components
    including database, stores, workers, encryption, and knowledge mound.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> health = client.health.check()
        >>> if health['status'] == 'healthy':
        ...     print("Service is operational")
        >>> deep = client.health.deep()
        >>> for component, status in deep['components'].items():
        ...     print(f"{component}: {status['status']}")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Core Health Checks
    # =========================================================================

    def check(self) -> dict[str, Any]:
        """
        Get overall health status (liveness check).

        Returns:
            Dict with status, version, uptime, and component health.
        """
        return self._client.request("GET", "/api/v1/health")

    def detailed(self) -> dict[str, Any]:
        """
        Get detailed health status with extended diagnostics.

        Returns:
            Dict with detailed health information including memory usage,
            connection pool status, and performance metrics.
        """
        return self._client.request("GET", "/api/v1/health/detailed")

    def deep(self) -> dict[str, Any]:
        """
        Run deep health check across all subsystems.

        This performs active checks against each component rather than
        relying on cached status.

        Returns:
            Dict with deep health check results for every subsystem.
        """
        return self._client.request("GET", "/api/v1/health/deep")

    def startup(self) -> dict[str, Any]:
        """
        Get startup health status.

        Indicates whether the server has completed its startup sequence.

        Returns:
            Dict with startup status and initialization progress.
        """
        return self._client.request("GET", "/api/v1/health/startup")

    def platform(self) -> dict[str, Any]:
        """
        Get platform-level health overview.

        Returns:
            Dict with overall platform health including all integrated
            services and their status.
        """
        return self._client.request("GET", "/api/v1/health/platform")

    # =========================================================================
    # Component Health
    # =========================================================================

    def components(self) -> dict[str, Any]:
        """
        Get detailed health status of all components.

        Returns:
            Dict with components map containing individual statuses.
        """
        return self._client.request("GET", "/api/v1/health/components")

    def database(self) -> dict[str, Any]:
        """
        Get database health status.

        Returns:
            Dict with database connectivity, latency, and pool status.
        """
        return self._client.request("GET", "/api/v1/health/database")

    def stores(self) -> dict[str, Any]:
        """
        Get data store health status.

        Returns:
            Dict with health status of all data stores (Redis, Postgres, etc.).
        """
        return self._client.request("GET", "/api/v1/health/stores")

    def encryption(self) -> dict[str, Any]:
        """
        Get encryption subsystem health.

        Returns:
            Dict with encryption key status, rotation schedule, and health.
        """
        return self._client.request("GET", "/api/v1/health/encryption")

    def knowledge_mound(self) -> dict[str, Any]:
        """
        Get Knowledge Mound health status.

        Returns:
            Dict with Knowledge Mound adapter statuses, sync status,
            and data integrity metrics.
        """
        return self._client.request("GET", "/api/v1/health/knowledge-mound")

    def circuits(self) -> dict[str, Any]:
        """
        Get circuit breaker status across all services.

        Returns:
            Dict with circuit breaker states (closed, open, half-open)
            for each protected service.
        """
        return self._client.request("GET", "/api/v1/health/circuits")

    def sync(self) -> dict[str, Any]:
        """
        Get sync health status.

        Returns:
            Dict with sync status for background synchronization tasks.
        """
        return self._client.request("GET", "/api/v1/health/sync")

    def cross_pollination(self) -> dict[str, Any]:
        """
        Get cross-pollination health status.

        Returns:
            Dict with cross-workspace knowledge sharing health.
        """
        return self._client.request("GET", "/api/v1/health/cross-pollination")

    def decay(self) -> dict[str, Any]:
        """
        Get confidence decay health status.

        Returns:
            Dict with confidence decay process health and metrics.
        """
        return self._client.request("GET", "/api/v1/health/decay")

    # =========================================================================
    # Workers & Jobs
    # =========================================================================

    def job_queue_status(self) -> dict[str, Any]:
        """
        Get job queue health status.

        Returns:
            Dict with job queue depth, processing rate, and error rate.
        """
        return self._client.request("GET", "/api/v1/health/job-queue")

    def list_workers(self) -> dict[str, Any]:
        """
        Get background worker health status.

        Returns:
            Dict with worker statuses, task counts, and last heartbeat.
        """
        return self._client.request("GET", "/api/v1/health/workers")

    def list_all_workers(self) -> dict[str, Any]:
        """
        Get combined workers and job queue health.

        Returns:
            Dict with comprehensive worker and queue health data.
        """
        return self._client.request("GET", "/api/v1/health/workers/all")

    # =========================================================================
    # Debate Health
    # =========================================================================

    def slow_debates(self) -> dict[str, Any]:
        """
        Get slow debate health status.

        Returns:
            Dict with slow-running debates, their durations,
            and potential bottlenecks.
        """
        return self._client.request("GET", "/api/v1/health/slow-debates")


class AsyncHealthAPI:
    """
    Asynchronous Health API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     health = await client.health.check()
        ...     deep = await client.health.deep()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Core Health Checks
    # =========================================================================

    async def check(self) -> dict[str, Any]:
        """Get overall health status (liveness check)."""
        return await self._client.request("GET", "/api/v1/health")

    async def detailed(self) -> dict[str, Any]:
        """Get detailed health status with extended diagnostics."""
        return await self._client.request("GET", "/api/v1/health/detailed")

    async def deep(self) -> dict[str, Any]:
        """Run deep health check across all subsystems."""
        return await self._client.request("GET", "/api/v1/health/deep")

    async def startup(self) -> dict[str, Any]:
        """Get startup health status."""
        return await self._client.request("GET", "/api/v1/health/startup")

    async def platform(self) -> dict[str, Any]:
        """Get platform-level health overview."""
        return await self._client.request("GET", "/api/v1/health/platform")

    # =========================================================================
    # Component Health
    # =========================================================================

    async def components(self) -> dict[str, Any]:
        """Get detailed health status of all components."""
        return await self._client.request("GET", "/api/v1/health/components")

    async def database(self) -> dict[str, Any]:
        """Get database health status."""
        return await self._client.request("GET", "/api/v1/health/database")

    async def stores(self) -> dict[str, Any]:
        """Get data store health status."""
        return await self._client.request("GET", "/api/v1/health/stores")

    async def encryption(self) -> dict[str, Any]:
        """Get encryption subsystem health."""
        return await self._client.request("GET", "/api/v1/health/encryption")

    async def knowledge_mound(self) -> dict[str, Any]:
        """Get Knowledge Mound health status."""
        return await self._client.request("GET", "/api/v1/health/knowledge-mound")

    async def circuits(self) -> dict[str, Any]:
        """Get circuit breaker status across all services."""
        return await self._client.request("GET", "/api/v1/health/circuits")

    async def sync(self) -> dict[str, Any]:
        """Get sync health status."""
        return await self._client.request("GET", "/api/v1/health/sync")

    async def cross_pollination(self) -> dict[str, Any]:
        """Get cross-pollination health status."""
        return await self._client.request("GET", "/api/v1/health/cross-pollination")

    async def decay(self) -> dict[str, Any]:
        """Get confidence decay health status."""
        return await self._client.request("GET", "/api/v1/health/decay")

    # =========================================================================
    # Workers & Jobs
    # =========================================================================

    async def job_queue_status(self) -> dict[str, Any]:
        """Get job queue health status."""
        return await self._client.request("GET", "/api/v1/health/job-queue")

    async def list_workers(self) -> dict[str, Any]:
        """Get background worker health status."""
        return await self._client.request("GET", "/api/v1/health/workers")

    async def list_all_workers(self) -> dict[str, Any]:
        """Get combined workers and job queue health."""
        return await self._client.request("GET", "/api/v1/health/workers/all")

    # =========================================================================
    # Debate Health
    # =========================================================================

    async def slow_debates(self) -> dict[str, Any]:
        """Get slow debate health status."""
        return await self._client.request("GET", "/api/v1/health/slow-debates")
