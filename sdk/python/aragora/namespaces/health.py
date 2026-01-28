"""
Health Namespace API

Provides methods for checking system health and readiness.

Features:
- Liveness checks
- Readiness checks
- Component health status
- Dependency health
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class HealthAPI:
    """
    Synchronous Health API.

    Provides methods for health checking:
    - Liveness probes (is the service running?)
    - Readiness probes (can the service handle requests?)
    - Component health (database, cache, external services)

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> health = client.health.check()
        >>> if health['status'] == 'healthy':
        ...     print("Service is operational")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def check(self) -> dict[str, Any]:
        """
        Get overall health status.

        Returns:
            Dict with status, version, uptime, and component health
        """
        return self._client.request("GET", "/api/v1/health")

    def liveness(self) -> dict[str, Any]:
        """
        Liveness probe - check if service is running.

        Used by Kubernetes/orchestrators to know if the process is alive.
        Returns 200 if alive, regardless of dependency health.

        Returns:
            Dict with status ('alive' or 'dead')
        """
        return self._client.request("GET", "/api/v1/health/liveness")

    def readiness(self) -> dict[str, Any]:
        """
        Readiness probe - check if service can handle requests.

        Used by load balancers to know if traffic can be routed.
        Checks all critical dependencies.

        Returns:
            Dict with status ('ready' or 'not_ready') and dependency states
        """
        return self._client.request("GET", "/api/v1/health/readiness")

    def components(self) -> dict[str, Any]:
        """
        Get detailed health status of all components.

        Returns:
            Dict with components map containing individual statuses
        """
        return self._client.request("GET", "/api/v1/health/components")

    def component(self, name: str) -> dict[str, Any]:
        """
        Get health status of a specific component.

        Args:
            name: Component name (database, redis, elasticsearch, etc.)

        Returns:
            Dict with component status, latency, and details
        """
        return self._client.request("GET", f"/api/v1/health/components/{name}")

    def metrics(self) -> dict[str, Any]:
        """
        Get health metrics.

        Returns:
            Dict with request_rate, error_rate, latency_p50, latency_p99, etc.
        """
        return self._client.request("GET", "/api/v1/health/metrics")


class AsyncHealthAPI:
    """
    Asynchronous Health API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     health = await client.health.check()
        ...     print(f"Status: {health['status']}")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def check(self) -> dict[str, Any]:
        """Get overall health status."""
        return await self._client.request("GET", "/api/v1/health")

    async def liveness(self) -> dict[str, Any]:
        """Liveness probe - check if service is running."""
        return await self._client.request("GET", "/api/v1/health/liveness")

    async def readiness(self) -> dict[str, Any]:
        """Readiness probe - check if service can handle requests."""
        return await self._client.request("GET", "/api/v1/health/readiness")

    async def components(self) -> dict[str, Any]:
        """Get detailed health status of all components."""
        return await self._client.request("GET", "/api/v1/health/components")

    async def component(self, name: str) -> dict[str, Any]:
        """Get health status of a specific component."""
        return await self._client.request("GET", f"/api/v1/health/components/{name}")

    async def metrics(self) -> dict[str, Any]:
        """Get health metrics."""
        return await self._client.request("GET", "/api/v1/health/metrics")
