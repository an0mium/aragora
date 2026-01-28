"""
Metrics Namespace API

Provides methods for system and application metrics:
- System health and performance
- Cache statistics
- Application metrics
- Prometheus metrics export
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class MetricsAPI:
    """
    Synchronous Metrics API.

    Provides methods for accessing system and application metrics.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> health = client.metrics.get_health()
        >>> system = client.metrics.get_system()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # General Metrics
    # ===========================================================================

    def get(self) -> dict[str, Any]:
        """
        Get general application metrics.

        Returns:
            Dict with application metrics
        """
        return self._client.request("GET", "/api/v1/metrics")

    def get_health(self) -> dict[str, Any]:
        """
        Get health metrics.

        Returns:
            Dict with health status and checks
        """
        return self._client.request("GET", "/api/v1/metrics/health")

    def get_cache(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache hit/miss ratios, size, etc.
        """
        return self._client.request("GET", "/api/v1/metrics/cache")

    def get_system(self) -> dict[str, Any]:
        """
        Get system metrics.

        Returns:
            Dict with CPU, memory, disk, network metrics
        """
        return self._client.request("GET", "/api/v1/metrics/system")

    def get_prometheus(self) -> str:
        """
        Get Prometheus-format metrics.

        Returns:
            Prometheus text format metrics
        """
        return self._client.request("GET", "/api/v1/metrics/prometheus")

    # ===========================================================================
    # Specialized Metrics
    # ===========================================================================

    def get_agents(self) -> dict[str, Any]:
        """
        Get agent performance metrics.

        Returns:
            Dict with agent-specific metrics
        """
        return self._client.request("GET", "/api/v1/metrics/agents")

    def get_debates(self) -> dict[str, Any]:
        """
        Get debate metrics.

        Returns:
            Dict with debate throughput, latency, etc.
        """
        return self._client.request("GET", "/api/v1/metrics/debates")

    def get_api(self) -> dict[str, Any]:
        """
        Get API metrics.

        Returns:
            Dict with request counts, latencies, error rates
        """
        return self._client.request("GET", "/api/v1/metrics/api")

    def get_knowledge(self) -> dict[str, Any]:
        """
        Get knowledge mound metrics.

        Returns:
            Dict with KM-specific metrics
        """
        return self._client.request("GET", "/api/v1/metrics/knowledge")

    def get_billing(self) -> dict[str, Any]:
        """
        Get billing metrics.

        Returns:
            Dict with billing and usage metrics
        """
        return self._client.request("GET", "/api/v1/metrics/billing")


class AsyncMetricsAPI:
    """
    Asynchronous Metrics API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     health = await client.metrics.get_health()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # General Metrics
    async def get(self) -> dict[str, Any]:
        """Get general application metrics."""
        return await self._client.request("GET", "/api/v1/metrics")

    async def get_health(self) -> dict[str, Any]:
        """Get health metrics."""
        return await self._client.request("GET", "/api/v1/metrics/health")

    async def get_cache(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self._client.request("GET", "/api/v1/metrics/cache")

    async def get_system(self) -> dict[str, Any]:
        """Get system metrics."""
        return await self._client.request("GET", "/api/v1/metrics/system")

    async def get_prometheus(self) -> str:
        """Get Prometheus-format metrics."""
        return await self._client.request("GET", "/api/v1/metrics/prometheus")

    # Specialized Metrics
    async def get_agents(self) -> dict[str, Any]:
        """Get agent performance metrics."""
        return await self._client.request("GET", "/api/v1/metrics/agents")

    async def get_debates(self) -> dict[str, Any]:
        """Get debate metrics."""
        return await self._client.request("GET", "/api/v1/metrics/debates")

    async def get_api(self) -> dict[str, Any]:
        """Get API metrics."""
        return await self._client.request("GET", "/api/v1/metrics/api")

    async def get_knowledge(self) -> dict[str, Any]:
        """Get knowledge mound metrics."""
        return await self._client.request("GET", "/api/v1/metrics/knowledge")

    async def get_billing(self) -> dict[str, Any]:
        """Get billing metrics."""
        return await self._client.request("GET", "/api/v1/metrics/billing")
