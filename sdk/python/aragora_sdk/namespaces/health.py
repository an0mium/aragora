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

    def components(self) -> dict[str, Any]:
        """
        Get detailed health status of all components.

        Returns:
            Dict with components map containing individual statuses
        """
        return self._client.request("GET", "/api/v1/health/components")

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

    async def components(self) -> dict[str, Any]:
        """Get detailed health status of all components."""
        return await self._client.request("GET", "/api/v1/health/components")

