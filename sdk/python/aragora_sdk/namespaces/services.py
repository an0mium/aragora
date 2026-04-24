"""
Services Namespace API

Provides methods for service discovery:
- Service discovery and listing
- Service details
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ServicesAPI:
    """
    Synchronous Services API.

    Provides methods for DevOps service discovery.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> services = client.services.list()
        >>> for svc in services["services"]:
        ...     print(f"{svc['name']}: {svc['status']}")
        >>> details = client.services.get("api-gateway")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, status: str | None = None) -> dict[str, Any]:
        """
        List all registered services.

        Args:
            status: Optional filter by service status
                (healthy, degraded, down).

        Returns:
            Dict with services list and their health statuses.
        """
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/services", params=params or None)

    def get(self, service_id: str) -> dict[str, Any]:
        """
        Get detailed information for a specific service.

        Args:
            service_id: Service identifier or name.

        Returns:
            Dict with service details including health, configuration,
            endpoints, and dependencies.
        """
        return self._client.request("GET", f"/api/v1/services/{service_id}")


class AsyncServicesAPI:
    """
    Asynchronous Services API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     services = await client.services.list()
        ...     details = await client.services.get("api-gateway")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, status: str | None = None) -> dict[str, Any]:
        """List all registered services."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/services", params=params or None)

    async def get(self, service_id: str) -> dict[str, Any]:
        """Get detailed information for a specific service."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}")
