"""
Status Namespace API

Provides methods for system status and health:
- Service health checks
- System metrics
- Operational status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class StatusAPI:
    """Synchronous Status API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        return self._client.request("GET", "/api/v1/status/health")

    def get_ready(self) -> dict[str, Any]:
        """Check if system is ready."""
        return self._client.request("GET", "/api/v1/status/ready")

    def get_live(self) -> dict[str, Any]:
        """Check if system is live."""
        return self._client.request("GET", "/api/v1/status/live")

    def get_services(self) -> dict[str, Any]:
        """Get status of all services."""
        return self._client.request("GET", "/api/v1/status/services")

    def get_service(self, service_name: str) -> dict[str, Any]:
        """Get status of a specific service."""
        return self._client.request("GET", f"/api/v1/status/services/{service_name}")

    def get_version(self) -> dict[str, Any]:
        """Get system version information."""
        return self._client.request("GET", "/api/v1/status/version")

    def get_metrics(self) -> dict[str, Any]:
        """Get system metrics."""
        return self._client.request("GET", "/api/v1/status/metrics")


class AsyncStatusAPI:
    """Asynchronous Status API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        return await self._client.request("GET", "/api/v1/status/health")

    async def get_ready(self) -> dict[str, Any]:
        """Check if system is ready."""
        return await self._client.request("GET", "/api/v1/status/ready")

    async def get_live(self) -> dict[str, Any]:
        """Check if system is live."""
        return await self._client.request("GET", "/api/v1/status/live")

    async def get_services(self) -> dict[str, Any]:
        """Get status of all services."""
        return await self._client.request("GET", "/api/v1/status/services")

    async def get_service(self, service_name: str) -> dict[str, Any]:
        """Get status of a specific service."""
        return await self._client.request("GET", f"/api/v1/status/services/{service_name}")

    async def get_version(self) -> dict[str, Any]:
        """Get system version information."""
        return await self._client.request("GET", "/api/v1/status/version")

    async def get_metrics(self) -> dict[str, Any]:
        """Get system metrics."""
        return await self._client.request("GET", "/api/v1/status/metrics")
