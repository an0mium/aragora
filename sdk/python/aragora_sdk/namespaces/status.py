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

    def get_status(self) -> dict[str, Any]:
        """Get overall system status."""
        return self._client.request("GET", "/api/v1/status")

    def get_components(self) -> dict[str, Any]:
        """Get status of individual components."""
        return self._client.request("GET", "/api/v1/status/components")

    def get_history(self) -> dict[str, Any]:
        """Get status history."""
        return self._client.request("GET", "/api/v1/status/history")

    def get_incidents(self) -> dict[str, Any]:
        """Get current and recent incidents."""
        return self._client.request("GET", "/api/v1/status/incidents")

    def get_summary(self) -> dict[str, Any]:
        """Get status summary."""
        return self._client.request("GET", "/api/v1/status/summary")


class AsyncStatusAPI:
    """Asynchronous Status API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get overall system status."""
        return await self._client.request("GET", "/api/v1/status")

    async def get_components(self) -> dict[str, Any]:
        """Get status of individual components."""
        return await self._client.request("GET", "/api/v1/status/components")

    async def get_history(self) -> dict[str, Any]:
        """Get status history."""
        return await self._client.request("GET", "/api/v1/status/history")

    async def get_incidents(self) -> dict[str, Any]:
        """Get current and recent incidents."""
        return await self._client.request("GET", "/api/v1/status/incidents")

    async def get_summary(self) -> dict[str, Any]:
        """Get status summary."""
        return await self._client.request("GET", "/api/v1/status/summary")
