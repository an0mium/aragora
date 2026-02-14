"""
Services Namespace API

Provides methods for service management:
- Service discovery
- Health monitoring
- Configuration management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class ServicesAPI:
    """Synchronous Services API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, status: str | None = None) -> dict[str, Any]:
        """List all services."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/services", params=params)

    def get(self, service_id: str) -> dict[str, Any]:
        """Get service by ID."""
        return self._client.request("GET", f"/api/v1/services/{service_id}")

class AsyncServicesAPI:
    """Asynchronous Services API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, status: str | None = None) -> dict[str, Any]:
        """List all services."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/services", params=params)

    async def get(self, service_id: str) -> dict[str, Any]:
        """Get service by ID."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}")

