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

    def get_health(self, service_id: str) -> dict[str, Any]:
        """Get service health."""
        return self._client.request("GET", f"/api/v1/services/{service_id}/health")

    def get_config(self, service_id: str) -> dict[str, Any]:
        """Get service configuration."""
        return self._client.request("GET", f"/api/v1/services/{service_id}/config")

    def update_config(self, service_id: str, **config: Any) -> dict[str, Any]:
        """Update service configuration."""
        return self._client.request("PATCH", f"/api/v1/services/{service_id}/config", json=config)

    def restart(self, service_id: str) -> dict[str, Any]:
        """Restart a service."""
        return self._client.request("POST", f"/api/v1/services/{service_id}/restart")

    def get_logs(self, service_id: str, limit: int = 100) -> dict[str, Any]:
        """Get service logs."""
        return self._client.request(
            "GET", f"/api/v1/services/{service_id}/logs", params={"limit": limit}
        )

    def get_metrics(self, service_id: str) -> dict[str, Any]:
        """Get service metrics."""
        return self._client.request("GET", f"/api/v1/services/{service_id}/metrics")


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

    async def get_health(self, service_id: str) -> dict[str, Any]:
        """Get service health."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}/health")

    async def get_config(self, service_id: str) -> dict[str, Any]:
        """Get service configuration."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}/config")

    async def update_config(self, service_id: str, **config: Any) -> dict[str, Any]:
        """Update service configuration."""
        return await self._client.request(
            "PATCH", f"/api/v1/services/{service_id}/config", json=config
        )

    async def restart(self, service_id: str) -> dict[str, Any]:
        """Restart a service."""
        return await self._client.request("POST", f"/api/v1/services/{service_id}/restart")

    async def get_logs(self, service_id: str, limit: int = 100) -> dict[str, Any]:
        """Get service logs."""
        return await self._client.request(
            "GET", f"/api/v1/services/{service_id}/logs", params={"limit": limit}
        )

    async def get_metrics(self, service_id: str) -> dict[str, Any]:
        """Get service metrics."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}/metrics")
