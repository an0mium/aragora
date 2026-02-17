"""
Features Namespace API

Provides access to feature flags and feature discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FeaturesAPI:
    """Synchronous Features API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """List all features."""
        return self._client.request("GET", "/api/v1/features")

    def list_all(self) -> dict[str, Any]:
        """List all features with full details."""
        return self._client.request("GET", "/api/v1/features/all")

    def list_available(self) -> dict[str, Any]:
        """List available features."""
        return self._client.request("GET", "/api/v1/features/available")

    def get_config(self) -> dict[str, Any]:
        """Get feature configuration."""
        return self._client.request("GET", "/api/v1/features/config")

    def discover(self) -> dict[str, Any]:
        """Discover features."""
        return self._client.request("GET", "/api/v1/features/discover")

    def list_endpoints(self) -> dict[str, Any]:
        """List feature endpoints."""
        return self._client.request("GET", "/api/v1/features/endpoints")

    def list_handlers(self) -> dict[str, Any]:
        """List feature handlers."""
        return self._client.request("GET", "/api/v1/features/handlers")

    def get(self, feature_id: str) -> dict[str, Any]:
        """Get a specific feature by ID."""
        return self._client.request("GET", f"/api/v1/features/{feature_id}")


class AsyncFeaturesAPI:
    """Asynchronous Features API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all features."""
        return await self._client.request("GET", "/api/v1/features")

    async def list_all(self) -> dict[str, Any]:
        """List all features with full details."""
        return await self._client.request("GET", "/api/v1/features/all")

    async def list_available(self) -> dict[str, Any]:
        """List available features."""
        return await self._client.request("GET", "/api/v1/features/available")

    async def get_config(self) -> dict[str, Any]:
        """Get feature configuration."""
        return await self._client.request("GET", "/api/v1/features/config")

    async def discover(self) -> dict[str, Any]:
        """Discover features."""
        return await self._client.request("GET", "/api/v1/features/discover")

    async def list_endpoints(self) -> dict[str, Any]:
        """List feature endpoints."""
        return await self._client.request("GET", "/api/v1/features/endpoints")

    async def list_handlers(self) -> dict[str, Any]:
        """List feature handlers."""
        return await self._client.request("GET", "/api/v1/features/handlers")

    async def get(self, feature_id: str) -> dict[str, Any]:
        """Get a specific feature by ID."""
        return await self._client.request("GET", f"/api/v1/features/{feature_id}")
