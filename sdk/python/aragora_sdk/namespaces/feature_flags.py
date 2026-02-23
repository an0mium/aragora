"""
Feature Flags Namespace API

Provides methods for reading feature flag configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FeatureFlagsAPI:
    """Synchronous Feature Flags API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, **params: Any) -> dict[str, Any]:
        """List all feature flags and their current states.

        Args:
            **params: Optional query parameters (category, enabled, etc.).

        Returns:
            Dict with feature flags and values.
        """
        return self._client.request("GET", "/api/v1/feature-flags", params=params)

    def get(self, name: str) -> dict[str, Any]:
        """Get a specific feature flag by name.

        Args:
            name: Feature flag name.

        Returns:
            Dict with flag details (name, enabled, description, etc.).
        """
        return self._client.request("GET", f"/api/v1/feature-flags/{name}")


class AsyncFeatureFlagsAPI:
    """Asynchronous Feature Flags API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, **params: Any) -> dict[str, Any]:
        """List all feature flags and their current states."""
        return await self._client.request(
            "GET", "/api/v1/feature-flags", params=params
        )

    async def get(self, name: str) -> dict[str, Any]:
        """Get a specific feature flag by name."""
        return await self._client.request(
            "GET", f"/api/v1/feature-flags/{name}"
        )
