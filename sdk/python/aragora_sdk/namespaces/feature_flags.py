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

    def list(self) -> dict[str, Any]:
        """List all feature flags and their current states.

        Returns:
            Dict with feature flags and values.
        """
        return self._client.request("GET", "/api/v1/feature-flags")


class AsyncFeatureFlagsAPI:
    """Asynchronous Feature Flags API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all feature flags and their current states."""
        return await self._client.request("GET", "/api/v1/feature-flags")
