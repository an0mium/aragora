"""
Selection Namespace API

Provides methods for agent selection:
- Selection criteria management
- Agent matching
- Performance-based selection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SelectionAPI:
    """Synchronous Selection API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_defaults(self, **kwargs: Any) -> dict[str, Any]:
        """Get default agent selection settings."""
        return self._client.request("POST", "/api/v1/selection/defaults", json=kwargs)

    def get_plugins(self, **kwargs: Any) -> dict[str, Any]:
        """Get available selection plugins."""
        return self._client.request("POST", "/api/v1/selection/plugins", json=kwargs)

    def score(self, **kwargs: Any) -> dict[str, Any]:
        """Score agents for a task."""
        return self._client.request("POST", "/api/v1/selection/score", json=kwargs)

    def select_team(self, **kwargs: Any) -> dict[str, Any]:
        """Select a team of agents for a task."""
        return self._client.request("POST", "/api/v1/selection/team", json=kwargs)


class AsyncSelectionAPI:
    """Asynchronous Selection API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_defaults(self, **kwargs: Any) -> dict[str, Any]:
        """Get default agent selection settings."""
        return await self._client.request("POST", "/api/v1/selection/defaults", json=kwargs)

    async def get_plugins(self, **kwargs: Any) -> dict[str, Any]:
        """Get available selection plugins."""
        return await self._client.request("POST", "/api/v1/selection/plugins", json=kwargs)

    async def score(self, **kwargs: Any) -> dict[str, Any]:
        """Score agents for a task."""
        return await self._client.request("POST", "/api/v1/selection/score", json=kwargs)

    async def select_team(self, **kwargs: Any) -> dict[str, Any]:
        """Select a team of agents for a task."""
        return await self._client.request("POST", "/api/v1/selection/team", json=kwargs)
