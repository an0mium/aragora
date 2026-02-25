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

    def list_scorers(self) -> dict[str, Any]:
        """List available scorer plugins."""
        return self._client.request("GET", "/api/v1/selection/scorers/")

    def get_scorer(self, name: str) -> dict[str, Any]:
        """Get details for a specific scorer plugin."""
        return self._client.request("GET", f"/api/v1/selection/scorers/{name}")

    def list_team_selectors(self) -> dict[str, Any]:
        """List available team selector plugins."""
        return self._client.request("GET", "/api/v1/selection/team-selectors/")

    def get_team_selector(self, name: str) -> dict[str, Any]:
        """Get details for a specific team selector plugin."""
        return self._client.request("GET", f"/api/v1/selection/team-selectors/{name}")

    def list_role_assigners(self) -> dict[str, Any]:
        """List available role assigner plugins."""
        return self._client.request("GET", "/api/v1/selection/role-assigners/")

    def get_role_assigner(self, name: str) -> dict[str, Any]:
        """Get details for a specific role assigner plugin."""
        return self._client.request("GET", f"/api/v1/selection/role-assigners/{name}")


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

    async def list_scorers(self) -> dict[str, Any]:
        """List available scorer plugins."""
        return await self._client.request("GET", "/api/v1/selection/scorers/")

    async def get_scorer(self, name: str) -> dict[str, Any]:
        """Get details for a specific scorer plugin."""
        return await self._client.request("GET", f"/api/v1/selection/scorers/{name}")

    async def list_team_selectors(self) -> dict[str, Any]:
        """List available team selector plugins."""
        return await self._client.request("GET", "/api/v1/selection/team-selectors/")

    async def get_team_selector(self, name: str) -> dict[str, Any]:
        """Get details for a specific team selector plugin."""
        return await self._client.request("GET", f"/api/v1/selection/team-selectors/{name}")

    async def list_role_assigners(self) -> dict[str, Any]:
        """List available role assigner plugins."""
        return await self._client.request("GET", "/api/v1/selection/role-assigners/")

    async def get_role_assigner(self, name: str) -> dict[str, Any]:
        """Get details for a specific role assigner plugin."""
        return await self._client.request("GET", f"/api/v1/selection/role-assigners/{name}")
