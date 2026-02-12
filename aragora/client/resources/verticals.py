"""
Verticals API resource for the Aragora client.

Provides access to vertical specialist configuration and utilities:
- List verticals
- Get/update vertical configs
- Fetch tools/compliance frameworks
- Suggest a vertical for a task
- Create vertical-specific agents or debates
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraClient


class VerticalsAPI:
    """API interface for vertical specialist operations."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, keyword: str | None = None) -> list[dict[str, Any]]:
        """List available verticals."""
        params: dict[str, Any] = {}
        if keyword:
            params["keyword"] = keyword
        response = self._client._get("/api/v1/verticals", params)
        return response.get("verticals", [])

    async def list_async(self, keyword: str | None = None) -> builtins.list[dict[str, Any]]:
        """Async version of list()."""
        params: dict[str, Any] = {}
        if keyword:
            params["keyword"] = keyword
        response = await self._client._get_async("/api/v1/verticals", params)
        return response.get("verticals", [])

    def get(self, vertical_id: str) -> dict[str, Any]:
        """Get a vertical configuration."""
        return self._client._get(f"/api/v1/verticals/{vertical_id}")

    async def get_async(self, vertical_id: str) -> dict[str, Any]:
        """Async version of get()."""
        return await self._client._get_async(f"/api/v1/verticals/{vertical_id}")

    def tools(self, vertical_id: str) -> dict[str, Any]:
        """Get tools for a vertical."""
        return self._client._get(f"/api/v1/verticals/{vertical_id}/tools")

    async def tools_async(self, vertical_id: str) -> dict[str, Any]:
        """Async version of tools()."""
        return await self._client._get_async(f"/api/v1/verticals/{vertical_id}/tools")

    def compliance(self, vertical_id: str) -> dict[str, Any]:
        """Get compliance frameworks for a vertical."""
        return self._client._get(f"/api/v1/verticals/{vertical_id}/compliance")

    async def compliance_async(self, vertical_id: str) -> dict[str, Any]:
        """Async version of compliance()."""
        return await self._client._get_async(f"/api/v1/verticals/{vertical_id}/compliance")

    def suggest(self, task: str) -> dict[str, Any]:
        """Suggest a vertical for a task."""
        params = {"task": task}
        return self._client._get("/api/v1/verticals/suggest", params)

    async def suggest_async(self, task: str) -> dict[str, Any]:
        """Async version of suggest()."""
        params = {"task": task}
        return await self._client._get_async("/api/v1/verticals/suggest", params)

    def update_config(self, vertical_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Update vertical configuration."""
        return self._client._put(f"/api/v1/verticals/{vertical_id}/config", config)

    async def update_config_async(self, vertical_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Async version of update_config()."""
        return await self._client._put_async(f"/api/v1/verticals/{vertical_id}/config", config)

    def create_agent(self, vertical_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a vertical specialist agent instance."""
        return self._client._post(f"/api/v1/verticals/{vertical_id}/agent", payload)

    async def create_agent_async(self, vertical_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Async version of create_agent()."""
        return await self._client._post_async(f"/api/v1/verticals/{vertical_id}/agent", payload)

    def create_debate(self, vertical_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a vertical-specific debate."""
        return self._client._post(f"/api/v1/verticals/{vertical_id}/debate", payload)

    async def create_debate_async(
        self, vertical_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Async version of create_debate()."""
        return await self._client._post_async(f"/api/v1/verticals/{vertical_id}/debate", payload)
