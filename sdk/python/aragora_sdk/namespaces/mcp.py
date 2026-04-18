"""
MCP namespace API.

Provides SDK access to Aragora's MCP tool discovery endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class MCPAPI:
    """Synchronous MCP tool discovery API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list_tools(self, category: str | None = None) -> dict[str, Any]:
        """List MCP tools, optionally filtered by category prefix."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        return self._client.request("GET", "/api/v1/mcp/tools", params=params or None)

    def get_tool(self, name: str) -> dict[str, Any]:
        """Get one MCP tool by name."""
        return self._client.request("GET", f"/api/v1/mcp/tools/{name}")


class AsyncMCPAPI:
    """Asynchronous MCP tool discovery API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_tools(self, category: str | None = None) -> dict[str, Any]:
        """List MCP tools, optionally filtered by category prefix."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        return await self._client.request("GET", "/api/v1/mcp/tools", params=params or None)

    async def get_tool(self, name: str) -> dict[str, Any]:
        """Get one MCP tool by name."""
        return await self._client.request("GET", f"/api/v1/mcp/tools/{name}")
