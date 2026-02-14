"""
Graph Debates Namespace API

Provides methods for graph-structured debates with branching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list


class GraphDebatesAPI:
    """
    Synchronous Graph Debates API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> debates = client.graph_debates.list()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """List graph-structured debates."""
        return self._client.request("GET", "/api/v1/graph-debates")

    def get(self, debate_id: str) -> dict[str, Any]:
        """Get a graph debate by ID."""
        return self._client.request("GET", f"/api/v1/graph-debates/{debate_id}")

    def create(
        self,
        task: str,
        agents: _List[str] | None = None,
        max_rounds: int | None = None,
        branch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new graph-structured debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        if max_rounds is not None:
            data["max_rounds"] = max_rounds
        if branch_policy:
            data["branch_policy"] = branch_policy
        return self._client.request("POST", "/api/v1/debates/graph", json=data)

    def get_branches(self, debate_id: str) -> dict[str, Any]:
        """Get all branches for a graph debate."""
        return self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/branches")

    def get_nodes(self, debate_id: str) -> dict[str, Any]:
        """Get all nodes in a graph debate."""
        return self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/nodes")


class AsyncGraphDebatesAPI:
    """
    Asynchronous Graph Debates API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     debates = await client.graph_debates.list()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List graph-structured debates."""
        return await self._client.request("GET", "/api/v1/graph-debates")

    async def get(self, debate_id: str) -> dict[str, Any]:
        """Get a graph debate by ID."""
        return await self._client.request("GET", f"/api/v1/graph-debates/{debate_id}")

    async def create(
        self,
        task: str,
        agents: _List[str] | None = None,
        max_rounds: int | None = None,
        branch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new graph-structured debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        if max_rounds is not None:
            data["max_rounds"] = max_rounds
        if branch_policy:
            data["branch_policy"] = branch_policy
        return await self._client.request("POST", "/api/v1/debates/graph", json=data)

    async def get_branches(self, debate_id: str) -> dict[str, Any]:
        """Get all branches for a graph debate."""
        return await self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/branches")

    async def get_nodes(self, debate_id: str) -> dict[str, Any]:
        """Get all nodes in a graph debate."""
        return await self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/nodes")
