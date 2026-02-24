"""DAG operations namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DAGOperationsAPI:
    """Synchronous DAG operations API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        """Get a DAG graph by ID."""
        return self._client.request("GET", f"/api/v1/pipeline/dag/{graph_id}")

    def debate_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        agents: list[str] | None = None,
        rounds: int = 3,
    ) -> dict[str, Any]:
        """Run a debate operation for a node."""
        payload: dict[str, Any] = {"rounds": rounds}
        if agents is not None:
            payload["agents"] = agents
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/debate",
            json=payload,
        )

    def decompose_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Decompose a node into child nodes."""
        return self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/decompose",
        )

    def prioritize_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Prioritize children for a node."""
        return self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/prioritize",
        )

    def assign_agents(
        self,
        graph_id: str,
        node_id: str,
        *,
        node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assign agents for a node or a list of node IDs."""
        payload: dict[str, Any] = {}
        if node_ids is not None:
            payload["node_ids"] = node_ids
        if payload:
            return self._client.request(
                "POST",
                f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/assign-agents",
                json=payload,
            )
        return self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/assign-agents",
        )

    def execute_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Execute a node."""
        return self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/execute",
        )

    def find_precedents(
        self,
        graph_id: str,
        node_id: str,
        *,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Find precedent nodes for a node."""
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/find-precedents",
            json={"max_results": max_results},
        )

    def cluster_ideas(
        self,
        graph_id: str,
        ideas: list[str],
        *,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Cluster a set of ideas into graph nodes."""
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/cluster-ideas",
            json={"ideas": ideas, "threshold": threshold},
        )

    def auto_flow(
        self,
        graph_id: str,
        ideas: list[str],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the end-to-end DAG auto-flow operation."""
        payload: dict[str, Any] = {"ideas": ideas}
        if config is not None:
            payload["config"] = config
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/auto-flow",
            json=payload,
        )


class AsyncDAGOperationsAPI:
    """Asynchronous DAG operations API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_graph(self, graph_id: str) -> dict[str, Any]:
        """Get a DAG graph by ID."""
        return await self._client.request("GET", f"/api/v1/pipeline/dag/{graph_id}")

    async def debate_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        agents: list[str] | None = None,
        rounds: int = 3,
    ) -> dict[str, Any]:
        """Run a debate operation for a node."""
        payload: dict[str, Any] = {"rounds": rounds}
        if agents is not None:
            payload["agents"] = agents
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/debate",
            json=payload,
        )

    async def decompose_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Decompose a node into child nodes."""
        return await self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/decompose",
        )

    async def prioritize_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Prioritize children for a node."""
        return await self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/prioritize",
        )

    async def assign_agents(
        self,
        graph_id: str,
        node_id: str,
        *,
        node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assign agents for a node or a list of node IDs."""
        payload: dict[str, Any] = {}
        if node_ids is not None:
            payload["node_ids"] = node_ids
        if payload:
            return await self._client.request(
                "POST",
                f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/assign-agents",
                json=payload,
            )
        return await self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/assign-agents",
        )

    async def execute_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Execute a node."""
        return await self._client.request(
            "POST", f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/execute",
        )

    async def find_precedents(
        self,
        graph_id: str,
        node_id: str,
        *,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Find precedent nodes for a node."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/nodes/{node_id}/find-precedents",
            json={"max_results": max_results},
        )

    async def cluster_ideas(
        self,
        graph_id: str,
        ideas: list[str],
        *,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Cluster a set of ideas into graph nodes."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/cluster-ideas",
            json={"ideas": ideas, "threshold": threshold},
        )

    async def auto_flow(
        self,
        graph_id: str,
        ideas: list[str],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the end-to-end DAG auto-flow operation."""
        payload: dict[str, Any] = {"ideas": ideas}
        if config is not None:
            payload["config"] = config
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/dag/{graph_id}/auto-flow",
            json=payload,
        )
