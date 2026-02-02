"""
Relationships Namespace API

Provides access to agent relationship analysis, including rivalries and alliances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class RelationshipsAPI:
    """Synchronous Relationships API for agent relationship analysis."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_summary(self) -> dict[str, Any]:
        """Get global relationship overview.

        Returns:
            Summary including strongest rivalry/alliance, most connected agent,
            and average scores.
        """
        return self._client.request("GET", "/api/v1/relationships/summary")

    def get_graph(
        self,
        min_debates: int = 3,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Get full relationship graph for visualizations.

        Args:
            min_debates: Minimum debates between agents to include (default: 3).
            min_score: Minimum rivalry/alliance score to include (default: 0.0).

        Returns:
            Graph structure with nodes (agents) and edges (relationships).
        """
        params = {"min_debates": min_debates, "min_score": min_score}
        return self._client.request("GET", "/api/v1/relationships/graph", params=params)

    def get_stats(self) -> dict[str, Any]:
        """Get relationship system statistics.

        Returns:
            Statistics including rivalry/alliance counts, most debated pair,
            and highest agreement pair.
        """
        return self._client.request("GET", "/api/v1/relationships/stats")

    def get_pair(
        self,
        agent_a: str,
        agent_b: str,
    ) -> dict[str, Any]:
        """Get detailed relationship between two specific agents.

        Args:
            agent_a: First agent name.
            agent_b: Second agent name.

        Returns:
            Detailed relationship including debate count, agreement rate,
            rivalry/alliance scores, head-to-head record, and influence metrics.
        """
        return self._client.request("GET", f"/api/v1/relationship/{agent_a}/{agent_b}")


class AsyncRelationshipsAPI:
    """Asynchronous Relationships API for agent relationship analysis."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_summary(self) -> dict[str, Any]:
        """Get global relationship overview."""
        return await self._client.request("GET", "/api/v1/relationships/summary")

    async def get_graph(
        self,
        min_debates: int = 3,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Get full relationship graph for visualizations."""
        params = {"min_debates": min_debates, "min_score": min_score}
        return await self._client.request("GET", "/api/v1/relationships/graph", params=params)

    async def get_stats(self) -> dict[str, Any]:
        """Get relationship system statistics."""
        return await self._client.request("GET", "/api/v1/relationships/stats")

    async def get_pair(
        self,
        agent_a: str,
        agent_b: str,
    ) -> dict[str, Any]:
        """Get detailed relationship between two specific agents."""
        return await self._client.request("GET", f"/api/v1/relationship/{agent_a}/{agent_b}")
