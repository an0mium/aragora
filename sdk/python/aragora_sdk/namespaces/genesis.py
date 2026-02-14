"""
Genesis Namespace API

Provides access to evolution visibility, genome tracking, and lineage analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class GenesisAPI:
    """Synchronous Genesis API for evolution visibility and genome management."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_stats(self) -> dict[str, Any]:
        """Get overall genesis statistics for evolution visibility.

        Returns:
            Statistics including event counts, births, deaths, and fitness trends.
        """
        return self._client.request("GET", "/api/v1/genesis/stats")

    def get_events(
        self,
        limit: int = 20,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """Get recent genesis events.

        Args:
            limit: Maximum events to return (default: 20, max: 100).
            event_type: Filter by event type (optional).

        Returns:
            List of genesis events with timestamps and data.
        """
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return self._client.request("GET", "/api/v1/genesis/events", params=params)

    def get_genomes(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get all genomes with pagination.

        Args:
            limit: Maximum genomes to return (default: 50, max: 200).
            offset: Number of genomes to skip.

        Returns:
            Paginated list of genomes with fitness scores and traits.
        """
        params = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/genesis/genomes", params=params)

    def get_top_genomes(
        self,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get top genomes by fitness score.

        Args:
            limit: Maximum genomes to return (default: 10, max: 50).

        Returns:
            Top-performing genomes sorted by fitness.
        """
        params = {"limit": limit}
        return self._client.request("GET", "/api/v1/genesis/genomes/top", params=params)

    def get_genome(
        self,
        genome_id: str,
    ) -> dict[str, Any]:
        """Get a single genome by ID.

        Args:
            genome_id: The genome ID.

        Returns:
            Full genome details including traits, expertise, and lineage.
        """
        return self._client.request("GET", f"/api/v1/genesis/genomes/{genome_id}")

    def get_population(self) -> dict[str, Any]:
        """Get the active population and its status.

        Returns:
            Population details including genomes, generation, and average fitness.
        """
        return self._client.request("GET", "/api/v1/genesis/population")

    def get_lineage(
        self,
        genome_id: str,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Get the lineage (ancestry) of a genome.

        Args:
            genome_id: The genome to trace.
            max_depth: Maximum depth to trace (default: 10, max: 50).

        Returns:
            Lineage data including ancestors and their event types.
        """
        params = {"max_depth": max_depth}
        return self._client.request("GET", f"/api/v1/genesis/lineage/{genome_id}", params=params)

    def get_descendants(
        self,
        genome_id: str,
        max_depth: int = 5,
    ) -> dict[str, Any]:
        """Get all descendants of a genome.

        Args:
            genome_id: The genome to find descendants of.
            max_depth: Maximum depth to search (default: 5, max: 20).

        Returns:
            Descendants with depth, fitness, and parent relationships.
        """
        params = {"max_depth": max_depth}
        return self._client.request(
            "GET", f"/api/v1/genesis/descendants/{genome_id}", params=params
        )

    def get_debate_tree(
        self,
        debate_id: str,
    ) -> dict[str, Any]:
        """Get the fractal tree structure for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            Tree structure with nodes showing genome evolution during debate.
        """
        return self._client.request("GET", f"/api/v1/genesis/tree/{debate_id}")


class AsyncGenesisAPI:
    """Asynchronous Genesis API for evolution visibility and genome management."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_stats(self) -> dict[str, Any]:
        """Get overall genesis statistics for evolution visibility."""
        return await self._client.request("GET", "/api/v1/genesis/stats")

    async def get_events(
        self,
        limit: int = 20,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """Get recent genesis events."""
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return await self._client.request("GET", "/api/v1/genesis/events", params=params)

    async def get_genomes(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get all genomes with pagination."""
        params = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/genesis/genomes", params=params)

    async def get_top_genomes(
        self,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get top genomes by fitness score."""
        params = {"limit": limit}
        return await self._client.request("GET", "/api/v1/genesis/genomes/top", params=params)

    async def get_genome(
        self,
        genome_id: str,
    ) -> dict[str, Any]:
        """Get a single genome by ID."""
        return await self._client.request("GET", f"/api/v1/genesis/genomes/{genome_id}")

    async def get_population(self) -> dict[str, Any]:
        """Get the active population and its status."""
        return await self._client.request("GET", "/api/v1/genesis/population")

    async def get_lineage(
        self,
        genome_id: str,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Get the lineage (ancestry) of a genome."""
        params = {"max_depth": max_depth}
        return await self._client.request(
            "GET", f"/api/v1/genesis/lineage/{genome_id}", params=params
        )

    async def get_descendants(
        self,
        genome_id: str,
        max_depth: int = 5,
    ) -> dict[str, Any]:
        """Get all descendants of a genome."""
        params = {"max_depth": max_depth}
        return await self._client.request(
            "GET", f"/api/v1/genesis/descendants/{genome_id}", params=params
        )

    async def get_debate_tree(
        self,
        debate_id: str,
    ) -> dict[str, Any]:
        """Get the fractal tree structure for a debate."""
        return await self._client.request("GET", f"/api/v1/genesis/tree/{debate_id}")
