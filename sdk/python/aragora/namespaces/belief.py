"""
Belief Network Namespace API

Provides access to belief network analysis, cruxes, and provenance tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class BeliefAPI:
    """Synchronous Belief Network API for claim analysis and provenance."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_cruxes(
        self,
        debate_id: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Get key claims that would most impact the debate outcome.

        Args:
            debate_id: The debate ID.
            top_k: Number of top cruxes to return (1-10, default: 3).

        Returns:
            Cruxes with their impact scores and claim details.
        """
        params = {"top_k": top_k}
        return self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/cruxes", params=params
        )

    def get_load_bearing_claims(
        self,
        debate_id: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Get claims with highest centrality (most load-bearing).

        Args:
            debate_id: The debate ID.
            limit: Maximum claims to return (1-20, default: 5).

        Returns:
            Load-bearing claims with centrality scores.
        """
        params = {"limit": limit}
        return self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/load-bearing-claims", params=params
        )

    def get_graph(
        self,
        debate_id: str,
        include_cruxes: bool = True,
    ) -> dict[str, Any]:
        """Get belief network as a graph structure for visualization.

        Args:
            debate_id: The debate ID.
            include_cruxes: Whether to include crux detection (default: True).

        Returns:
            Graph structure with nodes (claims) and links (influences).
        """
        params = {"include_cruxes": str(include_cruxes).lower()}
        return self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/graph", params=params
        )

    def export(
        self,
        debate_id: str,
        format: Literal["json", "graphml", "csv"] = "json",
    ) -> dict[str, Any]:
        """Export belief network in various formats.

        Args:
            debate_id: The debate ID.
            format: Export format (json, graphml, csv).

        Returns:
            Exported belief network in requested format.
        """
        params = {"format": format}
        return self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/export", params=params
        )

    def get_claim_support(
        self,
        debate_id: str,
        claim_id: str,
    ) -> dict[str, Any]:
        """Get verification status of all evidence supporting a claim.

        Args:
            debate_id: The debate ID.
            claim_id: The claim ID.

        Returns:
            Claim support status and evidence details.
        """
        return self._client.request(
            "GET", f"/api/v1/provenance/{debate_id}/claims/{claim_id}/support"
        )

    def get_graph_stats(
        self,
        debate_id: str,
    ) -> dict[str, Any]:
        """Get argument graph statistics for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            Graph statistics including node counts, edge counts, and density.
        """
        return self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")


class AsyncBeliefAPI:
    """Asynchronous Belief Network API for claim analysis and provenance."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_cruxes(
        self,
        debate_id: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Get key claims that would most impact the debate outcome."""
        params = {"top_k": top_k}
        return await self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/cruxes", params=params
        )

    async def get_load_bearing_claims(
        self,
        debate_id: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Get claims with highest centrality (most load-bearing)."""
        params = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/load-bearing-claims", params=params
        )

    async def get_graph(
        self,
        debate_id: str,
        include_cruxes: bool = True,
    ) -> dict[str, Any]:
        """Get belief network as a graph structure for visualization."""
        params = {"include_cruxes": str(include_cruxes).lower()}
        return await self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/graph", params=params
        )

    async def export(
        self,
        debate_id: str,
        format: Literal["json", "graphml", "csv"] = "json",
    ) -> dict[str, Any]:
        """Export belief network in various formats."""
        params = {"format": format}
        return await self._client.request(
            "GET", f"/api/v1/belief-network/{debate_id}/export", params=params
        )

    async def get_claim_support(
        self,
        debate_id: str,
        claim_id: str,
    ) -> dict[str, Any]:
        """Get verification status of all evidence supporting a claim."""
        return await self._client.request(
            "GET", f"/api/v1/provenance/{debate_id}/claims/{claim_id}/support"
        )

    async def get_graph_stats(
        self,
        debate_id: str,
    ) -> dict[str, Any]:
        """Get argument graph statistics for a debate."""
        return await self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")
