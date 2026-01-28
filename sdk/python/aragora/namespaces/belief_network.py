"""
Belief Network Namespace API.

Provides access to debate belief networks, cruxes, and load-bearing claims.
These represent the epistemic structure of arguments within debates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ExportFormat = Literal["json", "graphml", "csv"]
NodeType = Literal["premise", "inference", "conclusion"]
RelationshipType = Literal["supports", "contradicts", "qualifies", "depends_on", "influences"]


class Crux(TypedDict, total=False):
    """A crux - a key disagreement point that would change positions if resolved."""

    id: str
    debate_id: str
    description: str
    agents_pro: list[str]
    agents_con: list[str]
    importance: float
    resolved: bool
    resolution: str | None
    identified_at: str


class LoadBearingClaim(TypedDict, total=False):
    """A load-bearing claim - a claim that many other claims depend on."""

    id: str
    debate_id: str
    claim: str
    agent_name: str
    confidence: float
    dependents_count: int
    dependent_claims: list[str]
    supporting_evidence: list[str]
    created_at: str


class BeliefNode(TypedDict, total=False):
    """Belief network node."""

    id: str
    claim: str
    agent_name: str
    confidence: float
    type: str
    supporting_nodes: list[str]
    opposing_nodes: list[str]


class BeliefGraphNode(TypedDict, total=False):
    """Belief network graph node for visualization."""

    id: str
    claim_id: str
    statement: str
    author: str
    centrality: float
    is_crux: bool
    crux_score: float | None
    entropy: float
    belief: dict[str, float] | None


class BeliefGraphEdge(TypedDict, total=False):
    """Belief network graph edge for visualization."""

    source: str
    target: str
    weight: float
    type: str


class BeliefGraphResponse(TypedDict, total=False):
    """Belief network graph response."""

    nodes: list[BeliefGraphNode]
    links: list[BeliefGraphEdge]
    metadata: dict[str, Any]


class BeliefNetworkAPI:
    """
    Synchronous Belief Network API.

    Provides methods for analyzing debate belief structures:
    - Identify cruxes (key disagreements)
    - Find load-bearing claims
    - Explore argument dependencies
    - Export network visualizations

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Get cruxes for a debate
        >>> result = client.belief_network.get_cruxes("debate-123")
        >>> for crux in result["cruxes"]:
        ...     print(f"Crux: {crux['description']} (importance: {crux['importance']})")
        >>> # Get load-bearing claims
        >>> claims = client.belief_network.get_load_bearing_claims("debate-123")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def get_cruxes(
        self,
        debate_id: str,
        resolved: bool | None = None,
        min_importance: float | None = None,
    ) -> dict[str, Any]:
        """
        Get cruxes (key disagreements) for a debate.

        Cruxes are points where, if one party changed their mind,
        it would significantly affect their overall position.

        Args:
            debate_id: The debate ID.
            resolved: Filter by resolution status.
            min_importance: Minimum importance score (0.0-1.0).

        Returns:
            Dict with 'cruxes' list and 'total' count.
        """
        params: dict[str, Any] = {}
        if resolved is not None:
            params["resolved"] = resolved
        if min_importance is not None:
            params["min_importance"] = min_importance
        return self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/cruxes",
            params=params if params else None,
        )

    def get_load_bearing_claims(
        self,
        debate_id: str,
        min_dependents: int | None = None,
        agent: str | None = None,
    ) -> dict[str, Any]:
        """
        Get load-bearing claims for a debate.

        Load-bearing claims are claims that many other claims depend on.
        If these are invalidated, the overall argument structure may collapse.

        Args:
            debate_id: The debate ID.
            min_dependents: Minimum number of dependent claims.
            agent: Filter by agent name.

        Returns:
            Dict with 'claims' list and 'total' count.
        """
        params: dict[str, Any] = {}
        if min_dependents is not None:
            params["min_dependents"] = min_dependents
        if agent:
            params["agent"] = agent
        return self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/load-bearing-claims",
            params=params if params else None,
        )

    def get_graph(
        self,
        debate_id: str,
        include_cruxes: bool = True,
    ) -> BeliefGraphResponse:
        """
        Get belief network as a graph structure for visualization.

        Returns nodes (claims) and links (influence relationships) suitable
        for force-directed graph rendering.

        Args:
            debate_id: The debate ID.
            include_cruxes: Include crux analysis in nodes (default: True).

        Returns:
            BeliefGraphResponse with nodes, links, and metadata.
        """
        params: dict[str, Any] = {}
        if not include_cruxes:
            params["include_cruxes"] = False
        return self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/graph",
            params=params if params else None,
        )

    def export(
        self,
        debate_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export belief network in various formats.

        Supported formats:
        - json: Full JSON structure with nodes, edges, and summary
        - graphml: GraphML format for Gephi/yEd visualization tools
        - csv: CSV-friendly structure with separate nodes and edges arrays

        Args:
            debate_id: The debate ID.
            format: Export format ('json', 'graphml', or 'csv').

        Returns:
            Export response with format-specific content.
        """
        return self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/export",
            params={"format": format},
        )


class AsyncBeliefNetworkAPI:
    """Asynchronous Belief Network API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def get_cruxes(
        self,
        debate_id: str,
        resolved: bool | None = None,
        min_importance: float | None = None,
    ) -> dict[str, Any]:
        """Get cruxes (key disagreements) for a debate."""
        params: dict[str, Any] = {}
        if resolved is not None:
            params["resolved"] = resolved
        if min_importance is not None:
            params["min_importance"] = min_importance
        return await self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/cruxes",
            params=params if params else None,
        )

    async def get_load_bearing_claims(
        self,
        debate_id: str,
        min_dependents: int | None = None,
        agent: str | None = None,
    ) -> dict[str, Any]:
        """Get load-bearing claims for a debate."""
        params: dict[str, Any] = {}
        if min_dependents is not None:
            params["min_dependents"] = min_dependents
        if agent:
            params["agent"] = agent
        return await self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/load-bearing-claims",
            params=params if params else None,
        )

    async def get_graph(
        self,
        debate_id: str,
        include_cruxes: bool = True,
    ) -> BeliefGraphResponse:
        """Get belief network as a graph structure for visualization."""
        params: dict[str, Any] = {}
        if not include_cruxes:
            params["include_cruxes"] = False
        return await self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/graph",
            params=params if params else None,
        )

    async def export(
        self,
        debate_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export belief network in various formats."""
        return await self._client.request(
            "GET",
            f"/api/v1/belief-network/{debate_id}/export",
            params={"format": format},
        )
