"""
Consensus Namespace API

Provides methods for consensus tracking and analysis:
- Find similar past debates
- Check settled topics
- Access domain history and consensus patterns
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ConsensusAPI:
    """
    Synchronous Consensus API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> similar = client.consensus.get_similar_debates("API rate limiting")
        >>> for debate in similar["debates"]:
        ...     print(debate["topic"], debate["similarity_score"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_similar_debates(
        self,
        topic: str,
        domain: str | None = None,
        min_similarity: float = 0.7,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Find debates similar to a given topic.

        Args:
            topic: Topic to search for
            domain: Filter by domain
            min_similarity: Minimum similarity threshold (0.0-1.0)
            limit: Maximum results

        Returns:
            Similar debates with similarity scores
        """
        params: dict[str, Any] = {
            "topic": topic,
            "min_similarity": min_similarity,
            "limit": limit,
        }
        if domain:
            params["domain"] = domain

        return self._client.request("GET", "/api/v1/consensus/similar", params=params)

    def get_settled_topics(
        self,
        domain: str | None = None,
        min_confidence: float = 0.8,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get topics with established consensus.

        Args:
            domain: Filter by domain
            min_confidence: Minimum consensus confidence
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Settled topics with consensus details
        """
        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "limit": limit,
            "offset": offset,
        }
        if domain:
            params["domain"] = domain

        return self._client.request("GET", "/api/v1/consensus/settled", params=params)

    def get_domain_history(
        self,
        domain: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get consensus history for a domain.

        Args:
            domain: Domain name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Domain history with debates and consensus evolution
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request(
            "GET", f"/api/v1/consensus/domains/{domain}/history", params=params
        )

    def check_consensus(self, debate_id: str) -> dict[str, Any]:
        """
        Check consensus status for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Consensus status including reached, confidence, dissent
        """
        return self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}")

    def get_consensus_proof(self, debate_id: str) -> dict[str, Any]:
        """
        Get cryptographic proof of consensus.

        Args:
            debate_id: Debate ID

        Returns:
            Consensus proof with hash and verification data
        """
        return self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}/proof")

    def get_dissenting_views(self, debate_id: str) -> dict[str, Any]:
        """
        Get dissenting views from a debate.

        Args:
            debate_id: Debate ID

        Returns:
            List of dissenting agents and their positions
        """
        return self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}/dissent")

    def get_stats(self, domain: str | None = None) -> dict[str, Any]:
        """
        Get consensus statistics.

        Args:
            domain: Filter by domain

        Returns:
            Statistics including consensus rates, average confidence, etc.
        """
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain

        return self._client.request("GET", "/api/v1/consensus/stats", params=params)


class AsyncConsensusAPI:
    """
    Asynchronous Consensus API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     similar = await client.consensus.get_similar_debates("API design")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_similar_debates(
        self,
        topic: str,
        domain: str | None = None,
        min_similarity: float = 0.7,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find debates similar to a given topic."""
        params: dict[str, Any] = {
            "topic": topic,
            "min_similarity": min_similarity,
            "limit": limit,
        }
        if domain:
            params["domain"] = domain

        return await self._client.request("GET", "/api/v1/consensus/similar", params=params)

    async def get_settled_topics(
        self,
        domain: str | None = None,
        min_confidence: float = 0.8,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get topics with established consensus."""
        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "limit": limit,
            "offset": offset,
        }
        if domain:
            params["domain"] = domain

        return await self._client.request("GET", "/api/v1/consensus/settled", params=params)

    async def get_domain_history(
        self,
        domain: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get consensus history for a domain."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request(
            "GET", f"/api/v1/consensus/domains/{domain}/history", params=params
        )

    async def check_consensus(self, debate_id: str) -> dict[str, Any]:
        """Check consensus status for a debate."""
        return await self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}")

    async def get_consensus_proof(self, debate_id: str) -> dict[str, Any]:
        """Get cryptographic proof of consensus."""
        return await self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}/proof")

    async def get_dissenting_views(self, debate_id: str) -> dict[str, Any]:
        """Get dissenting views from a debate."""
        return await self._client.request("GET", f"/api/v1/consensus/debates/{debate_id}/dissent")

    async def get_stats(self, domain: str | None = None) -> dict[str, Any]:
        """Get consensus statistics."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain

        return await self._client.request("GET", "/api/v1/consensus/stats", params=params)
