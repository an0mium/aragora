"""
Consensus API for the Aragora Python SDK.

Provides access to consensus memory, dissents, and risk warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient


@dataclass
class SimilarDebate:
    """A similar past debate."""

    id: str
    topic: str
    conclusion: str
    strength: str
    confidence: float
    similarity: float
    timestamp: str
    dissent_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarDebate":
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            conclusion=data.get("conclusion", ""),
            strength=data.get("strength", "unknown"),
            confidence=data.get("confidence", 0.0),
            similarity=data.get("similarity", 0.0),
            timestamp=data.get("timestamp", ""),
            dissent_count=data.get("dissent_count", 0),
        )


@dataclass
class SettledTopic:
    """A topic that has been settled by consensus."""

    topic: str
    conclusion: str
    confidence: float
    strength: str
    last_debated: str
    debate_count: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettledTopic":
        return cls(
            topic=data.get("topic", ""),
            conclusion=data.get("conclusion", ""),
            confidence=data.get("confidence", 0.0),
            strength=data.get("strength", "unknown"),
            last_debated=data.get("last_debated", ""),
            debate_count=data.get("debate_count", 1),
        )


@dataclass
class Dissent:
    """A dissenting view from a debate."""

    id: str
    debate_id: str
    agent_id: str
    dissent_type: str
    content: str
    reasoning: str
    confidence: float
    acknowledged: bool = False
    rebuttal: str = ""
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dissent":
        return cls(
            id=data.get("id", ""),
            debate_id=data.get("debate_id", ""),
            agent_id=data.get("agent_id", ""),
            dissent_type=data.get("dissent_type", ""),
            content=data.get("content", ""),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.0),
            acknowledged=data.get("acknowledged", False),
            rebuttal=data.get("rebuttal", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class RiskWarning:
    """A risk warning from debates."""

    id: str
    debate_id: str
    agent_id: str
    content: str
    reasoning: str
    severity: str = "medium"
    acknowledged: bool = False
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskWarning":
        return cls(
            id=data.get("id", ""),
            debate_id=data.get("debate_id", ""),
            agent_id=data.get("agent_id", ""),
            content=data.get("content", ""),
            reasoning=data.get("reasoning", ""),
            severity=data.get("severity", "medium"),
            acknowledged=data.get("acknowledged", False),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ConsensusStats:
    """Statistics about consensus memory."""

    total_consensuses: int
    total_dissents: int
    by_strength: Dict[str, int]
    by_domain: Dict[str, int]
    avg_confidence: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusStats":
        return cls(
            total_consensuses=data.get("total_consensuses", 0),
            total_dissents=data.get("total_dissents", 0),
            by_strength=data.get("by_strength", {}),
            by_domain=data.get("by_domain", {}),
            avg_confidence=data.get("avg_confidence", 0.0),
        )


class ConsensusAPI:
    """
    API interface for consensus memory.

    Provides access to past consensus decisions, dissenting views,
    and risk warnings from debates.

    Example:
        # Find similar past debates
        similar = client.consensus.find_similar("rate limiting")
        for debate in similar:
            print(f"{debate.topic}: {debate.conclusion}")

        # Get settled topics
        settled = client.consensus.get_settled(domain="architecture")

        # Get risk warnings
        warnings = client.consensus.get_risk_warnings(limit=10)
    """

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def find_similar(
        self,
        topic: str,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[SimilarDebate]:
        """
        Find similar past debates on a topic.

        Args:
            topic: The topic to search for
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return

        Returns:
            List of SimilarDebate objects
        """
        params = {
            "topic": topic,
            "limit": limit,
            "min_confidence": min_confidence,
        }
        if domain:
            params["domain"] = domain

        response = self._client._get("/api/consensus/similar", params=params)
        debates = response.get("debates", [])
        return [SimilarDebate.from_dict(d) for d in debates]

    async def find_similar_async(
        self,
        topic: str,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[SimilarDebate]:
        """Async version of find_similar."""
        params = {
            "topic": topic,
            "limit": limit,
            "min_confidence": min_confidence,
        }
        if domain:
            params["domain"] = domain

        response = await self._client._get_async("/api/consensus/similar", params=params)
        debates = response.get("debates", [])
        return [SimilarDebate.from_dict(d) for d in debates]

    def get_settled(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.7,
        limit: int = 20,
    ) -> List[SettledTopic]:
        """
        Get topics that have been settled by consensus.

        Args:
            domain: Optional domain filter
            min_confidence: Minimum confidence for settled topics
            limit: Maximum results to return

        Returns:
            List of SettledTopic objects
        """
        params = {"limit": limit, "min_confidence": min_confidence}
        if domain:
            params["domain"] = domain

        response = self._client._get("/api/consensus/settled", params=params)
        topics = response.get("topics", [])
        return [SettledTopic.from_dict(t) for t in topics]

    async def get_settled_async(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.7,
        limit: int = 20,
    ) -> List[SettledTopic]:
        """Async version of get_settled."""
        params = {"limit": limit, "min_confidence": min_confidence}
        if domain:
            params["domain"] = domain

        response = await self._client._get_async("/api/consensus/settled", params=params)
        topics = response.get("topics", [])
        return [SettledTopic.from_dict(t) for t in topics]

    def get_dissents(
        self,
        topic: Optional[str] = None,
        dissent_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dissent]:
        """
        Get dissenting views from debates.

        Args:
            topic: Optional topic filter
            dissent_type: Optional type filter (minor_quibble, alternative_approach,
                          fundamental_disagreement, edge_case_concern, risk_warning)
            limit: Maximum results to return

        Returns:
            List of Dissent objects
        """
        params: Dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if dissent_type:
            params["type"] = dissent_type

        response = self._client._get("/api/consensus/dissents", params=params)
        dissents = response.get("dissents", [])
        return [Dissent.from_dict(d) for d in dissents]

    async def get_dissents_async(
        self,
        topic: Optional[str] = None,
        dissent_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dissent]:
        """Async version of get_dissents."""
        params: Dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if dissent_type:
            params["type"] = dissent_type

        response = await self._client._get_async("/api/consensus/dissents", params=params)
        dissents = response.get("dissents", [])
        return [Dissent.from_dict(d) for d in dissents]

    def get_risk_warnings(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> List[RiskWarning]:
        """
        Get risk warnings from debates.

        Risk warnings are particularly valuable for organizational learning.

        Args:
            topic: Optional topic filter
            limit: Maximum results to return

        Returns:
            List of RiskWarning objects
        """
        params: Dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic

        response = self._client._get("/api/consensus/risk-warnings", params=params)
        warnings = response.get("warnings", [])
        return [RiskWarning.from_dict(w) for w in warnings]

    async def get_risk_warnings_async(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> List[RiskWarning]:
        """Async version of get_risk_warnings."""
        params: Dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic

        response = await self._client._get_async("/api/consensus/risk-warnings", params=params)
        warnings = response.get("warnings", [])
        return [RiskWarning.from_dict(w) for w in warnings]

    def get_contrarian_views(self, limit: int = 10) -> List[Dissent]:
        """
        Get fundamental disagreements from debates.

        These represent strongly held alternative views.

        Args:
            limit: Maximum results to return

        Returns:
            List of Dissent objects with fundamental_disagreement type
        """
        response = self._client._get("/api/consensus/contrarian", params={"limit": limit})
        views = response.get("views", [])
        return [Dissent.from_dict(v) for v in views]

    async def get_contrarian_views_async(self, limit: int = 10) -> List[Dissent]:
        """Async version of get_contrarian_views."""
        response = await self._client._get_async("/api/consensus/contrarian", params={"limit": limit})
        views = response.get("views", [])
        return [Dissent.from_dict(v) for v in views]

    def get_stats(self) -> ConsensusStats:
        """
        Get consensus memory statistics.

        Returns:
            ConsensusStats with aggregate metrics
        """
        response = self._client._get("/api/consensus/stats")
        return ConsensusStats.from_dict(response)

    async def get_stats_async(self) -> ConsensusStats:
        """Async version of get_stats."""
        response = await self._client._get_async("/api/consensus/stats")
        return ConsensusStats.from_dict(response)
