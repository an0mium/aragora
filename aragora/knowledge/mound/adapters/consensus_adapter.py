"""
ConsensusAdapter - Bridges ConsensusMemory to the Knowledge Mound.

This adapter enables the Knowledge Mound to query debate outcomes and
dissenting views stored in ConsensusMemory.

The adapter provides:
- Unified search interface (search_by_topic)
- Consensus-to-KnowledgeItem conversion
- Dissent tracking for organizational learning
- Domain-based filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from aragora.memory.consensus import (
        ConsensusMemory,
        ConsensusRecord,
        DissentRecord,
        SimilarDebate,
    )
    from aragora.knowledge.mound.types import KnowledgeItem, IngestionRequest

logger = logging.getLogger(__name__)


@dataclass
class ConsensusSearchResult:
    """Wrapper for consensus search results with similarity metadata."""

    record: "ConsensusRecord"
    similarity: float = 0.0
    dissents: List["DissentRecord"] = None

    def __post_init__(self) -> None:
        if self.dissents is None:
            self.dissents = []


class ConsensusAdapter:
    """
    Adapter that bridges ConsensusMemory to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_by_topic: Find similar debates and consensus outcomes
    - to_knowledge_item: Convert records to unified format
    - get_dissents: Retrieve dissenting views for a topic

    Usage:
        from aragora.memory.consensus import ConsensusMemory
        from aragora.knowledge.mound.adapters import ConsensusAdapter

        consensus = ConsensusMemory()
        adapter = ConsensusAdapter(consensus)

        # Search for similar debates
        results = await adapter.search_by_topic("rate limiting", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    def __init__(
        self,
        consensus: "ConsensusMemory",
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            consensus: The ConsensusMemory instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._consensus = consensus
        self._enable_dual_write = enable_dual_write

    @property
    def consensus(self) -> "ConsensusMemory":
        """Access the underlying ConsensusMemory."""
        return self._consensus

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        include_dissents: bool = True,
    ) -> List[ConsensusSearchResult]:
        """
        Search consensus memory by topic.

        This method wraps ConsensusMemory.find_similar_debates() to provide
        the interface expected by KnowledgeMound._query_consensus().

        Args:
            query: Topic to search for
            limit: Maximum results to return
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            include_dissents: Whether to include dissenting views

        Returns:
            List of ConsensusSearchResult with records and dissents
        """
        # Use ConsensusMemory's find_similar_debates method
        similar = self._consensus.find_similar_debates(
            topic=query,
            domain=domain,
            min_confidence=min_confidence,
            limit=limit,
        )

        results = []
        for debate in similar:
            # SimilarDebate has: consensus, dissents, similarity
            result = ConsensusSearchResult(
                record=debate.consensus,
                similarity=debate.similarity,
                dissents=debate.dissents if include_dissents else [],
            )
            results.append(result)

        return results

    def get(self, record_id: str) -> Optional["ConsensusRecord"]:
        """
        Get a specific consensus record by ID.

        Args:
            record_id: The record ID (may be prefixed with "cs_" from mound)

        Returns:
            ConsensusRecord or None
        """
        # Strip mound prefix if present
        if record_id.startswith("cs_"):
            record_id = record_id[3:]

        return self._consensus.get_consensus(record_id)

    async def get_async(self, record_id: str) -> Optional["ConsensusRecord"]:
        """Async version of get for compatibility."""
        return self.get(record_id)

    def to_knowledge_item(
        self,
        record: Union["ConsensusRecord", ConsensusSearchResult],
    ) -> "KnowledgeItem":
        """
        Convert a ConsensusRecord to a KnowledgeItem.

        Args:
            record: The consensus record or search result

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Handle both ConsensusRecord and ConsensusSearchResult
        if isinstance(record, ConsensusSearchResult):
            consensus = record.record
            similarity = record.similarity
        else:
            consensus = record
            similarity = 1.0

        # Map consensus strength to confidence level
        strength_to_confidence = {
            "unanimous": ConfidenceLevel.VERIFIED,
            "strong": ConfidenceLevel.HIGH,
            "moderate": ConfidenceLevel.MEDIUM,
            "weak": ConfidenceLevel.LOW,
            "split": ConfidenceLevel.LOW,
            "contested": ConfidenceLevel.UNVERIFIED,
        }
        confidence = strength_to_confidence.get(
            consensus.strength.value, ConfidenceLevel.MEDIUM
        )

        # Use conclusion as content, with topic as fallback
        content = consensus.conclusion or consensus.topic

        # Build metadata
        metadata: Dict[str, Any] = {
            "topic": consensus.topic,
            "strength": consensus.strength.value,
            "domain": consensus.domain,
            "tags": consensus.tags,
            "participating_agents": consensus.participating_agents,
            "agreeing_agents": consensus.agreeing_agents,
            "dissenting_agents": consensus.dissenting_agents,
            "key_claims": consensus.key_claims,
            "supporting_evidence": consensus.supporting_evidence,
            "dissent_ids": consensus.dissent_ids,
            "rounds": consensus.rounds,
            "debate_duration_seconds": consensus.debate_duration_seconds,
            "agreement_ratio": consensus.compute_agreement_ratio(),
            "similarity": similarity,
        }

        if consensus.supersedes:
            metadata["supersedes"] = consensus.supersedes
        if consensus.superseded_by:
            metadata["superseded_by"] = consensus.superseded_by

        return KnowledgeItem(
            id=f"cs_{consensus.id}",
            content=content,
            source=KnowledgeSource.CONSENSUS,
            source_id=consensus.id,
            confidence=confidence,
            created_at=consensus.timestamp,
            updated_at=consensus.timestamp,
            metadata=metadata,
            importance=consensus.confidence,
        )

    def dissent_to_knowledge_item(
        self,
        dissent: "DissentRecord",
    ) -> "KnowledgeItem":
        """
        Convert a DissentRecord to a KnowledgeItem.

        Dissenting views are valuable organizational knowledge that can
        inform future decisions and reveal edge cases.

        Args:
            dissent: The dissent record

        Returns:
            KnowledgeItem for the dissenting view
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # All dissents are initially low confidence since they're minority views
        # but they can be valuable for edge case detection
        confidence = ConfidenceLevel.LOW
        if dissent.dissent_type.value == "risk_warning":
            confidence = ConfidenceLevel.MEDIUM  # Risk warnings deserve attention

        metadata: Dict[str, Any] = {
            "debate_id": dissent.debate_id,
            "agent_id": dissent.agent_id,
            "dissent_type": dissent.dissent_type.value,
            "reasoning": dissent.reasoning,
            "acknowledged": dissent.acknowledged,
            "rebuttal": dissent.rebuttal,
        }

        return KnowledgeItem(
            id=f"ds_{dissent.id}",
            content=dissent.content,
            source=KnowledgeSource.CONSENSUS,  # Dissent is part of consensus process
            source_id=dissent.id,
            confidence=confidence,
            created_at=dissent.timestamp,
            updated_at=dissent.timestamp,
            metadata=metadata,
            importance=dissent.confidence,
        )

    def get_dissents_for_topic(
        self,
        topic: str,
        limit: int = 20,
    ) -> List["DissentRecord"]:
        """
        Get dissenting views related to a topic.

        Args:
            topic: Topic to search for dissents
            limit: Maximum dissents to return

        Returns:
            List of DissentRecord objects
        """
        return self._consensus.find_relevant_dissent(topic=topic, limit=limit)

    def get_risk_warnings(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> List["DissentRecord"]:
        """
        Get risk warnings from debates.

        Risk warnings are particularly valuable for organizational learning
        as they capture potential issues identified during debates.

        Args:
            topic: Optional topic filter
            limit: Maximum warnings to return

        Returns:
            List of DissentRecord with type RISK_WARNING
        """
        return self._consensus.find_risk_warnings(topic=topic, limit=limit)

    def get_contrarian_views(
        self,
        limit: int = 10,
    ) -> List["DissentRecord"]:
        """
        Get fundamental disagreements from debates.

        These represent strongly held alternative views that may indicate
        areas of genuine uncertainty or multiple valid approaches.

        Args:
            limit: Maximum views to return

        Returns:
            List of DissentRecord with type FUNDAMENTAL_DISAGREEMENT
        """
        return self._consensus.find_contrarian_views(limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the consensus memory."""
        return self._consensus.get_stats()
