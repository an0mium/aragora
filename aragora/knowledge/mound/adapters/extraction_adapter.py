"""
ExtractionAdapter - Bridges Knowledge Extraction to the Knowledge Mound.

This adapter enables integration between the knowledge extraction system
and the Knowledge Mound:

- Entity extraction from debate content
- Relationship extraction between concepts
- Knowledge graph operations (add/update/query)
- Batch processing for multiple debates
- Promotion of high-confidence claims to KM

The adapter provides:
- Unified extraction interface
- Bidirectional sync with Knowledge Mound
- Confidence-based filtering
- Topic and relationship tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.ops.extraction import (
    DebateKnowledgeExtractor,
    ExtractedClaim,
    ExtractedRelationship,
    ExtractionConfig,
    ExtractionResult,
    ExtractionType,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


class ExtractionAdapterError(Exception):
    """Base exception for extraction adapter errors."""

    pass


class ExtractionNotFoundError(ExtractionAdapterError):
    """Raised when an extraction result is not found."""

    pass


class ExtractionEngineUnavailableError(ExtractionAdapterError):
    """Raised when the extraction engine is not available."""

    pass


@dataclass
class ExtractionSearchResult:
    """Wrapper for extraction search results with adapter metadata."""

    claim: ExtractedClaim
    relevance_score: float = 0.0
    matched_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim.to_dict(),
            "relevance_score": self.relevance_score,
            "matched_topics": self.matched_topics,
        }


@dataclass
class RelationshipSearchResult:
    """Wrapper for relationship search results."""

    relationship: ExtractedRelationship
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relationship": self.relationship.to_dict(),
            "relevance_score": self.relevance_score,
        }


@dataclass
class KnowledgeGraphNode:
    """A node in the knowledge graph."""

    id: str
    concept: str
    claim_ids: list[str] = field(default_factory=list)
    relationship_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "concept": self.concept,
            "claim_ids": self.claim_ids,
            "relationship_ids": self.relationship_ids,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BatchExtractionResult:
    """Result of batch extraction processing."""

    debate_ids: list[str]
    total_claims: int
    total_relationships: int
    promoted_count: int
    failed_debate_ids: list[str]
    duration_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if batch was successful."""
        return len(self.errors) == 0 and len(self.failed_debate_ids) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_ids": self.debate_ids,
            "total_claims": self.total_claims,
            "total_relationships": self.total_relationships,
            "promoted_count": self.promoted_count,
            "failed_debate_ids": self.failed_debate_ids,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "success": self.success,
        }


class ExtractionAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges knowledge extraction to the Knowledge Mound.

    Provides methods for:
    - Entity extraction from debate messages
    - Relationship extraction between concepts
    - Knowledge graph construction and querying
    - Batch processing of multiple debates
    - Promotion of claims to Knowledge Mound

    Usage:
        from aragora.knowledge.mound.adapters import ExtractionAdapter

        adapter = ExtractionAdapter()

        # Extract from a single debate
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=[{"agent_id": "claude", "content": "..."}],
        )

        # Query extracted knowledge
        claims = await adapter.search_claims("machine learning", limit=10)
    """

    adapter_name = "extraction"
    ID_PREFIX = "ext_"

    def __init__(
        self,
        mound: KnowledgeMound | None = None,
        config: ExtractionConfig | None = None,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
        auto_promote: bool = False,
        min_confidence_for_promotion: float = 0.6,
    ):
        """
        Initialize the adapter.

        Args:
            mound: Optional KnowledgeMound instance for storage
            config: Optional extraction configuration
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events
            enable_resilience: If True, enables circuit breaker protection
            auto_promote: If True, automatically promote high-confidence claims to KM
            min_confidence_for_promotion: Minimum confidence for auto-promotion
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )

        self._mound = mound
        self._config = config or ExtractionConfig()
        self._extractor = DebateKnowledgeExtractor(self._config)
        self._auto_promote = auto_promote
        self._min_confidence_for_promotion = min_confidence_for_promotion

        # Knowledge graph storage
        self._graph_nodes: dict[str, KnowledgeGraphNode] = {}
        self._extraction_results: dict[str, ExtractionResult] = {}

        # Statistics
        self._total_debates_processed = 0
        self._total_claims_extracted = 0
        self._total_relationships_extracted = 0
        self._total_promoted = 0

    def set_mound(self, mound: KnowledgeMound) -> None:
        """Set the Knowledge Mound instance.

        Args:
            mound: KnowledgeMound instance
        """
        self._mound = mound

    def set_config(self, config: ExtractionConfig) -> None:
        """Set extraction configuration.

        Args:
            config: New extraction configuration
        """
        self._config = config
        self._extractor = DebateKnowledgeExtractor(config)

    async def extract_from_debate(
        self,
        debate_id: str,
        messages: list[dict[str, Any]],
        consensus_text: str | None = None,
        topic: str | None = None,
        workspace_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract knowledge from a debate.

        Args:
            debate_id: ID of the debate
            messages: List of debate messages
            consensus_text: Optional consensus conclusion
            topic: Optional debate topic
            workspace_id: Optional workspace ID for KM storage

        Returns:
            ExtractionResult with extracted claims and relationships

        Raises:
            ExtractionAdapterError: If extraction fails
        """
        start_time = time.time()
        success = False

        try:
            with self._timed_operation("extract", debate_id=debate_id):
                result = await self._extractor.extract_from_debate(
                    debate_id=debate_id,
                    messages=messages,
                    consensus_text=consensus_text,
                    topic=topic,
                )

                # Store result
                self._extraction_results[debate_id] = result

                # Update knowledge graph
                await self._update_graph(result)

                # Update stats
                self._total_debates_processed += 1
                self._total_claims_extracted += len(result.claims)
                self._total_relationships_extracted += len(result.relationships)

                # Auto-promote if enabled
                promoted = 0
                if self._auto_promote and self._mound and workspace_id:
                    promoted = await self.promote_claims(
                        workspace_id=workspace_id,
                        debate_id=debate_id,
                        min_confidence=self._min_confidence_for_promotion,
                    )
                    result.promoted_to_mound = promoted
                    self._total_promoted += promoted

                # Emit event
                self._emit_event(
                    "knowledge_extracted",
                    {
                        "debate_id": debate_id,
                        "claims_extracted": len(result.claims),
                        "relationships_extracted": len(result.relationships),
                        "promoted": promoted,
                        "duration_ms": result.extraction_duration_ms,
                    },
                )

                success = True
                return result

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.error(f"Failed to extract from debate {debate_id}: {e}")
            raise ExtractionAdapterError(f"Extraction failed: {e}") from e
        finally:
            self._record_metric("extract", success, time.time() - start_time)

    async def batch_extract(
        self,
        debates: list[dict[str, Any]],
        workspace_id: str | None = None,
        max_concurrent: int = 5,
    ) -> BatchExtractionResult:
        """
        Extract knowledge from multiple debates in batch.

        Args:
            debates: List of debate dicts with keys:
                - debate_id: Debate ID
                - messages: List of messages
                - consensus_text: Optional consensus
                - topic: Optional topic
            workspace_id: Optional workspace for promotion
            max_concurrent: Max concurrent extractions

        Returns:
            BatchExtractionResult with aggregate statistics
        """
        start_time = time.time()
        debate_ids = []
        failed_ids = []
        errors = []
        total_claims = 0
        total_relationships = 0
        promoted = 0

        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(debate: dict[str, Any]) -> ExtractionResult | None:
            async with semaphore:
                try:
                    return await self.extract_from_debate(
                        debate_id=debate["debate_id"],
                        messages=debate.get("messages", []),
                        consensus_text=debate.get("consensus_text"),
                        topic=debate.get("topic"),
                        workspace_id=workspace_id,
                    )
                except (RuntimeError, ValueError, AttributeError, KeyError) as e:  # noqa: BLE001 - adapter isolation
                    failed_ids.append(debate["debate_id"])
                    logger.warning("Extraction failed for debate %s: %s", debate["debate_id"], e)
                    errors.append(f"{debate['debate_id']}: extraction failed")
                    return None

        # Process all debates
        tasks = [extract_one(d) for d in debates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            debate_id = debates[i]["debate_id"]
            debate_ids.append(debate_id)

            if isinstance(result, ExtractionResult):
                total_claims += len(result.claims)
                total_relationships += len(result.relationships)
                promoted += result.promoted_to_mound
            elif isinstance(result, Exception):
                if debate_id not in failed_ids:
                    failed_ids.append(debate_id)
                    errors.append(f"{debate_id}: {str(result)}")

        duration_ms = (time.time() - start_time) * 1000

        return BatchExtractionResult(
            debate_ids=debate_ids,
            total_claims=total_claims,
            total_relationships=total_relationships,
            promoted_count=promoted,
            failed_debate_ids=failed_ids,
            duration_ms=duration_ms,
            errors=errors,
        )

    async def promote_claims(
        self,
        workspace_id: str,
        debate_id: str | None = None,
        min_confidence: float | None = None,
        claim_ids: list[str] | None = None,
    ) -> int:
        """
        Promote extracted claims to Knowledge Mound.

        Args:
            workspace_id: Workspace to store claims in
            debate_id: Optional filter by debate ID
            min_confidence: Minimum confidence threshold
            claim_ids: Optional specific claim IDs to promote

        Returns:
            Number of claims promoted

        Raises:
            ExtractionAdapterError: If promotion fails
        """
        if not self._mound:
            logger.warning("Cannot promote claims: Knowledge Mound not configured")
            return 0

        min_conf = min_confidence or self._min_confidence_for_promotion

        try:
            # Get claims to promote
            claims_to_promote: list[ExtractedClaim] = []

            if claim_ids:
                # Specific claims requested
                for result in self._extraction_results.values():
                    for claim in result.claims:
                        if claim.id in claim_ids and claim.confidence >= min_conf:
                            claims_to_promote.append(claim)
            elif debate_id:
                # Filter by debate
                result = self._extraction_results.get(debate_id)
                if result:
                    claims_to_promote = [c for c in result.claims if c.confidence >= min_conf]
            else:
                # All claims meeting threshold
                for result in self._extraction_results.values():
                    claims_to_promote.extend(c for c in result.claims if c.confidence >= min_conf)

            # Promote to mound
            promoted = await self._extractor.promote_to_mound(
                mound=self._mound,
                workspace_id=workspace_id,
                claims=claims_to_promote,
                min_confidence=min_conf,
            )

            self._total_promoted += promoted

            self._emit_event(
                "claims_promoted",
                {
                    "workspace_id": workspace_id,
                    "debate_id": debate_id,
                    "promoted_count": promoted,
                },
            )

            return promoted

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.error(f"Failed to promote claims: {e}")
            raise ExtractionAdapterError(f"Promotion failed: {e}") from e

    async def search_claims(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        extraction_types: list[ExtractionType] | None = None,
        debate_id: str | None = None,
    ) -> list[ExtractionSearchResult]:
        """
        Search extracted claims.

        Args:
            query: Search query (matches against content and topics)
            limit: Maximum results to return
            min_confidence: Minimum confidence threshold
            extraction_types: Optional filter by extraction types
            debate_id: Optional filter by debate ID

        Returns:
            List of ExtractionSearchResult
        """
        results: list[ExtractionSearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for extraction_result in self._extraction_results.values():
            if debate_id and extraction_result.debate_id != debate_id:
                continue

            for claim in extraction_result.claims:
                if claim.confidence < min_confidence:
                    continue

                if extraction_types and claim.extraction_type not in extraction_types:
                    continue

                # Calculate relevance
                content_lower = claim.content.lower()
                matched_topics = [t for t in claim.topics if query_lower in t.lower()]

                # Score based on query word matches
                content_words = set(content_lower.split())
                word_matches = len(query_words & content_words)
                relevance = word_matches / len(query_words) if query_words else 0.0

                # Boost for topic matches
                if matched_topics:
                    relevance = min(1.0, relevance + 0.3)

                # Boost for exact substring match
                if query_lower in content_lower:
                    relevance = min(1.0, relevance + 0.2)

                if relevance > 0 or not query:
                    results.append(
                        ExtractionSearchResult(
                            claim=claim,
                            relevance_score=relevance,
                            matched_topics=matched_topics,
                        )
                    )

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:limit]

    async def search_relationships(
        self,
        source_concept: str | None = None,
        target_concept: str | None = None,
        relationship_type: str | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[RelationshipSearchResult]:
        """
        Search extracted relationships.

        Args:
            source_concept: Optional filter by source concept
            target_concept: Optional filter by target concept
            relationship_type: Optional filter by relationship type
            limit: Maximum results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of RelationshipSearchResult
        """
        results: list[RelationshipSearchResult] = []

        for extraction_result in self._extraction_results.values():
            for rel in extraction_result.relationships:
                if rel.confidence < min_confidence:
                    continue

                # Apply filters
                if source_concept:
                    if source_concept.lower() not in rel.source_concept.lower():
                        continue

                if target_concept:
                    if target_concept.lower() not in rel.target_concept.lower():
                        continue

                if relationship_type:
                    if relationship_type.lower() != rel.relationship_type.lower():
                        continue

                # Calculate relevance based on match quality
                relevance = rel.confidence

                results.append(
                    RelationshipSearchResult(
                        relationship=rel,
                        relevance_score=relevance,
                    )
                )

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:limit]

    async def get_claim(self, claim_id: str) -> ExtractedClaim | None:
        """
        Get a specific claim by ID.

        Args:
            claim_id: The claim ID

        Returns:
            ExtractedClaim or None if not found
        """
        for result in self._extraction_results.values():
            for claim in result.claims:
                if claim.id == claim_id:
                    return claim
        return None

    async def get_relationship(self, relationship_id: str) -> ExtractedRelationship | None:
        """
        Get a specific relationship by ID.

        Args:
            relationship_id: The relationship ID

        Returns:
            ExtractedRelationship or None if not found
        """
        for result in self._extraction_results.values():
            for rel in result.relationships:
                if rel.id == relationship_id:
                    return rel
        return None

    async def get_extraction_result(self, debate_id: str) -> ExtractionResult | None:
        """
        Get the extraction result for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            ExtractionResult or None if not found
        """
        return self._extraction_results.get(debate_id)

    async def _update_graph(self, result: ExtractionResult) -> None:
        """
        Update the knowledge graph with extraction results.

        Args:
            result: The extraction result to add
        """
        now = datetime.now(timezone.utc)

        # Add nodes for topics discovered
        for topic in result.topics_discovered:
            node_id = f"topic_{topic.lower().replace(' ', '_')}"

            if node_id in self._graph_nodes:
                node = self._graph_nodes[node_id]
                node.last_updated = now
            else:
                self._graph_nodes[node_id] = KnowledgeGraphNode(
                    id=node_id,
                    concept=topic,
                    first_seen=now,
                    last_updated=now,
                )

        # Add nodes for relationship concepts
        for rel in result.relationships:
            for concept in [rel.source_concept, rel.target_concept]:
                node_id = f"concept_{concept.lower().replace(' ', '_')}"

                if node_id in self._graph_nodes:
                    node = self._graph_nodes[node_id]
                    node.relationship_ids.append(rel.id)
                    node.last_updated = now
                    # Update confidence if higher
                    if rel.confidence > node.confidence:
                        node.confidence = rel.confidence
                else:
                    self._graph_nodes[node_id] = KnowledgeGraphNode(
                        id=node_id,
                        concept=concept,
                        relationship_ids=[rel.id],
                        confidence=rel.confidence,
                        first_seen=now,
                        last_updated=now,
                    )

        # Associate claims with topic nodes
        for claim in result.claims:
            for topic in claim.topics:
                node_id = f"topic_{topic.lower().replace(' ', '_')}"
                if node_id in self._graph_nodes:
                    node = self._graph_nodes[node_id]
                    if claim.id not in node.claim_ids:
                        node.claim_ids.append(claim.id)
                    node.last_updated = now

    async def get_graph_node(self, node_id: str) -> KnowledgeGraphNode | None:
        """
        Get a knowledge graph node by ID.

        Args:
            node_id: The node ID

        Returns:
            KnowledgeGraphNode or None if not found
        """
        return self._graph_nodes.get(node_id)

    async def get_graph_nodes(
        self,
        concept_filter: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[KnowledgeGraphNode]:
        """
        Get knowledge graph nodes.

        Args:
            concept_filter: Optional concept name filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of KnowledgeGraphNode
        """
        nodes: list[KnowledgeGraphNode] = []

        for node in self._graph_nodes.values():
            if node.confidence < min_confidence:
                continue

            if concept_filter:
                if concept_filter.lower() not in node.concept.lower():
                    continue

            nodes.append(node)

        # Sort by confidence
        nodes.sort(key=lambda x: x.confidence, reverse=True)

        return nodes[:limit]

    async def get_related_concepts(
        self,
        concept: str,
        relationship_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        Get concepts related to a given concept.

        Args:
            concept: The source concept
            relationship_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of related concept dicts with relationship info
        """
        related = []
        concept_lower = concept.lower()

        for result in self._extraction_results.values():
            for rel in result.relationships:
                if relationship_type and rel.relationship_type != relationship_type:
                    continue

                if direction in ("outgoing", "both"):
                    if rel.source_concept.lower() == concept_lower:
                        related.append(
                            {
                                "concept": rel.target_concept,
                                "relationship_type": rel.relationship_type,
                                "direction": "outgoing",
                                "confidence": rel.confidence,
                                "relationship_id": rel.id,
                            }
                        )

                if direction in ("incoming", "both"):
                    if rel.target_concept.lower() == concept_lower:
                        related.append(
                            {
                                "concept": rel.source_concept,
                                "relationship_type": rel.relationship_type,
                                "direction": "incoming",
                                "confidence": rel.confidence,
                                "relationship_id": rel.id,
                            }
                        )

        return related

    async def clear_extraction(self, debate_id: str) -> bool:
        """
        Clear extraction results for a debate.

        Args:
            debate_id: The debate ID to clear

        Returns:
            True if cleared, False if not found
        """
        if debate_id in self._extraction_results:
            del self._extraction_results[debate_id]
            return True
        return False

    async def clear_all_extractions(self) -> int:
        """
        Clear all extraction results.

        Returns:
            Number of results cleared
        """
        count = len(self._extraction_results)
        self._extraction_results.clear()
        self._graph_nodes.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get extraction statistics.

        Returns:
            Dict with extraction statistics
        """
        extractor_stats = self._extractor.get_stats()

        return {
            "debates_processed": self._total_debates_processed,
            "total_claims_extracted": self._total_claims_extracted,
            "total_relationships_extracted": self._total_relationships_extracted,
            "total_promoted": self._total_promoted,
            "cached_results": len(self._extraction_results),
            "graph_nodes": len(self._graph_nodes),
            "mound_connected": self._mound is not None,
            "auto_promote_enabled": self._auto_promote,
            "min_confidence_for_promotion": self._min_confidence_for_promotion,
            "extractor_stats": extractor_stats,
        }

    def get_config(self) -> ExtractionConfig:
        """Get the current extraction configuration."""
        return self._config


__all__ = [
    "ExtractionAdapter",
    "ExtractionAdapterError",
    "ExtractionNotFoundError",
    "ExtractionEngineUnavailableError",
    "ExtractionSearchResult",
    "RelationshipSearchResult",
    "KnowledgeGraphNode",
    "BatchExtractionResult",
]
