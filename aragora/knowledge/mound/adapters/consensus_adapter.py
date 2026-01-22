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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from aragora.memory.consensus import (
        ConsensusMemory,
        ConsensusRecord,
        DissentRecord,
    )
    from aragora.knowledge.mound.types import KnowledgeItem

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]

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
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize the adapter.

        Args:
            consensus: The ConsensusMemory instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
        """
        self._consensus = consensus
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _record_metric(self, operation: str, success: bool, latency: float) -> None:
        """Record Prometheus metric for adapter operation and check SLOs.

        Args:
            operation: Operation name (search, store, sync, semantic_search)
            success: Whether operation succeeded
            latency: Operation latency in seconds
        """
        latency_ms = latency * 1000  # Convert to milliseconds

        try:
            from aragora.observability.metrics.km import (
                record_km_operation,
                record_km_adapter_sync,
            )

            record_km_operation(operation, success, latency)
            if operation in ("store", "sync"):
                record_km_adapter_sync("consensus", "forward", success)
        except ImportError:
            pass  # Metrics not available
        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")

        # Check SLOs and alert on violations
        try:
            from aragora.observability.metrics.slo import check_and_record_slo_with_recovery

            # Map operation to SLO name
            slo_mapping = {
                "search": "adapter_reverse",
                "store": "adapter_forward_sync",
                "sync": "adapter_sync",
                "semantic_search": "adapter_semantic_search",
            }
            slo_name = slo_mapping.get(operation, "adapter_sync")

            passed, message = check_and_record_slo_with_recovery(
                operation=slo_name,
                latency_ms=latency_ms,
                context={
                    "adapter": "consensus",
                    "operation": operation,
                    "success": success,
                },
            )
            if not passed:
                logger.debug(f"Consensus adapter SLO violation: {message}")
        except ImportError:
            pass  # SLO metrics not available
        except Exception as e:
            logger.debug(f"Failed to check SLO: {e}")

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
                similarity=debate.similarity,  # type: ignore[attr-defined]
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
        confidence = strength_to_confidence.get(consensus.strength.value, ConfidenceLevel.MEDIUM)

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
        return self._consensus.find_risk_warnings(topic=topic, limit=limit)  # type: ignore[attr-defined]

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
        return self._consensus.find_contrarian_views(limit=limit)  # type: ignore[attr-defined]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the consensus memory."""
        return self._consensus.get_stats()  # type: ignore[attr-defined]

    def search_similar(
        self,
        topic: str,
        limit: int = 5,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar consensus records for deduplication (reverse flow).

        Args:
            topic: Topic to find similar consensus for
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of similar consensus records as dicts
        """
        import time

        start = time.time()
        success = False

        try:
            similar = self._consensus.find_similar_debates(
                topic=topic,
                min_confidence=min_confidence,
                limit=limit,
            )

            # Convert to dict format for consistency
            results = [
                {
                    "id": d.consensus.id,
                    "topic": d.consensus.topic,
                    "conclusion": d.consensus.conclusion,
                    "strength": d.consensus.strength.value,
                    "confidence": d.consensus.confidence,
                    "domain": d.consensus.domain,
                    "similarity": d.similarity,  # type: ignore[attr-defined]
                    "timestamp": d.consensus.timestamp.isoformat()
                    if hasattr(d.consensus.timestamp, "isoformat")
                    else str(d.consensus.timestamp),
                }
                for d in similar
            ]

            # Emit dashboard event for reverse flow query
            self._emit_event(
                "km_adapter_reverse_query",
                {
                    "source": "consensus",
                    "topic_preview": topic[:50] + "..." if len(topic) > 50 else topic,
                    "results_count": len(results),
                    "limit": limit,
                },
            )

            success = True
            return results
        finally:
            self._record_metric("search", success, time.time() - start)

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search over consensus records.

        Uses the Knowledge Mound's SemanticStore for embedding-based similarity
        search, falling back to keyword search if embeddings aren't available.

        Args:
            query: The search query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            tenant_id: Optional tenant filter

        Returns:
            List of matching consensus records with similarity scores
        """
        import time

        start = time.time()
        success = False

        try:
            # Try semantic search first
            try:
                from aragora.knowledge.mound.semantic_store import SemanticStore

                # Get or create semantic store
                store = SemanticStore()  # type: ignore[call-arg]

                # Search using embeddings
                results = await store.search_similar(  # type: ignore[call-arg]
                    query=query,
                    tenant_id=tenant_id or "default",
                    limit=limit,
                    min_similarity=min_similarity,
                    source_type="consensus",
                )

                # Enrich results with full consensus records
                enriched = []
                for r in results:
                    # Try to get the full record from consensus memory
                    record_id = r.source_id
                    if record_id.startswith("cs_"):
                        record_id = record_id[3:]

                    record = self._consensus.get_consensus(record_id)
                    if record:
                        enriched.append(
                            {
                                "id": record.id,
                                "topic": record.topic,
                                "conclusion": record.conclusion,
                                "strength": record.strength.value,
                                "confidence": record.confidence,
                                "domain": record.domain,
                                "similarity": r.similarity,
                                "timestamp": record.timestamp.isoformat()
                                if hasattr(record.timestamp, "isoformat")
                                else str(record.timestamp),
                                "metadata": record.metadata,
                            }
                        )
                    else:
                        # Record may not be in memory
                        enriched.append(
                            {
                                "id": r.source_id,
                                "similarity": r.similarity,
                                "domain": r.domain,
                                "importance": r.importance,
                                "metadata": r.metadata,
                            }
                        )

                success = True
                logger.debug(f"Semantic search returned {len(enriched)} results for '{query[:50]}'")

                # Emit event
                self._emit_event(
                    "km_adapter_semantic_search",
                    {
                        "source": "consensus",
                        "query_preview": query[:50],
                        "results_count": len(enriched),
                        "search_type": "vector",
                    },
                )

                return enriched

            except ImportError:
                logger.debug("SemanticStore not available, falling back to keyword search")
            except Exception as e:
                logger.debug(f"Semantic search failed, falling back: {e}")

            # Fallback to keyword search
            results = self.search_similar(query, limit=limit, min_confidence=min_similarity)  # type: ignore[attr-defined,call-arg]
            success = True  # type: ignore[assignment,possibly-undefined]
            return results

        finally:
            self._record_metric("semantic_search", success, time.time() - start)

    def store_consensus(self, record: "ConsensusRecord") -> None:
        """
        Store a consensus record in the Knowledge Mound (forward flow).

        This is called by ConsensusMemory when a high-confidence consensus
        is stored and should be synced to KM for cross-session learning.

        Args:
            record: The ConsensusRecord to store in KM
        """
        from datetime import datetime

        # This method is a hook for KM sync. The actual KM storage happens
        # when sync_to_mound is called with a mound instance.
        logger.debug(
            f"Consensus marked for KM sync: {record.id} "
            f"(topic={record.topic[:50]}..., confidence={record.confidence:.2f})"
        )
        # Mark the record as pending KM sync in metadata
        if not record.metadata.get("km_sync_pending"):
            record.metadata["km_sync_pending"] = True
            record.metadata["km_sync_requested_at"] = datetime.now().isoformat()

        # Emit dashboard event for forward sync
        self._emit_event(
            "km_adapter_forward_sync",
            {
                "source": "consensus",
                "consensus_id": record.id,
                "topic_preview": record.topic[:50] + "..."
                if len(record.topic) > 50
                else record.topic,
                "confidence": record.confidence,
                "strength": record.strength.value,
            },
        )
