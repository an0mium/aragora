"""
TricksterAdapter - Bridges Trickster interventions to the Knowledge Mound.

This adapter enables the Knowledge Mound to query past hollow consensus
interventions for cross-debate learning and bias pattern detection.

The adapter provides:
- Unified search interface (search_by_topic)
- TricksterIntervention-to-KnowledgeItem conversion
- Domain-based filtering for intervention patterns
- Cross-debate intervention learning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.debate.trickster import (
        EvidencePoweredTrickster,
        TricksterIntervention,
    )
    from aragora.knowledge.mound.types import KnowledgeItem

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


logger = logging.getLogger(__name__)

# Import mixins for shared adapter functionality
from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._types import SyncResult


@dataclass
class TricksterSearchResult:
    """Wrapper for trickster intervention search results."""

    intervention: TricksterIntervention
    debate_id: str
    domain: str | None
    topic: str
    similarity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class InterventionRecord:
    """Persistent record of a trickster intervention."""

    id: str
    debate_id: str
    domain: str | None
    topic: str
    intervention_type: str
    round_num: int
    target_agents: list[str]
    challenge_text: str
    evidence_gaps: dict[str, list[str]]
    priority: float
    timestamp: datetime
    outcome: str | None = None  # "effective", "ineffective", "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


class TricksterAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Trickster interventions to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_by_topic: Find similar interventions from past debates
    - to_knowledge_item: Convert interventions to unified format
    - get_domain_patterns: Identify recurring intervention patterns by domain

    Usage:
        from aragora.debate.trickster import EvidencePoweredTrickster
        from aragora.knowledge.mound.adapters import TricksterAdapter

        trickster = EvidencePoweredTrickster()
        adapter = TricksterAdapter(trickster)

        # After a debate, persist interventions
        await adapter.persist_debate_interventions(debate_id="d123", topic="rate limiting")

        # Search for similar interventions from past debates
        results = await adapter.search_by_topic("rate limiting", limit=10)
    """

    adapter_name = "trickster"
    source_type = "trickster"

    def __init__(
        self,
        trickster: EvidencePoweredTrickster | None = None,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            trickster: The EvidencePoweredTrickster instance to wrap (optional)
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
            enable_resilience: If True, enables circuit breaker protection
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )

        self._trickster = trickster
        # In-memory store of intervention records (would be database-backed in prod)
        self._records: dict[str, InterventionRecord] = {}
        # Index by debate_id for fast lookup
        self._by_debate: dict[str, list[str]] = {}
        # Index by domain for pattern analysis
        self._by_domain: dict[str, list[str]] = {}

    @property
    def trickster(self) -> EvidencePoweredTrickster | None:
        """Access the underlying trickster."""
        return self._trickster

    def set_trickster(self, trickster: EvidencePoweredTrickster) -> None:
        """Set or update the trickster instance."""
        self._trickster = trickster

    async def persist_debate_interventions(
        self,
        debate_id: str,
        topic: str,
        domain: str | None = None,
        trickster: EvidencePoweredTrickster | None = None,
    ) -> int:
        """
        Persist all interventions from a debate to the adapter storage.

        Args:
            debate_id: ID of the debate
            topic: Topic of the debate
            domain: Optional domain for the debate
            trickster: Optional trickster instance (uses self._trickster if not provided)

        Returns:
            Number of interventions persisted
        """
        import time
        import uuid

        start = time.time()
        trickster = trickster or self._trickster

        if not trickster:
            logger.warning("No trickster instance available for persistence")
            return 0

        state = trickster._state
        count = 0

        for intervention in state.interventions:
            record_id = f"tr_{uuid.uuid4().hex[:12]}"

            record = InterventionRecord(
                id=record_id,
                debate_id=debate_id,
                domain=domain,
                topic=topic,
                intervention_type=intervention.intervention_type.value,
                round_num=intervention.round_num,
                target_agents=intervention.target_agents,
                challenge_text=intervention.challenge_text,
                evidence_gaps=intervention.evidence_gaps,
                priority=intervention.priority,
                timestamp=datetime.now(),
                metadata=intervention.metadata.copy(),
            )

            self._records[record_id] = record

            # Update indices
            if debate_id not in self._by_debate:
                self._by_debate[debate_id] = []
            self._by_debate[debate_id].append(record_id)

            if domain:
                if domain not in self._by_domain:
                    self._by_domain[domain] = []
                self._by_domain[domain].append(record_id)

            count += 1

        # Emit event for persistence
        self._emit_event(
            "km_adapter_trickster_persist",
            {
                "debate_id": debate_id,
                "topic_preview": topic[:50] + "..." if len(topic) > 50 else topic,
                "domain": domain,
                "interventions_count": count,
            },
        )

        self._record_metric("persist", True, time.time() - start)
        logger.info(f"Persisted {count} trickster interventions for debate {debate_id}")
        return count

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        domain: str | None = None,
        min_priority: float = 0.0,
    ) -> list[TricksterSearchResult]:
        """
        Search for similar interventions by topic.

        Args:
            query: Topic to search for
            limit: Maximum results to return
            domain: Optional domain filter
            min_priority: Minimum intervention priority threshold

        Returns:
            List of TricksterSearchResult with interventions and metadata
        """
        import time

        start = time.time()
        results: list[TricksterSearchResult] = []

        # Simple keyword matching (would use semantic search in prod)
        query_words = set(query.lower().split())

        for record in self._records.values():
            # Domain filter
            if domain and record.domain != domain:
                continue

            # Priority filter
            if record.priority < min_priority:
                continue

            # Simple similarity based on keyword overlap
            topic_words = set(record.topic.lower().split())
            challenge_words = set(record.challenge_text.lower().split())
            all_words = topic_words | challenge_words

            if not query_words:
                similarity = 0.0
            else:
                overlap = len(query_words & all_words)
                similarity = overlap / len(query_words)

            if similarity > 0:
                results.append(
                    TricksterSearchResult(
                        intervention=self._record_to_intervention(record),
                        debate_id=record.debate_id,
                        domain=record.domain,
                        topic=record.topic,
                        similarity=similarity,
                        timestamp=record.timestamp,
                    )
                )

        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity, reverse=True)
        results = results[:limit]

        self._emit_event(
            "km_adapter_trickster_search",
            {
                "query_preview": query[:50] + "..." if len(query) > 50 else query,
                "domain": domain,
                "results_count": len(results),
                "limit": limit,
            },
        )

        self._record_metric("search", True, time.time() - start)
        return results

    def _record_to_intervention(self, record: InterventionRecord) -> TricksterIntervention:
        """Convert a persisted record back to TricksterIntervention."""
        from aragora.debate.trickster import InterventionType, TricksterIntervention

        return TricksterIntervention(
            intervention_type=InterventionType(record.intervention_type),
            round_num=record.round_num,
            target_agents=record.target_agents,
            challenge_text=record.challenge_text,
            evidence_gaps=record.evidence_gaps,
            priority=record.priority,
            metadata=record.metadata,
        )

    def get(self, record_id: str) -> InterventionRecord | None:
        """
        Get a specific intervention record by ID.

        Args:
            record_id: The record ID (may be prefixed with "tr_")

        Returns:
            InterventionRecord or None
        """
        # Strip prefix if present
        if record_id.startswith("tr_"):
            pass  # Already prefixed
        else:
            record_id = f"tr_{record_id}"

        return self._records.get(record_id)

    async def get_async(self, record_id: str) -> InterventionRecord | None:
        """Async version of get for compatibility."""
        return self.get(record_id)

    def to_knowledge_item(
        self,
        record: InterventionRecord | TricksterSearchResult,
    ) -> KnowledgeItem:
        """
        Convert an intervention record to a KnowledgeItem.

        Args:
            record: The intervention record or search result

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Handle both record types
        if isinstance(record, TricksterSearchResult):
            intervention = record.intervention
            debate_id = record.debate_id
            domain = record.domain
            topic = record.topic
            timestamp = record.timestamp
            record_id = f"tr_{debate_id}_{intervention.round_num}"
        else:
            intervention = self._record_to_intervention(record)
            debate_id = record.debate_id
            domain = record.domain
            topic = record.topic
            timestamp = record.timestamp
            record_id = record.id

        # Map priority to confidence level
        if intervention.priority >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif intervention.priority >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        # Build content from intervention
        content = (
            f"Trickster intervention ({intervention.intervention_type.value}): "
            f"{intervention.challenge_text}"
        )

        metadata: dict[str, Any] = {
            "topic": topic,
            "domain": domain,
            "debate_id": debate_id,
            "intervention_type": intervention.intervention_type.value,
            "round_num": intervention.round_num,
            "target_agents": intervention.target_agents,
            "evidence_gaps": intervention.evidence_gaps,
            "priority": intervention.priority,
            "source_adapter": "trickster",
        }
        metadata.update(intervention.metadata)

        return KnowledgeItem(
            id=record_id,
            content=content,
            source=KnowledgeSource.INSIGHT,  # Interventions are insights about debate quality
            source_id=debate_id,
            confidence=confidence,
            created_at=timestamp,
            updated_at=timestamp,
            metadata=metadata,
            importance=intervention.priority,
        )

    def get_domain_patterns(
        self,
        domain: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get recurring intervention patterns for a domain.

        Useful for understanding what types of hollow consensus
        commonly occur in specific domains.

        Args:
            domain: The domain to analyze
            limit: Maximum patterns to return

        Returns:
            List of pattern dictionaries with type, frequency, and examples
        """
        if domain not in self._by_domain:
            return []

        # Count intervention types
        type_counts: dict[str, int] = {}
        type_examples: dict[str, list[str]] = {}

        for record_id in self._by_domain[domain]:
            record = self._records.get(record_id)
            if not record:
                continue

            int_type = record.intervention_type
            type_counts[int_type] = type_counts.get(int_type, 0) + 1

            if int_type not in type_examples:
                type_examples[int_type] = []
            if len(type_examples[int_type]) < 3:  # Keep up to 3 examples
                type_examples[int_type].append(record.challenge_text[:100])

        # Build pattern list
        patterns = []
        for int_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            patterns.append(
                {
                    "intervention_type": int_type,
                    "frequency": count,
                    "examples": type_examples.get(int_type, []),
                    "domain": domain,
                }
            )

        return patterns[:limit]

    def get_debate_interventions(
        self,
        debate_id: str,
    ) -> list[InterventionRecord]:
        """
        Get all interventions for a specific debate.

        Args:
            debate_id: The debate ID

        Returns:
            List of InterventionRecord objects
        """
        if debate_id not in self._by_debate:
            return []

        return [
            self._records[record_id]
            for record_id in self._by_debate[debate_id]
            if record_id in self._records
        ]

    def record_outcome(
        self,
        record_id: str,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Record the outcome of an intervention for learning.

        Args:
            record_id: The intervention record ID
            outcome: "effective", "ineffective", or "pending"
            metadata: Optional additional outcome metadata

        Returns:
            True if successfully recorded
        """
        record = self.get(record_id)
        if not record:
            return False

        record.outcome = outcome
        if metadata:
            record.metadata["outcome_data"] = metadata

        self._emit_event(
            "km_adapter_trickster_outcome",
            {
                "record_id": record_id,
                "outcome": outcome,
                "intervention_type": record.intervention_type,
            },
        )

        logger.debug(f"Recorded outcome '{outcome}' for intervention {record_id}")
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored interventions."""
        type_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {"effective": 0, "ineffective": 0, "pending": 0}

        for record in self._records.values():
            type_counts[record.intervention_type] = type_counts.get(record.intervention_type, 0) + 1
            if record.outcome:
                outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1

        return {
            "total_interventions": len(self._records),
            "debates_with_interventions": len(self._by_debate),
            "domains_with_interventions": len(self._by_domain),
            "intervention_types": type_counts,
            "outcomes": outcome_counts,
        }

    async def sync_to_km(
        self,
        mound: Any,
        min_priority: float = 0.5,
        batch_size: int = 50,
    ) -> SyncResult:
        """
        Sync intervention records to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance to sync to
            min_priority: Minimum priority threshold for syncing
            batch_size: Maximum records to sync in one call

        Returns:
            SyncResult with sync statistics
        """
        import time

        start = time.time()
        result = SyncResult()

        # Find records not yet synced
        pending_records: list[InterventionRecord] = []
        for record in self._records.values():
            if record.metadata.get("km_sync_pending", True) and record.priority >= min_priority:
                pending_records.append(record)
            elif record.priority < min_priority:
                result.records_skipped += 1

        if not pending_records:
            logger.debug("No pending trickster records to sync to KM")
            result.duration_ms = (time.time() - start) * 1000
            return result

        logger.info(f"Syncing {len(pending_records[:batch_size])} trickster records to KM")

        for record in pending_records[:batch_size]:
            try:
                km_item = self.to_knowledge_item(record)

                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                record.metadata["km_sync_pending"] = False
                record.metadata["km_synced_at"] = datetime.now().isoformat()
                record.metadata["km_item_id"] = km_item.id

                result.records_synced += 1

                self._emit_event(
                    "km_adapter_trickster_sync_complete",
                    {
                        "record_id": record.id,
                        "km_item_id": km_item.id,
                        "priority": record.priority,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                result.records_failed += 1
                logger.warning("Failed to sync trickster %s: %s", record.id, e)
                result.errors.append(f"Failed to sync trickster {record.id}")

                record.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"
                record.metadata["km_sync_failed_at"] = datetime.now().isoformat()

        result.duration_ms = (time.time() - start) * 1000

        self._record_metric(
            "sync",
            result.records_failed == 0,
            result.duration_ms / 1000,
        )

        logger.info(
            f"Trickster KM sync complete: "
            f"synced={result.records_synced}, "
            f"skipped={result.records_skipped}, "
            f"failed={result.records_failed}"
        )

        return result
