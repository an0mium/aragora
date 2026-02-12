"""
DebateAdapter - Bridges debate orchestrator outcomes to the Knowledge Mound.

Unlike ConsensusAdapter (which focuses on consensus outcomes from ConsensusMemory),
the DebateAdapter persists full debate transcripts, per-agent metrics, disagreement
reports, and evidence suggestions from DebateResult objects.

The adapter provides:
- Full debate outcome persistence (transcripts, votes, critiques)
- Per-agent performance metric extraction
- Disagreement and crux tracking for institutional learning
- Domain-based search for similar past debates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


logger = logging.getLogger(__name__)

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._types import SyncResult


@dataclass
class DebateSearchResult:
    """Wrapper for debate search results with similarity metadata."""

    debate_id: str
    task: str
    final_answer: str
    confidence: float
    consensus_reached: bool
    participants: list[str]
    rounds_used: int
    similarity: float = 0.0
    dissenting_views: list[str] = field(default_factory=list)
    debate_cruxes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DebateOutcome:
    """Lightweight representation of a debate outcome for adapter storage.

    This decouples the adapter from the full DebateResult dataclass,
    accepting only the fields needed for KM persistence.
    """

    debate_id: str
    task: str
    final_answer: str
    confidence: float
    consensus_reached: bool
    status: str = "completed"
    rounds_used: int = 0
    duration_seconds: float = 0.0
    participants: list[str] = field(default_factory=list)
    winner: str | None = None
    consensus_strength: str = ""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    per_agent_cost: dict[str, float] = field(default_factory=dict)
    dissenting_views: list[str] = field(default_factory=list)
    debate_cruxes: list[dict[str, Any]] = field(default_factory=list)
    evidence_suggestions: list[dict[str, Any]] = field(default_factory=list)
    convergence_similarity: float = 0.0
    per_agent_similarity: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_debate_result(cls, result: Any) -> DebateOutcome:
        """Create a DebateOutcome from a DebateResult object."""
        return cls(
            debate_id=getattr(result, "debate_id", getattr(result, "id", "")),
            task=getattr(result, "task", ""),
            final_answer=getattr(result, "final_answer", ""),
            confidence=getattr(result, "confidence", 0.0),
            consensus_reached=getattr(result, "consensus_reached", False),
            status=getattr(result, "status", "completed"),
            rounds_used=getattr(result, "rounds_used", 0),
            duration_seconds=getattr(result, "duration_seconds", 0.0),
            participants=getattr(result, "participants", []),
            winner=getattr(result, "winner", None),
            consensus_strength=getattr(result, "consensus_strength", ""),
            total_cost_usd=getattr(result, "total_cost_usd", 0.0),
            total_tokens=getattr(result, "total_tokens", 0),
            per_agent_cost=getattr(result, "per_agent_cost", {}),
            dissenting_views=getattr(result, "dissenting_views", []),
            debate_cruxes=getattr(result, "debate_cruxes", []),
            evidence_suggestions=getattr(result, "evidence_suggestions", []),
            convergence_similarity=getattr(result, "convergence_similarity", 0.0),
            per_agent_similarity=getattr(result, "per_agent_similarity", {}),
            metadata=getattr(result, "metadata", {}),
        )


class DebateAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges debate outcomes to the Knowledge Mound.

    Provides full debate persistence including transcripts, agent metrics,
    disagreement reports, and evidence suggestions. Complements ConsensusAdapter
    which focuses on consensus-level data.

    Resilience Features:
    - Circuit breaker protection for external service calls
    - Bulkhead isolation to prevent cascading failures
    - Automatic retry with exponential backoff

    Usage:
        adapter = DebateAdapter()
        adapter.store_outcome(debate_result)  # Mark for sync
        await adapter.sync_to_km(mound)       # Persist to KM
        results = await adapter.search_by_topic("rate limiting")
    """

    adapter_name = "debate"
    source_type = "debate"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._pending_outcomes: list[DebateOutcome] = []
        self._synced_outcomes: dict[str, DebateOutcome] = {}

    def store_outcome(self, result: Any) -> None:
        """Store a debate outcome for KM sync.

        Args:
            result: A DebateResult or DebateOutcome object.
        """
        if isinstance(result, DebateOutcome):
            outcome = result
        else:
            outcome = DebateOutcome.from_debate_result(result)

        outcome.metadata["km_sync_pending"] = True
        outcome.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_outcomes.append(outcome)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "debate_id": outcome.debate_id,
                "task": outcome.task[:100],
                "confidence": outcome.confidence,
            },
        )

    def get(self, record_id: str) -> DebateOutcome | None:
        """Get a debate outcome by ID."""
        clean_id = record_id[3:] if record_id.startswith("db_") else record_id
        return self._synced_outcomes.get(clean_id)

    async def get_async(self, record_id: str) -> DebateOutcome | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        include_cruxes: bool = True,
    ) -> list[DebateSearchResult]:
        """Search stored debate outcomes by topic similarity.

        Args:
            query: Search query (topic text)
            limit: Max results to return
            min_confidence: Minimum confidence threshold
            include_cruxes: Whether to include debate cruxes in results

        Returns:
            List of DebateSearchResult objects sorted by relevance.
        """
        results: list[DebateSearchResult] = []
        query_lower = query.lower()

        all_outcomes = list(self._synced_outcomes.values()) + self._pending_outcomes
        for outcome in all_outcomes:
            if outcome.confidence < min_confidence:
                continue

            # Simple text similarity (semantic search via mixin handles vector search)
            task_lower = outcome.task.lower()
            if query_lower in task_lower or any(word in task_lower for word in query_lower.split()):
                similarity = 0.8 if query_lower in task_lower else 0.5
                results.append(
                    DebateSearchResult(
                        debate_id=outcome.debate_id,
                        task=outcome.task,
                        final_answer=outcome.final_answer,
                        confidence=outcome.confidence,
                        consensus_reached=outcome.consensus_reached,
                        participants=outcome.participants,
                        rounds_used=outcome.rounds_used,
                        similarity=similarity,
                        dissenting_views=outcome.dissenting_views,
                        debate_cruxes=outcome.debate_cruxes if include_cruxes else [],
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, outcome: DebateOutcome) -> KnowledgeItem:
        """Convert a DebateOutcome to a KnowledgeItem for KM storage."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        content = f"Debate: {outcome.task}\n\nAnswer: {outcome.final_answer}"
        if outcome.dissenting_views:
            content += f"\n\nDissenting views: {'; '.join(outcome.dissenting_views[:3])}"

        return KnowledgeItem(
            id=f"db_{outcome.debate_id}",
            content=content,
            source=KnowledgeSource.DEBATE,
            source_id=outcome.debate_id,
            confidence=ConfidenceLevel.from_float(outcome.confidence),
            created_at=outcome.created_at,
            updated_at=outcome.created_at,
            metadata={
                "task": outcome.task,
                "consensus_reached": outcome.consensus_reached,
                "consensus_strength": outcome.consensus_strength,
                "rounds_used": outcome.rounds_used,
                "duration_seconds": outcome.duration_seconds,
                "participants": outcome.participants,
                "winner": outcome.winner,
                "total_cost_usd": outcome.total_cost_usd,
                "total_tokens": outcome.total_tokens,
                "convergence_similarity": outcome.convergence_similarity,
                "debate_cruxes_count": len(outcome.debate_cruxes),
                "dissenting_views_count": len(outcome.dissenting_views),
                "evidence_suggestions_count": len(outcome.evidence_suggestions),
                "status": outcome.status,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.3,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending debate outcomes to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance
            min_confidence: Minimum confidence to sync
            batch_size: Max records per batch

        Returns:
            SyncResult with sync statistics.
        """
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_outcomes[:batch_size]

        for outcome in pending:
            if outcome.confidence < min_confidence:
                skipped += 1
                continue

            try:
                km_item = self.to_knowledge_item(outcome)

                # Try multiple store methods for compatibility
                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                outcome.metadata["km_sync_pending"] = False
                outcome.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                outcome.metadata["km_item_id"] = km_item.id

                self._synced_outcomes[outcome.debate_id] = outcome
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "debate_id": outcome.debate_id,
                        "km_item_id": km_item.id,
                    },
                )

            except Exception as e:
                failed += 1
                error_msg = f"Failed to sync debate {outcome.debate_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                outcome.metadata["km_sync_error"] = str(e)

        # Remove successfully synced from pending
        synced_ids = {o.debate_id for o in pending if o.metadata.get("km_sync_pending") is False}
        self._pending_outcomes = [
            o for o in self._pending_outcomes if o.debate_id not in synced_ids
        ]

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored debate outcomes."""
        all_outcomes = list(self._synced_outcomes.values())
        return {
            "total_synced": len(self._synced_outcomes),
            "pending_sync": len(self._pending_outcomes),
            "avg_confidence": (
                sum(o.confidence for o in all_outcomes) / len(all_outcomes) if all_outcomes else 0.0
            ),
            "consensus_rate": (
                sum(1 for o in all_outcomes if o.consensus_reached) / len(all_outcomes)
                if all_outcomes
                else 0.0
            ),
            "total_cost_usd": sum(o.total_cost_usd for o in all_outcomes),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> DebateOutcome | None:
        """Get a debate outcome by ID (required by SemanticSearchMixin)."""
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        """Convert a debate outcome to dict (required by SemanticSearchMixin)."""
        return {
            "id": record.debate_id,
            "task": record.task,
            "final_answer": record.final_answer,
            "confidence": record.confidence,
            "consensus_reached": record.consensus_reached,
            "participants": record.participants,
            "rounds_used": record.rounds_used,
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> DebateOutcome | None:
        """Get a debate outcome for validation (required by ReverseFlowMixin)."""
        return self.get(source_id)

    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Apply KM validation to a debate outcome."""
        record.metadata["km_validated"] = True
        record.metadata["km_validation_confidence"] = km_confidence
        record.metadata["km_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if cross_refs:
            record.metadata["km_cross_references"] = cross_refs
        return True

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
        """Extract source ID from KM item."""
        source_id = item.get("source_id", "")
        if source_id.startswith("db_"):
            return source_id[3:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        """Adapters this adapter can fuse data from."""
        return ["consensus", "evidence", "elo"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        """Extract data that can be used for fusion."""
        if km_item.get("source") == "debate":
            return {
                "confidence": km_item.get("confidence", 0.0),
                "consensus_reached": km_item.get("metadata", {}).get("consensus_reached", False),
            }
        return None

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Apply a fusion result to a debate outcome."""
        record.metadata["fusion_applied"] = True
        record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
        if hasattr(fusion_result, "fused_confidence"):
            record.metadata["fused_confidence"] = fusion_result.fused_confidence
        return True
