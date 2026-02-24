"""
ExplainabilityAdapter - Bridges debate explanations to the Knowledge Mound.

Persists Decision entities (evidence chains, vote pivots, belief changes,
confidence attribution, counterfactuals) so that future debates can reference
past reasoning patterns.  This closes the explainability-to-memory loop:

    Debate → ExplanationBuilder → Decision → ExplainabilityAdapter → KM
    KM → Future debate prompt enrichment (via topic search)

The adapter provides:
- Ingestion of Decision objects as structured knowledge items
- Factor pattern storage for institutional reasoning memory
- Counterfactual tagging for sensitivity analysis history
- Topic-based retrieval of past explanations
- Sync to Knowledge Mound for cross-debate learning

"Every explanation teaches the system how it reasons."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.explainability.decision import Decision

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._types import SyncResult

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityEntry:
    """Lightweight representation of a Decision for adapter storage.

    Decouples the adapter from the full Decision dataclass, accepting
    only the fields needed for KM persistence.
    """

    decision_id: str
    debate_id: str
    task: str
    domain: str
    conclusion: str
    confidence: float
    consensus_reached: bool
    consensus_type: str = "majority"
    rounds_used: int = 0
    agents_participated: list[str] = field(default_factory=list)

    # Explainability components (serialized)
    evidence_count: int = 0
    vote_pivot_count: int = 0
    belief_change_count: int = 0
    factor_count: int = 0
    counterfactual_count: int = 0

    # Top factors for quick retrieval
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    top_counterfactuals: list[dict[str, Any]] = field(default_factory=list)

    # Summary scores
    evidence_quality_score: float = 0.0
    agent_agreement_score: float = 0.0
    belief_stability_score: float = 0.0

    # Full serialized Decision for deep retrieval
    decision_data: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_decision(cls, decision: Decision, task: str = "") -> ExplainabilityEntry:
        """Create an ExplainabilityEntry from a Decision object.

        Args:
            decision: The Decision entity from ExplanationBuilder.
            task: Optional task string (overrides decision.task if provided).
        """
        top_factors = [
            ca.to_dict()
            for ca in sorted(
                decision.confidence_attribution,
                key=lambda c: c.contribution,
                reverse=True,
            )[:5]
        ]

        top_counterfactuals = [
            cf.to_dict()
            for cf in sorted(
                decision.counterfactuals,
                key=lambda c: c.sensitivity,
                reverse=True,
            )[:3]
        ]

        return cls(
            decision_id=decision.decision_id,
            debate_id=decision.debate_id,
            task=task or decision.task,
            domain=decision.domain,
            conclusion=decision.conclusion,
            confidence=decision.confidence,
            consensus_reached=decision.consensus_reached,
            consensus_type=decision.consensus_type,
            rounds_used=decision.rounds_used,
            agents_participated=list(decision.agents_participated),
            evidence_count=len(decision.evidence_chain),
            vote_pivot_count=len(decision.vote_pivots),
            belief_change_count=len(decision.belief_changes),
            factor_count=len(decision.confidence_attribution),
            counterfactual_count=len(decision.counterfactuals),
            top_factors=top_factors,
            top_counterfactuals=top_counterfactuals,
            evidence_quality_score=decision.evidence_quality_score,
            agent_agreement_score=decision.agent_agreement_score,
            belief_stability_score=decision.belief_stability_score,
            decision_data=decision.to_dict(),
        )


@dataclass
class ExplainabilitySearchResult:
    """Wrapper for explainability search results with similarity metadata."""

    decision_id: str
    debate_id: str
    task: str
    domain: str
    conclusion: str
    confidence: float
    consensus_reached: bool
    top_factors: list[dict[str, Any]]
    top_counterfactuals: list[dict[str, Any]]
    similarity: float = 0.0


class ExplainabilityAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges debate explanations to the Knowledge Mound.

    Persists Decision entities so the system can learn from its own
    reasoning.  Stores factor patterns, counterfactuals, and evidence
    chains as structured knowledge items for future debate enrichment.

    Usage:
        adapter = ExplainabilityAdapter()
        adapter.store_explanation(decision, task="Design a rate limiter")
        await adapter.sync_to_km(mound)
        results = await adapter.search_by_topic("rate limiting")
    """

    adapter_name = "explainability"
    source_type = "explainability"

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
        self._pending_entries: list[ExplainabilityEntry] = []
        self._synced_entries: dict[str, ExplainabilityEntry] = {}

    def store_explanation(self, decision: Any, task: str = "") -> None:
        """Store a Decision explanation for KM sync.

        Accepts either a Decision object or an ExplainabilityEntry.

        Args:
            decision: A Decision or ExplainabilityEntry object.
            task: Optional task string to associate with the entry.
        """
        if isinstance(decision, ExplainabilityEntry):
            entry = decision
        else:
            entry = ExplainabilityEntry.from_decision(decision, task=task)

        entry.metadata["km_sync_pending"] = True
        entry.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_entries.append(entry)

        self._emit_event(
            "km_explainability_stored",
            {
                "adapter": self.adapter_name,
                "decision_id": entry.decision_id,
                "debate_id": entry.debate_id,
                "task": entry.task[:100],
                "confidence": entry.confidence,
                "factor_count": entry.factor_count,
                "counterfactual_count": entry.counterfactual_count,
            },
        )

    def ingest(self, explanation_data: dict[str, Any]) -> bool:
        """Synchronous convenience method to ingest an explanation from a dict.

        Used by FeedbackPhase to persist explanations without async.

        Args:
            explanation_data: Dict with explanation fields.

        Returns:
            True if ingestion succeeded, False otherwise.
        """
        try:
            entry = ExplainabilityEntry(
                decision_id=explanation_data.get("decision_id", ""),
                debate_id=explanation_data.get("debate_id", ""),
                task=explanation_data.get("task", ""),
                domain=explanation_data.get("domain", "general"),
                conclusion=explanation_data.get("conclusion", ""),
                confidence=explanation_data.get("confidence", 0.0),
                consensus_reached=explanation_data.get("consensus_reached", False),
                consensus_type=explanation_data.get("consensus_type", "majority"),
                rounds_used=explanation_data.get("rounds_used", 0),
                agents_participated=explanation_data.get("agents_participated", []),
                evidence_count=explanation_data.get("evidence_count", 0),
                vote_pivot_count=explanation_data.get("vote_pivot_count", 0),
                belief_change_count=explanation_data.get("belief_change_count", 0),
                factor_count=explanation_data.get("factor_count", 0),
                counterfactual_count=explanation_data.get("counterfactual_count", 0),
                top_factors=explanation_data.get("top_factors", []),
                top_counterfactuals=explanation_data.get("top_counterfactuals", []),
                evidence_quality_score=explanation_data.get("evidence_quality_score", 0.0),
                agent_agreement_score=explanation_data.get("agent_agreement_score", 0.0),
                belief_stability_score=explanation_data.get("belief_stability_score", 0.0),
                decision_data=explanation_data.get("decision_data", {}),
            )

            entry.metadata["km_sync_pending"] = True
            self._pending_entries.append(entry)

            self._emit_event(
                "km_explainability_ingested",
                {
                    "decision_id": entry.decision_id,
                    "debate_id": entry.debate_id,
                    "confidence": entry.confidence,
                },
            )

            logger.info(
                "[explainability_adapter] Ingested explanation for debate %s",
                entry.debate_id,
            )
            return True

        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[explainability_adapter] Ingest failed: %s", e)
            return False

    def get(self, record_id: str) -> ExplainabilityEntry | None:
        """Get an explanation entry by decision ID."""
        clean_id = record_id[4:] if record_id.startswith("exp_") else record_id
        return self._synced_entries.get(clean_id)

    async def get_async(self, record_id: str) -> ExplainabilityEntry | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[ExplainabilitySearchResult]:
        """Search stored explanations by topic similarity.

        Args:
            query: Search query (topic text).
            limit: Max results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of ExplainabilitySearchResult sorted by relevance.
        """
        results: list[ExplainabilitySearchResult] = []
        query_lower = query.lower()

        all_entries = list(self._synced_entries.values()) + self._pending_entries
        for entry in all_entries:
            if entry.confidence < min_confidence:
                continue

            task_lower = entry.task.lower()
            if query_lower in task_lower or any(word in task_lower for word in query_lower.split()):
                similarity = 0.8 if query_lower in task_lower else 0.5
                results.append(
                    ExplainabilitySearchResult(
                        decision_id=entry.decision_id,
                        debate_id=entry.debate_id,
                        task=entry.task,
                        domain=entry.domain,
                        conclusion=entry.conclusion,
                        confidence=entry.confidence,
                        consensus_reached=entry.consensus_reached,
                        top_factors=entry.top_factors,
                        top_counterfactuals=entry.top_counterfactuals,
                        similarity=similarity,
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, entry: ExplainabilityEntry) -> Any:
        """Convert an ExplainabilityEntry to a KnowledgeItem for KM storage."""
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Build content summary
        content_parts = [f"Explanation: {entry.task}"]
        if entry.conclusion:
            content_parts.append(f"\nConclusion: {entry.conclusion}")

        if entry.top_factors:
            factor_lines = []
            for f in entry.top_factors[:3]:
                factor_lines.append(
                    f"  - {f.get('factor', 'unknown')}: "
                    f"{f.get('contribution', 0):.0%} ({f.get('explanation', '')})"
                )
            if factor_lines:
                content_parts.append("\nKey factors:\n" + "\n".join(factor_lines))

        if entry.top_counterfactuals:
            cf_lines = []
            for cf in entry.top_counterfactuals[:2]:
                cf_lines.append(
                    f"  - If {cf.get('condition', '?')}: "
                    f"{cf.get('outcome_change', '?')} "
                    f"(sensitivity={cf.get('sensitivity', 0):.2f})"
                )
            if cf_lines:
                content_parts.append("\nCounterfactuals:\n" + "\n".join(cf_lines))

        content = "\n".join(content_parts)

        # Determine tags
        tags = [
            "explainability",
            f"domain:{entry.domain}",
            f"consensus:{entry.consensus_type}",
        ]
        if entry.counterfactual_count > 0:
            tags.append("has_counterfactuals")
        if entry.consensus_reached:
            tags.append("consensus_reached")
        if entry.evidence_quality_score >= 0.7:
            tags.append("high_evidence_quality")

        return KnowledgeItem(
            id=f"exp_{entry.decision_id}",
            content=content,
            source=KnowledgeSource.EXPLAINABILITY,
            source_id=entry.debate_id,
            confidence=ConfidenceLevel.from_float(entry.confidence),
            created_at=entry.created_at,
            updated_at=entry.created_at,
            metadata={
                "decision_id": entry.decision_id,
                "debate_id": entry.debate_id,
                "task": entry.task,
                "domain": entry.domain,
                "consensus_reached": entry.consensus_reached,
                "consensus_type": entry.consensus_type,
                "rounds_used": entry.rounds_used,
                "agents_participated": entry.agents_participated,
                "evidence_count": entry.evidence_count,
                "vote_pivot_count": entry.vote_pivot_count,
                "belief_change_count": entry.belief_change_count,
                "factor_count": entry.factor_count,
                "counterfactual_count": entry.counterfactual_count,
                "evidence_quality_score": entry.evidence_quality_score,
                "agent_agreement_score": entry.agent_agreement_score,
                "belief_stability_score": entry.belief_stability_score,
                "top_factors": entry.top_factors,
                "top_counterfactuals": entry.top_counterfactuals,
                "tags": tags,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.3,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending explanation entries to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance.
            min_confidence: Minimum confidence to sync.
            batch_size: Max records per batch.

        Returns:
            SyncResult with sync statistics.
        """
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_entries[:batch_size]

        for entry in pending:
            if entry.confidence < min_confidence:
                skipped += 1
                continue

            try:
                km_item = self.to_knowledge_item(entry)

                # Try multiple store methods for compatibility
                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                entry.metadata["km_sync_pending"] = False
                entry.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                entry.metadata["km_item_id"] = km_item.id

                self._synced_entries[entry.decision_id] = entry
                synced += 1

                self._emit_event(
                    "km_explainability_synced",
                    {
                        "adapter": self.adapter_name,
                        "decision_id": entry.decision_id,
                        "debate_id": entry.debate_id,
                        "km_item_id": km_item.id,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                failed += 1
                error_msg = f"Failed to sync explanation {entry.decision_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                entry.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"

        # Remove successfully synced from pending
        synced_ids = {e.decision_id for e in pending if e.metadata.get("km_sync_pending") is False}
        self._pending_entries = [
            e for e in self._pending_entries if e.decision_id not in synced_ids
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
        """Get statistics about stored explanations."""
        all_entries = list(self._synced_entries.values())
        return {
            "total_synced": len(self._synced_entries),
            "pending_sync": len(self._pending_entries),
            "avg_confidence": (
                sum(e.confidence for e in all_entries) / len(all_entries) if all_entries else 0.0
            ),
            "avg_factor_count": (
                sum(e.factor_count for e in all_entries) / len(all_entries) if all_entries else 0.0
            ),
            "total_counterfactuals": sum(e.counterfactual_count for e in all_entries),
            "consensus_rate": (
                sum(1 for e in all_entries if e.consensus_reached) / len(all_entries)
                if all_entries
                else 0.0
            ),
        }


# Module-level singleton for cross-module access
_explainability_adapter_singleton: ExplainabilityAdapter | None = None


def get_explainability_adapter() -> ExplainabilityAdapter:
    """Get or create the module-level ExplainabilityAdapter singleton.

    Returns:
        The singleton ExplainabilityAdapter instance.
    """
    global _explainability_adapter_singleton
    if _explainability_adapter_singleton is None:
        _explainability_adapter_singleton = ExplainabilityAdapter()
    return _explainability_adapter_singleton


__all__ = [
    "ExplainabilityAdapter",
    "ExplainabilityEntry",
    "ExplainabilitySearchResult",
    "get_explainability_adapter",
]
