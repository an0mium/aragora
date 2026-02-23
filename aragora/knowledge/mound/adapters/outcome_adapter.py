"""
OutcomeAdapter - Bridges Decision Outcomes to the Knowledge Mound.

This adapter enables closed-loop decision learning:

- Data flow IN: Outcome records are stored as knowledge items
- Data flow IN: KPI deltas are tracked for impact measurement
- Reverse flow: KM can retrieve past outcomes for similar decisions
- Semantic search: Find outcomes by topic, tags, or similarity

The adapter provides:
- Automatic extraction of outcome data to knowledge items
- Impact score tracking with KPI before/after deltas
- Lessons learned persistence for institutional memory
- Timeline queries for decision outcome chains

"Every outcome teaches; every lesson compounds."
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
)
from aragora.storage.governance.models import OutcomeRecord

logger = logging.getLogger(__name__)

OUTCOME_SOURCE = KnowledgeSource.DEBATE

EventCallback = Callable[[str, dict[str, Any]], None]


class OutcomeAdapterError(Exception):
    """Base exception for outcome adapter errors."""


class OutcomeNotFoundError(OutcomeAdapterError):
    """Raised when an outcome is not found."""


@dataclass
class OutcomeIngestionResult:
    """Result of ingesting a decision outcome into Knowledge Mound."""

    outcome_id: str
    items_ingested: int
    knowledge_item_ids: list[str]
    errors: list[str]

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.items_ingested > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "outcome_id": self.outcome_id,
            "items_ingested": self.items_ingested,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


class OutcomeAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Decision Outcomes to the Knowledge Mound.

    Provides methods to:
    - Ingest outcome records as knowledge items
    - Track impact scores and KPI deltas
    - Store lessons learned for institutional memory
    - Find similar past outcomes via semantic search
    - Build outcome timelines for decision chains

    Usage:
        from aragora.knowledge.mound.adapters.outcome_adapter import OutcomeAdapter
        from aragora.knowledge.mound.core import KnowledgeMound

        mound = KnowledgeMound()
        adapter = OutcomeAdapter(mound)

        result = adapter.ingest(outcome_data)
        similar = await adapter.find_similar_outcomes("vendor selection", limit=5)
    """

    adapter_name = "outcome"

    ID_PREFIX = "outc_"
    LESSON_PREFIX = "lsn_"

    def __init__(
        self,
        mound: Any | None = None,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        """Initialize the adapter.

        Args:
            mound: Optional KnowledgeMound instance to use
            enable_dual_write: If True, writes go to both outcome store and KM
            event_callback: Optional callback for emitting events
            enable_resilience: If True, enables circuit breaker and bulkhead protection
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )

        self._mound = mound
        self._ingested_outcomes: dict[str, OutcomeIngestionResult] = {}

    def set_mound(self, mound: Any) -> None:
        """Set the Knowledge Mound instance."""
        self._mound = mound

    @staticmethod
    def record_to_dict(record: OutcomeRecord) -> dict[str, Any]:
        """Convert an OutcomeRecord dataclass to the dict format expected by ingest().

        This bridges the governance data model to the adapter's ingestion format,
        keeping the feedback loop type-safe from handler through to Knowledge Mound.

        Args:
            record: A persisted OutcomeRecord from the governance store.

        Returns:
            Dict suitable for passing to ingest().
        """
        return record.to_dict()

    def ingest_record(self, record: OutcomeRecord) -> bool:
        """Ingest an OutcomeRecord directly into the Knowledge Mound.

        Convenience method that converts an OutcomeRecord dataclass to dict
        format and delegates to ingest(). Use this when you have a typed
        OutcomeRecord from the governance store.

        Args:
            record: OutcomeRecord instance.

        Returns:
            True if ingestion succeeded.
        """
        return self.ingest(self.record_to_dict(record))

    def ingest(self, outcome_data: dict[str, Any]) -> bool:
        """Synchronous convenience method to ingest an outcome from a plain dict.

        Creates a KnowledgeItem from the dict fields and stores it.

        Args:
            outcome_data: Dict with keys: outcome_id, decision_id, debate_id,
                outcome_type, outcome_description, impact_score, lessons_learned, tags.

        Returns:
            True if ingestion succeeded, False otherwise.
        """
        try:
            outcome_id = outcome_data.get("outcome_id", "")
            decision_id = outcome_data.get("decision_id", "")
            debate_id = outcome_data.get("debate_id", "")
            outcome_type = outcome_data.get("outcome_type", "unknown")
            description = outcome_data.get("outcome_description", "")
            impact_score = outcome_data.get("impact_score", 0.0)
            lessons = outcome_data.get("lessons_learned", "")
            tags = outcome_data.get("tags", [])
            kpis_before = outcome_data.get("kpis_before", {})
            kpis_after = outcome_data.get("kpis_after", {})

            # Map impact score to confidence
            if impact_score >= 0.7:
                conf_level = ConfidenceLevel.HIGH
            elif impact_score >= 0.4:
                conf_level = ConfidenceLevel.MEDIUM
            else:
                conf_level = ConfidenceLevel.LOW

            item_id = f"{self.ID_PREFIX}{hashlib.md5(outcome_id.encode(), usedforsecurity=False).hexdigest()[:12]}"
            now = datetime.now(timezone.utc)

            content = f"[Outcome:{outcome_type}] {description}"
            if lessons:
                content += f"\n\nLessons: {lessons}"

            # Compute KPI delta summary
            kpi_deltas = {}
            for key in set(list(kpis_before.keys()) + list(kpis_after.keys())):
                before_val = kpis_before.get(key)
                after_val = kpis_after.get(key)
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    kpi_deltas[key] = after_val - before_val

            item = KnowledgeItem(
                id=item_id,
                content=content,
                source=OUTCOME_SOURCE,
                source_id=decision_id,
                confidence=conf_level,
                created_at=now,
                updated_at=now,
                metadata={
                    "outcome_id": outcome_id,
                    "decision_id": decision_id,
                    "debate_id": debate_id,
                    "outcome_type": outcome_type,
                    "impact_score": impact_score,
                    "kpis_before": kpis_before,
                    "kpis_after": kpis_after,
                    "kpi_deltas": kpi_deltas,
                    "lessons_learned": lessons,
                    "tags": tags + ["decision_outcome", f"type:{outcome_type}"],
                    "item_type": "decision_outcome",
                },
            )

            if self._mound and hasattr(self._mound, "store_sync"):
                self._mound.store_sync(item)
            elif self._mound and hasattr(self._mound, "store"):
                pass  # async-only mound, dual-write handles persistence

            result = OutcomeIngestionResult(
                outcome_id=outcome_id,
                items_ingested=1,
                knowledge_item_ids=[item_id],
                errors=[],
            )
            self._ingested_outcomes[outcome_id] = result

            self._emit_event("outcome_ingested", {
                "outcome_id": outcome_id,
                "decision_id": decision_id,
                "outcome_type": outcome_type,
                "impact_score": impact_score,
            })

            logger.info(
                "[outcome_adapter] Ingested outcome %s as KM item %s",
                outcome_id,
                item_id,
            )
            return True

        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[outcome_adapter] Ingest failed: %s", e)
            return False

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to emit event %s: %s", event_type, e)

    async def find_similar_outcomes(
        self,
        query: str,
        workspace_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeItem]:
        """Find outcomes related to a query via semantic search.

        Args:
            query: Search query
            workspace_id: Optional workspace filter
            outcome_type: Optional filter by outcome type
            limit: Maximum results

        Returns:
            List of related outcome knowledge items
        """
        if not self._mound:
            return []

        try:
            tags = ["decision_outcome"]
            if outcome_type:
                tags.append(f"type:{outcome_type}")

            if hasattr(self._mound, "query"):
                results = await self._mound.query(
                    query=query,
                    tags=tags,
                    workspace_id=workspace_id,
                    limit=limit,
                )
                return results.items if hasattr(results, "items") else []
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[outcome_adapter] Search failed: %s", e)

        return []

    async def get_outcome_timeline(
        self,
        decision_id: str,
        limit: int = 20,
    ) -> list[KnowledgeItem]:
        """Get the outcome timeline for a decision chain.

        Args:
            decision_id: The decision to get outcomes for
            limit: Maximum results

        Returns:
            List of outcome items ordered by time
        """
        if not self._mound:
            return []

        try:
            if hasattr(self._mound, "query"):
                results = await self._mound.query(
                    query=f"decision_id:{decision_id}",
                    tags=["decision_outcome"],
                    limit=limit,
                )
                return results.items if hasattr(results, "items") else []
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[outcome_adapter] Timeline query failed: %s", e)

        return []

    def get_ingestion_result(self, outcome_id: str) -> OutcomeIngestionResult | None:
        """Get the ingestion result for an outcome."""
        return self._ingested_outcomes.get(outcome_id)

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        total_items = sum(r.items_ingested for r in self._ingested_outcomes.values())
        total_errors = sum(len(r.errors) for r in self._ingested_outcomes.values())
        type_counts: dict[str, int] = {}
        for result in self._ingested_outcomes.values():
            # Count by outcome type from the ingested data
            type_counts[result.outcome_id] = result.items_ingested

        return {
            "outcomes_processed": len(self._ingested_outcomes),
            "total_items_ingested": total_items,
            "total_errors": total_errors,
            "mound_connected": self._mound is not None,
        }


# Module-level singleton
_outcome_adapter_singleton: OutcomeAdapter | None = None


def get_outcome_adapter() -> OutcomeAdapter:
    """Get or create the module-level OutcomeAdapter singleton."""
    global _outcome_adapter_singleton
    if _outcome_adapter_singleton is None:
        _outcome_adapter_singleton = OutcomeAdapter()
    return _outcome_adapter_singleton


__all__ = [
    "OutcomeAdapter",
    "OutcomeAdapterError",
    "OutcomeNotFoundError",
    "OutcomeIngestionResult",
    "get_outcome_adapter",
]
