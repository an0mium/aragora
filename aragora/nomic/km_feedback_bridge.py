"""Knowledge Mound Feedback Bridge for the Nomic Loop.

After a successful self-improvement cycle, this bridge extracts
learned patterns and persists them in the Knowledge Mound so that
future cycles can query past experience.

Usage:
    from aragora.nomic.km_feedback_bridge import KMFeedbackBridge
    from aragora.nomic.cycle_telemetry import CycleRecord

    bridge = KMFeedbackBridge()
    bridge.persist_cycle_learnings(record)

    learnings = bridge.retrieve_relevant_learnings("improve test coverage")
    for item in learnings:
        print(item["content"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearningItem:
    """A single learning extracted from a cycle."""

    content: str
    tags: list[str] = field(default_factory=list)
    source: str = "nomic_cycle"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tags": self.tags,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class KMFeedbackBridge:
    """Bridge between the Nomic Loop and the Knowledge Mound.

    Responsibilities:
    1. **persist_cycle_learnings**: After a cycle, extract what worked,
       what failed, and which agents performed best, then store as
       KnowledgeItems with structured tags.
    2. **retrieve_relevant_learnings**: Before a cycle, query KM for
       past learnings relevant to the current goal.
    """

    def __init__(self, km: Any | None = None):
        """Initialize the bridge.

        Args:
            km: Optional Knowledge Mound instance. If None, will attempt
                to acquire one via ``get_knowledge_mound()`` at call time.
        """
        self._km = km
        self._in_memory_store: list[LearningItem] = []

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist_cycle_learnings(self, cycle_record: Any) -> list[LearningItem]:
        """Extract learnings from a cycle record and persist to KM.

        Args:
            cycle_record: A CycleRecord (from cycle_telemetry) or any
                          object with matching attributes.

        Returns:
            List of LearningItem objects that were persisted.
        """
        items = self._extract_learnings(cycle_record)
        if not items:
            return []

        km = self._get_km()
        persisted: list[LearningItem] = []

        for item in items:
            # Always store in memory first (guaranteed to succeed)
            self._in_memory_store.append(item)
            persisted.append(item)
            # Then attempt KM ingestion (best-effort)
            if km is not None:
                try:
                    self._ingest_to_km(km, item)
                except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as e:
                    logger.debug("km_feedback_persist_failed: %s", e)

        logger.info(
            "km_feedback_persisted count=%d cycle=%s goal=%s",
            len(persisted),
            getattr(cycle_record, "cycle_id", "unknown"),
            getattr(cycle_record, "goal", "")[:60],
        )
        return persisted

    def _extract_learnings(self, record: Any) -> list[LearningItem]:
        """Extract structured learnings from a cycle record."""
        items: list[LearningItem] = []
        cycle_id = getattr(record, "cycle_id", "unknown")
        goal = getattr(record, "goal", "")
        success = getattr(record, "success", False)
        agents = getattr(record, "agents_used", [])
        quality_delta = getattr(record, "quality_delta", 0.0)
        cost_usd = getattr(record, "cost_usd", 0.0)
        timestamp = getattr(record, "timestamp", time.time())

        base_tags = [
            "nomic_learned:true",
            f"cycle_id:{cycle_id}",
            f"goal:{goal[:80]}",
        ]

        # Learning 1: Outcome summary
        outcome = "succeeded" if success else "failed"
        items.append(
            LearningItem(
                content=(
                    f"Nomic cycle {cycle_id} {outcome} on goal: {goal}. "
                    f"Quality delta: {quality_delta:.4f}, cost: ${cost_usd:.4f}, "
                    f"agents: {', '.join(agents) if agents else 'none'}."
                ),
                tags=base_tags + [f"outcome:{outcome}"],
                source="nomic_cycle_summary",
                timestamp=timestamp,
            )
        )

        # Learning 2: Agent performance (if we have agent data)
        if agents:
            if success:
                items.append(
                    LearningItem(
                        content=(
                            f"Agents {', '.join(agents)} succeeded on goal type: {goal[:60]}. "
                            f"Consider reusing this team for similar goals."
                        ),
                        tags=base_tags + ["agent_success"] + [f"agent:{a}" for a in agents[:5]],
                        source="nomic_agent_performance",
                        timestamp=timestamp,
                    )
                )
            else:
                items.append(
                    LearningItem(
                        content=(
                            f"Agents {', '.join(agents)} failed on goal type: {goal[:60]}. "
                            f"Consider alternative agents or different decomposition."
                        ),
                        tags=base_tags + ["agent_failure"] + [f"agent:{a}" for a in agents[:5]],
                        source="nomic_agent_performance",
                        timestamp=timestamp,
                    )
                )

        # Learning 3: Cost efficiency
        if success and cost_usd > 0 and quality_delta > 0:
            efficiency = quality_delta / cost_usd
            items.append(
                LearningItem(
                    content=(
                        f"Cost efficiency for goal '{goal[:40]}': "
                        f"{efficiency:.2f} quality-per-dollar "
                        f"(delta={quality_delta:.4f}, cost=${cost_usd:.4f})."
                    ),
                    tags=base_tags + ["cost_efficiency"],
                    source="nomic_cost_analysis",
                    timestamp=timestamp,
                )
            )

        return items

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_relevant_learnings(
        self,
        goal_text: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Query KM for past learnings relevant to the given goal.

        Falls back to in-memory keyword search if KM is unavailable.

        Args:
            goal_text: The current goal to find relevant learnings for.
            limit: Maximum number of results.

        Returns:
            List of dicts with 'content', 'tags', 'source', 'timestamp'.
        """
        results: list[dict[str, Any]] = []

        # Try KM first
        km = self._get_km()
        if km is not None:
            try:
                km_results = self._search_km(km, goal_text, limit)
                results.extend(km_results)
            except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as e:
                logger.debug("km_feedback_retrieve_km_failed: %s", e)

        # Supplement with in-memory store
        if len(results) < limit:
            in_memory = self._search_in_memory(goal_text, limit - len(results))
            results.extend(in_memory)

        logger.info(
            "km_feedback_retrieved count=%d goal=%s",
            len(results),
            goal_text[:60],
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Internal: KM operations
    # ------------------------------------------------------------------

    def _get_km(self) -> Any:
        """Acquire a Knowledge Mound instance."""
        if self._km is not None:
            return self._km

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            return get_knowledge_mound()
        except ImportError:
            return None
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug("km_feedback_km_unavailable: %s", e)
            return None

    def _ingest_to_km(self, km: Any, item: LearningItem) -> None:
        """Ingest a single learning item into the KM."""
        try:
            from aragora.knowledge.mound.core import KnowledgeItem

            ki = KnowledgeItem(  # type: ignore[call-arg]
                content=item.content,
                source=item.source,  # type: ignore[arg-type]
                tags=item.tags,
            )

            # Synchronous ingestion (preferred for reliability)
            if hasattr(km, "ingest_sync"):
                km.ingest_sync(ki)
            elif hasattr(km, "ingest"):
                # Try fire-and-forget async
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(km.ingest(ki))
                except RuntimeError:
                    # No event loop - skip
                    pass
        except ImportError:
            logger.debug("KnowledgeItem not available for ingestion")

    def _search_km(
        self,
        km: Any,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search KM for learnings matching the query."""
        results: list[dict[str, Any]] = []

        # Try semantic search
        if hasattr(km, "search"):
            items = km.search(
                query=f"nomic_learned {query}",
                limit=limit,
                tags=["nomic_learned:true"],
            )
            if items:
                for item in items:
                    content = getattr(item, "content", str(item))
                    tags = getattr(item, "tags", [])
                    source = getattr(item, "source", "km")
                    timestamp = getattr(item, "timestamp", 0.0)
                    results.append(
                        {
                            "content": content,
                            "tags": tags,
                            "source": source,
                            "timestamp": timestamp,
                        }
                    )

        return results

    def _search_in_memory(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Keyword-based search over in-memory store."""
        query_words = set(query.lower().split())
        scored: list[tuple[float, LearningItem]] = []

        for item in self._in_memory_store:
            content_words = set(item.content.lower().split())
            tag_words = set(
                word for tag in item.tags for word in tag.lower().replace(":", " ").split()
            )
            all_words = content_words | tag_words
            overlap = len(query_words & all_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item.to_dict() for _, item in scored[:limit]]


__all__ = [
    "KMFeedbackBridge",
    "LearningItem",
]
