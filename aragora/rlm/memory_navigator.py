"""
RLM Memory Navigator - Unified memory exploration via REPL helpers.

Uses the Unified Memory Gateway to provide REPL-accessible helpers
for navigating across all 5 memory systems (ContinuumMemory, Knowledge Mound,
Supermemory, claude-mem, RLM).

Based on arXiv:2512.24601 "Recursive Language Models":
Context stored as Python variables in REPL, model writes code to
query/filter/navigate across memory systems programmatically.

Usage in TRUE RLM REPL:
    # Load unified memory
    nav = await build_context_hierarchy("rate limiting", max_items=20)

    # Search across all systems
    results = await search_all("rate limit")

    # Drill into a specific item
    detail = await drill_into("km", "km_42")

    # Get high-surprise items (Titans/MIRAS insight)
    surprising = await get_by_surprise(min_surprise=0.7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.memory.gateway import MemoryGateway, UnifiedMemoryResult
    from aragora.memory.retention_gate import RetentionGate

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMemoryItem:
    """A memory item formatted for RLM REPL navigation."""

    id: str
    source: str  # "continuum", "km", "supermemory", "claude_mem"
    content: str
    confidence: float
    surprise_score: float | None = None
    retention_action: str | None = None  # "retain", "demote", "forget", "consolidate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedMemoryContext:
    """Structured context for RLM REPL navigation across all memory systems.

    Memory items are stored as Python data structures that the
    LLM can query programmatically via REPL helpers.
    """

    items: list[UnifiedMemoryItem]
    by_source: dict[str, list[UnifiedMemoryItem]]
    by_id: dict[str, UnifiedMemoryItem]
    total_items: int
    source_counts: dict[str, int]
    query: str
    sources_queried: list[str] = field(default_factory=list)
    duplicates_removed: int = 0
    query_time_ms: float = 0.0


class RLMMemoryNavigator:
    """REPL helpers for programmatic memory exploration across all systems.

    Wraps the MemoryGateway to provide typed, REPL-friendly navigation
    functions that can be injected into TRUE RLM environments.
    """

    def __init__(
        self,
        gateway: MemoryGateway | None = None,
        retention_gate: RetentionGate | None = None,
    ):
        self._gateway = gateway
        self._retention_gate = retention_gate

    def get_repl_helpers(self) -> dict[str, Any]:
        """Get all unified memory REPL helpers as a dictionary.

        Returns:
            Dictionary of helper functions for RLM REPL injection
        """
        helpers: dict[str, Any] = {
            # Types
            "UnifiedMemoryItem": UnifiedMemoryItem,
            "UnifiedMemoryContext": UnifiedMemoryContext,
            # Async helpers
            "search_all": self.search_all,
            "build_context_hierarchy": self.build_context_hierarchy,
            "drill_into": self.drill_into,
            "get_by_surprise": self.get_by_surprise,
            # Sync filter helpers
            "filter_by_source": filter_by_source,
            "filter_by_confidence": filter_by_confidence,
            "sort_by_confidence": sort_by_confidence,
            "sort_by_surprise": sort_by_surprise,
        }
        return helpers

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        sources: list[str] | None = None,
    ) -> list[UnifiedMemoryItem]:
        """Search across all memory systems via the gateway.

        Args:
            query: Search query string
            limit: Maximum results
            sources: Optional source filter (e.g. ["km", "continuum"])

        Returns:
            List of UnifiedMemoryItem results
        """
        if not self._gateway:
            logger.warning("No gateway configured for memory navigator")
            return []

        from aragora.memory.gateway import UnifiedMemoryQuery

        resp = await self._gateway.query(
            UnifiedMemoryQuery(
                query=query,
                limit=limit,
                sources=sources,
            )
        )

        return [_to_unified_item(r) for r in resp.results]

    async def build_context_hierarchy(
        self,
        topic: str,
        max_items: int = 30,
        sources: list[str] | None = None,
    ) -> UnifiedMemoryContext:
        """Build a structured context hierarchy for a topic.

        Queries all memory systems and organizes results into
        a navigable context structure for RLM REPL use.

        Args:
            topic: Topic to build context for
            max_items: Maximum items to include
            sources: Optional source filter

        Returns:
            UnifiedMemoryContext with indexed access
        """
        if not self._gateway:
            return UnifiedMemoryContext(
                items=[],
                by_source={},
                by_id={},
                total_items=0,
                source_counts={},
                query=topic,
            )

        from aragora.memory.gateway import UnifiedMemoryQuery

        resp = await self._gateway.query(
            UnifiedMemoryQuery(
                query=topic,
                limit=max_items,
                sources=sources,
            )
        )

        items = [_to_unified_item(r) for r in resp.results]

        # Enrich with retention decisions if gate available
        if self._retention_gate and items:
            items = await self._enrich_with_retention(items)

        by_source: dict[str, list[UnifiedMemoryItem]] = {}
        by_id: dict[str, UnifiedMemoryItem] = {}
        for item in items:
            by_source.setdefault(item.source, []).append(item)
            if item.id:
                by_id[item.id] = item

        source_counts = {src: len(lst) for src, lst in by_source.items()}

        return UnifiedMemoryContext(
            items=items,
            by_source=by_source,
            by_id=by_id,
            total_items=len(items),
            source_counts=source_counts,
            query=topic,
            sources_queried=resp.sources_queried,
            duplicates_removed=resp.duplicates_removed,
            query_time_ms=resp.query_time_ms,
        )

    async def drill_into(
        self,
        source: str,
        item_id: str,
    ) -> dict[str, Any]:
        """Drill into a specific memory item for detailed view.

        Args:
            source: Source system ("km", "continuum", etc.)
            item_id: Item ID within that source

        Returns:
            Detailed item info dict
        """
        if not self._gateway:
            return {"error": "No gateway configured"}

        # Use source-specific access for deeper detail
        detail: dict[str, Any] = {
            "source": source,
            "item_id": item_id,
        }

        try:
            if source == "km" and self._gateway.knowledge_mound:
                mound = self._gateway.knowledge_mound
                if hasattr(mound, "get_item"):
                    item = await mound.get_item(item_id)
                    if item:
                        detail["content"] = getattr(item, "content", "")
                        detail["confidence"] = getattr(item, "confidence", 0.0)
                        detail["tags"] = getattr(item, "tags", [])
                        detail["created_at"] = str(
                            getattr(item, "created_at", "")
                        )
                        return detail

            elif source == "continuum" and self._gateway.continuum_memory:
                cm = self._gateway.continuum_memory
                if hasattr(cm, "get_entry"):
                    entry = cm.get_entry(item_id)
                    if entry:
                        detail["content"] = getattr(entry, "content", "")
                        detail["importance"] = getattr(entry, "importance", 0.0)
                        detail["surprise_score"] = getattr(
                            entry, "surprise_score", 0.0
                        )
                        detail["tier"] = getattr(entry, "tier", "")
                        return detail

        except (AttributeError, TypeError, ValueError) as e:
            detail["error"] = str(e)

        # Fallback: search for the item by ID
        detail["note"] = "Detailed lookup not available; use search_all instead"
        return detail

    async def get_by_surprise(
        self,
        min_surprise: float = 0.5,
        limit: int = 20,
    ) -> list[UnifiedMemoryItem]:
        """Get high-surprise items across all memory systems.

        Titans/MIRAS insight: surprising items are more likely to be
        important for learning and should be retained.

        Args:
            min_surprise: Minimum surprise score threshold
            limit: Maximum results

        Returns:
            List of high-surprise items sorted by surprise descending
        """
        if not self._gateway:
            return []

        from aragora.memory.gateway import UnifiedMemoryQuery

        # Query broadly then filter by surprise
        resp = await self._gateway.query(
            UnifiedMemoryQuery(
                query="*",
                limit=limit * 3,  # Over-fetch to compensate for filtering
                sources=["continuum"],  # Surprise scores mainly from continuum
            )
        )

        items = [_to_unified_item(r) for r in resp.results]
        # Filter by surprise score
        items = [
            item
            for item in items
            if item.surprise_score is not None
            and item.surprise_score >= min_surprise
        ]
        # Sort by surprise descending
        items.sort(
            key=lambda x: x.surprise_score if x.surprise_score is not None else 0.0,
            reverse=True,
        )
        return items[:limit]

    async def _enrich_with_retention(
        self, items: list[UnifiedMemoryItem]
    ) -> list[UnifiedMemoryItem]:
        """Enrich items with retention gate decisions."""
        if not self._retention_gate:
            return items

        try:
            batch_input = [
                {
                    "item_id": item.id,
                    "source": item.source,
                    "content": item.content,
                    "outcome_surprise": item.surprise_score or 0.0,
                    "current_confidence": item.confidence,
                }
                for item in items
            ]
            decisions = self._retention_gate.batch_evaluate(batch_input)
            for item, decision in zip(items, decisions):
                item.retention_action = decision.action
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Retention enrichment failed: %s", e)

        return items


# ---------------------------------------------------------------------------
# Stateless filter/sort helpers (for REPL injection)
# ---------------------------------------------------------------------------


def filter_by_source(
    items: list[UnifiedMemoryItem],
    source: str,
) -> list[UnifiedMemoryItem]:
    """Filter items by source system."""
    return [i for i in items if i.source == source]


def filter_by_confidence(
    items: list[UnifiedMemoryItem],
    threshold: float = 0.7,
) -> list[UnifiedMemoryItem]:
    """Filter items above a confidence threshold."""
    return [i for i in items if i.confidence >= threshold]


def sort_by_confidence(
    items: list[UnifiedMemoryItem],
    descending: bool = True,
) -> list[UnifiedMemoryItem]:
    """Sort items by confidence score."""
    return sorted(items, key=lambda i: i.confidence, reverse=descending)


def sort_by_surprise(
    items: list[UnifiedMemoryItem],
    descending: bool = True,
) -> list[UnifiedMemoryItem]:
    """Sort items by surprise score (None values sorted last)."""
    return sorted(
        items,
        key=lambda i: i.surprise_score if i.surprise_score is not None else -1.0,
        reverse=descending,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_unified_item(result: UnifiedMemoryResult) -> UnifiedMemoryItem:
    """Convert a UnifiedMemoryResult to an UnifiedMemoryItem for REPL use."""
    return UnifiedMemoryItem(
        id=result.id,
        source=result.source_system,
        content=result.content,
        confidence=result.confidence,
        surprise_score=result.surprise_score,
        metadata=result.metadata,
    )
