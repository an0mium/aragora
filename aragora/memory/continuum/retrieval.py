"""
Retrieval operations mixin for ContinuumMemory.

Provides retrieve, hybrid_search, and related search operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from aragora.memory.tier_manager import MemoryTier
from aragora.resilience.retry import PROVIDER_RETRY_POLICIES, with_retry
from aragora.utils.json_helpers import safe_json_loads

from .entry import AwaitableList, ContinuumMemoryEntry

if TYPE_CHECKING:
    from .core import ContinuumMemory

logger = logging.getLogger(__name__)

# Retry configuration for memory operations
_MEMORY_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["memory"]


class RetrievalMixin:
    """Mixin providing retrieval operations for ContinuumMemory."""

    def retrieve(
        self: "ContinuumMemory",
        query: str | None = None,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: str | MemoryTier | None = None,
    ) -> list[ContinuumMemoryEntry]:
        """
        Retrieve memories ranked by importance, surprise, and recency.

        The retrieval formula combines:
        - Tier-weighted importance
        - Surprise score (unexpected patterns are more valuable)
        - Time decay based on tier half-life

        Args:
            query: Optional query for relevance filtering
            tiers: Filter to specific tiers (default: all)
            limit: Maximum entries to return
            min_importance: Minimum importance threshold
            include_glacial: Whether to include glacial tier

        Returns:
            List of memory entries sorted by retrieval score
        """
        if tier is not None:
            target_tier: MemoryTier = MemoryTier(tier) if isinstance(tier, str) else tier
            if query:
                entry: ContinuumMemoryEntry | None = self.get(query)
                if entry and entry.tier == target_tier:
                    return AwaitableList([entry])
                return AwaitableList([])
            tiers = [target_tier]

        # Build tier filter
        if tiers is None:
            tiers = list(MemoryTier)
        if not include_glacial:
            tiers = [t for t in tiers if t != MemoryTier.GLACIAL]

        tier_values: list[str] = [t.value for t in tiers]
        placeholders: str = ",".join("?" * len(tier_values))

        # Build keyword filter clause for SQL (more efficient than Python filtering)
        keyword_clause: str = ""
        keyword_params: list[str] = []
        if query:
            # Split query into words and require at least one match
            # Limit to 50 keywords to prevent unbounded SQL condition generation
            MAX_QUERY_KEYWORDS: int = 50
            keywords: list[str] = [
                kw.strip().lower() for kw in query.split()[:MAX_QUERY_KEYWORDS] if kw.strip()
            ]
            if keywords:
                # Use INSTR for case-insensitive containment check (faster than LIKE)
                keyword_conditions: list[str] = ["INSTR(LOWER(content), ?) > 0" for _ in keywords]
                keyword_clause = f" AND ({' OR '.join(keyword_conditions)})"
                keyword_params = keywords

        with self.connection() as conn:
            cursor = conn.cursor()

            # Retrieval query with time-decay scoring
            # Score = importance * (1 + surprise) * decay_factor
            # Keyword filtering now done in SQL for efficiency
            cursor.execute(
                f"""
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at, metadata,
                       (importance * (1 + surprise_score) *
                        (1.0 / (1 + (julianday('now') - julianday(updated_at)) *
                         CASE tier
                           WHEN 'fast' THEN 24
                           WHEN 'medium' THEN 1
                           WHEN 'slow' THEN 0.14
                           WHEN 'glacial' THEN 0.03
                         END))) as score
                FROM continuum_memory
                WHERE tier IN ({placeholders})
                  AND importance >= ?
                  {keyword_clause}
                ORDER BY score DESC
                LIMIT ?
                """,
                (*tier_values, min_importance, *keyword_params, limit),
            )

            rows: list[tuple[Any, ...]] = cursor.fetchall()

        entries: list[ContinuumMemoryEntry] = []
        for row in rows:
            mem_entry: ContinuumMemoryEntry = ContinuumMemoryEntry(
                id=row[0],
                tier=MemoryTier(row[1]),
                content=row[2],
                importance=row[3],
                surprise_score=row[4],
                consolidation_score=row[5],
                update_count=row[6],
                success_count=row[7],
                failure_count=row[8],
                created_at=row[9],
                updated_at=row[10],
                metadata=safe_json_loads(row[11], {}),
            )
            entries.append(mem_entry)

        # Emit MEMORY_RECALL event if memories were retrieved
        if entries and self.event_emitter:
            try:
                tier_counts: dict[str, int] = {}
                for e in entries:
                    tier_counts[e.tier.value] = tier_counts.get(e.tier.value, 0) + 1

                self.event_emitter.emit_sync(
                    event_type="memory_recall",
                    debate_id="",
                    count=len(entries),
                    query=query[:100] if query else None,
                    tier_distribution=tier_counts,
                    top_importance=max(e.importance for e in entries),
                )
            except (ImportError, AttributeError, TypeError):
                pass  # Emitter not available or misconfigured

            # Also emit MEMORY_RETRIEVED for cross-subsystem tracking
            try:
                for entry in entries:
                    self.event_emitter.emit_sync(
                        event_type="memory_retrieved",
                        debate_id="",
                        memory_id=entry.id,
                        tier=entry.tier.value,
                        importance=entry.importance,
                        cache_hit=False,  # DB retrieval, not cache
                    )
            except (ImportError, AttributeError, TypeError):
                pass  # Emitter not available

        return AwaitableList(entries)

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def retrieve_async(
        self: "ContinuumMemory",
        query: str | None = None,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: str | MemoryTier | None = None,
    ) -> list[ContinuumMemoryEntry]:
        """Async wrapper for retrieve() - offloads blocking I/O to executor."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(
                query=query,
                tiers=tiers,
                limit=limit,
                min_importance=min_importance,
                include_glacial=include_glacial,
                tier=tier,
            ),
        )

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def hybrid_search(
        self: "ContinuumMemory",
        query: str,
        limit: int = 10,
        tiers: list[MemoryTier] | list[str] | None = None,
        vector_weight: float | None = None,
        min_importance: float = 0.0,
    ) -> list[Any]:
        """
        Perform hybrid search combining vector and keyword retrieval.

        Uses Reciprocal Rank Fusion (RRF) to combine results from vector
        similarity search (via KM adapter) and keyword search (via FTS5).

        Args:
            query: Search query text
            limit: Maximum results to return
            tiers: Optional tier filter (e.g., [MemoryTier.SLOW, MemoryTier.GLACIAL])
            vector_weight: Override default vector weight (0-1), rest goes to keyword
            min_importance: Minimum importance threshold

        Returns:
            List of MemorySearchResult objects sorted by combined score

        Example:
            results = await memory.hybrid_search(
                "circuit breaker pattern",
                limit=10,
                tiers=[MemoryTier.SLOW, MemoryTier.GLACIAL],
            )
            for result in results:
                print(f"{result.memory_id}: {result.combined_score:.3f}")
        """
        from aragora.memory.hybrid_search import (
            HybridMemorySearch,
            HybridMemoryConfig,
        )

        # Lazily create hybrid search instance
        if not hasattr(self, "_hybrid_search") or self._hybrid_search is None:
            self._hybrid_search = HybridMemorySearch(
                continuum_memory=self,
                config=HybridMemoryConfig(),
            )

        # Convert MemoryTier enum values to strings for hybrid search
        tier_strings: list[str] | None = None
        if tiers:
            tier_strings = [t.value if isinstance(t, MemoryTier) else str(t) for t in tiers]

        results = await self._hybrid_search.search(
            query=query,
            limit=limit,
            tiers=tier_strings,
            vector_weight=vector_weight,
            min_importance=min_importance,
        )

        # Emit event for hybrid search
        if results and self.event_emitter:
            try:
                self.event_emitter.emit_sync(
                    event_type="memory_recall",
                    debate_id="",
                    count=len(results),
                    query=query[:100] if query else None,
                    search_type="hybrid",
                    top_combined_score=max(r.combined_score for r in results),
                )
            except (ImportError, AttributeError, TypeError):
                pass

        return results

    def rebuild_keyword_index(self: "ContinuumMemory") -> int:
        """
        Rebuild the FTS5 keyword index for hybrid search.

        Call this after bulk data loading or if the index becomes out of sync.

        Returns:
            Number of entries indexed
        """
        from aragora.memory.hybrid_search import HybridMemorySearch, HybridMemoryConfig

        if not hasattr(self, "_hybrid_search") or self._hybrid_search is None:
            self._hybrid_search = HybridMemorySearch(
                continuum_memory=self,
                config=HybridMemoryConfig(),
            )

        count: int = self._hybrid_search.rebuild_keyword_index()
        logger.info(f"Rebuilt keyword index: {count} entries")
        return count


__all__ = ["RetrievalMixin"]
