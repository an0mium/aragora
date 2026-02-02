"""
Retrieval and search operations for Continuum Memory System.

Contains memory retrieval, hybrid search, and cache pre-warming logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aragora.memory.continuum.entry import AwaitableList, ContinuumMemoryEntry
from aragora.memory.tier_manager import MemoryTier
from aragora.resilience.retry import PROVIDER_RETRY_POLICIES, with_retry
from aragora.utils.json_helpers import safe_json_loads

if TYPE_CHECKING:
    from aragora.memory.continuum.core import ContinuumMemory

logger = logging.getLogger(__name__)

# Retry configuration for memory operations
_MEMORY_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["memory"]


class ContinuumRetrievalMixin:
    """
    Mixin providing retrieval and search operations for ContinuumMemory.

    Includes:
    - Basic retrieve() with tier filtering and keyword matching
    - Hybrid search combining vector and keyword retrieval
    - Cache pre-warming for cross-subsystem integration
    - KM reference invalidation
    """

    # These must be provided by the main class
    event_emitter: Any
    _km_adapter: Any
    _hybrid_search: Any
    hyperparams: dict[str, Any]

    def connection(self) -> Any:
        """Get database connection context manager."""
        ...

    def get(self, id: str) -> ContinuumMemoryEntry | None:
        """Get memory entry by ID."""
        ...

    def retrieve(
        self,
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
            cursor: sqlite3.Cursor = conn.cursor()

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
    async def hybrid_search(
        self,
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
                continuum_memory=self,  # type: ignore[arg-type]
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

    def rebuild_keyword_index(self) -> int:
        """
        Rebuild the FTS5 keyword index for hybrid search.

        Call this after bulk data loading or if the index becomes out of sync.

        Returns:
            Number of entries indexed
        """
        from aragora.memory.hybrid_search import HybridMemorySearch, HybridMemoryConfig

        if not hasattr(self, "_hybrid_search") or self._hybrid_search is None:
            self._hybrid_search = HybridMemorySearch(
                continuum_memory=self,  # type: ignore[arg-type]
                config=HybridMemoryConfig(),
            )

        count: int = self._hybrid_search.rebuild_keyword_index()
        logger.info(f"Rebuilt keyword index: {count} entries")
        return count

    def prewarm_for_query(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> int:
        """
        Pre-warm the memory cache for a given query.

        Called by KM->Memory cross-subscriber when Knowledge Mound is queried.
        This ensures related memories are loaded into faster access patterns.

        Args:
            query: The search query to pre-warm for
            workspace_id: Optional workspace filter
            limit: Maximum entries to pre-warm

        Returns:
            Number of entries pre-warmed
        """
        if not query:
            return 0

        try:
            # Retrieve relevant memories to warm cache
            entries: list[ContinuumMemoryEntry] = self.retrieve(
                query=query,
                limit=limit,
                min_importance=0.3,  # Only cache moderately important memories
            )

            if not entries:
                return 0

            # Batch update all entries in a single transaction using executemany
            prewarm_time: str = datetime.now().isoformat()
            current_time: str = datetime.now().isoformat()

            # Prepare batch update data
            update_data: list[tuple[str, str, str]] = []
            for entry in entries:
                if entry.metadata is None:
                    entry.metadata = {}
                entry.metadata["last_prewarm"] = prewarm_time
                metadata_json: str = json.dumps(entry.metadata)
                update_data.append((metadata_json, current_time, entry.id))

            with self.connection() as conn:
                cursor: sqlite3.Cursor = conn.cursor()
                # Use executemany for batch update (more efficient than N individual queries)
                cursor.executemany(
                    """
                    UPDATE continuum_memory
                    SET metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    update_data,
                )
                conn.commit()

            count: int = len(entries)
            logger.debug(f"Pre-warmed {count} memories for query: '{query[:50]}...'")
            return count

        except sqlite3.Error as e:
            logger.warning(f"Memory pre-warm failed (database): {e}")
            return 0
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Memory pre-warm failed (network): {e}")
            return 0
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Unexpected error during memory pre-warm: {e}")
            return 0

    def invalidate_reference(self, node_id: str) -> bool:
        """
        Invalidate any memory references to a KM node.

        Called when a KM node is deleted to clear stale cross-references.

        Args:
            node_id: The Knowledge Mound node ID to invalidate

        Returns:
            True if any references were invalidated
        """
        try:
            updated_count: int = 0
            # Find entries that reference this node and batch update
            with self.connection() as conn:
                cursor: sqlite3.Cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, metadata FROM continuum_memory
                    WHERE metadata LIKE ?
                    """,
                    (f"%{node_id}%",),
                )

                rows: list[tuple[Any, ...]] = cursor.fetchall()

                # Collect updates to perform in batch
                updates: list[tuple[str, str]] = []

                for row in rows:
                    entry_id: str = row[0]
                    metadata: dict[str, Any] = safe_json_loads(row[1], {})
                    modified: bool = False

                    # Remove km_node_id reference if present
                    if metadata.get("km_node_id") == node_id:
                        del metadata["km_node_id"]
                        metadata["km_synced"] = False
                        modified = True

                    # Remove from cross_references if present
                    # Use try/except (EAFP) to avoid O(n) in check + O(n) remove
                    cross_refs: list[str] = metadata.get("cross_references", [])
                    try:
                        cross_refs.remove(node_id)
                        metadata["cross_references"] = cross_refs
                        modified = True
                    except ValueError:
                        pass  # node_id was not in cross_refs

                    if modified:
                        updates.append((json.dumps(metadata), entry_id))
                        updated_count += 1

                # Batch update all modified entries in single transaction
                if updates:
                    cursor.executemany(
                        """
                        UPDATE continuum_memory
                        SET metadata = ?
                        WHERE id = ?
                        """,
                        updates,
                    )
                    conn.commit()

            if updated_count > 0:
                logger.debug(f"Invalidated {updated_count} references to KM node {node_id}")

            return updated_count > 0

        except sqlite3.Error as e:
            logger.warning(f"Failed to invalidate KM reference {node_id} (database): {e}")
            return False
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to invalidate KM reference {node_id} (data): {e}")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Unexpected error invalidating KM reference {node_id}: {e}")
            return False


# Async wrapper functions to be mixed into ContinuumMemory


@with_retry(_MEMORY_RETRY_CONFIG)
async def add_async(
    cms: "ContinuumMemory",
    id: str,
    content: str,
    tier: MemoryTier = MemoryTier.SLOW,
    importance: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> ContinuumMemoryEntry:
    """Async wrapper for add() - offloads blocking I/O to executor."""
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: cms.add(
            id=id,
            content=content,
            tier=tier,
            importance=importance,
            metadata=metadata,
        ),
    )


@with_retry(_MEMORY_RETRY_CONFIG)
async def store(
    cms: "ContinuumMemory",
    key: str,
    content: str,
    tier: str | MemoryTier = MemoryTier.SLOW,
    importance: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> ContinuumMemoryEntry:
    """Async wrapper for add() - offloads blocking I/O to executor."""
    normalized_tier: MemoryTier = MemoryTier(tier) if isinstance(tier, str) else tier
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: cms.add(
            id=key,
            content=content,
            tier=normalized_tier,
            importance=importance,
            metadata=metadata,
        ),
    )


@with_retry(_MEMORY_RETRY_CONFIG)
async def get_async(cms: "ContinuumMemory", id: str) -> ContinuumMemoryEntry | None:
    """Async wrapper for get() - offloads blocking I/O to executor."""
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, cms.get, id)


@with_retry(_MEMORY_RETRY_CONFIG)
async def retrieve_async(
    cms: "ContinuumMemory",
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
        lambda: cms.retrieve(
            query=query,
            tiers=tiers,
            limit=limit,
            min_importance=min_importance,
            include_glacial=include_glacial,
            tier=tier,
        ),
    )


__all__ = [
    "ContinuumRetrievalMixin",
    "add_async",
    "store",
    "get_async",
    "retrieve_async",
]
