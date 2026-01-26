"""
PostgreSQL implementation of Continuum Memory System.

Provides async PostgreSQL-backed storage for ContinuumMemory with
connection pooling for production deployments requiring horizontal
scaling and concurrent writes.

Usage:
    from aragora.memory.postgres_continuum import get_postgres_continuum_memory

    # Get the PostgreSQL continuum memory instance
    cms = await get_postgres_continuum_memory()

    # Store a memory (async)
    await cms.add("pattern_123", "Error handling pattern", tier="slow")

    # Retrieve memories
    entries = await cms.retrieve(query="error", limit=10)

    # Update outcome
    await cms.update_outcome("pattern_123", success=True)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aragora.storage.postgres_store import PostgresStore, ASYNCPG_AVAILABLE
from aragora.memory.tier_manager import (
    MemoryTier,
    TierManager,
    get_tier_manager,
    DEFAULT_TIER_CONFIGS,
)

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


# PostgreSQL schema version for continuum memory
POSTGRES_CONTINUUM_SCHEMA_VERSION = 1

# PostgreSQL-optimized schema
POSTGRES_CONTINUUM_SCHEMA = """
    -- Main continuum memory table
    CREATE TABLE IF NOT EXISTS continuum_memory (
        id TEXT PRIMARY KEY,
        tier TEXT NOT NULL DEFAULT 'slow',
        content TEXT NOT NULL,
        importance REAL DEFAULT 0.5,
        surprise_score REAL DEFAULT 0.0,
        consolidation_score REAL DEFAULT 0.0,
        update_count INTEGER DEFAULT 0,
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        semantic_centroid BYTEA,
        last_promotion_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        metadata JSONB DEFAULT '{}',
        expires_at TIMESTAMPTZ,
        red_line BOOLEAN DEFAULT FALSE,
        red_line_reason TEXT DEFAULT ''
    );

    -- Indexes for efficient tier-based retrieval
    CREATE INDEX IF NOT EXISTS idx_continuum_tier ON continuum_memory(tier);
    CREATE INDEX IF NOT EXISTS idx_continuum_surprise ON continuum_memory(surprise_score DESC);
    CREATE INDEX IF NOT EXISTS idx_continuum_importance ON continuum_memory(importance DESC);
    -- Composite index for cleanup_expired_memories() performance
    CREATE INDEX IF NOT EXISTS idx_continuum_tier_updated ON continuum_memory(tier, updated_at);
    -- Composite index for tier-filtered retrieval by importance
    CREATE INDEX IF NOT EXISTS idx_continuum_tier_importance ON continuum_memory(tier, importance DESC);
    -- Composite index for promotion queries (tier + surprise_score)
    CREATE INDEX IF NOT EXISTS idx_continuum_tier_surprise ON continuum_memory(tier, surprise_score DESC);
    -- Index for TTL-based cleanup queries
    CREATE INDEX IF NOT EXISTS idx_continuum_expires ON continuum_memory(expires_at);
    -- Index for red line protected entries
    CREATE INDEX IF NOT EXISTS idx_continuum_red_line ON continuum_memory(red_line);

    -- Meta-learning state table for hyperparameter tracking
    CREATE TABLE IF NOT EXISTS meta_learning_state (
        id SERIAL PRIMARY KEY,
        hyperparams JSONB NOT NULL,
        learning_efficiency REAL,
        pattern_retention_rate REAL,
        forgetting_rate REAL,
        cycles_evaluated INTEGER DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Tier transition history for analysis
    CREATE TABLE IF NOT EXISTS tier_transitions (
        id SERIAL PRIMARY KEY,
        memory_id TEXT NOT NULL REFERENCES continuum_memory(id) ON DELETE CASCADE,
        from_tier TEXT NOT NULL,
        to_tier TEXT NOT NULL,
        reason TEXT,
        surprise_score REAL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Archive table for deleted memories
    CREATE TABLE IF NOT EXISTS continuum_memory_archive (
        id TEXT PRIMARY KEY,
        tier TEXT NOT NULL,
        content TEXT NOT NULL,
        importance REAL,
        surprise_score REAL,
        consolidation_score REAL,
        update_count INTEGER,
        success_count INTEGER,
        failure_count INTEGER,
        semantic_centroid BYTEA,
        created_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ,
        archived_at TIMESTAMPTZ DEFAULT NOW(),
        archive_reason TEXT,
        metadata JSONB
    );

    -- Indexes for archive queries
    CREATE INDEX IF NOT EXISTS idx_archive_tier ON continuum_memory_archive(tier);
    CREATE INDEX IF NOT EXISTS idx_archive_archived_at ON continuum_memory_archive(archived_at);
"""


class ContinuumMemoryEntryDict(Dict[str, Any]):
    """Dictionary representation of a continuum memory entry with helper properties."""

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.get("success_count", 0) + self.get("failure_count", 0)
        return self.get("success_count", 0) / total if total > 0 else 0.5

    @property
    def stability_score(self) -> float:
        """Inverse of surprise - how predictable this pattern is."""
        return 1.0 - self.get("surprise_score", 0.0)

    @property
    def cross_references(self) -> List[str]:
        """Get list of cross-reference IDs linked to this entry."""
        metadata = self.get("metadata", {})
        if isinstance(metadata, str):
            import json

            metadata = json.loads(metadata)
        return metadata.get("cross_references", [])

    @property
    def knowledge_mound_id(self) -> str:
        """Get the Knowledge Mound ID for this entry."""
        return f"cm_{self.get('id', '')}"


class PostgresContinuumMemory(PostgresStore):
    """
    PostgreSQL implementation of Continuum Memory System.

    Provides async operations for multi-tier memory with:
    - Connection pooling for horizontal scaling
    - JSONB for efficient metadata queries
    - TIMESTAMPTZ for proper timestamp handling
    - Batch operations for consolidation

    Usage:
        pool = await get_postgres_pool()
        cms = PostgresContinuumMemory(pool)
        await cms.initialize()

        # Add memory
        entry = await cms.add("id", "content", tier=MemoryTier.FAST)

        # Retrieve memories
        entries = await cms.retrieve(query="error", limit=10)
    """

    SCHEMA_NAME = "continuum_memory"
    SCHEMA_VERSION = POSTGRES_CONTINUUM_SCHEMA_VERSION
    INITIAL_SCHEMA = POSTGRES_CONTINUUM_SCHEMA

    def __init__(
        self,
        pool: "Pool",
        tier_manager: Optional[TierManager] = None,
    ):
        """
        Initialize PostgreSQL continuum memory.

        Args:
            pool: asyncpg connection pool
            tier_manager: Optional tier manager for promotion/demotion decisions
        """
        super().__init__(pool)
        self._tier_manager = tier_manager or get_tier_manager()

        # Hyperparameters (can be modified by MetaLearner)
        self.hyperparams: Dict[str, Any] = {
            "surprise_weight_success": 0.3,
            "surprise_weight_semantic": 0.3,
            "surprise_weight_temporal": 0.2,
            "surprise_weight_agent": 0.2,
            "consolidation_threshold": 100.0,
            "promotion_cooldown_hours": 24.0,
            "max_entries_per_tier": {
                "fast": 1000,
                "medium": 5000,
                "slow": 10000,
                "glacial": 50000,
            },
            "retention_multiplier": 2.0,
        }

    @property
    def tier_manager(self) -> TierManager:
        """Get the tier manager instance."""
        return self._tier_manager

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    async def add(
        self,
        memory_id: str,
        content: str,
        tier: MemoryTier | str = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContinuumMemoryEntryDict:
        """
        Add a new memory entry to the continuum.

        Args:
            memory_id: Unique identifier for the memory
            content: The memory content
            tier: Initial memory tier
            importance: 0-1 importance score
            metadata: Optional additional data

        Returns:
            The created memory entry as a dict
        """
        tier_value = tier.value if isinstance(tier, MemoryTier) else tier
        now = datetime.now()
        meta_json = json.dumps(metadata or {})

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO continuum_memory
                (id, tier, content, importance, surprise_score, consolidation_score,
                 update_count, success_count, failure_count, created_at, updated_at, metadata)
                VALUES ($1, $2, $3, $4, 0.0, 0.0, 1, 0, 0, $5, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    tier = EXCLUDED.tier,
                    importance = EXCLUDED.importance,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                """,
                memory_id,
                tier_value,
                content,
                importance,
                now,
                meta_json,
            )

        return ContinuumMemoryEntryDict(
            {
                "id": memory_id,
                "tier": tier_value,
                "content": content,
                "importance": importance,
                "surprise_score": 0.0,
                "consolidation_score": 0.0,
                "update_count": 1,
                "success_count": 0,
                "failure_count": 0,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "metadata": metadata or {},
                "red_line": False,
                "red_line_reason": "",
            }
        )

    async def store(
        self,
        key: str,
        content: str,
        tier: str | MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContinuumMemoryEntryDict:
        """Alias for add() for interface compatibility."""
        return await self.add(key, content, tier, importance, metadata)

    async def get(self, memory_id: str) -> Optional[ContinuumMemoryEntryDict]:
        """
        Get a memory entry by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory entry dict or None if not found
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at,
                       metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, '')
                FROM continuum_memory
                WHERE id = $1
                """,
                memory_id,
            )

        if not row:
            return None

        return self._row_to_entry(row)

    async def get_entry(self, memory_id: str) -> Optional[ContinuumMemoryEntryDict]:
        """Alias for get() for interface compatibility."""
        return await self.get(memory_id)

    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        surprise_score: Optional[float] = None,
        consolidation_score: Optional[float] = None,
    ) -> bool:
        """
        Update specific fields of a memory entry.

        Args:
            memory_id: The ID of the memory entry to update
            content: New content (optional)
            importance: New importance score (optional)
            metadata: New metadata dict (optional)
            surprise_score: New surprise score (optional)
            consolidation_score: New consolidation score (optional)

        Returns:
            True if the entry was updated, False if not found
        """
        updates = []
        params = []
        param_idx = 1

        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1
        if importance is not None:
            updates.append(f"importance = ${param_idx}")
            params.append(importance)  # type: ignore[arg-type]
            param_idx += 1
        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1
        if surprise_score is not None:
            updates.append(f"surprise_score = ${param_idx}")
            params.append(surprise_score)  # type: ignore[arg-type]
            param_idx += 1
        if consolidation_score is not None:
            updates.append(f"consolidation_score = ${param_idx}")
            params.append(consolidation_score)  # type: ignore[arg-type]
            param_idx += 1

        if not updates:
            return False

        updates.append(f"updated_at = ${param_idx}")
        params.append(datetime.now())  # type: ignore[arg-type]
        param_idx += 1

        params.append(memory_id)

        async with self.connection() as conn:
            result = await conn.execute(
                f"""
                UPDATE continuum_memory
                SET {", ".join(updates)}
                WHERE id = ${param_idx}
                """,
                *params,
            )
            return result == "UPDATE 1"

    async def delete(
        self,
        memory_id: str,
        archive: bool = True,
        reason: str = "user_deleted",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a specific memory entry by ID.

        Args:
            memory_id: The ID of the memory to delete
            archive: If True, archive before deletion
            reason: Reason for deletion (stored in archive)
            force: If True, delete even if entry is red-lined

        Returns:
            Dict with result
        """
        async with self.transaction() as conn:
            # Check if exists and if red-lined
            row = await conn.fetchrow(
                "SELECT red_line FROM continuum_memory WHERE id = $1",
                memory_id,
            )

            if not row:
                return {"deleted": False, "archived": False, "id": memory_id, "blocked": False}

            if row["red_line"] and not force:
                return {"deleted": False, "archived": False, "id": memory_id, "blocked": True}

            # Archive if requested
            archived = False
            if archive:
                await conn.execute(
                    """
                    INSERT INTO continuum_memory_archive
                    (id, tier, content, importance, surprise_score, consolidation_score,
                     update_count, success_count, failure_count, semantic_centroid,
                     created_at, updated_at, archive_reason, metadata)
                    SELECT id, tier, content, importance, surprise_score, consolidation_score,
                           update_count, success_count, failure_count, semantic_centroid,
                           created_at, updated_at, $2, metadata
                    FROM continuum_memory
                    WHERE id = $1
                    """,
                    memory_id,
                    reason,
                )
                archived = True

            # Delete
            await conn.execute("DELETE FROM continuum_memory WHERE id = $1", memory_id)

        return {"deleted": True, "archived": archived, "id": memory_id, "blocked": False}

    # =========================================================================
    # Retrieval Operations
    # =========================================================================

    async def retrieve(
        self,
        query: Optional[str] = None,
        tiers: Optional[List[MemoryTier]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: Optional[str | MemoryTier] = None,
    ) -> List[ContinuumMemoryEntryDict]:
        """
        Retrieve memories ranked by importance, surprise, and recency.

        Args:
            query: Optional query for keyword filtering
            tiers: Filter to specific tiers (default: all)
            limit: Maximum entries to return
            min_importance: Minimum importance threshold
            include_glacial: Whether to include glacial tier
            tier: Single tier filter (convenience param)

        Returns:
            List of memory entries sorted by retrieval score
        """
        # Handle single tier parameter
        if tier is not None:
            target_tier = tier.value if isinstance(tier, MemoryTier) else tier
            tiers = [MemoryTier(target_tier)]

        # Build tier filter
        if tiers is None:
            tiers = list(MemoryTier)
        if not include_glacial:
            tiers = [t for t in tiers if t != MemoryTier.GLACIAL]

        tier_values = [t.value for t in tiers]

        # Build query
        base_query = """
            SELECT id, tier, content, importance, surprise_score, consolidation_score,
                   update_count, success_count, failure_count, created_at, updated_at,
                   metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, ''),
                   (importance * (1 + surprise_score) *
                    (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400 *
                     CASE tier
                       WHEN 'fast' THEN 24
                       WHEN 'medium' THEN 1
                       WHEN 'slow' THEN 0.14
                       WHEN 'glacial' THEN 0.03
                     END))) as score
            FROM continuum_memory
            WHERE tier = ANY($1)
              AND importance >= $2
        """

        params: List[Any] = [tier_values, min_importance]
        param_idx = 3

        # Add keyword filter if query provided
        if query:
            keywords = [kw.strip().lower() for kw in query.split()[:50] if kw.strip()]
            if keywords:
                keyword_conditions = [
                    f"LOWER(content) LIKE ${param_idx + i}" for i in range(len(keywords))
                ]
                base_query += f" AND ({' OR '.join(keyword_conditions)})"
                params.extend([f"%{kw}%" for kw in keywords])
                param_idx += len(keywords)

        base_query += f" ORDER BY score DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.connection() as conn:
            rows = await conn.fetch(base_query, *params)

        return [self._row_to_entry(row) for row in rows]

    async def get_by_tier(
        self,
        tier: MemoryTier | str,
        limit: int = 100,
        min_importance: float = 0.0,
    ) -> List[ContinuumMemoryEntryDict]:
        """
        Get memories for a specific tier.

        Args:
            tier: The tier to query
            limit: Maximum entries to return
            min_importance: Minimum importance threshold

        Returns:
            List of memory entries
        """
        tier_value = tier.value if isinstance(tier, MemoryTier) else tier

        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at,
                       metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, '')
                FROM continuum_memory
                WHERE tier = $1 AND importance >= $2
                ORDER BY importance DESC
                LIMIT $3
                """,
                tier_value,
                min_importance,
                limit,
            )

        return [self._row_to_entry(row) for row in rows]

    # =========================================================================
    # Surprise & Outcome Updates
    # =========================================================================

    async def update_outcome(
        self,
        memory_id: str,
        success: bool,
        agent_prediction_error: Optional[float] = None,
    ) -> float:
        """
        Update memory after observing outcome.

        Implements surprise-based learning where the surprise score is
        updated based on how unexpected the outcome was.

        Args:
            memory_id: Memory ID
            success: Whether the pattern led to success
            agent_prediction_error: Optional agent's prediction error

        Returns:
            Updated surprise score
        """
        async with self.transaction() as conn:
            # Get current state
            row = await conn.fetchrow(
                """
                SELECT success_count, failure_count, surprise_score, tier
                FROM continuum_memory
                WHERE id = $1
                FOR UPDATE
                """,
                memory_id,
            )

            if not row:
                return 0.0

            success_count = row["success_count"]
            failure_count = row["failure_count"]
            old_surprise = row["surprise_score"] or 0.0
            total = success_count + failure_count

            # Calculate expected success rate (base rate)
            expected_rate = success_count / total if total > 0 else 0.5

            # Actual outcome
            actual = 1.0 if success else 0.0

            # Success rate surprise component
            success_surprise = abs(actual - expected_rate)

            # Combine surprise signals
            new_surprise = self.hyperparams[
                "surprise_weight_success"
            ] * success_surprise + self.hyperparams["surprise_weight_agent"] * (
                agent_prediction_error or 0.0
            )

            # Exponential moving average for surprise
            alpha = 0.3
            updated_surprise = old_surprise * (1 - alpha) + new_surprise * alpha

            # Update consolidation score
            update_count = total + 1
            consolidation = min(
                1.0,
                math.log(1 + update_count) / math.log(self.hyperparams["consolidation_threshold"]),
            )

            # Update database
            if success:
                await conn.execute(
                    """
                    UPDATE continuum_memory
                    SET success_count = success_count + 1,
                        update_count = update_count + 1,
                        surprise_score = $1,
                        consolidation_score = $2,
                        updated_at = $3
                    WHERE id = $4
                    """,
                    updated_surprise,
                    consolidation,
                    datetime.now(),
                    memory_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE continuum_memory
                    SET failure_count = failure_count + 1,
                        update_count = update_count + 1,
                        surprise_score = $1,
                        consolidation_score = $2,
                        updated_at = $3
                    WHERE id = $4
                    """,
                    updated_surprise,
                    consolidation,
                    datetime.now(),
                    memory_id,
                )

        return updated_surprise

    # =========================================================================
    # Tier Promotion/Demotion
    # =========================================================================

    async def promote(self, memory_id: str) -> Optional[MemoryTier]:
        """
        Promote a memory to a faster tier.

        Returns the new tier if promoted, None otherwise.
        """
        async with self.transaction() as conn:
            row = await conn.fetchrow(
                """
                SELECT tier, surprise_score, last_promotion_at
                FROM continuum_memory
                WHERE id = $1
                FOR UPDATE
                """,
                memory_id,
            )

            if not row:
                return None

            current_tier = MemoryTier(row["tier"])
            surprise_score = row["surprise_score"]
            last_promotion = row["last_promotion_at"]

            # Check if promotion is allowed
            if not self._tier_manager.should_promote(current_tier, surprise_score, last_promotion):
                return None

            # Get next tier
            new_tier = self._tier_manager.get_next_tier(current_tier, "faster")
            if new_tier is None:
                return None

            now = datetime.now()

            # Update tier
            await conn.execute(
                """
                UPDATE continuum_memory
                SET tier = $1, last_promotion_at = $2, updated_at = $2
                WHERE id = $3
                """,
                new_tier.value,
                now,
                memory_id,
            )

            # Record transition
            await conn.execute(
                """
                INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
                VALUES ($1, $2, $3, 'high_surprise', $4)
                """,
                memory_id,
                current_tier.value,
                new_tier.value,
                surprise_score,
            )

        # Record metrics
        self._tier_manager.record_promotion(current_tier, new_tier)
        return new_tier

    async def demote(self, memory_id: str) -> Optional[MemoryTier]:
        """
        Demote a memory to a slower tier.

        Returns the new tier if demoted, None otherwise.
        """
        async with self.transaction() as conn:
            row = await conn.fetchrow(
                """
                SELECT tier, surprise_score, update_count
                FROM continuum_memory
                WHERE id = $1
                FOR UPDATE
                """,
                memory_id,
            )

            if not row:
                return None

            current_tier = MemoryTier(row["tier"])
            surprise_score = row["surprise_score"]
            update_count = row["update_count"]

            # Check if demotion is allowed
            if not self._tier_manager.should_demote(current_tier, surprise_score, update_count):
                return None

            # Get next tier
            new_tier = self._tier_manager.get_next_tier(current_tier, "slower")
            if new_tier is None:
                return None

            now = datetime.now()

            # Update tier
            await conn.execute(
                """
                UPDATE continuum_memory
                SET tier = $1, updated_at = $2
                WHERE id = $3
                """,
                new_tier.value,
                now,
                memory_id,
            )

            # Record transition
            await conn.execute(
                """
                INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
                VALUES ($1, $2, $3, 'high_stability', $4)
                """,
                memory_id,
                current_tier.value,
                new_tier.value,
                surprise_score,
            )

        # Record metrics
        self._tier_manager.record_demotion(current_tier, new_tier)
        return new_tier

    async def promote_entry(self, memory_id: str, new_tier: MemoryTier) -> bool:
        """Promote an entry to a specific tier."""
        async with self.connection() as conn:
            result = await conn.execute(
                """
                UPDATE continuum_memory
                SET tier = $1, updated_at = $2
                WHERE id = $3
                """,
                new_tier.value,
                datetime.now(),
                memory_id,
            )
            return result == "UPDATE 1"

    async def demote_entry(self, memory_id: str, new_tier: MemoryTier) -> bool:
        """Demote an entry to a specific tier."""
        return await self.promote_entry(memory_id, new_tier)

    # =========================================================================
    # Red Line Protection
    # =========================================================================

    async def mark_red_line(
        self,
        memory_id: str,
        reason: str,
        promote_to_glacial: bool = True,
    ) -> bool:
        """
        Mark a memory entry as a red line - cannot be forgotten or overwritten.

        Args:
            memory_id: The ID of the memory to protect
            reason: Why this entry is critical (for auditing)
            promote_to_glacial: If True, promote to glacial tier

        Returns:
            True if the entry was marked, False if entry not found
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT tier FROM continuum_memory WHERE id = $1",
                memory_id,
            )

            if not row:
                return False

            current_tier = row["tier"]
            now = datetime.now()

            if promote_to_glacial and current_tier != MemoryTier.GLACIAL.value:
                await conn.execute(
                    """
                    UPDATE continuum_memory
                    SET red_line = TRUE, red_line_reason = $1, tier = $2,
                        importance = 1.0, updated_at = $3
                    WHERE id = $4
                    """,
                    reason,
                    MemoryTier.GLACIAL.value,
                    now,
                    memory_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE continuum_memory
                    SET red_line = TRUE, red_line_reason = $1, importance = 1.0, updated_at = $2
                    WHERE id = $3
                    """,
                    reason,
                    now,
                    memory_id,
                )

        return True

    async def get_red_line_memories(self) -> List[ContinuumMemoryEntryDict]:
        """Get all red-lined memory entries."""
        async with self.connection() as conn:
            rows = await conn.fetch("""
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at,
                       metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, '')
                FROM continuum_memory
                WHERE red_line = TRUE
                ORDER BY created_at ASC
                """)

        return [self._row_to_entry(row) for row in rows]

    # =========================================================================
    # Statistics & Consolidation
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory system."""
        async with self.connection() as conn:
            # Get counts by tier
            tier_counts = await conn.fetch("""
                SELECT tier, COUNT(*) as count, AVG(importance) as avg_importance,
                       AVG(surprise_score) as avg_surprise
                FROM continuum_memory
                GROUP BY tier
                """)

            total = await conn.fetchrow("SELECT COUNT(*) as count FROM continuum_memory")

        tier_stats = {}
        for row in tier_counts:
            tier_stats[row["tier"]] = {
                "count": row["count"],
                "avg_importance": float(row["avg_importance"]) if row["avg_importance"] else 0.0,
                "avg_surprise": float(row["avg_surprise"]) if row["avg_surprise"] else 0.0,
            }

        return {
            "total_entries": total["count"] if total else 0,
            "by_tier": tier_stats,
            "hyperparams": self.hyperparams,
        }

    async def count(self, tier: Optional[MemoryTier] = None) -> int:  # type: ignore[override]
        """Count memory entries, optionally filtered by tier."""
        async with self.connection() as conn:
            if tier:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM continuum_memory WHERE tier = $1",
                    tier.value,
                )
            else:
                row = await conn.fetchrow("SELECT COUNT(*) as count FROM continuum_memory")

        return row["count"] if row else 0

    async def consolidate(self) -> Dict[str, int]:
        """
        Run tier consolidation: promote/demote memories based on surprise.

        Returns counts of promotions and demotions by tier.
        """
        results: Dict[str, int] = {"promoted": 0, "demoted": 0}

        async with self.connection() as conn:
            # Find candidates for promotion (high surprise)
            promotion_candidates = await conn.fetch("""
                SELECT id, tier, surprise_score
                FROM continuum_memory
                WHERE tier != 'fast'
                  AND surprise_score > 0.7
                ORDER BY surprise_score DESC
                LIMIT 100
                """)

            # Find candidates for demotion (high stability, many updates)
            demotion_candidates = await conn.fetch("""
                SELECT id, tier, surprise_score, update_count
                FROM continuum_memory
                WHERE tier != 'glacial'
                  AND surprise_score < 0.3
                  AND update_count > 10
                ORDER BY surprise_score ASC
                LIMIT 100
                """)

        # Process promotions concurrently (batch of 20 at a time to avoid connection exhaustion)
        promotion_ids = [row["id"] for row in promotion_candidates]
        for i in range(0, len(promotion_ids), 20):
            batch = promotion_ids[i : i + 20]
            promotion_results = await asyncio.gather(
                *[self.promote(memory_id) for memory_id in batch],
                return_exceptions=True,
            )
            results["promoted"] += sum(
                1 for r in promotion_results if r and not isinstance(r, Exception)
            )

        # Process demotions concurrently (batch of 20 at a time)
        demotion_ids = [row["id"] for row in demotion_candidates]
        for i in range(0, len(demotion_ids), 20):
            batch = demotion_ids[i : i + 20]
            demotion_results = await asyncio.gather(
                *[self.demote(memory_id) for memory_id in batch],
                return_exceptions=True,
            )
            results["demoted"] += sum(
                1 for r in demotion_results if r and not isinstance(r, Exception)
            )

        logger.info(f"Consolidation complete: {results}")
        return results

    async def cleanup_expired_memories(
        self,
        tier: Optional[MemoryTier] = None,
        archive: bool = True,
        max_age_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Remove or archive expired memories based on tier retention policies.

        Args:
            tier: Specific tier to cleanup (None = all tiers)
            archive: If True, move to archive table
            max_age_hours: Override default retention

        Returns:
            Dict with counts
        """
        results: Dict[str, Any] = {"archived": 0, "deleted": 0, "by_tier": {}}

        tiers_to_process = [tier] if tier else list(MemoryTier)

        for t in tiers_to_process:
            tier_config = DEFAULT_TIER_CONFIGS[t]
            age_hours = max_age_hours or (
                tier_config.half_life_hours * self.hyperparams["retention_multiplier"]
            )
            cutoff = datetime.now().timestamp() - (age_hours * 3600)

            async with self.transaction() as conn:
                # Get expired entries (excluding red-lined)
                expired = await conn.fetch(
                    """
                    SELECT id FROM continuum_memory
                    WHERE tier = $1
                      AND red_line = FALSE
                      AND EXTRACT(EPOCH FROM updated_at) < $2
                    LIMIT 1000
                    """,
                    t.value,
                    cutoff,
                )

                expired_ids = [row["id"] for row in expired]
                tier_count = len(expired_ids)

                if tier_count > 0 and archive:
                    # Batch archive all expired entries in single query
                    await conn.execute(
                        """
                        INSERT INTO continuum_memory_archive
                        (id, tier, content, importance, surprise_score, consolidation_score,
                         update_count, success_count, failure_count, semantic_centroid,
                         created_at, updated_at, archive_reason, metadata)
                        SELECT id, tier, content, importance, surprise_score, consolidation_score,
                               update_count, success_count, failure_count, semantic_centroid,
                               created_at, updated_at, 'expired', metadata
                        FROM continuum_memory
                        WHERE id = ANY($1)
                        """,
                        expired_ids,
                    )
                    results["archived"] += tier_count

                if tier_count > 0:
                    # Batch delete all expired entries in single query
                    await conn.execute(
                        "DELETE FROM continuum_memory WHERE id = ANY($1)",
                        expired_ids,
                    )
                    if not archive:
                        results["deleted"] += tier_count

                results["by_tier"][t.value] = tier_count

        return results

    async def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived memories."""
        async with self.connection() as conn:
            stats = await conn.fetchrow("""
                SELECT COUNT(*) as total,
                       MIN(archived_at) as oldest,
                       MAX(archived_at) as newest
                FROM continuum_memory_archive
                """)

            by_tier = await conn.fetch("""
                SELECT tier, COUNT(*) as count
                FROM continuum_memory_archive
                GROUP BY tier
                """)

        return {
            "total": stats["total"] if stats else 0,
            "oldest": stats["oldest"].isoformat() if stats and stats["oldest"] else None,
            "newest": stats["newest"].isoformat() if stats and stats["newest"] else None,
            "by_tier": {row["tier"]: row["count"] for row in by_tier},
        }

    # =========================================================================
    # Glacial Tier Access
    # =========================================================================

    async def get_glacial_insights(
        self,
        limit: int = 10,
        min_importance: float = 0.7,
    ) -> List[ContinuumMemoryEntryDict]:
        """
        Get high-importance glacial tier memories for cross-session insights.

        Args:
            limit: Maximum entries to return
            min_importance: Minimum importance threshold

        Returns:
            List of glacial tier memory entries
        """
        return await self.get_by_tier(MemoryTier.GLACIAL, limit, min_importance)

    async def get_cross_session_patterns(
        self,
        pattern_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[ContinuumMemoryEntryDict]:
        """
        Get cross-session patterns from glacial tier.

        Args:
            pattern_type: Filter by pattern type in metadata
            limit: Maximum entries to return

        Returns:
            List of pattern entries
        """
        async with self.connection() as conn:
            if pattern_type:
                rows = await conn.fetch(
                    """
                    SELECT id, tier, content, importance, surprise_score, consolidation_score,
                           update_count, success_count, failure_count, created_at, updated_at,
                           metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, '')
                    FROM continuum_memory
                    WHERE tier = 'glacial'
                      AND metadata->>'pattern_type' = $1
                    ORDER BY importance DESC
                    LIMIT $2
                    """,
                    pattern_type,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, tier, content, importance, surprise_score, consolidation_score,
                           update_count, success_count, failure_count, created_at, updated_at,
                           metadata, COALESCE(red_line, FALSE), COALESCE(red_line_reason, '')
                    FROM continuum_memory
                    WHERE tier = 'glacial'
                    ORDER BY importance DESC
                    LIMIT $1
                    """,
                    limit,
                )

        return [self._row_to_entry(row) for row in rows]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_entry(self, row: Any) -> ContinuumMemoryEntryDict:
        """Convert a database row to a ContinuumMemoryEntryDict."""
        metadata = row[11]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        created_at = row[9]
        updated_at = row[10]

        return ContinuumMemoryEntryDict(
            {
                "id": row[0],
                "tier": row[1],
                "content": row[2],
                "importance": float(row[3]) if row[3] else 0.5,
                "surprise_score": float(row[4]) if row[4] else 0.0,
                "consolidation_score": float(row[5]) if row[5] else 0.0,
                "update_count": row[6] or 0,
                "success_count": row[7] or 0,
                "failure_count": row[8] or 0,
                "created_at": (
                    created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
                ),
                "updated_at": (
                    updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at)
                ),
                "metadata": metadata or {},
                "red_line": bool(row[12]),
                "red_line_reason": row[13] or "",
            }
        )


# =========================================================================
# Factory Function
# =========================================================================

_postgres_continuum_memory: Optional[PostgresContinuumMemory] = None


async def get_postgres_continuum_memory(
    pool: Optional["Pool"] = None,
    tier_manager: Optional[TierManager] = None,
) -> PostgresContinuumMemory:
    """
    Get or create the PostgreSQL continuum memory instance.

    Creates a new instance if one doesn't exist. The instance is cached
    globally for reuse across the application.

    Args:
        pool: Optional asyncpg pool. If not provided, gets from settings.
        tier_manager: Optional tier manager instance

    Returns:
        PostgresContinuumMemory instance

    Raises:
        RuntimeError: If asyncpg is not installed or PostgreSQL not configured
    """
    global _postgres_continuum_memory

    if not ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install aragora[postgres] or pip install asyncpg"
        )

    if _postgres_continuum_memory is not None:
        return _postgres_continuum_memory

    # Get pool from settings if not provided
    if pool is None:
        from aragora.storage.postgres_store import get_postgres_pool_from_settings

        pool = await get_postgres_pool_from_settings()

    _postgres_continuum_memory = PostgresContinuumMemory(pool, tier_manager)
    await _postgres_continuum_memory.initialize()

    return _postgres_continuum_memory
