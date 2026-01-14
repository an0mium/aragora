"""
Continuum Memory System (CMS) for Nested Learning.

Implements Google Research's Nested Learning paradigm with multi-timescale
memory updates. Memory is treated as a spectrum where different modules
update at different frequencies, enabling continual learning without
catastrophic forgetting.

Based on: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

Key concepts:
- Fast tier: Updates on every event (immediate patterns, 1h half-life)
- Medium tier: Updates per debate round (tactical learning, 24h half-life)
- Slow tier: Updates per nomic cycle (strategic learning, 7d half-life)
- Glacial tier: Updates monthly (foundational knowledge, 30d half-life)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

from aragora.config import DB_MEMORY_PATH
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierConfig,  # noqa: F401 - re-exported for backwards compatibility
    TierManager,
    get_tier_manager,
)
from aragora.storage.base_store import SQLiteStore
from aragora.storage.schema import SchemaManager
from aragora.utils.json_helpers import safe_json_loads

# Schema version for ContinuumMemory
CONTINUUM_SCHEMA_VERSION = 2

# Default retention multiplier (entries older than multiplier * half_life are eligible for cleanup)
DEFAULT_RETENTION_MULTIPLIER = 2.0

# Re-export for backwards compatibility (use DEFAULT_TIER_CONFIGS from tier_manager)
TIER_CONFIGS = DEFAULT_TIER_CONFIGS


class MaxEntriesPerTier(TypedDict):
    """Type definition for max entries per tier configuration."""

    fast: int
    medium: int
    slow: int
    glacial: int


class ContinuumHyperparams(TypedDict):
    """
    Type definition for ContinuumMemory hyperparameters.

    These parameters control memory consolidation, tier transitions,
    and retention policies. They can be modified by MetaLearner.
    """

    surprise_weight_success: float  # Weight for success rate surprise
    surprise_weight_semantic: float  # Weight for semantic novelty
    surprise_weight_temporal: float  # Weight for timing surprise
    surprise_weight_agent: float  # Weight for agent prediction error
    consolidation_threshold: float  # Updates to reach full consolidation
    promotion_cooldown_hours: float  # Minimum time between promotions
    max_entries_per_tier: MaxEntriesPerTier  # Max entries per tier
    retention_multiplier: float  # multiplier * half_life for cleanup


@dataclass
class ContinuumMemoryEntry:
    """A single entry in the continuum memory system."""

    id: str
    tier: MemoryTier
    content: str
    importance: float
    surprise_score: float
    consolidation_score: float  # 0-1, how consolidated/stable the memory is
    update_count: int
    success_count: int
    failure_count: int
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def stability_score(self) -> float:
        """Inverse of surprise - how predictable this pattern is."""
        return 1.0 - self.surprise_score

    def should_promote(self) -> bool:
        """Check if this entry should be promoted to a faster tier."""
        if self.tier == MemoryTier.FAST:
            return False  # Already at fastest
        config = TIER_CONFIGS[self.tier]
        return self.surprise_score > config.promotion_threshold

    def should_demote(self) -> bool:
        """Check if this entry should be demoted to a slower tier."""
        if self.tier == MemoryTier.GLACIAL:
            return False  # Already at slowest
        config = TIER_CONFIGS[self.tier]
        return self.stability_score > config.demotion_threshold and self.update_count > 10


class AwaitableList(list):
    """List wrapper that can be awaited for async compatibility."""

    def __await__(self):
        async def _wrap():
            return self

        return _wrap().__await__()


class ContinuumMemory(SQLiteStore):
    """
    Continuum Memory System with multi-timescale updates.

    This implements Google's Nested Learning paradigm where memory is
    treated as a spectrum with modules updating at different frequency rates.

    Usage:
        cms = ContinuumMemory()

        # Add a fast-tier memory (immediate pattern)
        cms.add("error_pattern_123", "TypeError in agent response",
                tier=MemoryTier.FAST, importance=0.8)

        # Retrieve memories for a specific context
        relevant = cms.retrieve(query="type errors", tiers=[MemoryTier.FAST, MemoryTier.MEDIUM])

        # Update surprise score after observing outcome
        cms.update_surprise("error_pattern_123", observed_success=True)

        # Periodic tier consolidation
        cms.consolidate()
    """

    SCHEMA_NAME = "continuum_memory"
    SCHEMA_VERSION = CONTINUUM_SCHEMA_VERSION

    INITIAL_SCHEMA = """
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
            semantic_centroid BLOB,
            last_promotion_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            expires_at TEXT
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

        -- Meta-learning state table for hyperparameter tracking
        CREATE TABLE IF NOT EXISTS meta_learning_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hyperparams TEXT NOT NULL,
            learning_efficiency REAL,
            pattern_retention_rate REAL,
            forgetting_rate REAL,
            cycles_evaluated INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Tier transition history for analysis
        CREATE TABLE IF NOT EXISTS tier_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            from_tier TEXT NOT NULL,
            to_tier TEXT NOT NULL,
            reason TEXT,
            surprise_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES continuum_memory(id)
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
            semantic_centroid BLOB,
            created_at TEXT,
            updated_at TEXT,
            archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
            archive_reason TEXT,
            metadata TEXT
        );

        -- Indexes for archive queries
        CREATE INDEX IF NOT EXISTS idx_archive_tier ON continuum_memory_archive(tier);
        CREATE INDEX IF NOT EXISTS idx_archive_archived_at ON continuum_memory_archive(archived_at);
    """

    def __init__(
        self,
        db_path: str = DB_MEMORY_PATH,
        tier_manager: Optional[TierManager] = None,
        event_emitter: Any = None,
        storage_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ):
        resolved_path: str | Path = db_path
        base_path = storage_path or base_dir
        if base_path:
            base = Path(base_path)
            if base.suffix:
                resolved_path = base
            else:
                resolved_path = base / "continuum_memory.db"

        # Initialize SQLiteStore base class (handles schema creation)
        super().__init__(resolved_path)

        # Use provided TierManager or get the shared instance
        self._tier_manager = tier_manager or get_tier_manager()

        # Optional event emitter for WebSocket streaming
        self.event_emitter = event_emitter

        # Hyperparameters (can be modified by MetaLearner)
        self.hyperparams: ContinuumHyperparams = {
            "surprise_weight_success": 0.3,  # Weight for success rate surprise
            "surprise_weight_semantic": 0.3,  # Weight for semantic novelty
            "surprise_weight_temporal": 0.2,  # Weight for timing surprise
            "surprise_weight_agent": 0.2,  # Weight for agent prediction error
            "consolidation_threshold": 100.0,  # Updates to reach full consolidation
            "promotion_cooldown_hours": 24.0,  # Minimum time between promotions
            # Retention policy settings
            "max_entries_per_tier": {
                "fast": 1000,
                "medium": 5000,
                "slow": 10000,
                "glacial": 50000,
            },
            "retention_multiplier": DEFAULT_RETENTION_MULTIPLIER,  # multiplier * half_life for cleanup
        }

        # Sync tier manager settings with hyperparams
        self._tier_manager.promotion_cooldown_hours = self.hyperparams["promotion_cooldown_hours"]

        # Lock for atomic tier transitions (prevents TOCTOU race in promote/demote)
        self._tier_lock = threading.Lock()

    def register_migrations(self, manager: SchemaManager) -> None:
        """Register schema migrations for ContinuumMemory."""
        # Migration from v1 to v2 is handled by including archive table in INITIAL_SCHEMA
        # since v2 is the current version. Old v1 databases will get the archive table
        # automatically when the schema is applied.
        pass

    @property
    def tier_manager(self) -> TierManager:
        """Get the tier manager instance."""
        return self._tier_manager

    def get_tier_metrics(self) -> Dict[str, Any]:
        """Get tier transition metrics from the tier manager."""
        return self._tier_manager.get_metrics_dict()

    def add(
        self,
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """
        Add a new memory entry to the continuum.

        Args:
            id: Unique identifier for the memory
            content: The memory content
            tier: Initial memory tier
            importance: 0-1 importance score
            metadata: Optional additional data

        Returns:
            The created memory entry
        """
        now = datetime.now().isoformat()

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO continuum_memory
                (id, tier, content, importance, surprise_score, consolidation_score,
                 update_count, success_count, failure_count, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, 0.0, 0.0, 1, 0, 0, ?, ?, ?)
                """,
                (id, tier.value, content, importance, now, now, json.dumps(metadata or {})),
            )
            conn.commit()

        return ContinuumMemoryEntry(
            id=id,
            tier=tier,
            content=content,
            importance=importance,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    async def store(
        self,
        key: str,
        content: str,
        tier: str | MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """Async wrapper for add() for compatibility."""
        normalized_tier = MemoryTier(tier) if isinstance(tier, str) else tier
        return self.add(
            id=key,
            content=content,
            tier=normalized_tier,
            importance=importance,
            metadata=metadata,
        )

    def get(self, id: str) -> Optional[ContinuumMemoryEntry]:
        """Get a memory entry by ID."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at, metadata
                FROM continuum_memory
                WHERE id = ?
                """,
                (id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return ContinuumMemoryEntry(
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

    def retrieve(
        self,
        query: str | None = None,
        tiers: List[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: str | MemoryTier | None = None,
    ) -> List[ContinuumMemoryEntry]:
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
            target_tier = MemoryTier(tier) if isinstance(tier, str) else tier
            if query:
                entry = self.get(query)
                if entry and entry.tier == target_tier:
                    return AwaitableList([entry])
                return AwaitableList([])
            tiers = [target_tier]

        # Build tier filter
        if tiers is None:
            tiers = list(MemoryTier)
        if not include_glacial:
            tiers = [t for t in tiers if t != MemoryTier.GLACIAL]

        tier_values = [t.value for t in tiers]
        placeholders = ",".join("?" * len(tier_values))

        # Build keyword filter clause for SQL (more efficient than Python filtering)
        keyword_clause = ""
        keyword_params: list = []
        if query:
            # Split query into words and require at least one match
            # Limit to 50 keywords to prevent unbounded SQL condition generation
            MAX_QUERY_KEYWORDS = 50
            keywords = [
                kw.strip().lower() for kw in query.split()[:MAX_QUERY_KEYWORDS] if kw.strip()
            ]
            if keywords:
                # Use INSTR for case-insensitive containment check (faster than LIKE)
                keyword_conditions = ["INSTR(LOWER(content), ?) > 0" for _ in keywords]
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

            rows = cursor.fetchall()

        entries = []
        for row in rows:
            entry = ContinuumMemoryEntry(
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
            entries.append(entry)

        return AwaitableList(entries)

    def update_outcome(
        self,
        id: str,
        success: bool,
        agent_prediction_error: float | None = None,
    ) -> float:
        """
        Update memory after observing outcome.

        This implements surprise-based learning: the surprise score is
        updated based on how unexpected the outcome was.

        Uses BEGIN IMMEDIATE to prevent race conditions by acquiring
        a write lock before reading the current state.

        Args:
            id: Memory ID
            success: Whether the pattern led to success
            agent_prediction_error: Optional agent's prediction error

        Returns:
            Updated surprise score
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            # Use BEGIN IMMEDIATE to acquire write lock before reading.
            # This prevents race conditions in read-modify-write operations.
            # Note: get_wal_connection sets busy_timeout (default 30s) to prevent
            # indefinite blocking - will raise sqlite3.OperationalError if timeout
            # exceeded waiting for the lock.
            cursor.execute("BEGIN IMMEDIATE")

            try:
                # Get current state (now protected by write lock)
                cursor.execute(
                    """
                    SELECT success_count, failure_count, surprise_score, tier
                    FROM continuum_memory WHERE id = ?
                    """,
                    (id,),
                )
                row = cursor.fetchone()
                if not row:
                    cursor.execute("ROLLBACK")
                    return 0.0

                success_count, failure_count, old_surprise, tier = row
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
                    math.log(1 + update_count)
                    / math.log(self.hyperparams["consolidation_threshold"]),
                )

                # Update database
                if success:
                    cursor.execute(
                        """
                        UPDATE continuum_memory
                        SET success_count = success_count + 1,
                            update_count = update_count + 1,
                            surprise_score = ?,
                            consolidation_score = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (updated_surprise, consolidation, datetime.now().isoformat(), id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE continuum_memory
                        SET failure_count = failure_count + 1,
                            update_count = update_count + 1,
                            surprise_score = ?,
                            consolidation_score = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (updated_surprise, consolidation, datetime.now().isoformat(), id),
                    )

                cursor.execute("COMMIT")
            except sqlite3.Error as e:
                logger.error(f"Database error updating surprise score: {e}", exc_info=True)
                cursor.execute("ROLLBACK")
                raise
            except Exception as e:
                # Rollback on any exception, then re-raise unchanged
                logger.warning(
                    f"Non-database exception during surprise update, rolling back: {type(e).__name__}: {e}"
                )
                cursor.execute("ROLLBACK")
                raise

        return updated_surprise

    def get_learning_rate(self, tier: MemoryTier, update_count: int) -> float:
        """
        Get tier-specific learning rate with decay.

        HOPE-inspired: fast tiers have high initial LR with rapid decay,
        slow tiers have low initial LR with gradual decay.
        """
        config = TIER_CONFIGS[tier]
        return config.base_learning_rate * (config.decay_rate**update_count)

    def promote(self, id: str) -> Optional[MemoryTier]:
        """
        Promote a memory to a faster tier.

        Uses TierManager for decision logic and records metrics.
        Thread-safe: uses lock to prevent TOCTOU race conditions.

        Returns the new tier if promoted, None otherwise.
        """
        with self._tier_lock, self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT tier, surprise_score, last_promotion_at FROM continuum_memory WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            current_tier = MemoryTier(row[0])
            surprise_score = row[1]
            last_promotion = row[2]

            # Use TierManager for decision
            tm_current = MemoryTier(current_tier.value)
            if not self._tier_manager.should_promote(tm_current, surprise_score, last_promotion):
                logger.debug(
                    f"[memory] Promotion denied for {id}: tier={current_tier.value}, "
                    f"surprise={surprise_score:.3f}, last_promotion={last_promotion}"
                )
                return None

            # Get next tier using TierManager
            tm_new = self._tier_manager.get_next_tier(tm_current, "faster")
            if tm_new is None:
                logger.debug(
                    f"[memory] No faster tier available for {id} (already at {current_tier.value})"
                )
                return None

            new_tier = MemoryTier(tm_new.value)
            now = datetime.now().isoformat()

            logger.info(
                f"[memory] Promoting {id}: {current_tier.value} -> {new_tier.value} "
                f"(surprise={surprise_score:.3f})"
            )

            # Update tier
            cursor.execute(
                """
                UPDATE continuum_memory
                SET tier = ?, last_promotion_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_tier.value, now, now, id),
            )

            # Record transition in database
            cursor.execute(
                """
                INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
                VALUES (?, ?, ?, 'high_surprise', ?)
                """,
                (id, current_tier.value, new_tier.value, surprise_score),
            )

            conn.commit()

        # Record metrics in TierManager
        self._tier_manager.record_promotion(tm_current, tm_new)

        # Emit promotion event if event_emitter is available
        self._emit_tier_event("promotion", id, current_tier, new_tier, surprise_score)

        return new_tier

    def demote(self, id: str) -> Optional[MemoryTier]:
        """
        Demote a memory to a slower tier.

        Uses TierManager for decision logic and records metrics.
        Thread-safe: uses lock to prevent TOCTOU race conditions.

        Returns the new tier if demoted, None otherwise.
        """
        with self._tier_lock, self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT tier, surprise_score, update_count FROM continuum_memory WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            current_tier = MemoryTier(row[0])
            surprise_score = row[1]
            update_count = row[2]

            # Use TierManager for decision
            tm_current = MemoryTier(current_tier.value)
            if not self._tier_manager.should_demote(tm_current, surprise_score, update_count):
                logger.debug(
                    f"[memory] Demotion denied for {id}: tier={current_tier.value}, "
                    f"surprise={surprise_score:.3f}, updates={update_count}"
                )
                return None

            # Get next tier using TierManager
            tm_new = self._tier_manager.get_next_tier(tm_current, "slower")
            if tm_new is None:
                logger.debug(
                    f"[memory] No slower tier available for {id} (already at {current_tier.value})"
                )
                return None

            new_tier = MemoryTier(tm_new.value)
            now = datetime.now().isoformat()

            logger.info(
                f"[memory] Demoting {id}: {current_tier.value} -> {new_tier.value} "
                f"(surprise={surprise_score:.3f}, updates={update_count})"
            )

            # Update tier
            cursor.execute(
                """
                UPDATE continuum_memory
                SET tier = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_tier.value, now, id),
            )

            # Record transition in database
            cursor.execute(
                """
                INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
                VALUES (?, ?, ?, 'high_stability', ?)
                """,
                (id, current_tier.value, new_tier.value, surprise_score),
            )

            conn.commit()

        # Record metrics in TierManager
        self._tier_manager.record_demotion(tm_current, tm_new)

        # Emit demotion event if event_emitter is available
        self._emit_tier_event("demotion", id, current_tier, new_tier, surprise_score)

        return new_tier

    def _emit_tier_event(
        self,
        event_type: str,
        memory_id: str,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        surprise_score: float,
    ) -> None:
        """Emit MEMORY_TIER_PROMOTION or MEMORY_TIER_DEMOTION event."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            stream_type = (
                StreamEventType.MEMORY_TIER_PROMOTION
                if event_type == "promotion"
                else StreamEventType.MEMORY_TIER_DEMOTION
            )

            self.event_emitter.emit(
                StreamEvent(
                    type=stream_type,
                    data={
                        "memory_id": memory_id,
                        "from_tier": from_tier.value,
                        "to_tier": to_tier.value,
                        "surprise_score": surprise_score,
                    },
                )
            )
        except ImportError:
            # Stream module not available - expected in minimal installations
            logger.debug("[memory] Stream module not available for tier event emission")
        except (AttributeError, TypeError) as e:
            # event_emitter not properly configured or emit() signature mismatch
            logger.debug(f"[memory] Event emitter configuration error: {e}")
        except (ValueError, KeyError) as e:
            # Invalid event data or missing StreamEventType
            logger.warning(f"[memory] Invalid tier event data: {e}")
        except (ConnectionError, OSError) as e:
            # Network/IO errors during event emission - non-critical
            logger.debug(f"[memory] Event emission network error: {e}")

    def _promote_batch(
        self,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        ids: List[str],
    ) -> int:
        """
        Batch promote memories from one tier to another.

        Uses executemany for efficient batch updates instead of N+1 queries.
        Thread-safe: uses lock to prevent race conditions with single-item operations.

        Args:
            from_tier: Source tier
            to_tier: Target tier (must be one level faster)
            ids: List of memory IDs to promote

        Returns:
            Number of successfully promoted entries
        """
        if not ids:
            return 0

        now = datetime.now().isoformat()
        cooldown_hours = self.hyperparams["promotion_cooldown_hours"]
        cutoff_time = (datetime.now() - timedelta(hours=cooldown_hours)).isoformat()

        with self._tier_lock, self.connection() as conn:
            cursor = conn.cursor()

            # Batch UPDATE with cooldown check
            # Only promote entries where last_promotion_at is NULL or older than cooldown
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"""
                UPDATE continuum_memory
                SET tier = ?, last_promotion_at = ?, updated_at = ?
                WHERE id IN ({placeholders})
                  AND tier = ?
                  AND (last_promotion_at IS NULL OR last_promotion_at < ?)
                """,
                (to_tier.value, now, now, *ids, from_tier.value, cutoff_time),
            )
            promoted_count = cursor.rowcount

            # Batch INSERT tier transitions for promoted entries
            # Only insert for entries that were actually updated
            if promoted_count > 0:
                cursor.execute(
                    f"""
                    SELECT id, surprise_score FROM continuum_memory
                    WHERE id IN ({placeholders}) AND tier = ?
                    """,
                    (*ids, to_tier.value),
                )
                promoted_entries = cursor.fetchall()

                if promoted_entries:
                    cursor.executemany(
                        """
                        INSERT INTO tier_transitions
                        (memory_id, from_tier, to_tier, reason, surprise_score)
                        VALUES (?, ?, ?, 'high_surprise', ?)
                        """,
                        [
                            (entry[0], from_tier.value, to_tier.value, entry[1])
                            for entry in promoted_entries
                        ],
                    )

            conn.commit()

        if promoted_count > 0:
            logger.info(
                f"[memory] Batch promoted {promoted_count}/{len(ids)} entries: "
                f"{from_tier.value} -> {to_tier.value}"
            )

        return promoted_count

    def _demote_batch(
        self,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        ids: List[str],
    ) -> int:
        """
        Batch demote memories from one tier to another.

        Uses executemany for efficient batch updates instead of N+1 queries.
        Thread-safe: uses lock to prevent race conditions with single-item operations.

        Args:
            from_tier: Source tier
            to_tier: Target tier (must be one level slower)
            ids: List of memory IDs to demote

        Returns:
            Number of successfully demoted entries
        """
        if not ids:
            return 0

        now = datetime.now().isoformat()

        with self._tier_lock, self.connection() as conn:
            cursor = conn.cursor()

            # Batch UPDATE - update_count check is already done in candidate selection
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"""
                UPDATE continuum_memory
                SET tier = ?, updated_at = ?
                WHERE id IN ({placeholders}) AND tier = ?
                """,
                (to_tier.value, now, *ids, from_tier.value),
            )
            demoted_count = cursor.rowcount

            # Batch INSERT tier transitions for demoted entries
            if demoted_count > 0:
                cursor.execute(
                    f"""
                    SELECT id, surprise_score FROM continuum_memory
                    WHERE id IN ({placeholders}) AND tier = ?
                    """,
                    (*ids, to_tier.value),
                )
                demoted_entries = cursor.fetchall()

                if demoted_entries:
                    cursor.executemany(
                        """
                        INSERT INTO tier_transitions
                        (memory_id, from_tier, to_tier, reason, surprise_score)
                        VALUES (?, ?, ?, 'high_stability', ?)
                        """,
                        [
                            (entry[0], from_tier.value, to_tier.value, entry[1])
                            for entry in demoted_entries
                        ],
                    )

            conn.commit()

        if demoted_count > 0:
            logger.info(
                f"[memory] Batch demoted {demoted_count}/{len(ids)} entries: "
                f"{from_tier.value} -> {to_tier.value}"
            )

        return demoted_count

    def consolidate(self) -> Dict[str, int]:
        """
        Run tier consolidation: promote/demote memories based on surprise.

        This should be called periodically (e.g., after each nomic cycle).

        Uses batch operations to avoid N+1 query patterns for better performance
        with large memory stores.

        Each entry is only promoted/demoted once per consolidate call (one level
        at a time), matching the behavior of the individual promote/demote methods.

        Returns:
            Dict with counts of promotions and demotions
        """
        logger.debug("[memory] Starting tier consolidation")
        promotions = 0
        demotions = 0

        # Tier order for promotions: glacial -> slow -> medium -> fast
        promotion_pairs = [
            (MemoryTier.GLACIAL, MemoryTier.SLOW),
            (MemoryTier.SLOW, MemoryTier.MEDIUM),
            (MemoryTier.MEDIUM, MemoryTier.FAST),
        ]

        # Tier order for demotions: fast -> medium -> slow -> glacial
        demotion_pairs = [
            (MemoryTier.FAST, MemoryTier.MEDIUM),
            (MemoryTier.MEDIUM, MemoryTier.SLOW),
            (MemoryTier.SLOW, MemoryTier.GLACIAL),
        ]

        # Collect ALL candidates upfront before any processing
        # This ensures each entry only moves one level per consolidate call
        promotion_candidates: Dict[tuple, List[str]] = {}
        demotion_candidates: Dict[tuple, List[str]] = {}

        with self.connection() as conn:
            cursor = conn.cursor()

            # Collect promotion candidates for all tier pairs
            # Limit to 1000 per tier to prevent memory issues with large databases
            batch_limit = 1000
            for from_tier, to_tier in promotion_pairs:
                config = TIER_CONFIGS[from_tier]
                cursor.execute(
                    """
                    SELECT id FROM continuum_memory
                    WHERE tier = ? AND surprise_score > ?
                    ORDER BY surprise_score DESC
                    LIMIT ?
                    """,
                    (from_tier.value, config.promotion_threshold, batch_limit),
                )
                ids = [row[0] for row in cursor.fetchall()]
                if ids:
                    promotion_candidates[(from_tier, to_tier)] = ids

            # Collect demotion candidates for all tier pairs
            for from_tier, to_tier in demotion_pairs:
                config = TIER_CONFIGS[from_tier]
                cursor.execute(
                    """
                    SELECT id FROM continuum_memory
                    WHERE tier = ?
                      AND (1.0 - surprise_score) > ?
                      AND update_count > 10
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    (from_tier.value, config.demotion_threshold, batch_limit),
                )
                ids = [row[0] for row in cursor.fetchall()]
                if ids:
                    demotion_candidates[(from_tier, to_tier)] = ids

        # Process all promotions (outside the collection connection)
        for (from_tier, to_tier), ids in promotion_candidates.items():
            count = self._promote_batch(from_tier, to_tier, ids)
            promotions += count
            logger.debug(
                f"Promoted {count}/{len(ids)} entries from {from_tier.value} to {to_tier.value}"
            )

        # Process all demotions (outside the collection connection)
        for (from_tier, to_tier), ids in demotion_candidates.items():
            count = self._demote_batch(from_tier, to_tier, ids)
            demotions += count
            logger.debug(
                f"Demoted {count}/{len(ids)} entries from {from_tier.value} to {to_tier.value}"
            )

        if promotions > 0 or demotions > 0:
            logger.info(
                f"[memory] Consolidation complete: {promotions} promotions, {demotions} demotions"
            )
        else:
            logger.debug("[memory] Consolidation complete: no tier changes")

        return {"promotions": promotions, "demotions": demotions}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory system."""
        with self.connection() as conn:
            cursor = conn.cursor()

            stats: Dict[str, Any] = {}

            # Count by tier
            cursor.execute(
                """
                SELECT tier, COUNT(*), AVG(importance), AVG(surprise_score), AVG(consolidation_score)
                FROM continuum_memory
                GROUP BY tier
            """
            )
            stats["by_tier"] = {
                row[0]: {
                    "count": row[1],
                    "avg_importance": row[2] or 0,
                    "avg_surprise": row[3] or 0,
                    "avg_consolidation": row[4] or 0,
                }
                for row in cursor.fetchall()
            }

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM continuum_memory")
            row = cursor.fetchone()
            stats["total_memories"] = row[0] if row else 0

            # Transition history
            cursor.execute(
                """
                SELECT from_tier, to_tier, COUNT(*)
                FROM tier_transitions
                GROUP BY from_tier, to_tier
            """
            )
            stats["transitions"] = [
                {"from": row[0], "to": row[1], "count": row[2]} for row in cursor.fetchall()
            ]

        return stats

    def export_for_tier(self, tier: MemoryTier) -> List[Dict[str, Any]]:
        """Export all memories for a specific tier."""
        entries = self.retrieve(tiers=[tier], limit=1000)
        return [
            {
                "id": e.id,
                "content": e.content,
                "importance": e.importance,
                "surprise_score": e.surprise_score,
                "consolidation_score": e.consolidation_score,
                "success_rate": e.success_rate,
                "update_count": e.update_count,
            }
            for e in entries
        ]

    def get_memory_pressure(self) -> float:
        """
        Calculate memory pressure as a 0-1 score based on tier utilization.

        Returns the highest utilization ratio across all tiers, where:
        - 0.0 = All tiers are empty
        - 1.0 = At least one tier is at or above its max_entries limit

        Use this to trigger cleanup when pressure exceeds a threshold (e.g., 0.8).

        Returns:
            Float between 0.0 and 1.0 indicating memory pressure level.
        """
        max_entries = self.hyperparams["max_entries_per_tier"]
        if not max_entries:
            return 0.0

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tier, COUNT(*)
                FROM continuum_memory
                GROUP BY tier
            """
            )
            tier_counts: Dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

        # Calculate utilization for each tier
        max_pressure = 0.0
        tier_names = ["fast", "medium", "slow", "glacial"]
        for tier_name in tier_names:
            limit: int = max_entries[tier_name]  # type: ignore[literal-required]
            if limit <= 0:
                continue
            count = tier_counts.get(tier_name, 0)
            pressure = count / limit
            max_pressure = max(max_pressure, pressure)

        return min(max_pressure, 1.0)

    def cleanup_expired_memories(
        self,
        tier: Optional[MemoryTier] = None,
        archive: bool = True,
        max_age_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Remove or archive expired memories based on tier retention policies.

        Memories are considered expired when they are older than:
        tier_half_life * retention_multiplier (default 2x)

        Args:
            tier: Specific tier to cleanup (None = all tiers)
            archive: If True, move to archive table; if False, delete permanently
            max_age_hours: Override default retention (uses tier half-life * multiplier if None)

        Returns:
            Dict with counts: {"archived": N, "deleted": N, "by_tier": {...}}
        """
        results: Dict[str, Any] = {"archived": 0, "deleted": 0, "by_tier": {}}
        tiers_to_process = [tier] if tier else list(MemoryTier)
        retention_multiplier = self.hyperparams["retention_multiplier"]

        with self.connection() as conn:
            cursor = conn.cursor()

            for t in tiers_to_process:
                config = TIER_CONFIGS[t]
                tier_name = t.value

                # Calculate cutoff time
                if max_age_hours is not None:
                    retention_hours = max_age_hours
                else:
                    retention_hours = config.half_life_hours * retention_multiplier

                cutoff = datetime.now() - timedelta(hours=retention_hours)
                cutoff_str = cutoff.isoformat()

                if archive:
                    # Archive expired entries
                    cursor.execute(
                        """
                        INSERT INTO continuum_memory_archive
                            (id, tier, content, importance, surprise_score,
                             consolidation_score, update_count, success_count,
                             failure_count, semantic_centroid, created_at,
                             updated_at, archive_reason, metadata)
                        SELECT id, tier, content, importance, surprise_score,
                               consolidation_score, update_count, success_count,
                               failure_count, semantic_centroid, created_at,
                               updated_at, 'expired', metadata
                        FROM continuum_memory
                        WHERE tier = ?
                          AND datetime(updated_at) < datetime(?)
                        """,
                        (tier_name, cutoff_str),
                    )
                    archived_count = cursor.rowcount
                else:
                    archived_count = 0

                # Delete from main table
                cursor.execute(
                    """
                    DELETE FROM continuum_memory
                    WHERE tier = ?
                      AND datetime(updated_at) < datetime(?)
                    """,
                    (tier_name, cutoff_str),
                )
                deleted_count = cursor.rowcount

                results["by_tier"][tier_name] = {
                    "archived": archived_count if archive else 0,
                    "deleted": deleted_count,
                    "cutoff_hours": retention_hours,
                }
                results["archived"] += archived_count if archive else 0
                results["deleted"] += deleted_count

            conn.commit()

        logger.info(
            "Memory cleanup: archived=%d, deleted=%d",
            results["archived"],
            results["deleted"],
        )
        return results

    def delete(
        self,
        memory_id: str,
        archive: bool = True,
        reason: str = "user_deleted",
    ) -> Dict[str, Any]:
        """
        Delete a specific memory entry by ID.

        Args:
            memory_id: The ID of the memory to delete
            archive: If True, archive before deletion; if False, delete permanently
            reason: Reason for deletion (stored in archive)

        Returns:
            Dict with result: {"deleted": bool, "archived": bool, "id": str}
        """
        result: Dict[str, Any] = {"deleted": False, "archived": False, "id": memory_id}

        with self.connection() as conn:
            cursor = conn.cursor()

            # Check if memory exists
            cursor.execute("SELECT id FROM continuum_memory WHERE id = ?", (memory_id,))
            if not cursor.fetchone():
                logger.debug("Memory %s not found for deletion", memory_id)
                return result

            if archive:
                # Archive the entry before deletion
                cursor.execute(
                    """
                    INSERT INTO continuum_memory_archive
                        (id, tier, content, importance, surprise_score,
                         consolidation_score, update_count, success_count,
                         failure_count, semantic_centroid, created_at,
                         updated_at, archive_reason, metadata)
                    SELECT id, tier, content, importance, surprise_score,
                           consolidation_score, update_count, success_count,
                           failure_count, semantic_centroid, created_at,
                           updated_at, ?, metadata
                    FROM continuum_memory
                    WHERE id = ?
                    """,
                    (reason, memory_id),
                )
                result["archived"] = cursor.rowcount > 0

            # Delete from main table
            cursor.execute(
                "DELETE FROM continuum_memory WHERE id = ?",
                (memory_id,),
            )
            result["deleted"] = cursor.rowcount > 0

            conn.commit()

        if result["deleted"]:
            logger.info(
                "Memory %s deleted (archived=%s, reason=%s)", memory_id, result["archived"], reason
            )

        return result

    def enforce_tier_limits(
        self,
        tier: Optional[MemoryTier] = None,
        archive: bool = True,
    ) -> Dict[str, int]:
        """
        Enforce max entries per tier by removing lowest importance entries.

        When a tier exceeds its limit, the lowest importance entries are
        archived (or deleted) until the tier is within limits.

        Args:
            tier: Specific tier to enforce (None = all tiers)
            archive: If True, archive excess; if False, delete permanently

        Returns:
            Dict with counts of removed entries by tier
        """
        results: Dict[str, int] = {}
        tiers_to_process = [tier] if tier else list(MemoryTier)
        max_entries: MaxEntriesPerTier = self.hyperparams["max_entries_per_tier"]

        with self.connection() as conn:
            cursor = conn.cursor()

            for t in tiers_to_process:
                tier_name = t.value
                limit: int = max_entries.get(tier_name, 10000)  # type: ignore[assignment]

                # Count current entries
                cursor.execute(
                    "SELECT COUNT(*) FROM continuum_memory WHERE tier = ?",
                    (tier_name,),
                )
                row = cursor.fetchone()
                count: int = row[0] if row else 0

                if count <= limit:
                    results[tier_name] = 0
                    continue

                excess: int = count - limit

                if archive:
                    # Archive lowest importance entries
                    cursor.execute(
                        """
                        INSERT INTO continuum_memory_archive
                            (id, tier, content, importance, surprise_score,
                             consolidation_score, update_count, success_count,
                             failure_count, semantic_centroid, created_at,
                             updated_at, archive_reason, metadata)
                        SELECT id, tier, content, importance, surprise_score,
                               consolidation_score, update_count, success_count,
                               failure_count, semantic_centroid, created_at,
                               updated_at, 'tier_limit', metadata
                        FROM continuum_memory
                        WHERE tier = ?
                        ORDER BY importance ASC, updated_at ASC
                        LIMIT ?
                        """,
                        (tier_name, excess),
                    )

                # Delete excess entries (lowest importance first)
                cursor.execute(
                    """
                    DELETE FROM continuum_memory
                    WHERE id IN (
                        SELECT id FROM continuum_memory
                        WHERE tier = ?
                        ORDER BY importance ASC, updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (tier_name, excess),
                )

                results[tier_name] = cursor.rowcount
                logger.info(
                    "Tier limit enforced: tier=%s, removed=%d (limit=%d)",
                    tier_name,
                    cursor.rowcount,
                    limit,
                )

            conn.commit()

        return results

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived memories."""
        with self.connection() as conn:
            cursor = conn.cursor()

            stats: Dict[str, Any] = {}

            # Count by tier and reason
            cursor.execute(
                """
                SELECT tier, archive_reason, COUNT(*)
                FROM continuum_memory_archive
                GROUP BY tier, archive_reason
            """
            )
            by_tier_reason: Dict[str, Dict[str, int]] = {}
            for row in cursor.fetchall():
                tier, reason, count = row
                if tier not in by_tier_reason:
                    by_tier_reason[tier] = {}
                by_tier_reason[tier][reason or "unknown"] = count
            stats["by_tier_reason"] = by_tier_reason

            # Total archived
            cursor.execute("SELECT COUNT(*) FROM continuum_memory_archive")
            row = cursor.fetchone()
            stats["total_archived"] = row[0] if row else 0

            # Oldest and newest archived
            cursor.execute(
                """
                SELECT MIN(archived_at), MAX(archived_at)
                FROM continuum_memory_archive
            """
            )
            row = cursor.fetchone()
            stats["oldest_archived"] = row[0]
            stats["newest_archived"] = row[1]

        return stats
