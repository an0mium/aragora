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

import asyncio
import json
import logging
import math
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, TypedDict

if TYPE_CHECKING:
    from aragora.types.protocols import EventEmitterProtocol

logger = logging.getLogger(__name__)

from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierConfig,  # noqa: F401 - re-exported for backwards compatibility
    TierManager,
    get_tier_manager,
)

# Import extracted modules for delegation
from aragora.memory import continuum_consolidation as _consolidation
from aragora.memory import continuum_stats as _stats
from aragora.storage.base_store import SQLiteStore
from aragora.storage.schema import SchemaManager
from aragora.utils.json_helpers import safe_json_loads

# Schema version for ContinuumMemory
CONTINUUM_SCHEMA_VERSION = 3

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
    red_line: bool = False  # If True, entry cannot be deleted/forgotten
    red_line_reason: str = ""  # Why this entry is protected

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

    # Cross-reference support for Knowledge Mound integration

    @property
    def cross_references(self) -> List[str]:
        """Get list of cross-reference IDs linked to this entry."""
        return self.metadata.get("cross_references", [])

    @cross_references.setter
    def cross_references(self, refs: List[str]) -> None:
        """Set cross-reference IDs for this entry."""
        self.metadata["cross_references"] = refs

    def add_cross_reference(self, ref_id: str) -> None:
        """Add a cross-reference to another knowledge item."""
        refs = self.cross_references
        if ref_id not in refs:
            refs.append(ref_id)
            self.cross_references = refs

    def remove_cross_reference(self, ref_id: str) -> None:
        """Remove a cross-reference from this entry."""
        refs = self.cross_references
        if ref_id in refs:
            refs.remove(ref_id)
            self.cross_references = refs

    @property
    def knowledge_mound_id(self) -> str:
        """Get the Knowledge Mound ID for this entry."""
        return f"cm_{self.id}"

    @property
    def tags(self) -> List[str]:
        """Get tags associated with this entry."""
        return self.metadata.get("tags", [])

    @tags.setter
    def tags(self, tag_list: List[str]) -> None:
        """Set tags for this entry."""
        self.metadata["tags"] = tag_list

    @property
    def last_updated(self) -> str:
        """Alias for updated_at for Knowledge Mound compatibility."""
        return self.updated_at


class AwaitableList(list):
    """List wrapper that can be awaited for async compatibility."""

    def __await__(self) -> "Generator[Any, None, AwaitableList]":
        async def _wrap() -> "AwaitableList":
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
            expires_at TEXT,
            red_line INTEGER DEFAULT 0,
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

        -- Index for red line protected entries
        CREATE INDEX IF NOT EXISTS idx_continuum_red_line ON continuum_memory(red_line);
    """

    # Migration from version 2 to version 3: Add red_line columns
    SCHEMA_MIGRATIONS = {
        3: """
            ALTER TABLE continuum_memory ADD COLUMN red_line INTEGER DEFAULT 0;
            ALTER TABLE continuum_memory ADD COLUMN red_line_reason TEXT DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_continuum_red_line ON continuum_memory(red_line);
        """
    }

    def __init__(
        self,
        db_path: str | Path | None = None,
        tier_manager: Optional[TierManager] = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
        storage_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ):
        if db_path is None:
            db_path = get_db_path(DatabaseType.CONTINUUM_MEMORY)
        resolved_path: str | Path = db_path
        base_path = storage_path or base_dir
        if base_path:
            base = Path(base_path)
            if base.suffix:
                resolved_path = base
            else:
                resolved_path = base / "continuum_memory.db"
        else:
            candidate = Path(db_path)
            if candidate.exists() and candidate.is_dir():
                resolved_path = candidate / "continuum_memory.db"

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

    async def add_async(
        self,
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """Async wrapper for add() - offloads blocking I/O to executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.add(
                id=id,
                content=content,
                tier=tier,
                importance=importance,
                metadata=metadata,
            ),
        )

    async def store(
        self,
        key: str,
        content: str,
        tier: str | MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """Async wrapper for add() - offloads blocking I/O to executor."""
        normalized_tier = MemoryTier(tier) if isinstance(tier, str) else tier
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.add(
                id=key,
                content=content,
                tier=normalized_tier,
                importance=importance,
                metadata=metadata,
            ),
        )

    async def get_async(self, id: str) -> Optional[ContinuumMemoryEntry]:
        """Async wrapper for get() - offloads blocking I/O to executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, id)

    async def retrieve_async(
        self,
        query: str | None = None,
        tiers: List[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: str | MemoryTier | None = None,
    ) -> List[ContinuumMemoryEntry]:
        """Async wrapper for retrieve() - offloads blocking I/O to executor."""
        loop = asyncio.get_event_loop()
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

    async def update_outcome_async(
        self,
        id: str,
        success: bool,
        agent_prediction_error: float | None = None,
    ) -> float:
        """Async wrapper for update_outcome() - offloads blocking I/O to executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.update_outcome(id, success, agent_prediction_error),
        )

    def get(self, id: str) -> Optional[ContinuumMemoryEntry]:
        """Get a memory entry by ID."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at, metadata,
                       COALESCE(red_line, 0), COALESCE(red_line_reason, '')
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
            red_line=bool(row[12]),
            red_line_reason=row[13],
        )

    def mark_red_line(
        self,
        memory_id: str,
        reason: str,
        promote_to_glacial: bool = True,
    ) -> bool:
        """
        Mark a memory entry as a red line - cannot be forgotten or overwritten.

        Red line entries are critical memories that should never be deleted,
        such as safety-critical decisions, irreversible actions taken, or
        foundational knowledge that must be preserved.

        Args:
            memory_id: The ID of the memory to protect
            reason: Why this entry is critical (for auditing)
            promote_to_glacial: If True, promote to glacial tier for maximum retention

        Returns:
            True if the entry was marked, False if entry not found
        """
        now = datetime.now().isoformat()

        with self.connection() as conn:
            cursor = conn.cursor()

            # Check if memory exists
            cursor.execute("SELECT tier FROM continuum_memory WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning("Cannot mark non-existent memory as red line: %s", memory_id)
                return False

            current_tier = row[0]

            # Mark as red line
            if promote_to_glacial and current_tier != MemoryTier.GLACIAL.value:
                cursor.execute(
                    """
                    UPDATE continuum_memory
                    SET red_line = 1, red_line_reason = ?, tier = ?,
                        importance = 1.0, updated_at = ?
                    WHERE id = ?
                    """,
                    (reason, MemoryTier.GLACIAL.value, now, memory_id),
                )
                logger.info(
                    "Marked memory %s as red line and promoted to glacial tier (reason: %s)",
                    memory_id,
                    reason,
                )
            else:
                cursor.execute(
                    """
                    UPDATE continuum_memory
                    SET red_line = 1, red_line_reason = ?, importance = 1.0, updated_at = ?
                    WHERE id = ?
                    """,
                    (reason, now, memory_id),
                )
                logger.info("Marked memory %s as red line (reason: %s)", memory_id, reason)

            conn.commit()
            return True

    def get_red_line_memories(self) -> List[ContinuumMemoryEntry]:
        """Get all red-lined memory entries.

        Returns:
            List of all protected memory entries
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at, metadata,
                       COALESCE(red_line, 0), COALESCE(red_line_reason, '')
                FROM continuum_memory
                WHERE red_line = 1
                ORDER BY created_at ASC
                """
            )
            rows = cursor.fetchall()

        return [
            ContinuumMemoryEntry(
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
                red_line=bool(row[12]),
                red_line_reason=row[13],
            )
            for row in rows
        ]

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

        # Emit MEMORY_RECALL event if memories were retrieved
        if entries and self.event_emitter:
            try:
                from aragora.server.stream.events import StreamEvent, StreamEventType

                tier_counts = {}
                for e in entries:
                    tier_counts[e.tier.value] = tier_counts.get(e.tier.value, 0) + 1

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.MEMORY_RECALL,
                        data={
                            "count": len(entries),
                            "query": query[:100] if query else None,
                            "tier_distribution": tier_counts,
                            "top_importance": max(e.importance for e in entries),
                        },
                    )
                )
            except (ImportError, AttributeError, TypeError):
                pass  # Stream module not available or emitter misconfigured

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

                success_count, failure_count, old_surprise_raw, tier = row
                old_surprise: float = float(old_surprise_raw) if old_surprise_raw else 0.0
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
        _consolidation.emit_tier_event(
            self, event_type, memory_id, from_tier, to_tier, surprise_score
        )

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
        """
        return _consolidation.promote_batch(self, from_tier, to_tier, ids)

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
        """
        return _consolidation.demote_batch(self, from_tier, to_tier, ids)

    def consolidate(self) -> Dict[str, int]:
        """
        Run tier consolidation: promote/demote memories based on surprise.

        This should be called periodically (e.g., after each nomic cycle).
        Uses batch operations for better performance with large memory stores.
        """
        return _consolidation.consolidate(self)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory system."""
        return _stats.get_stats(self)

    def export_for_tier(self, tier: MemoryTier) -> List[Dict[str, Any]]:
        """Export all memories for a specific tier."""
        return _stats.export_for_tier(self, tier)

    def get_memory_pressure(self) -> float:
        """
        Calculate memory pressure as a 0-1 score based on tier utilization.

        Returns the highest utilization ratio across all tiers, where:
        - 0.0 = All tiers are empty
        - 1.0 = At least one tier is at or above its max_entries limit

        Use this to trigger cleanup when pressure exceeds a threshold (e.g., 0.8).
        """
        return _stats.get_memory_pressure(self)

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
        return _stats.cleanup_expired_memories(self, tier, archive, max_age_hours)

    def delete(
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
            archive: If True, archive before deletion; if False, delete permanently
            reason: Reason for deletion (stored in archive)
            force: If True, delete even if entry is red-lined (dangerous!)

        Returns:
            Dict with result: {"deleted": bool, "archived": bool, "id": str, "blocked": bool}
        """
        return _stats.delete_memory(self, memory_id, archive, reason, force)

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
        return _stats.enforce_tier_limits(self, tier, archive)

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived memories."""
        return _stats.get_archive_stats(self)

    # === Glacial Tier Consumption Methods ===
    # These methods enable cross-session learning by retrieving long-term patterns

    def get_glacial_insights(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.3,
    ) -> List[ContinuumMemoryEntry]:
        """
        Retrieve long-term patterns from the glacial tier for cross-session learning.

        The glacial tier stores foundational knowledge that persists across cycles
        (30-day half-life). This method provides targeted access to these insights
        for context gathering in debates and nomic cycles.

        Args:
            topic: Optional topic/query to filter relevant insights
            limit: Maximum entries to return (default 10)
            min_importance: Minimum importance threshold (default 0.3)

        Returns:
            List of glacial tier entries sorted by importance and relevance

        Example:
            # In debate orchestrator context phase:
            insights = await cms.get_glacial_insights_async(topic="error handling")
            for insight in insights:
                context.add_background(insight.content)
        """
        return self.retrieve(
            query=topic,
            tiers=[MemoryTier.GLACIAL],
            limit=limit,
            min_importance=min_importance,
            include_glacial=True,
        )

    async def get_glacial_insights_async(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.3,
    ) -> List[ContinuumMemoryEntry]:
        """Async wrapper for get_glacial_insights() for use in async contexts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_glacial_insights(
                topic=topic,
                limit=limit,
                min_importance=min_importance,
            ),
        )

    def get_cross_session_patterns(
        self,
        domain: Optional[str] = None,
        include_slow: bool = True,
        limit: int = 20,
    ) -> List[ContinuumMemoryEntry]:
        """
        Get patterns that persist across sessions (slow + glacial tiers).

        Useful for:
        - Nomic loop context gathering (cross-cycle learning)
        - Agent team selection (historical performance patterns)
        - Debate strategy (recurring themes and successful approaches)

        Args:
            domain: Optional domain filter (e.g., 'code', 'research', 'creative')
            include_slow: Include slow tier in addition to glacial
            limit: Maximum entries to return

        Returns:
            Combined list from slow and glacial tiers, sorted by importance
        """
        tiers = [MemoryTier.GLACIAL]
        if include_slow:
            tiers.append(MemoryTier.SLOW)

        entries = self.retrieve(
            query=domain,
            tiers=tiers,
            limit=limit,
            min_importance=0.2,
            include_glacial=True,
        )

        # Sort by importance (highest first)
        return sorted(entries, key=lambda e: e.importance, reverse=True)

    async def get_cross_session_patterns_async(
        self,
        domain: Optional[str] = None,
        include_slow: bool = True,
        limit: int = 20,
    ) -> List[ContinuumMemoryEntry]:
        """Async wrapper for get_cross_session_patterns()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_cross_session_patterns(
                domain=domain,
                include_slow=include_slow,
                limit=limit,
            ),
        )

    def get_glacial_tier_stats(self) -> Dict[str, Any]:
        """
        Get statistics specifically for the glacial tier.

        Useful for monitoring long-term memory health and deciding
        when to promote entries from slow to glacial tier.
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            # Get counts and averages for glacial tier
            cursor.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(importance) as avg_importance,
                    AVG(surprise_score) as avg_surprise,
                    AVG(consolidation_score) as avg_consolidation,
                    AVG(update_count) as avg_updates,
                    SUM(CASE WHEN red_line = 1 THEN 1 ELSE 0 END) as red_line_count,
                    MIN(created_at) as oldest_entry,
                    MAX(updated_at) as newest_update
                FROM continuum_memory
                WHERE tier = 'glacial'
                """
            )
            row = cursor.fetchone()

            # Get top tags in glacial tier
            cursor.execute(
                """
                SELECT metadata FROM continuum_memory
                WHERE tier = 'glacial'
                LIMIT 100
                """
            )
            tag_counts: Dict[str, int] = {}
            for (metadata_json,) in cursor.fetchall():
                metadata = safe_json_loads(metadata_json, {})
                for tag in metadata.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "tier": "glacial",
            "count": row[0] or 0,
            "avg_importance": round(row[1] or 0, 3),
            "avg_surprise": round(row[2] or 0, 3),
            "avg_consolidation": round(row[3] or 0, 3),
            "avg_updates": round(row[4] or 0, 1),
            "red_line_count": row[5] or 0,
            "oldest_entry": row[6],
            "newest_update": row[7],
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "max_entries": self.hyperparams["max_entries_per_tier"]["glacial"],
            "utilization": round((row[0] or 0) / self.hyperparams["max_entries_per_tier"]["glacial"], 3),
        }
