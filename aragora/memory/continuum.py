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

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class MemoryTier(Enum):
    """Memory update frequency tiers."""
    FAST = "fast"       # Updates on every event
    MEDIUM = "medium"   # Updates per debate round
    SLOW = "slow"       # Updates per nomic cycle
    GLACIAL = "glacial" # Updates monthly


@dataclass
class TierConfig:
    """Configuration for a memory tier."""
    name: str
    half_life_hours: float
    update_frequency: str
    base_learning_rate: float
    decay_rate: float
    promotion_threshold: float  # Surprise score to promote to faster tier
    demotion_threshold: float   # Stability score to demote to slower tier


# Default tier configurations (HOPE-inspired)
TIER_CONFIGS: Dict[MemoryTier, TierConfig] = {
    MemoryTier.FAST: TierConfig(
        name="fast",
        half_life_hours=1,
        update_frequency="event",
        base_learning_rate=0.3,
        decay_rate=0.95,
        promotion_threshold=1.0,  # Can't promote higher
        demotion_threshold=0.2,   # Very stable patterns demote
    ),
    MemoryTier.MEDIUM: TierConfig(
        name="medium",
        half_life_hours=24,
        update_frequency="round",
        base_learning_rate=0.1,
        decay_rate=0.99,
        promotion_threshold=0.7,  # High surprise promotes to fast
        demotion_threshold=0.3,
    ),
    MemoryTier.SLOW: TierConfig(
        name="slow",
        half_life_hours=168,  # 7 days
        update_frequency="cycle",
        base_learning_rate=0.03,
        decay_rate=0.999,
        promotion_threshold=0.6,  # Medium surprise promotes to medium
        demotion_threshold=0.4,
    ),
    MemoryTier.GLACIAL: TierConfig(
        name="glacial",
        half_life_hours=720,  # 30 days
        update_frequency="monthly",
        base_learning_rate=0.01,
        decay_rate=0.9999,
        promotion_threshold=0.5,  # Low bar to promote (rarely happens)
        demotion_threshold=1.0,   # Can't demote lower
    ),
}


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


class ContinuumMemory:
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

    def __init__(self, db_path: str = "aragora_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()
        # Hyperparameters (can be modified by MetaLearner)
        self.hyperparams = {
            "surprise_weight_success": 0.3,  # Weight for success rate surprise
            "surprise_weight_semantic": 0.3,  # Weight for semantic novelty
            "surprise_weight_temporal": 0.2,  # Weight for timing surprise
            "surprise_weight_agent": 0.2,    # Weight for agent prediction error
            "consolidation_threshold": 100,   # Updates to reach full consolidation
            "promotion_cooldown_hours": 24,   # Minimum time between promotions
        }

    def _init_db(self):
        """Initialize the continuum memory tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main continuum memory table
        cursor.execute("""
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
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Indexes for efficient tier-based retrieval
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_continuum_tier
            ON continuum_memory(tier)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_continuum_surprise
            ON continuum_memory(surprise_score DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_continuum_importance
            ON continuum_memory(importance DESC)
        """)

        # Meta-learning state table for hyperparameter tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta_learning_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hyperparams TEXT NOT NULL,
                learning_efficiency REAL,
                pattern_retention_rate REAL,
                forgetting_rate REAL,
                cycles_evaluated INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tier transition history for analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tier_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                from_tier TEXT NOT NULL,
                to_tier TEXT NOT NULL,
                reason TEXT,
                surprise_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES continuum_memory(id)
            )
        """)

        conn.commit()
        conn.close()

    def add(
        self,
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

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
        conn.close()

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

    def get(self, id: str) -> Optional[ContinuumMemoryEntry]:
        """Get a memory entry by ID."""
        conn = sqlite3.connect(self.db_path)
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
        conn.close()

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
            metadata=json.loads(row[11]) if row[11] else {},
        )

    def retrieve(
        self,
        query: str = None,
        tiers: List[MemoryTier] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build tier filter
        if tiers is None:
            tiers = list(MemoryTier)
        if not include_glacial:
            tiers = [t for t in tiers if t != MemoryTier.GLACIAL]

        tier_values = [t.value for t in tiers]
        placeholders = ",".join("?" * len(tier_values))

        # Retrieval query with time-decay scoring
        # Score = importance * (1 + surprise) * decay_factor
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
            ORDER BY score DESC
            LIMIT ?
            """,
            (*tier_values, min_importance, limit * 2),  # Fetch extra for filtering
        )

        entries = []
        for row in cursor.fetchall():
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
                metadata=json.loads(row[11]) if row[11] else {},
            )

            # Simple keyword relevance filter if query provided
            if query:
                query_lower = query.lower()
                content_lower = entry.content.lower()
                if not any(word in content_lower for word in query_lower.split()):
                    continue

            entries.append(entry)
            if len(entries) >= limit:
                break

        conn.close()
        return entries

    def update_outcome(
        self,
        id: str,
        success: bool,
        agent_prediction_error: float = None,
    ) -> float:
        """
        Update memory after observing outcome.

        This implements surprise-based learning: the surprise score is
        updated based on how unexpected the outcome was.

        Args:
            id: Memory ID
            success: Whether the pattern led to success
            agent_prediction_error: Optional agent's prediction error

        Returns:
            Updated surprise score
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current state
        cursor.execute(
            """
            SELECT success_count, failure_count, surprise_score, tier
            FROM continuum_memory WHERE id = ?
            """,
            (id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
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
        weights = self.hyperparams
        new_surprise = (
            weights["surprise_weight_success"] * success_surprise +
            weights["surprise_weight_agent"] * (agent_prediction_error or 0.0)
        )

        # Exponential moving average for surprise
        alpha = 0.3
        updated_surprise = old_surprise * (1 - alpha) + new_surprise * alpha

        # Update consolidation score
        update_count = total + 1
        consolidation = min(1.0, math.log(1 + update_count) / math.log(
            self.hyperparams["consolidation_threshold"]
        ))

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

        conn.commit()
        conn.close()
        return updated_surprise

    def get_learning_rate(self, tier: MemoryTier, update_count: int) -> float:
        """
        Get tier-specific learning rate with decay.

        HOPE-inspired: fast tiers have high initial LR with rapid decay,
        slow tiers have low initial LR with gradual decay.
        """
        config = TIER_CONFIGS[tier]
        return config.base_learning_rate * (config.decay_rate ** update_count)

    def promote(self, id: str) -> Optional[MemoryTier]:
        """
        Promote a memory to a faster tier.

        Returns the new tier if promoted, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT tier, surprise_score, last_promotion_at FROM continuum_memory WHERE id = ?",
            (id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        current_tier = MemoryTier(row[0])
        surprise_score = row[1]
        last_promotion = row[2]

        # Check cooldown
        if last_promotion:
            last_dt = datetime.fromisoformat(last_promotion)
            hours_since = (datetime.now() - last_dt).total_seconds() / 3600
            if hours_since < self.hyperparams["promotion_cooldown_hours"]:
                conn.close()
                return None

        # Determine new tier
        tier_order = [MemoryTier.GLACIAL, MemoryTier.SLOW, MemoryTier.MEDIUM, MemoryTier.FAST]
        current_idx = tier_order.index(current_tier)
        if current_idx >= len(tier_order) - 1:
            conn.close()
            return None  # Already at fastest

        new_tier = tier_order[current_idx + 1]
        now = datetime.now().isoformat()

        # Update tier
        cursor.execute(
            """
            UPDATE continuum_memory
            SET tier = ?, last_promotion_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_tier.value, now, now, id),
        )

        # Record transition
        cursor.execute(
            """
            INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
            VALUES (?, ?, ?, 'high_surprise', ?)
            """,
            (id, current_tier.value, new_tier.value, surprise_score),
        )

        conn.commit()
        conn.close()
        return new_tier

    def demote(self, id: str) -> Optional[MemoryTier]:
        """
        Demote a memory to a slower tier.

        Returns the new tier if demoted, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT tier, surprise_score, update_count FROM continuum_memory WHERE id = ?",
            (id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        current_tier = MemoryTier(row[0])
        surprise_score = row[1]
        update_count = row[2]

        # Need enough updates to be confident about stability
        if update_count < 10:
            conn.close()
            return None

        tier_order = [MemoryTier.GLACIAL, MemoryTier.SLOW, MemoryTier.MEDIUM, MemoryTier.FAST]
        current_idx = tier_order.index(current_tier)
        if current_idx <= 0:
            conn.close()
            return None  # Already at slowest

        new_tier = tier_order[current_idx - 1]
        now = datetime.now().isoformat()

        # Update tier
        cursor.execute(
            """
            UPDATE continuum_memory
            SET tier = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_tier.value, now, id),
        )

        # Record transition
        cursor.execute(
            """
            INSERT INTO tier_transitions (memory_id, from_tier, to_tier, reason, surprise_score)
            VALUES (?, ?, ?, 'high_stability', ?)
            """,
            (id, current_tier.value, new_tier.value, surprise_score),
        )

        conn.commit()
        conn.close()
        return new_tier

    def consolidate(self) -> Dict[str, int]:
        """
        Run tier consolidation: promote/demote memories based on surprise.

        This should be called periodically (e.g., after each nomic cycle).

        Returns:
            Dict with counts of promotions and demotions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        promotions = 0
        demotions = 0

        # Find candidates for promotion (high surprise)
        for tier in [MemoryTier.GLACIAL, MemoryTier.SLOW, MemoryTier.MEDIUM]:
            config = TIER_CONFIGS[tier]
            cursor.execute(
                """
                SELECT id FROM continuum_memory
                WHERE tier = ? AND surprise_score > ?
                """,
                (tier.value, config.promotion_threshold),
            )
            for (id,) in cursor.fetchall():
                if self.promote(id):
                    promotions += 1

        # Find candidates for demotion (high stability)
        for tier in [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW]:
            config = TIER_CONFIGS[tier]
            cursor.execute(
                """
                SELECT id FROM continuum_memory
                WHERE tier = ?
                  AND (1.0 - surprise_score) > ?
                  AND update_count > 10
                """,
                (tier.value, config.demotion_threshold),
            )
            for (id,) in cursor.fetchall():
                if self.demote(id):
                    demotions += 1

        conn.close()
        return {"promotions": promotions, "demotions": demotions}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Count by tier
        cursor.execute("""
            SELECT tier, COUNT(*), AVG(importance), AVG(surprise_score), AVG(consolidation_score)
            FROM continuum_memory
            GROUP BY tier
        """)
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
        stats["total_memories"] = cursor.fetchone()[0]

        # Transition history
        cursor.execute("""
            SELECT from_tier, to_tier, COUNT(*)
            FROM tier_transitions
            GROUP BY from_tier, to_tier
        """)
        stats["transitions"] = [
            {"from": row[0], "to": row[1], "count": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()
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
