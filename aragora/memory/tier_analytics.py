"""
Memory tier analytics for ROI tracking.

Tracks whether tier promotions actually improve debate quality,
enabling data-driven memory management decisions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

from aragora.config import DB_TIMEOUT_SECONDS, resolve_db_path
from aragora.memory.tier_manager import MemoryTier

logger = logging.getLogger(__name__)


@dataclass
class TierStats:
    """Statistics for a single memory tier."""

    tier: MemoryTier
    entries: int = 0
    total_hits: int = 0
    avg_hits: float = 0.0
    total_quality_impact: float = 0.0
    avg_quality_impact: float = 0.0
    promotions_in: int = 0  # Entries promoted to this tier
    promotions_out: int = 0  # Entries promoted from this tier
    demotions_in: int = 0
    demotions_out: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "entries": self.entries,
            "total_hits": self.total_hits,
            "avg_hits": round(self.avg_hits, 2),
            "total_quality_impact": round(self.total_quality_impact, 4),
            "avg_quality_impact": round(self.avg_quality_impact, 4),
            "promotions_in": self.promotions_in,
            "promotions_out": self.promotions_out,
            "demotions_in": self.demotions_in,
            "demotions_out": self.demotions_out,
        }


@dataclass
class MemoryUsageEvent:
    """Record of memory being used in a debate."""

    memory_id: str
    tier: MemoryTier
    debate_id: str
    quality_before: float
    quality_after: float
    used_at: str = ""

    def __post_init__(self):
        if not self.used_at:
            self.used_at = datetime.utcnow().isoformat()

    @property
    def quality_impact(self) -> float:
        """Calculate quality impact of using this memory."""
        return self.quality_after - self.quality_before


@dataclass
class TierMovement:
    """Record of a memory moving between tiers."""

    memory_id: str
    from_tier: MemoryTier
    to_tier: MemoryTier
    reason: str  # 'promotion' or 'demotion'
    moved_at: str = ""

    def __post_init__(self):
        if not self.moved_at:
            self.moved_at = datetime.utcnow().isoformat()


@dataclass
class MemoryAnalytics:
    """Aggregated memory analytics."""

    tier_stats: dict[str, TierStats]
    promotion_effectiveness: float  # 0-1, how effective promotions are
    learning_velocity: float  # Rate of new patterns learned
    total_entries: int
    total_hits: int
    overall_quality_impact: float
    recommendations: list[str]
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        """Convert to API response format."""
        return {
            "tier_stats": {tier: stats.to_dict() for tier, stats in self.tier_stats.items()},
            "promotion_effectiveness": round(self.promotion_effectiveness, 3),
            "learning_velocity": round(self.learning_velocity, 3),
            "total_entries": self.total_entries,
            "total_hits": self.total_hits,
            "overall_quality_impact": round(self.overall_quality_impact, 4),
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


class TierAnalyticsTracker:
    """
    Tracks memory tier analytics for ROI analysis.

    Records:
    - Memory usage events (when memory is used in debates)
    - Tier movements (promotions/demotions)
    - Quality impact measurements

    Provides:
    - Per-tier statistics
    - Promotion effectiveness metrics
    - Recommendations for memory management

    Uses SQLiteStore internally for standardized schema management.
    """

    SCHEMA_NAME = "tier_analytics"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS memory_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            debate_id TEXT NOT NULL,
            quality_before REAL,
            quality_after REAL,
            used_at TEXT,
            UNIQUE(memory_id, debate_id)
        );

        CREATE TABLE IF NOT EXISTS tier_movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            from_tier TEXT NOT NULL,
            to_tier TEXT NOT NULL,
            reason TEXT NOT NULL,
            moved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS tier_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            tier TEXT NOT NULL,
            entry_count INTEGER,
            total_hits INTEGER,
            avg_quality_impact REAL,
            UNIQUE(snapshot_date, tier)
        );
    """

    def __init__(self, db_path: str = "memory_analytics.db"):
        """
        Initialize the analytics tracker.

        Args:
            db_path: Path to SQLite database file
        """
        from aragora.storage.base_store import SQLiteStore

        class _AnalyticsDB(SQLiteStore):
            SCHEMA_NAME = TierAnalyticsTracker.SCHEMA_NAME
            SCHEMA_VERSION = TierAnalyticsTracker.SCHEMA_VERSION
            INITIAL_SCHEMA = TierAnalyticsTracker.INITIAL_SCHEMA

        resolved_path = resolve_db_path(db_path)
        self.db_path = Path(resolved_path)
        self._db = _AnalyticsDB(resolved_path, timeout=DB_TIMEOUT_SECONDS)

    def record_usage(
        self,
        memory_id: str,
        tier: MemoryTier,
        debate_id: str,
        quality_before: float,
        quality_after: float,
    ):
        """
        Record a memory usage event.

        Args:
            memory_id: ID of the memory entry
            tier: Current tier of the memory
            debate_id: ID of the debate where memory was used
            quality_before: Quality score before using this memory
            quality_after: Quality score after using this memory
        """
        event = MemoryUsageEvent(
            memory_id=memory_id,
            tier=tier,
            debate_id=debate_id,
            quality_before=quality_before,
            quality_after=quality_after,
        )

        with self._db.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO memory_usage
                    (memory_id, tier, debate_id, quality_before, quality_after, used_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.memory_id,
                        event.tier.value,
                        event.debate_id,
                        event.quality_before,
                        event.quality_after,
                        event.used_at,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                logger.debug(f"Memory {memory_id} already recorded for debate {debate_id}")

    def record_tier_movement(
        self,
        memory_id: str,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        reason: str = "promotion",
    ):
        """
        Record a tier movement event.

        Args:
            memory_id: ID of the memory entry
            from_tier: Original tier
            to_tier: New tier
            reason: 'promotion' or 'demotion'
        """
        movement = TierMovement(
            memory_id=memory_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
        )

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tier_movements
                (memory_id, from_tier, to_tier, reason, moved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    movement.memory_id,
                    movement.from_tier.value,
                    movement.to_tier.value,
                    movement.reason,
                    movement.moved_at,
                ),
            )
            conn.commit()

    def get_tier_stats(
        self,
        tier: MemoryTier,
        days: int = 30,
    ) -> TierStats:
        """
        Get statistics for a specific tier.

        Args:
            tier: The memory tier to analyze
            days: Number of days to look back

        Returns:
            TierStats with aggregated metrics
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()

            # Get usage stats
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT memory_id) as entries,
                    COUNT(*) as total_hits,
                    AVG(quality_after - quality_before) as avg_impact
                FROM memory_usage
                WHERE tier = ? AND used_at > ?
                """,
                (tier.value, cutoff),
            )
            row = cursor.fetchone()
            entries = row[0] or 0
            total_hits = row[1] or 0
            avg_impact = row[2] or 0.0

            # Get movement stats
            cursor.execute(
                """
                SELECT
                    SUM(CASE WHEN to_tier = ? THEN 1 ELSE 0 END) as promotions_in,
                    SUM(CASE WHEN from_tier = ? AND reason = 'promotion' THEN 1 ELSE 0 END) as promotions_out,
                    SUM(CASE WHEN to_tier = ? AND reason = 'demotion' THEN 1 ELSE 0 END) as demotions_in,
                    SUM(CASE WHEN from_tier = ? AND reason = 'demotion' THEN 1 ELSE 0 END) as demotions_out
                FROM tier_movements
                WHERE moved_at > ?
                """,
                (tier.value, tier.value, tier.value, tier.value, cutoff),
            )
            mov_row = cursor.fetchone()

            return TierStats(
                tier=tier,
                entries=entries,
                total_hits=total_hits,
                avg_hits=total_hits / entries if entries > 0 else 0.0,
                total_quality_impact=avg_impact * total_hits,
                avg_quality_impact=avg_impact,
                promotions_in=mov_row[0] or 0,
                promotions_out=mov_row[1] or 0,
                demotions_in=mov_row[2] or 0,
                demotions_out=mov_row[3] or 0,
            )

    def get_promotion_effectiveness(self, days: int = 30) -> float:
        """
        Calculate how effective promotions are.

        Measures whether promoted memories perform better
        in their new tier.

        Returns:
            Effectiveness score 0-1
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()

            # Get promoted memories
            cursor.execute(
                """
                SELECT memory_id, to_tier, moved_at
                FROM tier_movements
                WHERE reason = 'promotion' AND moved_at > ?
                """,
                (cutoff,),
            )
            promotions = cursor.fetchall()

            if not promotions:
                return 0.5  # Neutral if no data

            effective_count = 0
            total_count = 0

            for memory_id, to_tier, moved_at in promotions:
                # Check if promoted memory was used with positive impact
                cursor.execute(
                    """
                    SELECT AVG(quality_after - quality_before)
                    FROM memory_usage
                    WHERE memory_id = ? AND used_at > ?
                    """,
                    (memory_id, moved_at),
                )
                result = cursor.fetchone()
                if result[0] is not None:
                    total_count += 1
                    if result[0] > 0:
                        effective_count += 1

            if total_count == 0:
                return 0.5

            return effective_count / total_count

    def get_learning_velocity(self, days: int = 7) -> float:
        """
        Calculate the rate of new patterns being learned.

        Returns:
            Average new entries per day
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(DISTINCT memory_id)
                FROM memory_usage
                WHERE used_at > ?
                """,
                (cutoff,),
            )
            new_entries = cursor.fetchone()[0] or 0

            return new_entries / days

    def generate_recommendations(self, analytics: dict) -> list[str]:
        """
        Generate recommendations based on analytics.

        Args:
            analytics: Dict with tier stats

        Returns:
            List of recommendation strings
        """
        recommendations = []

        tier_stats = analytics.get("tier_stats", {})

        # Check fast tier usage
        fast = tier_stats.get("fast", {})
        if fast.get("avg_hits", 0) < 1.0:
            recommendations.append(
                "Fast tier underutilized. Consider lowering promotion threshold."
            )

        # Check glacial tier ROI
        glacial = tier_stats.get("glacial", {})
        if glacial.get("entries", 0) > 1000 and glacial.get("avg_hits", 0) < 0.1:
            recommendations.append(
                "Glacial tier has many entries with low hit rate. " "Consider pruning old entries."
            )

        # Check promotion effectiveness
        effectiveness = analytics.get("promotion_effectiveness", 0.5)
        if effectiveness < 0.4:
            recommendations.append("Low promotion effectiveness. Review promotion criteria.")
        elif effectiveness > 0.8:
            recommendations.append(
                "High promotion effectiveness. Consider more aggressive promotion."
            )

        # Check learning velocity
        velocity = analytics.get("learning_velocity", 0)
        if velocity < 1.0:
            recommendations.append("Low learning velocity. System may not be capturing patterns.")

        # Check tier balance
        fast_entries = fast.get("entries", 0)
        medium = tier_stats.get("medium", {})
        medium_entries = medium.get("entries", 0)

        if fast_entries > medium_entries * 2:
            recommendations.append(
                "Fast tier may be overloaded. Consider stricter promotion criteria."
            )

        if not recommendations:
            recommendations.append("Memory tiers are balanced. No action needed.")

        return recommendations

    def get_analytics(self, days: int = 30) -> MemoryAnalytics:
        """
        Generate comprehensive analytics report.

        Args:
            days: Number of days to analyze

        Returns:
            MemoryAnalytics with all metrics
        """
        tier_stats = {}
        total_entries = 0
        total_hits = 0
        total_quality = 0.0

        for tier in MemoryTier:
            stats = self.get_tier_stats(tier, days)
            tier_stats[tier.value] = stats
            total_entries += stats.entries
            total_hits += stats.total_hits
            total_quality += stats.total_quality_impact

        promotion_effectiveness = self.get_promotion_effectiveness(days)
        learning_velocity = self.get_learning_velocity(min(days, 7))

        # Build analytics dict for recommendations
        analytics_dict = {
            "tier_stats": {k: v.to_dict() for k, v in tier_stats.items()},
            "promotion_effectiveness": promotion_effectiveness,
            "learning_velocity": learning_velocity,
        }

        recommendations = self.generate_recommendations(analytics_dict)

        return MemoryAnalytics(
            tier_stats=tier_stats,
            promotion_effectiveness=promotion_effectiveness,
            learning_velocity=learning_velocity,
            total_entries=total_entries,
            total_hits=total_hits,
            overall_quality_impact=total_quality,
            recommendations=recommendations,
        )

    def take_snapshot(self):
        """Take a daily snapshot for trend analysis."""
        today = datetime.utcnow().date().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()

            for tier in MemoryTier:
                stats = self.get_tier_stats(tier, days=1)

                try:
                    cursor.execute(
                        """
                        INSERT INTO tier_snapshots
                        (snapshot_date, tier, entry_count, total_hits, avg_quality_impact)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            today,
                            tier.value,
                            stats.entries,
                            stats.total_hits,
                            stats.avg_quality_impact,
                        ),
                    )
                except sqlite3.IntegrityError:
                    # Already have snapshot for today
                    pass

            conn.commit()
