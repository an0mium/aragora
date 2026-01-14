"""
SQLite persistence for scheduled debate outcomes.

Enables:
- Duplicate detection across scheduler runs
- Analytics on which topics generate good debates
- Historical trending topic tracking
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class ScheduledDebateRecord:
    """Record of a scheduled debate from a trending topic."""

    id: str
    topic_hash: str
    topic_text: str
    platform: str
    category: str
    volume: int
    debate_id: Optional[str]
    created_at: float
    consensus_reached: Optional[bool]
    confidence: Optional[float]
    rounds_used: int
    scheduler_run_id: str

    @property
    def hours_ago(self) -> float:
        """Hours since this debate was created."""
        return (time.time() - self.created_at) / 3600


class ScheduledDebateStore(SQLiteStore):
    """
    SQLite persistence for scheduled debate outcomes.

    Tracks all debates created by the PulseDebateScheduler for:
    - Duplicate detection (avoid re-debating recent topics)
    - Analytics on trending topic debate success rates
    - Historical tracking of scheduler activity

    Usage:
        store = ScheduledDebateStore("data/scheduled_debates.db")
        store.record_scheduled_debate(record)
        recent = store.get_recent_topics(hours=24)
    """

    SCHEMA_NAME = "scheduled_debates"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS scheduled_debates (
            id TEXT PRIMARY KEY,
            topic_hash TEXT NOT NULL,
            topic_text TEXT NOT NULL,
            platform TEXT NOT NULL,
            category TEXT,
            volume INTEGER DEFAULT 0,
            debate_id TEXT,
            created_at REAL NOT NULL,
            consensus_reached INTEGER,
            confidence REAL,
            rounds_used INTEGER DEFAULT 0,
            scheduler_run_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_scheduled_topic_hash
        ON scheduled_debates(topic_hash);

        CREATE INDEX IF NOT EXISTS idx_scheduled_created_at
        ON scheduled_debates(created_at);

        CREATE INDEX IF NOT EXISTS idx_scheduled_platform
        ON scheduled_debates(platform);

        CREATE INDEX IF NOT EXISTS idx_scheduled_category
        ON scheduled_debates(category);
    """

    def __init__(self, db_path: Union[str, Path], **kwargs):
        """Initialize the scheduled debate store."""
        super().__init__(db_path, **kwargs)
        logger.info(f"ScheduledDebateStore initialized: {db_path}")

    @staticmethod
    def hash_topic(topic_text: str) -> str:
        """Generate a hash for a topic for deduplication."""
        # Normalize: lowercase, strip, remove extra whitespace
        normalized = " ".join(topic_text.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def record_scheduled_debate(self, record: ScheduledDebateRecord) -> None:
        """
        Persist a scheduled debate record.

        Args:
            record: ScheduledDebateRecord to save
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_debates (
                    id, topic_hash, topic_text, platform, category, volume,
                    debate_id, created_at, consensus_reached, confidence,
                    rounds_used, scheduler_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.topic_hash,
                    record.topic_text,
                    record.platform,
                    record.category,
                    record.volume,
                    record.debate_id,
                    record.created_at,
                    (
                        1
                        if record.consensus_reached
                        else 0
                        if record.consensus_reached is not None
                        else None
                    ),
                    record.confidence,
                    record.rounds_used,
                    record.scheduler_run_id,
                ),
            )

        logger.debug(f"Recorded scheduled debate: {record.topic_text[:50]}...")

    def get_recent_topics(self, hours: int = 24) -> List[ScheduledDebateRecord]:
        """
        Get topics debated within the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of ScheduledDebateRecord objects
        """
        cutoff = time.time() - (hours * 3600)
        rows = self.fetch_all(
            """
            SELECT id, topic_hash, topic_text, platform, category, volume,
                   debate_id, created_at, consensus_reached, confidence,
                   rounds_used, scheduler_run_id
            FROM scheduled_debates
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (cutoff,),
        )

        return [self._row_to_record(row) for row in rows]

    def is_duplicate(self, topic_text: str, hours: int = 24) -> bool:
        """
        Check if a topic was recently debated.

        Args:
            topic_text: The topic to check
            hours: Deduplication window in hours

        Returns:
            True if the topic was debated within the window
        """
        topic_hash = self.hash_topic(topic_text)
        cutoff = time.time() - (hours * 3600)

        row = self.fetch_one(
            """
            SELECT 1 FROM scheduled_debates
            WHERE topic_hash = ? AND created_at >= ?
            LIMIT 1
            """,
            (topic_hash, cutoff),
        )

        return row is not None

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        platform: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[ScheduledDebateRecord]:
        """
        Get historical scheduled debates.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            platform: Optional platform filter
            category: Optional category filter

        Returns:
            List of ScheduledDebateRecord objects
        """
        query = "SELECT id, topic_hash, topic_text, platform, category, volume, "
        query += "debate_id, created_at, consensus_reached, confidence, "
        query += "rounds_used, scheduler_run_id FROM scheduled_debates WHERE 1=1 "
        params: list = []

        if platform:
            query += "AND platform = ? "
            params.append(platform)

        if category:
            query += "AND category = ? "
            params.append(category)

        query += "ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.fetch_all(query, tuple(params))
        return [self._row_to_record(row) for row in rows]

    def count_total(self) -> int:
        """Get total count of scheduled debates."""
        row = self.fetch_one("SELECT COUNT(*) FROM scheduled_debates")
        return row[0] if row else 0

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on scheduled debates.

        Returns:
            Dict with counts by platform, category, consensus rates, etc.
        """
        analytics: Dict[str, Any] = {}

        # Total count
        analytics["total"] = self.count_total()

        # Consensus rate
        row = self.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN consensus_reached = 1 THEN 1 ELSE 0 END) as consensus_count,
                AVG(confidence) as avg_confidence,
                AVG(rounds_used) as avg_rounds
            FROM scheduled_debates
            WHERE debate_id IS NOT NULL
            """
        )
        if row and row[0] > 0:
            analytics["completed_debates"] = row[0]
            analytics["consensus_rate"] = row[1] / row[0] if row[0] > 0 else 0
            analytics["avg_confidence"] = row[2] or 0
            analytics["avg_rounds"] = row[3] or 0

        # By platform
        rows = self.fetch_all(
            """
            SELECT platform, COUNT(*) as count,
                   SUM(CASE WHEN consensus_reached = 1 THEN 1 ELSE 0 END) as consensus_count
            FROM scheduled_debates
            GROUP BY platform
            """
        )
        analytics["by_platform"] = {
            row[0]: {"total": row[1], "consensus_count": row[2]} for row in rows
        }

        # By category
        rows = self.fetch_all(
            """
            SELECT category, COUNT(*) as count,
                   SUM(CASE WHEN consensus_reached = 1 THEN 1 ELSE 0 END) as consensus_count
            FROM scheduled_debates
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            """
        )
        analytics["by_category"] = {
            row[0]: {"total": row[1], "consensus_count": row[2]} for row in rows
        }

        # Recent activity (last 7 days by day)
        rows = self.fetch_all(
            """
            SELECT DATE(created_at, 'unixepoch') as day, COUNT(*) as count
            FROM scheduled_debates
            WHERE created_at >= ?
            GROUP BY day
            ORDER BY day DESC
            """,
            (time.time() - 7 * 24 * 3600,),
        )
        analytics["daily_counts"] = {row[0]: row[1] for row in rows}

        return analytics

    def cleanup_old(self, days: int = 30) -> int:
        """
        Remove records older than N days.

        Args:
            days: Age threshold in days

        Returns:
            Number of records deleted
        """
        cutoff = time.time() - (days * 24 * 3600)
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM scheduled_debates WHERE created_at < ?",
                (cutoff,),
            )
            removed = cursor.rowcount

        if removed > 0:
            logger.info(f"Cleaned up {removed} old scheduled debate records")

        return removed

    def _row_to_record(self, row: tuple) -> ScheduledDebateRecord:
        """Convert a database row to a ScheduledDebateRecord."""
        return ScheduledDebateRecord(
            id=row[0],
            topic_hash=row[1],
            topic_text=row[2],
            platform=row[3],
            category=row[4] or "",
            volume=row[5] or 0,
            debate_id=row[6],
            created_at=row[7],
            consensus_reached=bool(row[8]) if row[8] is not None else None,
            confidence=row[9],
            rounds_used=row[10] or 0,
            scheduler_run_id=row[11] or "",
        )


__all__ = ["ScheduledDebateStore", "ScheduledDebateRecord"]
