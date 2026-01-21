"""Cycle Learning Store - persists Nomic cycle records for cross-cycle learning.

Provides SQLite-backed storage for NomicCycleRecord with:
- Save/load cycle records
- Query recent cycles
- Topic-based similarity search
- Agent trajectory tracking
- Pattern aggregation
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.nomic.cycle_record import NomicCycleRecord

logger = logging.getLogger(__name__)


class CycleLearningStore:
    """SQLite-backed storage for Nomic cycle records.

    Persists cycle data for cross-cycle learning, enabling:
    - Historical context injection
    - Agent performance trajectory
    - Pattern success tracking
    - Topic similarity queries

    Example:
        store = CycleLearningStore()

        # Save a cycle
        record = NomicCycleRecord(cycle_id="abc", started_at=time.time())
        record.mark_complete(success=True)
        store.save_cycle(record)

        # Query recent cycles
        recent = store.get_recent_cycles(5)
        for r in recent:
            print(f"Cycle {r.cycle_id}: {r.success}")
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the store.

        Args:
            db_path: Path to SQLite database. Defaults to .nomic/cycles.db
        """
        if db_path is None:
            data_dir = os.environ.get("ARAGORA_DATA_DIR", ".nomic")
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            db_path = str(Path(data_dir) / "cycles.db")

        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    duration_seconds REAL,
                    success INTEGER DEFAULT 0,
                    topics_json TEXT,
                    data_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cycles_started
                ON cycles(started_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cycles_success
                ON cycles(success)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_cycle(self, record: NomicCycleRecord) -> None:
        """Save a cycle record.

        Args:
            record: The cycle record to save
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Serialize topics for search
            topics_json = json.dumps(record.topics_debated)
            data_json = json.dumps(record.to_dict())

            conn.execute(
                """
                INSERT OR REPLACE INTO cycles
                (cycle_id, started_at, completed_at, duration_seconds, success, topics_json, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.cycle_id,
                    record.started_at,
                    record.completed_at,
                    record.duration_seconds,
                    1 if record.success else 0,
                    topics_json,
                    data_json,
                ),
            )
            conn.commit()
            logger.debug(f"cycle_saved cycle_id={record.cycle_id} success={record.success}")
        finally:
            conn.close()

    def load_cycle(self, cycle_id: str) -> Optional[NomicCycleRecord]:
        """Load a specific cycle by ID.

        Args:
            cycle_id: The cycle ID to load

        Returns:
            NomicCycleRecord or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT data_json FROM cycles WHERE cycle_id = ?",
                (cycle_id,),
            )
            row = cursor.fetchone()
            if row:
                return NomicCycleRecord.from_dict(json.loads(row[0]))
            return None
        finally:
            conn.close()

    def get_recent_cycles(self, n: int = 10) -> List[NomicCycleRecord]:
        """Get the N most recent cycles.

        Args:
            n: Number of cycles to retrieve

        Returns:
            List of cycle records, most recent first
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT data_json FROM cycles ORDER BY started_at DESC LIMIT ?",
                (n,),
            )
            return [NomicCycleRecord.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_successful_cycles(self, n: int = 10) -> List[NomicCycleRecord]:
        """Get the N most recent successful cycles.

        Args:
            n: Number of cycles to retrieve

        Returns:
            List of successful cycle records
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT data_json FROM cycles
                WHERE success = 1
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (n,),
            )
            return [NomicCycleRecord.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
        finally:
            conn.close()

    def query_by_topic(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[NomicCycleRecord]:
        """Find cycles that addressed similar topics.

        Uses simple substring matching on topics.
        For semantic similarity, use with embedding-based search.

        Args:
            topic: Topic to search for
            limit: Maximum results to return

        Returns:
            List of matching cycle records
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Simple substring match on topics
            cursor = conn.execute(
                """
                SELECT data_json FROM cycles
                WHERE topics_json LIKE ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (f"%{topic}%", limit),
            )
            return [NomicCycleRecord.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_agent_trajectory(
        self,
        agent_name: str,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get performance trajectory for an agent across cycles.

        Args:
            agent_name: Name of the agent
            n: Number of recent cycles to analyze

        Returns:
            List of performance snapshots for the agent
        """
        cycles = self.get_recent_cycles(n)
        trajectory: List[Dict[str, Any]] = []

        for cycle in cycles:
            if agent_name in cycle.agent_contributions:
                contrib = cycle.agent_contributions[agent_name]
                trajectory.append(
                    {
                        "cycle_id": cycle.cycle_id,
                        "timestamp": cycle.started_at,
                        "proposals_made": contrib.proposals_made,
                        "proposals_accepted": contrib.proposals_accepted,
                        "acceptance_rate": (
                            contrib.proposals_accepted / contrib.proposals_made
                            if contrib.proposals_made > 0
                            else 0.0
                        ),
                        "critiques_given": contrib.critiques_given,
                        "critiques_valuable": contrib.critiques_valuable,
                        "quality_score": contrib.quality_score,
                        "cycle_success": cycle.success,
                    }
                )

        return trajectory

    def get_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate pattern success statistics across all cycles.

        Returns:
            Dict mapping pattern types to success rates and counts
        """
        cycles = self.get_recent_cycles(100)
        stats: Dict[str, Dict[str, Any]] = {}

        for cycle in cycles:
            for reinforcement in cycle.pattern_reinforcements:
                pattern = reinforcement.pattern_type
                if pattern not in stats:
                    stats[pattern] = {
                        "success_count": 0,
                        "failure_count": 0,
                        "total_confidence": 0.0,
                        "examples": [],
                    }

                if reinforcement.success:
                    stats[pattern]["success_count"] += 1
                else:
                    stats[pattern]["failure_count"] += 1

                stats[pattern]["total_confidence"] += reinforcement.confidence

                # Keep a few examples
                if len(stats[pattern]["examples"]) < 3:
                    stats[pattern]["examples"].append(reinforcement.description)

        # Calculate success rates
        for pattern, data in stats.items():
            total = data["success_count"] + data["failure_count"]
            data["success_rate"] = data["success_count"] / total if total > 0 else 0.0
            data["avg_confidence"] = data["total_confidence"] / total if total > 0 else 0.0

        return stats

    def get_surprise_summary(self, n: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Get summary of surprise events grouped by phase.

        Args:
            n: Number of recent cycles to analyze

        Returns:
            Dict mapping phases to lists of surprise events
        """
        cycles = self.get_recent_cycles(n)
        summary: Dict[str, List[Dict[str, Any]]] = {}

        for cycle in cycles:
            for surprise in cycle.surprise_events:
                if surprise.phase not in summary:
                    summary[surprise.phase] = []

                summary[surprise.phase].append(
                    {
                        "cycle_id": cycle.cycle_id,
                        "description": surprise.description,
                        "expected": surprise.expected,
                        "actual": surprise.actual,
                        "impact": surprise.impact,
                    }
                )

        return summary

    def cleanup_old_cycles(self, keep_count: int = 100) -> int:
        """Remove old cycles, keeping the most recent ones.

        Args:
            keep_count: Number of cycles to retain

        Returns:
            Number of cycles deleted
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Find the cutoff
            cursor = conn.execute(
                """
                SELECT started_at FROM cycles
                ORDER BY started_at DESC
                LIMIT 1 OFFSET ?
                """,
                (keep_count - 1,),
            )
            row = cursor.fetchone()
            if not row:
                return 0

            cutoff = row[0]

            # Delete older cycles
            cursor = conn.execute(
                "DELETE FROM cycles WHERE started_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()

            if deleted > 0:
                logger.info(f"cycles_cleaned deleted={deleted} retained={keep_count}")

            return deleted
        finally:
            conn.close()

    def get_cycle_count(self) -> int:
        """Get total number of stored cycles."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM cycles")
            return cursor.fetchone()[0]
        finally:
            conn.close()


# Module-level singleton
_cycle_store: Optional[CycleLearningStore] = None


def get_cycle_store() -> CycleLearningStore:
    """Get or create the singleton CycleLearningStore instance."""
    global _cycle_store
    if _cycle_store is None:
        _cycle_store = CycleLearningStore()
    return _cycle_store


def save_cycle(record: NomicCycleRecord) -> None:
    """Convenience function to save a cycle."""
    get_cycle_store().save_cycle(record)


def get_recent_cycles(n: int = 10) -> List[NomicCycleRecord]:
    """Convenience function to get recent cycles."""
    return get_cycle_store().get_recent_cycles(n)
