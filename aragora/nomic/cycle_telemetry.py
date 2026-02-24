"""Cycle Telemetry Collector - production instrumentation for the Nomic Loop.

Records per-cycle metrics (timing, cost, quality, agents) in SQLite and
exposes aggregation queries for dashboards and stopping-rule evaluation.

Usage:
    from aragora.nomic.cycle_telemetry import CycleTelemetryCollector, CycleRecord

    collector = CycleTelemetryCollector()
    collector.record_cycle(CycleRecord(
        cycle_id="abc123",
        goal="Improve error handling",
        cycle_time_seconds=42.5,
        success=True,
        quality_delta=0.12,
        cost_usd=0.08,
        agents_used=["claude", "gemini"],
        debate_ids=["d_001"],
        branch_name="nomic/improve-errors",
        commit_sha="a1b2c3d",
    ))

    rate = collector.get_success_rate(window_days=7)
    avg_cost = collector.get_avg_cost_per_improvement()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    """Telemetry record for a single Nomic Loop cycle."""

    cycle_id: str = ""
    goal: str = ""
    cycle_time_seconds: float = 0.0
    success: bool = False
    quality_delta: float = 0.0
    cost_usd: float = 0.0
    agents_used: list[str] = field(default_factory=list)
    debate_ids: list[str] = field(default_factory=list)
    branch_name: str = ""
    commit_sha: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.cycle_id:
            self.cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleRecord:
        """Deserialize from a dictionary."""
        return cls(
            cycle_id=data.get("cycle_id", ""),
            goal=data.get("goal", ""),
            cycle_time_seconds=data.get("cycle_time_seconds", 0.0),
            success=bool(data.get("success", False)),
            quality_delta=data.get("quality_delta", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            agents_used=data.get("agents_used", []),
            debate_ids=data.get("debate_ids", []),
            branch_name=data.get("branch_name", ""),
            commit_sha=data.get("commit_sha", ""),
            timestamp=data.get("timestamp", 0.0),
        )


class CycleTelemetryCollector:
    """SQLite-backed telemetry collector for Nomic Loop cycles.

    Stores per-cycle records and provides aggregation queries
    for dashboards, stopping rules, and cross-cycle analysis.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the telemetry collector.

        Args:
            db_path: Path to SQLite database. Defaults to
                     {data_dir}/nomic_telemetry.db.
        """
        if db_path is None:
            try:
                from aragora.persistence.db_config import get_nomic_dir

                data_dir = get_nomic_dir()
                data_dir.mkdir(parents=True, exist_ok=True)
                db_path = str(data_dir / "nomic_telemetry.db")
            except ImportError:
                db_path = ":memory:"

        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the telemetry table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cycle_telemetry (
                    cycle_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    cycle_time_seconds REAL NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    quality_delta REAL NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    agents_used_json TEXT NOT NULL DEFAULT '[]',
                    debate_ids_json TEXT NOT NULL DEFAULT '[]',
                    branch_name TEXT NOT NULL DEFAULT '',
                    commit_sha TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
                ON cycle_telemetry(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_success
                ON cycle_telemetry(success)
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_cycle(self, record: CycleRecord) -> None:
        """Persist a cycle record to the database.

        Args:
            record: The CycleRecord to store.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO cycle_telemetry
                    (cycle_id, goal, cycle_time_seconds, success, quality_delta,
                     cost_usd, agents_used_json, debate_ids_json,
                     branch_name, commit_sha, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.cycle_id,
                    record.goal,
                    record.cycle_time_seconds,
                    1 if record.success else 0,
                    record.quality_delta,
                    record.cost_usd,
                    json.dumps(record.agents_used),
                    json.dumps(record.debate_ids),
                    record.branch_name,
                    record.commit_sha,
                    record.timestamp,
                ),
            )
            conn.commit()
            logger.info(
                "cycle_telemetry_recorded cycle_id=%s goal=%s success=%s cost=%.4f",
                record.cycle_id,
                record.goal[:60],
                record.success,
                record.cost_usd,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read / Query
    # ------------------------------------------------------------------

    def get_recent_cycles(self, n: int = 20) -> list[CycleRecord]:
        """Return the *n* most recent cycle records.

        Args:
            n: Maximum number of records to return.

        Returns:
            List of CycleRecord ordered by timestamp descending.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM cycle_telemetry ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_success_rate(self, window_days: int = 7) -> float:
        """Compute the success rate over a time window.

        Args:
            window_days: Number of days to look back.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 0.0 if no cycles exist in the window.
        """
        cutoff = time.time() - (window_days * 86400)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(success) AS successes
                FROM cycle_telemetry
                WHERE timestamp >= ?
                """,
                (cutoff,),
            ).fetchone()
            total = row[0] or 0
            successes = row[1] or 0
            return successes / total if total > 0 else 0.0
        finally:
            conn.close()

    def get_avg_cost_per_improvement(self) -> float:
        """Compute average cost for successful cycles.

        Returns:
            Average cost_usd for cycles where success=True.
            Returns 0.0 if no successful cycles exist.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT AVG(cost_usd) FROM cycle_telemetry
                WHERE success = 1 AND cost_usd > 0
                """
            ).fetchone()
            return row[0] or 0.0
        finally:
            conn.close()

    def get_top_goals_by_impact(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the goals with the highest quality_delta.

        Args:
            n: Maximum number of goals to return.

        Returns:
            List of dicts with goal, quality_delta, and cycle_id.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT cycle_id, goal, quality_delta, cost_usd, success
                FROM cycle_telemetry
                WHERE success = 1
                ORDER BY quality_delta DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
            return [
                {
                    "cycle_id": r["cycle_id"],
                    "goal": r["goal"],
                    "quality_delta": r["quality_delta"],
                    "cost_usd": r["cost_usd"],
                    "success": bool(r["success"]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_total_cost(self) -> float:
        """Return the cumulative cost across all recorded cycles."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM cycle_telemetry"
            ).fetchone()
            return row[0] or 0.0
        finally:
            conn.close()

    def get_cycle_count(self) -> int:
        """Return the total number of recorded cycles."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM cycle_telemetry"
            ).fetchone()
            return row[0] or 0
        finally:
            conn.close()

    def get_consecutive_failures(self) -> int:
        """Count consecutive failures from the most recent cycle backwards.

        Returns:
            Number of consecutive failed cycles at the tail. 0 if the
            most recent cycle was successful or no cycles exist.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT success FROM cycle_telemetry ORDER BY timestamp DESC"
            ).fetchall()
            count = 0
            for (success,) in rows:
                if success:
                    break
                count += 1
            return count
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, n: int | None = None) -> str:
        """Export recent cycle records as a JSON string.

        Args:
            n: Number of records to export. None exports all.

        Returns:
            JSON string of cycle records.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if n is not None:
                rows = conn.execute(
                    "SELECT * FROM cycle_telemetry ORDER BY timestamp DESC LIMIT ?",
                    (n,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cycle_telemetry ORDER BY timestamp DESC"
                ).fetchall()
            records = [self._row_to_record(r).to_dict() for r in rows]
            return json.dumps(records, indent=2)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CycleRecord:
        """Convert a database row to a CycleRecord."""
        return CycleRecord(
            cycle_id=row["cycle_id"],
            goal=row["goal"],
            cycle_time_seconds=row["cycle_time_seconds"],
            success=bool(row["success"]),
            quality_delta=row["quality_delta"],
            cost_usd=row["cost_usd"],
            agents_used=json.loads(row["agents_used_json"]),
            debate_ids=json.loads(row["debate_ids_json"]),
            branch_name=row["branch_name"],
            commit_sha=row["commit_sha"],
            timestamp=row["timestamp"],
        )


__all__ = [
    "CycleRecord",
    "CycleTelemetryCollector",
]
