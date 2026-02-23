"""SQLite-backed persistent store for strategic assessments.

Stores StrategicAssessment objects across self-improvement sessions so the
MetaPlanner can learn from past scans and prioritize recurring findings.

Uses the same PlanStore pattern (aragora/pipeline/plan_store.py) for SQLite
setup, WAL mode, and connection management.

Usage:
    store = StrategicMemoryStore()
    store.save(assessment)
    recent = store.get_latest(limit=3)
    recurring = store.get_recurring_findings(min_occurrences=2)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from aragora.nomic.strategic_scanner import StrategicAssessment, StrategicFinding

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = os.environ.get("ARAGORA_DATA_DIR", str(Path.home() / ".aragora"))
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "strategic_memory.db")


def _get_db_path() -> str:
    """Resolve the strategic memory database path."""
    try:
        from aragora.config import resolve_db_path

        return resolve_db_path("strategic_memory.db")
    except ImportError:
        return _DEFAULT_DB_PATH


class StrategicMemoryStore:
    """SQLite store for strategic assessments across sessions.

    Thread-safe via SQLite WAL mode. Each method creates its own
    connection to support concurrent access.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _get_db_path()
        self._ensure_dir()
        self._ensure_table()

    def _ensure_dir(self) -> None:
        """Create parent directory if needed."""
        parent = Path(self._db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create the assessments table if it does not exist."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategic_assessments (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    focus_areas_json TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assessments_timestamp
                ON strategic_assessments(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assessments_objective
                ON strategic_assessments(objective)
            """)
            conn.commit()
        finally:
            conn.close()

    def save(self, assessment: StrategicAssessment) -> str:
        """Save a strategic assessment.

        Args:
            assessment: The assessment to persist.

        Returns:
            The generated assessment ID.
        """
        assessment_id = f"sa-{uuid.uuid4().hex[:12]}"
        findings_data = [
            {
                "category": f.category,
                "severity": f.severity,
                "file_path": f.file_path,
                "description": f.description,
                "evidence": f.evidence,
                "suggested_action": f.suggested_action,
                "track": f.track,
            }
            for f in assessment.findings
        ]

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO strategic_assessments
                    (id, objective, timestamp, metrics_json, focus_areas_json,
                     findings_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    assessment_id,
                    assessment.objective,
                    assessment.timestamp or time.time(),
                    json.dumps(assessment.metrics),
                    json.dumps(assessment.focus_areas),
                    json.dumps(findings_data),
                ),
            )
            conn.commit()
            logger.debug(
                "Strategic assessment saved: %s (%d findings)",
                assessment_id,
                len(assessment.findings),
            )
            return assessment_id
        finally:
            conn.close()

    def get_latest(self, limit: int = 3) -> list[StrategicAssessment]:
        """Get the most recent assessments.

        Args:
            limit: Maximum number of assessments to return.

        Returns:
            List of StrategicAssessment ordered by most recent first.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT objective, timestamp, metrics_json, focus_areas_json,
                       findings_json
                FROM strategic_assessments
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._row_to_assessment(row) for row in rows]
        finally:
            conn.close()

    def get_for_objective(
        self, objective: str, limit: int = 3
    ) -> list[StrategicAssessment]:
        """Get assessments matching an objective (substring match).

        Args:
            objective: Objective text to match against.
            limit: Maximum number of assessments to return.

        Returns:
            List of matching StrategicAssessment.
        """
        conn = self._connect()
        try:
            # Use LIKE for substring matching on objective
            rows = conn.execute(
                """
                SELECT objective, timestamp, metrics_json, focus_areas_json,
                       findings_json
                FROM strategic_assessments
                WHERE objective LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (f"%{objective}%", limit),
            ).fetchall()
            return [self._row_to_assessment(row) for row in rows]
        finally:
            conn.close()

    def get_recurring_findings(
        self, min_occurrences: int = 2
    ) -> list[StrategicFinding]:
        """Find findings that appear across multiple assessments.

        Groups findings by (category, file_path) and returns those that
        appear in at least ``min_occurrences`` separate assessments.

        Args:
            min_occurrences: Minimum number of assessments a finding must
                appear in to be considered recurring.

        Returns:
            List of recurring StrategicFinding (one per group, most recent).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT findings_json, timestamp
                FROM strategic_assessments
                ORDER BY timestamp DESC
                """,
            ).fetchall()
        finally:
            conn.close()

        # Count occurrences by (category, file_path)
        occurrence_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            findings_data = json.loads(row["findings_json"])
            for fd in findings_data:
                key = (fd["category"], fd["file_path"])
                if key not in occurrence_map:
                    occurrence_map[key] = []
                occurrence_map[key].append(fd)

        recurring: list[StrategicFinding] = []
        for (_cat, _path), occurrences in occurrence_map.items():
            if len(occurrences) >= min_occurrences:
                # Return the most recent instance
                most_recent = occurrences[0]
                recurring.append(
                    StrategicFinding(
                        category=most_recent["category"],
                        severity=most_recent["severity"],
                        file_path=most_recent["file_path"],
                        description=most_recent["description"],
                        evidence=most_recent["evidence"],
                        suggested_action=most_recent["suggested_action"],
                        track=most_recent["track"],
                    )
                )

        return recurring

    @staticmethod
    def _row_to_assessment(row: sqlite3.Row) -> StrategicAssessment:
        """Convert a database row to a StrategicAssessment."""
        findings_data = json.loads(row["findings_json"])
        findings = [
            StrategicFinding(
                category=fd["category"],
                severity=fd["severity"],
                file_path=fd["file_path"],
                description=fd["description"],
                evidence=fd["evidence"],
                suggested_action=fd["suggested_action"],
                track=fd["track"],
            )
            for fd in findings_data
        ]
        return StrategicAssessment(
            findings=findings,
            metrics=json.loads(row["metrics_json"]),
            focus_areas=json.loads(row["focus_areas_json"]),
            objective=row["objective"],
            timestamp=row["timestamp"],
        )
