"""
Evolution Repository for tracking nomic loop history.

Provides storage and retrieval for:
- NomicRollback: When and why cycles were rolled back
- CycleEvolution: What changed in each cycle
- CycleFileChange: Which files were modified

Enables surfacing "what changed and why" for debugging and learning.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.persistence.models import (
    CycleEvolution,
    CycleFileChange,
    NomicRollback,
)

logger = logging.getLogger(__name__)

EVOLUTION_SCHEMA_VERSION = 1


class EvolutionRepository:
    """
    Repository for nomic loop evolution history.

    Tracks rollbacks, cycle outcomes, and file changes to provide
    visibility into what changed and why over time.

    Uses SQLiteStore internally for standardized schema management.

    Usage:
        repo = EvolutionRepository()

        # Record a rollback
        rollback = NomicRollback(
            id=str(uuid.uuid4()),
            loop_id="loop-123",
            cycle_number=5,
            phase="verify",
            reason="verify_failure",
            severity="high",
            error_message="Tests failed: 3 assertions",
        )
        repo.record_rollback(rollback)

        # Query rollbacks
        rollbacks = repo.get_rollbacks_for_loop("loop-123")
    """

    SCHEMA_NAME = "evolution_repository"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS nomic_rollbacks (
            id TEXT PRIMARY KEY,
            loop_id TEXT NOT NULL,
            cycle_number INTEGER NOT NULL,
            phase TEXT NOT NULL,
            reason TEXT NOT NULL,
            severity TEXT NOT NULL,
            rolled_back_commit TEXT,
            preserved_branch TEXT,
            files_affected TEXT,
            diff_summary TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_rollbacks_loop
        ON nomic_rollbacks(loop_id, cycle_number);

        CREATE INDEX IF NOT EXISTS idx_rollbacks_severity
        ON nomic_rollbacks(severity);

        CREATE TABLE IF NOT EXISTS cycle_evolutions (
            id TEXT PRIMARY KEY,
            loop_id TEXT NOT NULL,
            cycle_number INTEGER NOT NULL,
            debate_artifact_id TEXT,
            winning_proposal_summary TEXT,
            files_changed TEXT,
            git_commit TEXT,
            rollback_id TEXT REFERENCES nomic_rollbacks(id),
            created_at TEXT NOT NULL,
            UNIQUE(loop_id, cycle_number)
        );

        CREATE TABLE IF NOT EXISTS cycle_file_changes (
            loop_id TEXT NOT NULL,
            cycle_number INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            change_type TEXT NOT NULL,
            insertions INTEGER DEFAULT 0,
            deletions INTEGER DEFAULT 0,
            PRIMARY KEY (loop_id, cycle_number, file_path)
        );

        CREATE INDEX IF NOT EXISTS idx_file_changes_path
        ON cycle_file_changes(file_path);
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the evolution repository.

        Args:
            db_path: Path to database file (defaults to evolution.db)
        """
        from aragora.storage.base_store import SQLiteStore

        if db_path is None:
            db_path = get_db_path(DatabaseType.EVOLUTION)

        self.db_path = Path(db_path)

        # Create SQLiteStore-based database wrapper
        class _EvolutionDB(SQLiteStore):
            SCHEMA_NAME = EvolutionRepository.SCHEMA_NAME
            SCHEMA_VERSION = EvolutionRepository.SCHEMA_VERSION
            INITIAL_SCHEMA = EvolutionRepository.INITIAL_SCHEMA

        self._db = _EvolutionDB(str(db_path), timeout=30.0)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with guaranteed cleanup."""
        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            yield conn

    # =========================================================================
    # Rollback Operations
    # =========================================================================

    def record_rollback(self, rollback: NomicRollback) -> str:
        """Record a rollback event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO nomic_rollbacks
                (id, loop_id, cycle_number, phase, reason, severity,
                 rolled_back_commit, preserved_branch, files_affected,
                 diff_summary, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rollback.id,
                    rollback.loop_id,
                    rollback.cycle_number,
                    rollback.phase,
                    rollback.reason,
                    rollback.severity,
                    rollback.rolled_back_commit,
                    rollback.preserved_branch,
                    json.dumps(rollback.files_affected),
                    rollback.diff_summary,
                    rollback.error_message,
                    rollback.created_at.isoformat(),
                ),
            )

            conn.commit()
            logger.info(
                f"rollback_recorded id={rollback.id} loop={rollback.loop_id} "
                f"cycle={rollback.cycle_number} reason={rollback.reason}"
            )

            return rollback.id

    def get_rollback(self, rollback_id: str) -> Optional[NomicRollback]:
        """Get a rollback by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM nomic_rollbacks WHERE id = ?", (rollback_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_rollback(row)

    def get_rollbacks_for_loop(
        self,
        loop_id: str,
        severity: Optional[str] = None,
    ) -> list[NomicRollback]:
        """Get all rollbacks for a loop, optionally filtered by severity."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if severity:
                cursor.execute(
                    "SELECT * FROM nomic_rollbacks WHERE loop_id = ? AND severity = ? "
                    "ORDER BY cycle_number DESC",
                    (loop_id, severity),
                )
            else:
                cursor.execute(
                    "SELECT * FROM nomic_rollbacks WHERE loop_id = ? " "ORDER BY cycle_number DESC",
                    (loop_id,),
                )

            return [self._row_to_rollback(row) for row in cursor.fetchall()]

    def get_recent_rollbacks(
        self,
        limit: int = 10,
        severity: Optional[str] = None,
    ) -> list[NomicRollback]:
        """Get most recent rollbacks across all loops."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if severity:
                cursor.execute(
                    "SELECT * FROM nomic_rollbacks WHERE severity = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (severity, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM nomic_rollbacks ORDER BY created_at DESC LIMIT ?", (limit,)
                )

            return [self._row_to_rollback(row) for row in cursor.fetchall()]

    def _row_to_rollback(self, row: sqlite3.Row) -> NomicRollback:
        """Convert a database row to a NomicRollback."""
        return NomicRollback(
            id=row["id"],
            loop_id=row["loop_id"],
            cycle_number=row["cycle_number"],
            phase=row["phase"],
            reason=row["reason"],
            severity=row["severity"],
            rolled_back_commit=row["rolled_back_commit"],
            preserved_branch=row["preserved_branch"],
            files_affected=json.loads(row["files_affected"]) if row["files_affected"] else [],
            diff_summary=row["diff_summary"] or "",
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # =========================================================================
    # Evolution Operations
    # =========================================================================

    def record_evolution(self, evolution: CycleEvolution) -> str:
        """Record a cycle evolution event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO cycle_evolutions
                (id, loop_id, cycle_number, debate_artifact_id,
                 winning_proposal_summary, files_changed, git_commit,
                 rollback_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    evolution.id,
                    evolution.loop_id,
                    evolution.cycle_number,
                    evolution.debate_artifact_id,
                    evolution.winning_proposal_summary,
                    json.dumps(evolution.files_changed),
                    evolution.git_commit,
                    evolution.rollback_id,
                    evolution.created_at.isoformat(),
                ),
            )

            conn.commit()
            logger.debug(
                f"evolution_recorded id={evolution.id} loop={evolution.loop_id} "
                f"cycle={evolution.cycle_number}"
            )

            return evolution.id

    def get_evolution(self, loop_id: str, cycle_number: int) -> Optional[CycleEvolution]:
        """Get evolution record for a specific cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM cycle_evolutions WHERE loop_id = ? AND cycle_number = ?",
                (loop_id, cycle_number),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_evolution(row)

    def get_evolution_timeline(
        self,
        loop_id: str,
        limit: int = 50,
    ) -> list[CycleEvolution]:
        """Get evolution timeline for a loop."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM cycle_evolutions WHERE loop_id = ? "
                "ORDER BY cycle_number DESC LIMIT ?",
                (loop_id, limit),
            )

            return [self._row_to_evolution(row) for row in cursor.fetchall()]

    def _row_to_evolution(self, row: sqlite3.Row) -> CycleEvolution:
        """Convert a database row to a CycleEvolution."""
        return CycleEvolution(
            id=row["id"],
            loop_id=row["loop_id"],
            cycle_number=row["cycle_number"],
            debate_artifact_id=row["debate_artifact_id"],
            winning_proposal_summary=row["winning_proposal_summary"],
            files_changed=json.loads(row["files_changed"]) if row["files_changed"] else [],
            git_commit=row["git_commit"],
            rollback_id=row["rollback_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # =========================================================================
    # File Change Operations
    # =========================================================================

    def record_file_changes(
        self,
        loop_id: str,
        cycle_number: int,
        changes: list[CycleFileChange],
    ) -> None:
        """Record file changes for a cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for change in changes:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cycle_file_changes
                    (loop_id, cycle_number, file_path, change_type, insertions, deletions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        loop_id,
                        cycle_number,
                        change.file_path,
                        change.change_type,
                        change.insertions,
                        change.deletions,
                    ),
                )

            conn.commit()
            logger.debug(
                f"file_changes_recorded loop={loop_id} cycle={cycle_number} "
                f"count={len(changes)}"
            )

    def get_cycles_touching_file(
        self,
        file_path: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get cycles that modified a specific file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT fc.*, ce.winning_proposal_summary, ce.git_commit
                FROM cycle_file_changes fc
                LEFT JOIN cycle_evolutions ce
                    ON fc.loop_id = ce.loop_id AND fc.cycle_number = ce.cycle_number
                WHERE fc.file_path = ?
                ORDER BY fc.loop_id DESC, fc.cycle_number DESC
                LIMIT ?
            """,
                (file_path, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "loop_id": row["loop_id"],
                        "cycle_number": row["cycle_number"],
                        "change_type": row["change_type"],
                        "insertions": row["insertions"],
                        "deletions": row["deletions"],
                        "summary": row["winning_proposal_summary"],
                        "commit": row["git_commit"],
                    }
                )

            return results

    def get_file_changes_for_cycle(
        self,
        loop_id: str,
        cycle_number: int,
    ) -> list[CycleFileChange]:
        """Get all file changes for a specific cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM cycle_file_changes "
                "WHERE loop_id = ? AND cycle_number = ? "
                "ORDER BY file_path",
                (loop_id, cycle_number),
            )

            return [
                CycleFileChange(
                    loop_id=row["loop_id"],
                    cycle_number=row["cycle_number"],
                    file_path=row["file_path"],
                    change_type=row["change_type"],
                    insertions=row["insertions"],
                    deletions=row["deletions"],
                )
                for row in cursor.fetchall()
            ]

    # =========================================================================
    # Summary Operations
    # =========================================================================

    def get_loop_summary(self, loop_id: str) -> dict:
        """Get summary statistics for a loop."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count rollbacks by severity
            cursor.execute(
                """
                SELECT severity, COUNT(*) as count
                FROM nomic_rollbacks
                WHERE loop_id = ?
                GROUP BY severity
            """,
                (loop_id,),
            )
            rollback_counts = {row["severity"]: row["count"] for row in cursor.fetchall()}

            # Count cycles
            cursor.execute(
                """
                SELECT COUNT(*) as total, MAX(cycle_number) as latest
                FROM cycle_evolutions
                WHERE loop_id = ?
            """,
                (loop_id,),
            )
            row = cursor.fetchone()
            total_cycles = row["total"] if row else 0
            latest_cycle = row["latest"] if row else 0

            # Count successful commits
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM cycle_evolutions
                WHERE loop_id = ? AND git_commit IS NOT NULL AND rollback_id IS NULL
            """,
                (loop_id,),
            )
            row = cursor.fetchone()
            successful_commits = row["count"] if row else 0

            return {
                "loop_id": loop_id,
                "total_cycles": total_cycles,
                "latest_cycle": latest_cycle,
                "successful_commits": successful_commits,
                "rollback_counts": rollback_counts,
                "total_rollbacks": sum(rollback_counts.values()),
            }


__all__ = ["EvolutionRepository"]
